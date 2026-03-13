from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ml_models.models import build_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _get_device() -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))

    # MAPE: skip near-zero true values to avoid division instability
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(residuals[mask] / y_true[mask])) * 100) \
        if mask.sum() > 0 else float("nan")

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "rmse": round(rmse, 6),
        "mae":  round(mae,  6),
        "mape": round(mape, 4),
        "r2":   round(r2,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:

    def __init__(
        self,
        patience:  int   = 15,
        min_delta: float = 1e-5,
        mode:      str   = "min",
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best_score: Optional[float] = None
        self.best_weights: Optional[dict] = None
        self.stop = False

    def step(self, score: float, model: nn.Module) -> bool:

        improved = (
            self.best_score is None
            or (self.mode == "min" and score < self.best_score - self.min_delta)
            or (self.mode == "max" and score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score   = score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter      = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop

    def restore_best(self, model: nn.Module) -> None:
        """Load the best saved weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(
                f"Restored best weights (val_loss={self.best_score:.6f})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Master ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
    ) -> None:
        self.cfg      = _load_config(config_path)
        self.root     = Path(config_path).parent.parent
        self.proc_dir = self.root / self.cfg["paths"]["data_processed"]
        self.out_dir  = _ensure_dir(
            self.root / self.cfg["paths"]["outputs_models"]
        )
        self.device   = _get_device()
        self.seed     = int(self.cfg.get("project", {}).get("version", "42")
                            .replace(".", "")) if False else 42
        _set_seed(self.seed)
        logger.info(f"ModelTrainer | device: {self.device}")

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _make_loaders(
        self,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader,
               np.ndarray, np.ndarray, np.ndarray]:

        X_train = np.load(self.proc_dir / "X_train.npy")
        y_train = np.load(self.proc_dir / "y_train.npy")
        X_val   = np.load(self.proc_dir / "X_val.npy")
        y_val   = np.load(self.proc_dir / "y_val.npy")
        X_test  = np.load(self.proc_dir / "X_test.npy")
        y_test  = np.load(self.proc_dir / "y_test.npy")

        def _to_loader(X, y, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1),
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )

        return (
            _to_loader(X_train, y_train, shuffle=True),
            _to_loader(X_val,   y_val,   shuffle=False),
            _to_loader(X_test,  y_test,  shuffle=False),
            y_train, y_val, y_test,
        )

    def _input_size(self) -> int:
        """Infer input feature size from saved X_train array."""
        X = np.load(self.proc_dir / "X_train.npy")
        return X.shape[2]   # (n_samples, seq_len, n_features)

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
    ) -> tuple[float, float]:

        model.train()
        total_loss = 0.0
        total_mae  = 0.0
        n_batches  = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()

            total_loss += loss.item()
            total_mae  += torch.mean(torch.abs(preds - y_batch)).item()
            n_batches  += 1

        return total_loss / n_batches, total_mae / n_batches

    @torch.no_grad()
    def _eval_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float, np.ndarray]:

        model.eval()
        total_loss = 0.0
        total_mae  = 0.0
        n_batches  = 0
        all_preds  = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            preds = model(X_batch)
            loss  = criterion(preds, y_batch)

            total_loss += loss.item()
            total_mae  += torch.mean(torch.abs(preds - y_batch)).item()
            n_batches  += 1
            all_preds.append(preds.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0).flatten()
        return total_loss / n_batches, total_mae / n_batches, predictions

    # ── Main train method ─────────────────────────────────────────────────────

    def train(
        self,
        model_type: str = "lstm",
        epochs: Optional[int] = None,
    ) -> tuple[nn.Module, pd.DataFrame]:

        m_cfg       = self.cfg["ml_models"][model_type]
        batch_size  = m_cfg["batch_size"]
        max_epochs  = epochs or m_cfg["epochs"]
        patience    = m_cfg["patience"]
        lr          = m_cfg["learning_rate"]

        logger.info("=" * 60)
        logger.info(f"Training {model_type.upper()} | "
                    f"epochs={max_epochs} | bs={batch_size} | lr={lr}")
        logger.info("=" * 60)

        # ── Build data loaders ────────────────────────────────────────────────
        (train_loader, val_loader, _,
         y_train_np, y_val_np, _) = self._make_loaders(batch_size)

        # ── Build model ───────────────────────────────────────────────────────
        input_size = self._input_size()
        model = build_model(
            model_type, input_size, self.cfg["ml_models"]
        ).to(self.device)
        logger.info(
            f"Model: {model.__class__.__name__} | "
            f"params: {model.count_parameters():,}"
        )

        # ── Optimiser + scheduler + loss ──────────────────────────────────────
        optimiser = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
        )
        criterion     = nn.HuberLoss(delta=0.1)   # robust to outliers
        early_stopper = EarlyStopping(patience=patience, mode="min")

        # ── Training loop ─────────────────────────────────────────────────────
        history_rows = []
        t_start      = time.time()

        for epoch in range(1, max_epochs + 1):
            train_loss, train_mae = self._train_epoch(
                model, train_loader, criterion, optimiser
            )
            val_loss, val_mae, _ = self._eval_epoch(
                model, val_loader, criterion
            )
            current_lr = optimiser.param_groups[0]["lr"]

            scheduler.step(val_loss)

            history_rows.append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "train_mae":  round(train_mae,  6),
                "val_mae":    round(val_mae,    6),
                "lr":         current_lr,
            })

            # Log every 10 epochs or at end
            if epoch % 10 == 0 or epoch == max_epochs:
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch:4d}/{max_epochs} | "
                    f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                    f"val_mae={val_mae:.5f}  lr={current_lr:.2e}  "
                    f"({elapsed:.0f}s elapsed)"
                )

            # Early stopping
            if early_stopper.step(val_loss, model):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val_loss={early_stopper.best_score:.6f})"
                )
                break

        # Restore best weights
        early_stopper.restore_best(model)

        history = pd.DataFrame(history_rows)
        logger.info(
            f"Training complete | "
            f"{len(history)} epochs | "
            f"best val_loss={early_stopper.best_score:.6f}"
        )
        return model, history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model:      nn.Module,
        model_type: str,
        scaler=None,
    ) -> dict:

        m_cfg      = self.cfg["ml_models"][model_type]
        batch_size = m_cfg["batch_size"]

        (_, _, test_loader,
         _, _, y_test_np) = self._make_loaders(batch_size)
        criterion = nn.HuberLoss(delta=0.1)

        _, _, preds_scaled = self._eval_epoch(model, test_loader, criterion)

        metrics_scaled = compute_metrics(y_test_np, preds_scaled)
        logger.info(f"Test metrics (scaled): {metrics_scaled}")

        result = {
            "test_metrics_scaled": metrics_scaled,
            "y_true_scaled":       y_test_np,
            "y_pred_scaled":       preds_scaled,
        }

        if scaler is not None:
            y_true_phys = scaler.inverse_transform_target(y_test_np)
            y_pred_phys = scaler.inverse_transform_target(preds_scaled)
            metrics_phys = compute_metrics(y_true_phys, y_pred_phys)
            logger.info(f"Test metrics (M km²): {metrics_phys}")
            result["test_metrics_physical"] = metrics_phys
            result["y_true_physical"]       = y_true_phys
            result["y_pred_physical"]       = y_pred_phys

        return result

    # ── Future forecasting ────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast_future(
        self,
        model:   nn.Module,
        scaler,
        horizon: int = 24,
    ) -> pd.DataFrame:

        model.eval()
        X_test = np.load(self.proc_dir / "X_test.npy")

        # Seed with the very last window in the test set
        seed_window = torch.tensor(
            X_test[-1:], dtype=torch.float32
        ).to(self.device)   # shape: (1, seq_len, n_features)

        # Index of the target feature in the feature vector
        with open(self.proc_dir / "feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        target_idx = feature_cols.index("sea_ice_extent_mkm2")

        # ── MC-Dropout for uncertainty estimation ─────────────────────────────
        def _enable_dropout(m: nn.Module) -> None:
            """Set Dropout layers to train mode (enables MC-Dropout)."""
            for module in m.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

        n_mc     = 30    # Monte Carlo samples
        all_runs = np.zeros((n_mc, horizon))

        for run in range(n_mc):
            _enable_dropout(model)
            window = seed_window.clone()

            for step in range(horizon):
                pred_scaled = model(window).cpu().numpy().flatten()[0]
                all_runs[run, step] = pred_scaled

                # Slide window: drop first step, append new prediction
                new_step        = window[:, -1:, :].clone()
                new_step[0, 0, target_idx] = float(pred_scaled)
                window = torch.cat([window[:, 1:, :], new_step], dim=1)

        model.eval()   # reset to eval (disables dropout again)

        # Aggregate MC samples
        mean_preds = all_runs.mean(axis=0)
        ci_lo      = np.percentile(all_runs, 2.5,  axis=0)
        ci_hi      = np.percentile(all_runs, 97.5, axis=0)

        # Inverse transform to physical units
        mean_phys  = scaler.inverse_transform_target(mean_preds)
        lo_phys    = scaler.inverse_transform_target(ci_lo)
        hi_phys    = scaler.inverse_transform_target(ci_hi)

        forecast_df = pd.DataFrame({
            "step":              np.arange(1, horizon + 1),
            "predicted_mkm2":   np.round(mean_phys, 4),
            "lower_ci_95":      np.round(lo_phys,   4),
            "upper_ci_95":      np.round(hi_phys,   4),
        })

        logger.info(
            f"Forecast generated: {horizon} months ahead | "
            f"range [{mean_phys.min():.2f}, {mean_phys.max():.2f}] M km²"
        )
        return forecast_df

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(
        self,
        model:      nn.Module,
        model_type: str,
        history:    pd.DataFrame,
        eval_results: dict,
    ) -> Path:

        prefix      = self.out_dir / model_type
        weights_path = Path(f"{prefix}_weights.pt")

        torch.save(model.state_dict(), weights_path)
        history.to_csv(f"{prefix}_history.csv", index=False)

        metadata = {
            "model_type":   model_type,
            "architecture": model.__class__.__name__,
            "n_parameters": model.count_parameters(),
            "input_size":   model.input_size,
            "hyperparams":  self.cfg["ml_models"][model_type],
            "test_metrics_scaled":   eval_results.get("test_metrics_scaled"),
            "test_metrics_physical": eval_results.get("test_metrics_physical"),
        }
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved → {weights_path}")
        return weights_path

    def load(self, model_type: str) -> nn.Module:

        weights_path = self.out_dir / f"{model_type}_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No saved weights at {weights_path}. "
                "Train the model first with ModelTrainer.run()."
            )
        input_size = self._input_size()
        model = build_model(
            model_type, input_size, self.cfg["ml_models"]
        ).to(self.device)
        model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        model.eval()
        logger.info(f"Model loaded from {weights_path}")
        return model

    # ── Main run (full pipeline) ───────────────────────────────────────────────

    def run(
        self,
        model_type: str = "lstm",
        epochs: Optional[int] = None,
    ) -> dict:

        logger.info("=" * 60)
        logger.info(f"ArcticVision ML Pipeline START  [{model_type.upper()}]")
        logger.info("=" * 60)

        # Load scaler for inverse transformation
        scaler_path = self.proc_dir / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # ── Train ─────────────────────────────────────────────────────────────
        model, history = self.train(model_type, epochs=epochs)

        # ── Evaluate ──────────────────────────────────────────────────────────
        eval_results = self.evaluate(model, model_type, scaler=scaler)

        # ── Future forecast ───────────────────────────────────────────────────
        forecast = pd.DataFrame()
        if scaler is not None:
            forecast = self.forecast_future(
                model, scaler,
                horizon=self.cfg["ml_models"]["forecast_horizon"],
            )
            forecast.to_csv(
                self.out_dir / f"{model_type}_forecast.csv", index=False
            )

        # ── Save ──────────────────────────────────────────────────────────────
        weights_path = self.save(model, model_type, history, eval_results)

        # ── Summary ───────────────────────────────────────────────────────────
        self._log_summary(model_type, history, eval_results)

        return {
            "model":        model,
            "history":      history,
            "eval":         eval_results,
            "forecast":     forecast,
            "weights_path": weights_path,
        }

    # ── Logging helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(
        model_type: str,
        history:    pd.DataFrame,
        eval_results: dict,
    ) -> None:
        logger.info("=" * 60)
        logger.info(f"ML PIPELINE SUMMARY  [{model_type.upper()}]")
        logger.info("=" * 60)
        logger.info(f"  Epochs trained      : {len(history)}")
        logger.info(f"  Best val_loss       : {history['val_loss'].min():.6f}")
        logger.info(
            f"  Final lr            : {history['lr'].iloc[-1]:.2e}"
        )

        scaled = eval_results.get("test_metrics_scaled", {})
        logger.info(
            f"  Test RMSE (scaled)  : {scaled.get('rmse', 'N/A')}"
        )
        logger.info(
            f"  Test MAE  (scaled)  : {scaled.get('mae',  'N/A')}"
        )
        logger.info(
            f"  Test R²   (scaled)  : {scaled.get('r2',   'N/A')}"
        )

        if "test_metrics_physical" in eval_results:
            phys = eval_results["test_metrics_physical"]
            logger.info(
                f"  Test RMSE (M km²)   : {phys.get('rmse', 'N/A')}"
            )
            logger.info(
                f"  Test MAE  (M km²)   : {phys.get('mae',  'N/A')}"
            )
            logger.info(
                f"  Test R²   (M km²)   : {phys.get('r2',   'N/A')}"
            )
            logger.info(
                f"  Test MAPE (%)       : {phys.get('mape', 'N/A')}"
            )
        logger.info("=" * 60)