from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compare_models(models_dir: str | Path = "outputs/models") -> pd.DataFrame:

    import json
    p = Path(models_dir)
    rows = []
    for meta_file in sorted(p.glob("*_metadata.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        row = {
            "model":       meta.get("model_type"),
            "architecture": meta.get("architecture"),
            "n_parameters": meta.get("n_parameters"),
        }
        for split in ("scaled", "physical"):
            metrics = meta.get(f"test_metrics_{split}", {})
            for k, v in metrics.items():
                row[f"{split}_{k}"] = v
        rows.append(row)

    if not rows:
        logger.warning(f"No model metadata found in {p}")
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("model")


def persistence_baseline(y_true: np.ndarray) -> dict:

    from ml_models.trainer import compute_metrics
    y_actual  = y_true[1:]        # shifted target
    y_persist = y_true[:-1]       # persist previous value
    metrics = compute_metrics(y_actual, y_persist)
    logger.info(f"Persistence baseline metrics: {metrics}")
    return metrics


def skill_score(model_rmse: float, baseline_rmse: float) -> float:

    if baseline_rmse == 0:
        return float("nan")
    ss = 1.0 - (model_rmse ** 2) / (baseline_rmse ** 2)
    return round(float(ss), 4)


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    residuals = np.array(y_true) - np.array(y_pred)
    result = {
        "mean_residual":  round(float(residuals.mean()), 6),
        "std_residual":   round(float(residuals.std()),  6),
        "min_residual":   round(float(residuals.min()),  6),
        "max_residual":   round(float(residuals.max()),  6),
    }

    try:
        from scipy.stats import shapiro
        stat, p = shapiro(residuals[:5000])   # Shapiro-Wilk, capped at 5000
        result["shapiro_stat"]     = round(float(stat), 4)
        result["shapiro_p"]        = round(float(p), 6)
        result["residuals_normal"] = p > 0.05
    except ImportError:
        pass

    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(residuals)
        result["durbin_watson"]      = round(float(dw), 4)
        result["no_autocorrelation"] = (1.5 < dw < 2.5)
    except ImportError:
        pass

    return result