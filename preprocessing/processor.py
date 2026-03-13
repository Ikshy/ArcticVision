"""
preprocessing/processor.py
===========================
Core preprocessing pipeline for ArcticVision.

Responsibilities:
  1. Load raw combined dataset from data_pipeline output
  2. Clean: handle missing values, remove outliers, type-cast
  3. Feature engineering: lag features, rolling statistics, anomaly flags,
     cyclical month encoding, NAO / PDO placeholders
  4. Normalise / scale features for ML consumption
  5. Build supervised time-series windows (X, y) for LSTM / Transformer
  6. Persist processed artefacts to data/processed/

Entry point:
    from preprocessing.processor import DataPreprocessor
    proc = DataPreprocessor("configs/config.yaml")
    dataset = proc.run()

Author  : ArcticVision Research Team
Version : 1.0.0
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cleaner
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Cleans the raw combined DataFrame produced by DataFetcher.

    Steps performed:
      - Parse and sort dates
      - Drop columns with excessive NaN rates
      - Interpolate remaining NaNs (time-aware linear interpolation)
      - Cap physical outliers using IQR-based Winsorisation
      - Enforce correct dtypes
    """

    # Physical plausibility windows (hard clamp after Winsorisation)
    HARD_BOUNDS: dict[str, tuple[float, float]] = {
        "sea_ice_extent_mkm2": (0.0,  20.0),
        "sea_ice_area_mkm2":   (0.0,  18.0),
        "lst_mean_celsius":    (-65.0, 30.0),
        "lst_std_celsius":     (0.0,   25.0),
        "era5_t2m_celsius":    (-65.0, 30.0),
        "arctic_sst_celsius":  (-2.1,  15.0),
    }

    def __init__(self, nan_threshold: float = 0.30) -> None:
        """
        Args:
            nan_threshold: Drop columns with NaN rate above this fraction.
        """
        self.nan_threshold = nan_threshold
        self.dropped_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw DataFrame in-place (returns a copy).

        Args:
            df: Raw combined DataFrame from DataFetcher

        Returns:
            Cleaned DataFrame sorted by date.
        """
        df = df.copy()
        logger.info(f"Cleaner input shape: {df.shape}")

        # ── 1. Date parsing + sort ────────────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # ── 2. Drop non-feature columns ───────────────────────────────────────
        meta_cols = {"source"}
        df = df.drop(columns=[c for c in meta_cols if c in df.columns])

        # ── 3. Drop high-NaN columns ──────────────────────────────────────────
        nan_rates = df.isna().mean()
        self.dropped_cols = nan_rates[nan_rates > self.nan_threshold].index.tolist()
        if self.dropped_cols:
            logger.warning(f"Dropping high-NaN columns: {self.dropped_cols}")
            df = df.drop(columns=self.dropped_cols)

        # ── 4. Time-aware interpolation for remaining NaNs ─────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df.set_index("date")
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="time", limit_direction="both"
        )
        df = df.reset_index()

        # ── 5. Winsorise outliers (IQR × 3) ──────────────────────────────────
        for col in numeric_cols:
            if col in ("year", "month"):
                continue
            q1, q3 = df[col].quantile([0.01, 0.99])
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_clipped:
                logger.debug(f"Winsorised {n_clipped} values in '{col}'")
            df[col] = df[col].clip(lower=lo, upper=hi)

        # ── 6. Hard physical bounds (absolute safety net) ─────────────────────
        for col, (lo, hi) in self.HARD_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        # ── 7. Type enforcement ───────────────────────────────────────────────
        df["year"]  = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        logger.info(
            f"Cleaner output shape: {df.shape} | "
            f"NaN remaining: {df.isna().sum().sum()}"
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Constructs a rich feature matrix from the cleaned time series.

    Features added:
      ┌─ Lag features       : t-1, t-3, t-6, t-12 month lags of ice extent
      ├─ Rolling statistics  : 3-month and 12-month rolling mean & std
      ├─ Year-over-year diff : current month vs same month prior year
      ├─ Cyclical encoding   : sin/cos of month (avoids Dec→Jan discontinuity)
      ├─ Trend index         : fractional years since record start
      ├─ Anomaly column      : deviation from 1979-2000 monthly climatology
      └─ Interaction terms   : ice × temperature, temperature gradient
    """

    TARGET_COL = "sea_ice_extent_mkm2"

    def __init__(
        self,
        baseline_start: str = "1979-01-01",
        baseline_end: str   = "2000-12-31",
        smoothing_window: int = 3,
    ) -> None:
        self.baseline_start   = baseline_start
        self.baseline_end     = baseline_end
        self.smoothing_window = smoothing_window
        self.climatology_: Optional[pd.Series] = None   # fitted monthly means

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Compute the baseline climatology (monthly means over reference period).
        Must be called before transform() to avoid data leakage.

        Args:
            df: Cleaned DataFrame containing TARGET_COL and 'month'.

        Returns:
            self (for chaining)
        """
        baseline = df[
            (df["date"] >= self.baseline_start) &
            (df["date"] <= self.baseline_end)
        ]
        if len(baseline) < 24:
            logger.warning(
                "Baseline period too short (<24 months). "
                "Using full dataset for climatology."
            )
            baseline = df
        self.climatology_ = (
            baseline.groupby("month")[self.TARGET_COL].mean()
        )
        logger.info(
            f"Climatology fitted on {len(baseline)} rows "
            f"({self.baseline_start} → {self.baseline_end})"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to the cleaned DataFrame.

        Args:
            df: Cleaned DataFrame (from DataCleaner.fit_transform)

        Returns:
            Feature-enriched DataFrame.

        Raises:
            RuntimeError: If fit() has not been called first.
        """
        if self.climatology_ is None:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy().sort_values("date").reset_index(drop=True)
        ice = df[self.TARGET_COL]

        # ── Smoothed target ───────────────────────────────────────────────────
        df["ice_extent_smooth"] = (
            ice.rolling(self.smoothing_window, center=True, min_periods=1).mean()
        )

        # ── Lag features ──────────────────────────────────────────────────────
        for lag in [1, 3, 6, 12]:
            df[f"ice_lag_{lag}m"] = ice.shift(lag)

        # ── Rolling statistics ────────────────────────────────────────────────
        for window in [3, 12]:
            df[f"ice_roll_mean_{window}m"] = (
                ice.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"ice_roll_std_{window}m"] = (
                ice.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )

        # ── Year-over-year change ─────────────────────────────────────────────
        df["ice_yoy_change"] = ice - ice.shift(12)
        df["ice_mom_change"] = ice - ice.shift(1)   # month-over-month

        # ── Ice anomaly (vs baseline climatology) ─────────────────────────────
        df["ice_anomaly"] = ice - df["month"].map(self.climatology_)

        # ── Cyclical month encoding ───────────────────────────────────────────
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # ── Trend index (fractional years from record start) ──────────────────
        t0 = df["date"].min()
        df["trend_years"] = (df["date"] - t0).dt.days / 365.25

        # ── Temperature features (if available) ───────────────────────────────
        temp_cols = [c for c in ("era5_t2m_celsius", "lst_mean_celsius",
                                  "arctic_sst_celsius") if c in df.columns]
        for tc in temp_cols:
            df[f"{tc}_lag1"]     = df[tc].shift(1)
            df[f"{tc}_roll3m"]   = df[tc].shift(1).rolling(3, min_periods=1).mean()

        # ── Interaction: ice × temperature ────────────────────────────────────
        if "era5_t2m_celsius" in df.columns:
            df["ice_x_t2m"] = ice * df["era5_t2m_celsius"]

        # ── Temperature gradient (t2m - sst) ─────────────────────────────────
        if "era5_t2m_celsius" in df.columns and "arctic_sst_celsius" in df.columns:
            df["temp_gradient"] = (
                df["era5_t2m_celsius"] - df["arctic_sst_celsius"]
            )

        logger.info(
            f"Feature engineering complete: "
            f"{df.shape[1]} columns ({df.shape[0]} rows)"
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scaler
# ─────────────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """
    Scales numeric features for ML model consumption.

    Two strategies:
      - 'minmax' : MinMaxScaler → [0, 1]  (default, good for LSTM)
      - 'zscore' : StandardScaler → μ=0, σ=1

    The target column (sea_ice_extent_mkm2) gets its own dedicated scaler
    so predictions can be inverse-transformed back to physical units.

    Attributes:
        feature_scaler_: fitted scaler for input features
        target_scaler_:  fitted scaler for target column only
        feature_cols_:   list of feature column names used at fit time
    """

    EXCLUDE_FROM_SCALING = {"date", "year", "month",
                              "month_sin", "month_cos"}

    def __init__(self, strategy: str = "minmax") -> None:
        """
        Args:
            strategy: 'minmax' | 'zscore'
        """
        self.strategy = strategy
        self.feature_scaler_: Optional[MinMaxScaler | StandardScaler] = None
        self.target_scaler_:  Optional[MinMaxScaler | StandardScaler] = None
        self.feature_cols_: list[str] = []

    def _make_scaler(self):
        if self.strategy == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.strategy == "zscore":
            return StandardScaler()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. "
                             "Use 'minmax' or 'zscore'.")

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        Fit scalers on training data only (call BEFORE split to avoid leakage).

        Args:
            df: Feature-engineered DataFrame

        Returns:
            self
        """
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols_ = [
            c for c in numeric if c not in self.EXCLUDE_FROM_SCALING
        ]
        self.feature_scaler_ = self._make_scaler()
        self.feature_scaler_.fit(df[self.feature_cols_].fillna(0))

        # Dedicated target scaler
        target = FeatureEngineer.TARGET_COL
        self.target_scaler_ = self._make_scaler()
        self.target_scaler_.fit(df[[target]].fillna(0))

        logger.info(
            f"FeatureScaler fitted ({self.strategy}) on "
            f"{len(self.feature_cols_)} features."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features; return new DataFrame with scaled values."""
        if self.feature_scaler_ is None:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        df[self.feature_cols_] = self.feature_scaler_.transform(
            df[self.feature_cols_].fillna(0)
        )
        return df

    def inverse_transform_target(
        self, values: np.ndarray
    ) -> np.ndarray:
        """
        Inverse-scale predicted target values back to physical units (M km²).

        Args:
            values: Scaled predictions, shape (n,) or (n, 1)

        Returns:
            Array in original units.
        """
        if self.target_scaler_ is None:
            raise RuntimeError("Scaler not fitted.")
        arr = np.array(values).reshape(-1, 1)
        return self.target_scaler_.inverse_transform(arr).flatten()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sequence Builder
# ─────────────────────────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Converts a scaled tabular DataFrame into (X, y) numpy arrays
    suitable for LSTM / Transformer training.

    Each sample X[i] is a window of `seq_len` consecutive monthly rows.
    The corresponding label y[i] is the TARGET_COL value `horizon` steps
    ahead of the window's last row.

    Example (seq_len=24, horizon=1):
      X[0] = rows 0..23   →  y[0] = row 24  (next month's ice extent)
      X[1] = rows 1..24   →  y[1] = row 25

    Args:
        seq_len: Length of historical lookback window (months)
        horizon: How many steps ahead to forecast
        target_col: Name of the target feature column
    """

    def __init__(
        self,
        seq_len: int = 24,
        horizon: int = 1,
        target_col: str = FeatureEngineer.TARGET_COL,
    ) -> None:
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.target_col = target_col

    def build(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build (X, y) numpy arrays from a scaled DataFrame.

        Args:
            df:           Scaled feature DataFrame (must be time-sorted)
            feature_cols: Columns to include in X; defaults to all numerics
                          except date/year/month

        Returns:
            X: float32 array of shape (n_samples, seq_len, n_features)
            y: float32 array of shape (n_samples,)
            cols: list of feature column names used (for reproducibility)
        """
        exclude = {"date", "year", "month"}
        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        data = df[feature_cols].values.astype(np.float32)
        target_idx = feature_cols.index(self.target_col)

        X_list, y_list = [], []
        total = len(data) - self.seq_len - self.horizon + 1
        for i in range(total):
            X_list.append(data[i : i + self.seq_len])
            y_list.append(data[i + self.seq_len + self.horizon - 1, target_idx])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(
            f"Sequences built: X={X.shape}, y={y.shape} | "
            f"seq_len={self.seq_len}, horizon={self.horizon}"
        )
        return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train/Val/Test Splitter
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesSplitter:
    """
    Temporal (non-shuffled) train / validation / test split.

    Splits are done in chronological order to prevent future data leakage:
      [────── train ──────][── val ──][── test ──]

    Args:
        train_frac: Fraction for training (default 0.80)
        val_frac:   Fraction for validation (default 0.10)
                    Remaining fraction goes to test.
    """

    def __init__(
        self,
        train_frac: float = 0.80,
        val_frac:   float = 0.10,
    ) -> None:
        assert train_frac + val_frac < 1.0, \
            "train_frac + val_frac must be < 1.0"
        self.train_frac = train_frac
        self.val_frac   = val_frac

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Split (X, y) arrays chronologically.

        Args:
            X: shape (n_samples, seq_len, n_features)
            y: shape (n_samples,)

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n = len(X)
        i_train = int(n * self.train_frac)
        i_val   = int(n * (self.train_frac + self.val_frac))

        splits = {
            "train": (X[:i_train],        y[:i_train]),
            "val":   (X[i_train:i_val],   y[i_train:i_val]),
            "test":  (X[i_val:],          y[i_val:]),
        }
        for name, (Xs, ys) in splits.items():
            logger.info(f"  {name:5s}: X={Xs.shape}  y={ys.shape}")
        return splits["train"], splits["val"], splits["test"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Master DataPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """
    Master preprocessing orchestrator for ArcticVision.

    Chains:
        raw parquet  →  DataCleaner  →  FeatureEngineer
                     →  FeatureScaler  →  SequenceBuilder
                     →  TimeSeriesSplitter  →  saved artefacts

    Saves to data/processed/:
      - arctic_features.parquet      (cleaned + engineered, unscaled)
      - arctic_scaled.parquet        (scaled features)
      - X_train.npy / y_train.npy
      - X_val.npy   / y_val.npy
      - X_test.npy  / y_test.npy
      - feature_cols.pkl             (list of feature column names)
      - scaler.pkl                   (fitted FeatureScaler for inference)

    Args:
        config_path: Path to configs/config.yaml

    Example:
        >>> proc = DataPreprocessor("configs/config.yaml")
        >>> dataset = proc.run()
        >>> dataset["X_train"].shape
        (380, 24, 28)
    """

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.cfg      = _load_config(config_path)
        self.root     = Path(config_path).parent.parent
        self.raw_dir  = self.root / self.cfg["paths"]["data_raw"]
        self.proc_dir = _ensure_dir(
            self.root / self.cfg["paths"]["data_processed"]
        )

        pp = self.cfg["preprocessing"]
        cl = self.cfg["climate_analysis"]

        self.cleaner = DataCleaner(
            nan_threshold=pp["missing_value_threshold"]
        )
        self.engineer = FeatureEngineer(
            baseline_start=cl["baseline_period"][0],
            baseline_end=cl["baseline_period"][1],
            smoothing_window=pp["smoothing_window"],
        )
        self.scaler = FeatureScaler(strategy=pp["normalization"])
        self.seq_builder = SequenceBuilder(
            seq_len=pp["sequence_length"],
            horizon=1,
        )
        self.splitter = TimeSeriesSplitter(
            train_frac=pp["train_test_split"],
            val_frac=pp["validation_split"],
        )

    # ── IO helpers ────────────────────────────────────────────────────────────

    def _load_raw(self) -> pd.DataFrame:
        """Load the combined raw parquet from data_pipeline output."""
        raw_path = self.raw_dir / "arctic_combined_raw.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {raw_path}. "
                "Run DataFetcher.run() first."
            )
        df = pd.read_parquet(raw_path)
        logger.info(f"Raw data loaded: {df.shape} from {raw_path}")
        return df

    def _save_artefacts(
        self,
        df_feat:   pd.DataFrame,
        df_scaled: pd.DataFrame,
        splits:    dict,
        feat_cols: list[str],
    ) -> None:
        """Persist all preprocessing outputs."""
        df_feat.to_parquet(self.proc_dir / "arctic_features.parquet",   index=False)
        df_scaled.to_parquet(self.proc_dir / "arctic_scaled.parquet",   index=False)

        for split_name, (Xs, ys) in splits.items():
            np.save(self.proc_dir / f"X_{split_name}.npy", Xs)
            np.save(self.proc_dir / f"y_{split_name}.npy", ys)

        with open(self.proc_dir / "feature_cols.pkl", "wb") as f:
            pickle.dump(feat_cols, f)

        with open(self.proc_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(f"All preprocessing artefacts saved to {self.proc_dir}")

    # ── Summary reporting ─────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(
        df_clean:  pd.DataFrame,
        df_feat:   pd.DataFrame,
        df_scaled: pd.DataFrame,
        X_train:   np.ndarray,
        X_val:     np.ndarray,
        X_test:    np.ndarray,
        y_train:   np.ndarray,
    ) -> None:
        """Print a concise processing summary to the logger."""
        logger.info("=" * 60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Cleaned rows         : {len(df_clean)}")
        logger.info(f"  Engineered features  : {df_feat.shape[1]} columns")
        logger.info(f"  Scaled features      : {df_scaled.shape[1]} columns")
        logger.info(f"  Sequence shape (X)   : {X_train.shape[1:]} "
                    f"(seq_len × n_features)")
        logger.info(f"  Train samples        : {len(X_train)}")
        logger.info(f"  Val   samples        : {len(X_val)}")
        logger.info(f"  Test  samples        : {len(X_test)}")
        logger.info(f"  Target range (train) : "
                    f"[{y_train.min():.4f}, {y_train.max():.4f}] (scaled)")
        logger.info("=" * 60)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full preprocessing pipeline end-to-end.

        Returns:
            dict with keys:
              'df_clean'     : cleaned unscaled DataFrame
              'df_features'  : feature-engineered unscaled DataFrame
              'df_scaled'    : scaled DataFrame
              'X_train', 'y_train'
              'X_val',   'y_val'
              'X_test',  'y_test'
              'feature_cols' : list of feature column names
              'scaler'       : fitted FeatureScaler instance
        """
        logger.info("=" * 60)
        logger.info("ArcticVision Preprocessing Pipeline START")
        logger.info("=" * 60)

        # ── 1. Load raw data ──────────────────────────────────────────────────
        df_raw = self._load_raw()

        # ── 2. Clean ──────────────────────────────────────────────────────────
        logger.info("Step 1/5: Cleaning...")
        df_clean = self.cleaner.fit_transform(df_raw)

        # ── 3. Feature engineering ────────────────────────────────────────────
        logger.info("Step 2/5: Feature engineering...")
        df_feat = self.engineer.fit_transform(df_clean)

        # ── 4. Drop rows with NaN from lag/rolling features ───────────────────
        logger.info("Step 3/5: Dropping NaN rows from lag features...")
        n_before = len(df_feat)
        df_feat = df_feat.dropna().reset_index(drop=True)
        logger.info(f"  Dropped {n_before - len(df_feat)} NaN rows. "
                    f"Remaining: {len(df_feat)}")

        # ── 5. Scale ──────────────────────────────────────────────────────────
        logger.info("Step 4/5: Scaling features...")
        df_scaled = self.scaler.fit_transform(df_feat)

        # ── 6. Build sequences + split ────────────────────────────────────────
        logger.info("Step 5/5: Building sequences and splitting...")
        X, y, feat_cols = self.seq_builder.build(df_scaled)

        train, val, test = self.splitter.split(X, y)
        X_train, y_train = train
        X_val,   y_val   = val
        X_test,  y_test  = test

        # ── 7. Save ───────────────────────────────────────────────────────────
        splits = {
            "train": (X_train, y_train),
            "val":   (X_val,   y_val),
            "test":  (X_test,  y_test),
        }
        self._save_artefacts(df_feat, df_scaled, splits, feat_cols)
        self._print_summary(
            df_clean, df_feat, df_scaled, X_train, X_val, X_test, y_train
        )

        logger.info("Preprocessing Pipeline COMPLETE")

        return {
            "df_clean":    df_clean,
            "df_features": df_feat,
            "df_scaled":   df_scaled,
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "feature_cols": feat_cols,
            "scaler":       self.scaler,
        }

    # ── Convenience loader (for downstream modules) ───────────────────────────

    @classmethod
    def load_processed(
        cls,
        proc_dir: str | Path = "data/processed",
    ) -> dict:
        """
        Load all saved preprocessing artefacts from disk.
        Use this in climate_analysis / ml_models instead of re-running.

        Args:
            proc_dir: Path to data/processed directory

        Returns:
            dict matching the structure of run()
        """
        p = Path(proc_dir)
        required = [
            "arctic_features.parquet", "arctic_scaled.parquet",
            "X_train.npy", "y_train.npy",
            "X_val.npy",   "y_val.npy",
            "X_test.npy",  "y_test.npy",
            "feature_cols.pkl", "scaler.pkl",
        ]
        missing = [f for f in required if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing processed files: {missing}. "
                "Run DataPreprocessor().run() first."
            )

        with open(p / "feature_cols.pkl", "rb") as f:
            feat_cols = pickle.load(f)
        with open(p / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return {
            "df_features":  pd.read_parquet(p / "arctic_features.parquet"),
            "df_scaled":    pd.read_parquet(p / "arctic_scaled.parquet"),
            "X_train": np.load(p / "X_train.npy"),
            "y_train": np.load(p / "y_train.npy"),
            "X_val":   np.load(p / "X_val.npy"),
            "y_val":   np.load(p / "y_val.npy"),
            "X_test":  np.load(p / "X_test.npy"),
            "y_test":  np.load(p / "y_test.npy"),
            "feature_cols": feat_cols,
            "scaler":       scaler,
        }