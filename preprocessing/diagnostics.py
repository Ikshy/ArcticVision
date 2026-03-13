

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a rich descriptive statistics table for all numeric columns.
    Extends pd.describe() with skewness, kurtosis, and NaN count.
    """
    numeric = df.select_dtypes(include=[np.number])
    desc = numeric.describe().T
    desc["skewness"]  = numeric.skew()
    desc["kurtosis"]  = numeric.kurt()
    desc["nan_count"] = numeric.isna().sum()
    desc["nan_pct"]   = (numeric.isna().mean() * 100).round(2)
    return desc.round(4)


def check_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test for stationarity.

    A p-value < 0.05 suggests the series is stationary (no unit root).
    Non-stationary ice extent series typically need differencing before
    feeding to classical ARIMA; LSTM handles non-stationarity better.

    Args:
        series: pandas Series (should have no NaNs)
        name:   Label for logging

    Returns:
        dict with 'adf_stat', 'p_value', 'is_stationary'
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed. Skipping ADF test.")
        return {}

    s = series.dropna()
    adf_stat, p_value, _, _, crit, _ = adfuller(s, autolag="AIC")
    is_stationary = p_value < 0.05
    logger.info(
        f"ADF [{name}]: stat={adf_stat:.4f}, p={p_value:.4f} "
        f"→ {'stationary' if is_stationary else 'NON-stationary'}"
    )
    return {
        "series":        name,
        "adf_stat":      round(adf_stat, 4),
        "p_value":       round(p_value, 6),
        "critical_5pct": round(crit["5%"], 4),
        "is_stationary": is_stationary,
    }


def autocorrelation_summary(series: pd.Series, lags: int = 24) -> pd.DataFrame:
    """
    Compute autocorrelation and partial autocorrelation up to `lags` months.

    High ACF at lag-12 confirms strong seasonality in ice extent data.

    Args:
        series: Monthly time series (no NaNs)
        lags:   Maximum lag months to compute

    Returns:
        DataFrame with columns [lag, acf, pacf]
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        logger.warning("statsmodels not installed. Skipping ACF/PACF.")
        return pd.DataFrame()

    s = series.dropna().values
    acf_vals  = acf(s,  nlags=lags, fft=True)[1:]    # skip lag-0
    pacf_vals = pacf(s, nlags=lags, method="ols")[1:]

    return pd.DataFrame({
        "lag":  range(1, lags + 1),
        "acf":  np.round(acf_vals,  4),
        "pacf": np.round(pacf_vals, 4),
    })


def sequence_shape_report(proc_dir: str | Path = "data/processed") -> None:
    """
    Print a quick shape and value-range report for all saved .npy arrays.

    Args:
        proc_dir: Path to data/processed directory.
    """
    p = Path(proc_dir)
    for split in ("train", "val", "test"):
        xp = p / f"X_{split}.npy"
        yp = p / f"y_{split}.npy"
        if xp.exists():
            X = np.load(xp)
            y = np.load(yp)
            logger.info(
                f"  {split:5s} → X={X.shape}  y={y.shape}  "
                f"y_range=[{y.min():.4f}, {y.max():.4f}]"
            )
        else:
            logger.warning(f"  {split}: file not found at {xp}")