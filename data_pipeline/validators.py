
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Expected schema for the combined raw dataset
COMBINED_RAW_SCHEMA = {
    "date":                  "datetime64[ns]",
    "sea_ice_extent_mkm2":   float,
    "sea_ice_area_mkm2":     float,
    "lst_mean_celsius":      float,
    "era5_t2m_celsius":      float,
    "arctic_sst_celsius":    float,
    "year":                  int,
    "month":                 int,
}

PHYSICAL_BOUNDS = {
    "sea_ice_extent_mkm2":  (0.0,  20.0),
    "sea_ice_area_mkm2":    (0.0,  18.0),
    "lst_mean_celsius":     (-60.0, 30.0),
    "era5_t2m_celsius":     (-60.0, 30.0),
    "arctic_sst_celsius":   (-2.0,  15.0),
}


def validate_combined_raw(df: pd.DataFrame) -> dict:

    issues = []
    summary = {}

    # ── 1. Required columns ───────────────────────────────────────────────────
    missing_cols = [c for c in COMBINED_RAW_SCHEMA if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # ── 2. Row count ──────────────────────────────────────────────────────────
    if len(df) < 12:
        issues.append(f"Too few rows: {len(df)} (expected ≥ 12 for ≥1 year)")

    # ── 3. NaN rates ─────────────────────────────────────────────────────────
    for col in df.columns:
        nan_rate = df[col].isna().mean()
        summary[col] = {"nan_rate": round(nan_rate, 3)}
        if nan_rate > 0.30 and col not in ("source", "lst_std_celsius"):
            issues.append(f"High NaN rate in '{col}': {nan_rate:.1%}")

    # ── 4. Physical bounds ────────────────────────────────────────────────────
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        out = df[col].dropna()
        n_low  = (out < lo).sum()
        n_high = (out > hi).sum()
        summary[col]["out_of_range"] = int(n_low + n_high)
        if n_low + n_high > 0:
            issues.append(
                f"Physical bounds violation in '{col}': "
                f"{n_low} below {lo}, {n_high} above {hi}"
            )

    # ── 5. Monotonicity of dates ───────────────────────────────────────────────
    if "date" in df.columns:
        if not df["date"].is_monotonic_increasing:
            issues.append("'date' column is not monotonically increasing.")

    passed = len(issues) == 0
    if passed:
        logger.info("Dataset validation PASSED ✓")
    else:
        for iss in issues:
            logger.warning(f"Validation issue: {iss}")

    return {"passed": passed, "issues": issues, "summary": summary}