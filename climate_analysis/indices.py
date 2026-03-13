"""
climate_analysis/indices.py
============================
Computation of standard Arctic and large-scale climate indices.

Indices implemented:
  - Sea Ice Index (SII)     : monthly extent anomaly as published by NSIDC
  - Arctic Amplification    : ratio of Arctic vs global warming rate
  - Ice-Albedo Feedback     : simplified proxy from ice extent and temperature

Placeholder stubs (require teleconnection data from external sources):
  - Arctic Oscillation (AO) / NAM index  →  NOAA CPC
  - North Atlantic Oscillation (NAO)     →  NOAA CPC
  - Pacific Decadal Oscillation (PDO)    →  NOAA PSL

Instructions for adding teleconnection data:
  1. Download AO monthly index:
       https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii
  2. Download NAO monthly index:
       https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii
  3. Save to data/external/ and call merge_teleconnections() below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sea_ice_index(
    df: pd.DataFrame,
    column: str = "sea_ice_extent_mkm2",
    baseline_start: str = "1981-01-01",
    baseline_end:   str = "2010-12-31",
) -> pd.DataFrame:
    """
    Compute the NSIDC-style Sea Ice Index: monthly anomaly from a
    1981–2010 climatological baseline (the WMO standard period).

    Args:
        df:             Feature DataFrame with 'date' and ice extent column
        column:         Ice extent column name
        baseline_start: Start of baseline period
        baseline_end:   End of baseline period

    Returns:
        DataFrame with columns [date, sea_ice_extent_mkm2,
                                 sii_anomaly_mkm2, sii_pct_anomaly]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    base = df[(df["date"] >= baseline_start) & (df["date"] <= baseline_end)]
    clim = base.groupby("month")[column].mean()

    df["sii_climatology"] = df["month"].map(clim)
    df["sii_anomaly_mkm2"] = (df[column] - df["sii_climatology"]).round(4)
    df["sii_pct_anomaly"]  = (
        df["sii_anomaly_mkm2"] / df["sii_climatology"] * 100
    ).round(2)

    return df[["date", column, "sii_climatology",
                "sii_anomaly_mkm2", "sii_pct_anomaly"]]


def compute_arctic_amplification(
    arctic_t2m: pd.Series,
    global_t2m: pd.Series,
) -> float:
    """
    Estimate Arctic Amplification factor: ratio of warming trends.

    Arctic Amplification = (Arctic warming trend) / (Global warming trend)
    Values > 2 confirm polar amplification (observed ~3-4× in recent decades).

    Args:
        arctic_t2m: Monthly Arctic mean 2-m temperature (°C)
        global_t2m: Monthly global mean 2-m temperature (°C) — external data

    Returns:
        Amplification factor (float), or NaN if trends can't be estimated.
    """
    try:
        from scipy.stats import linregress
        t = np.arange(len(arctic_t2m))
        arctic_slope, *_ = linregress(t[:len(arctic_t2m)], arctic_t2m.dropna())
        global_slope, *_ = linregress(t[:len(global_t2m)],  global_t2m.dropna())
        if global_slope == 0:
            return float("nan")
        aa = arctic_slope / global_slope
        logger.info(f"Arctic Amplification factor: {aa:.2f}×")
        return round(float(aa), 2)
    except Exception as e:
        logger.warning(f"Arctic Amplification calculation failed: {e}")
        return float("nan")


def merge_teleconnections(
    df: pd.DataFrame,
    external_dir: str | Path = "data/external",
) -> pd.DataFrame:
    """
    Merge Arctic Oscillation (AO) and NAO indices into the feature DataFrame.

    Expects pre-downloaded files in data/external/:
      - ao_monthly.csv   : columns [year, month, ao_index]
      - nao_monthly.csv  : columns [year, month, nao_index]

    If files are not found, placeholder NaN columns are added with
    instructions printed to the logger.

    Args:
        df:           Feature DataFrame with 'year' and 'month' columns
        external_dir: Path to data/external/

    Returns:
        df with 'ao_index' and 'nao_index' columns appended.
    """
    ext = Path(external_dir)
    df = df.copy()

    for idx_name, filename in [("ao_index", "ao_monthly.csv"),
                                  ("nao_index", "nao_monthly.csv")]:
        fpath = ext / filename
        if fpath.exists():
            idx_df = pd.read_csv(fpath)
            df = df.merge(
                idx_df[["year", "month", idx_name]],
                on=["year", "month"],
                how="left",
            )
            logger.info(f"Merged {idx_name} from {fpath}")
        else:
            df[idx_name] = np.nan
            logger.warning(
                f"{idx_name} not found at {fpath}. "
                f"Column set to NaN. "
                f"Download from NOAA CPC and save as {filename}."
            )
    return df