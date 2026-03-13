"""
preprocessing/ice_extractor.py
===============================
Utilities for extracting sea ice coverage metrics from raw
NASA NSIDC HDF/NetCDF raster files when real satellite data is downloaded.

For users who download actual NSIDC-0051 HDF files via NASAFetcher,
this module:
  - Parses the HDF4/HDF5 concentration grids
  - Converts raw DN values to ice concentration (%)
  - Computes sea ice extent and area per month
  - Handles the polar hole (missing data near North Pole)

Note:
    This module is only needed when working with raw binary granules.
    If using the synthetic or ERSSTv5 pipeline path, this is skipped.

Dependencies:
    pip install h5py pyhdf rasterio  (HDF4 also needs libhdf4 system lib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NSIDC-0051 grid constants (25 km EASE-Grid, Northern Hemisphere)
NSIDC_GRID_ROWS      = 448
NSIDC_GRID_COLS      = 304
NSIDC_CELL_AREA_KM2  = 625.0          # 25 km × 25 km
NSIDC_FILL_VALUE     = 255            # land / no data
NSIDC_POLE_HOLE      = 251            # polar hole fill value
NSIDC_SCALE_FACTOR   = 0.01          # raw DN × 0.01 = fractional concentration
ICE_EXTENT_THRESHOLD = 0.15          # ≥15% concentration = "ice covered"


def parse_nsidc_hdf(file_path: Path) -> Optional[np.ndarray]:
    """
    Parse a single NSIDC-0051 HDF5 granule file and return the
    sea ice concentration grid as a float32 array.

    Raw values: 0-250 (concentration %), 251=pole hole, 255=land/no-data

    Args:
        file_path: Path to .he5 or .hdf file

    Returns:
        2D float32 array of shape (448, 304) with concentration [0, 1].
        Returns None if the file cannot be parsed.
    """
    try:
        import h5py
        with h5py.File(file_path, "r") as f:
            # NSIDC-0051 v2 HDF5 path
            keys = list(f.keys())
            logger.debug(f"HDF5 root keys: {keys}")

            # Navigate to concentration dataset
            # Typical path: /HDFEOS/GRIDS/NpPolarGrid25km/Data Fields/SI_25km_NH_ICECON_DAY
            grid_path = (
                "HDFEOS/GRIDS/NpPolarGrid25km/Data Fields/"
                "SI_25km_NH_ICECON_DAY"
            )
            if grid_path not in f:
                # Try alternative paths in older versions
                alt = [k for k in f.visit(lambda k: k) if "ICECON" in k.upper()]
                if not alt:
                    logger.error(f"Cannot find ice concentration dataset in {file_path}")
                    return None
                grid_path = alt[0]

            raw = f[grid_path][:]   # uint8 array
    except ImportError:
        logger.error("h5py not installed. Run: pip install h5py")
        return None
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return None

    # Convert to float, mask special values
    conc = raw.astype(np.float32)
    conc[raw == NSIDC_FILL_VALUE] = np.nan   # land / missing
    conc[raw == NSIDC_POLE_HOLE]  = np.nan   # polar hole

    # Scale: raw 0-250 → concentration 0.0-1.0  (via 0-100 intermediate)
    valid = ~np.isnan(conc)
    conc[valid] = conc[valid] * NSIDC_SCALE_FACTOR  # 0-2.50 → clip to 1.0
    conc = np.clip(conc, 0.0, 1.0)

    return conc


def compute_extent_and_area(
    conc_grid: np.ndarray,
    cell_area_km2: float = NSIDC_CELL_AREA_KM2,
    threshold: float = ICE_EXTENT_THRESHOLD,
) -> tuple[float, float]:
    """
    Compute sea ice extent and area from a concentration grid.

    Definitions (NSIDC standard):
      Extent = total area of grid cells with concentration ≥ threshold
      Area   = sum of (cell_area × concentration) for ice-covered cells

    Args:
        conc_grid:    2D float array of ice concentration [0, 1]
        cell_area_km2: Area of each grid cell in km²
        threshold:    Minimum concentration to count as ice-covered (default 0.15)

    Returns:
        (extent_mkm2, area_mkm2) — values in millions of km²
    """
    valid = ~np.isnan(conc_grid)
    ice_covered = valid & (conc_grid >= threshold)

    extent_km2 = ice_covered.sum() * cell_area_km2
    area_km2   = (conc_grid[ice_covered] * cell_area_km2).sum()

    return extent_km2 / 1e6, area_km2 / 1e6   # convert to M km²


def batch_extract_monthly(
    hdf_dir: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Process all HDF granules in a directory and assemble a monthly
    sea ice extent / area time series.

    For monthly NSIDC files, each file represents one month.
    For daily files, this function averages all days within each month.

    Args:
        hdf_dir:     Directory containing .he5 / .hdf files
        output_path: If given, save resulting DataFrame to parquet

    Returns:
        DataFrame with columns [date, sea_ice_extent_mkm2, sea_ice_area_mkm2]
    """
    files = sorted(hdf_dir.glob("*.he5")) + sorted(hdf_dir.glob("*.hdf"))
    if not files:
        logger.warning(f"No HDF files found in {hdf_dir}. "
                       "Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "sea_ice_extent_mkm2",
                                      "sea_ice_area_mkm2"])

    logger.info(f"Processing {len(files)} HDF granules from {hdf_dir}...")
    records = []

    for fp in files:
        conc = parse_nsidc_hdf(fp)
        if conc is None:
            continue
        extent, area = compute_extent_and_area(conc)

        # Extract date from NSIDC filename convention:
        # NSIDC0051_SEAICE_PS_N25km_YYYYMMDD_v2.0.he5
        date_str = _extract_date_from_filename(fp.name)
        records.append({
            "date":                 date_str,
            "sea_ice_extent_mkm2":  round(extent, 4),
            "sea_ice_area_mkm2":    round(area,   4),
            "file":                 fp.name,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    # Group to monthly means if daily files were processed
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))
        [["sea_ice_extent_mkm2", "sea_ice_area_mkm2"]]
        .mean()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly = monthly.sort_values("date").reset_index(drop=True)

    if output_path:
        monthly.to_parquet(output_path, index=False)
        logger.info(f"Monthly ice series saved → {output_path}")

    return monthly


def _extract_date_from_filename(filename: str) -> str:
    """
    Extract date from NSIDC filename using regex.

    Pattern: NSIDC0051_SEAICE_PS_N25km_YYYYMMDD_v2.0.he5

    Args:
        filename: Base filename string

    Returns:
        'YYYY-MM-DD' string or '1970-01-01' fallback.
    """
    import re
    match = re.search(r"(\d{8})", filename)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    logger.warning(f"Could not parse date from filename: {filename}")
    return "1970-01-01"