

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# ── Optional heavy imports (graceful degradation for offline testing) ─────────
try:
    import earthaccess  # NASA EarthData search + download
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False

try:
    import ee          # Google Earth Engine
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ── Logger setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)




def _load_config(config_path: str | Path) -> dict:
    """Load and return the YAML configuration dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> Path:
    """Create directory tree if it doesn't exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _date_range_monthly(start: str, end: str) -> list[str]:

    periods = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in periods]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NASA EarthData Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class NASAFetcher:

    NSIDC_BASE = "https://n5eil01u.ecs.nsidc.org/SNSIMMS/NSIDC-0051.002/"

    def __init__(
        self,
        username: str,
        password: str,
        raw_dir: Path,
        short_name: str = "NSIDC-0051",
    ) -> None:
        self.username = username
        self.password = password
        self.raw_dir = _ensure_dir(raw_dir / "nasa_seaice")
        self.short_name = short_name
        self._session: Optional[requests.Session] = None

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate(self) -> None:

        if EARTHACCESS_AVAILABLE:
            logger.info("Authenticating via earthaccess...")
            earthaccess.login(
                strategy="environment",   # reads env vars automatically
                persist=True,             # caches token in ~/.netrc
            )
            logger.info("earthaccess authentication successful.")
        else:
            logger.warning(
                "earthaccess not installed. Falling back to basic auth. "
                "Install with: pip install earthaccess"
            )
            self._session = requests.Session()
            self._session.auth = (self.username, self.password)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_granules(
        self,
        start_date: str,
        end_date: str,
        bounding_box: list[float],
    ) -> list:

        if not EARTHACCESS_AVAILABLE:
            logger.warning("earthaccess unavailable — skipping granule search.")
            return []

        logger.info(
            f"Searching {self.short_name} granules: {start_date} → {end_date}"
        )
        results = earthaccess.search_data(
            short_name=self.short_name,
            temporal=(start_date, end_date),
            bounding_box=tuple(bounding_box),
            count=-1,   # return all matching results
        )
        logger.info(f"Found {len(results)} granules.")
        return results

    # ── Download ──────────────────────────────────────────────────────────────

    def download(
        self,
        granules: list,
        max_files: Optional[int] = None,
    ) -> list[Path]:

        if not EARTHACCESS_AVAILABLE or not granules:
            logger.warning("No granules to download or earthaccess missing.")
            return []

        subset = granules[:max_files] if max_files else granules
        logger.info(f"Downloading {len(subset)} files to {self.raw_dir} ...")
        downloaded = earthaccess.download(subset, local_path=str(self.raw_dir))
        logger.info(f"Download complete: {len(downloaded)} files.")
        return [Path(p) for p in downloaded]

    # ── Synthetic fallback (for offline / CI use) ─────────────────────────────

    def generate_synthetic_data(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:

        logger.info("Generating synthetic NASA sea ice data...")
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n = len(dates)

        # Seasonal cycle: peak March (month=3), trough Sept (month=9)
        month_idx = np.array([d.month for d in dates])
        seasonal = 3.5 * np.cos(2 * np.pi * (month_idx - 3) / 12)

        # Long-term decline: ~0.043 M km² / year
        years_elapsed = np.arange(n) / 12.0
        trend = -0.043 * years_elapsed

        # Base extent ≈ 12.5 M km² (1979 mean)
        base = 12.5
        noise = np.random.normal(0, 0.25, n)

        extent = base + seasonal + trend + noise
        extent = np.clip(extent, 2.5, 16.0)   # physical bounds

        df = pd.DataFrame({
            "date": dates,
            "sea_ice_extent_mkm2": np.round(extent, 4),
            "sea_ice_area_mkm2":   np.round(extent * 0.87, 4),  # ~87% of extent
            "source": "synthetic_nsidc0051",
        })

        if save:
            out_path = self.raw_dir / "sea_ice_extent_synthetic.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Synthetic sea ice data saved → {out_path}")

        return df




class GEEFetcher:


    ARCTIC_REGION_COORDS = [   # Simplified Arctic polygon (60°N–90°N)
        [-180, 60], [180, 60], [180, 90], [-180, 90], [-180, 60]
    ]

    def __init__(
        self,
        project_id: str,
        raw_dir: Path,
        cfg: dict,
    ) -> None:
        self.project_id = project_id
        self.raw_dir = _ensure_dir(raw_dir / "gee")
        self.cfg = cfg
        self._initialized = False

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
 
        if not GEE_AVAILABLE:
            logger.warning(
                "earthengine-api not installed. "
                "Install with:  pip install earthengine-api geemap"
            )
            return False
        try:
            ee.Initialize(project=self.project_id)
            self._initialized = True
            logger.info(f"GEE initialized (project: {self.project_id})")
            return True
        except Exception as e:
            logger.error(f"GEE initialization failed: {e}")
            logger.error(
                "Run `earthengine authenticate` in your terminal first."
            )
            return False

    # ── MODIS Land Surface Temperature ───────────────────────────────────────

    def fetch_modis_lst(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
  
        if not self._initialized:
            logger.warning("GEE not initialized — returning synthetic LST.")
            return self._synthetic_lst(start_date, end_date)

        try:
            arctic = ee.Geometry.Polygon(self.ARCTIC_REGION_COORDS)
            collection = (
                ee.ImageCollection(self.cfg["gee"]["collection_modis_lst"])
                .filterDate(start_date, end_date)
                .filterBounds(arctic)
                .select("LST_Day_1km")
            )

            def _monthly_mean(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        ee.Reducer.stdDev(), sharedInputs=True
                    ),
                    geometry=arctic,
                    scale=5000,       # 5 km aggregation for speed
                    maxPixels=1e9,
                )
                return image.set("stats", stats).set(
                    "system:time_start", image.get("system:time_start")
                )

            results = collection.map(_monthly_mean).getInfo()
            rows = []
            for feat in results["features"]:
                ts = feat["properties"].get("system:time_start", 0)
                date = datetime.utcfromtimestamp(ts / 1000)
                stats = feat["properties"].get("stats", {})
                mean_k = stats.get("LST_Day_1km_mean", None)
                std_k  = stats.get("LST_Day_1km_stdDev", None)
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "lst_mean_celsius": round(mean_k * 0.02 - 273.15, 3) if mean_k else None,
                    "lst_std_celsius":  round(std_k  * 0.02, 3)          if std_k  else None,
                    "source": "MODIS_MOD11A2",
                })
            df = pd.DataFrame(rows)
            out = self.raw_dir / "modis_lst.parquet"
            df.to_parquet(out, index=False)
            logger.info(f"MODIS LST saved → {out}  ({len(df)} records)")
            return df

        except Exception as e:
            logger.error(f"MODIS LST fetch error: {e}")
            return self._synthetic_lst(start_date, end_date)

    # ── ERA5 Reanalysis ────────────────────────────────────────────────────────

    def fetch_era5_temperature(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch monthly ERA5 2-metre air temperature over the Arctic.

        Collection: ECMWF/ERA5/MONTHLY
        Band used : mean_2m_air_temperature (Kelvin)

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            DataFrame[date, era5_t2m_celsius, source]
        """
        if not self._initialized:
            logger.warning("GEE not initialized — returning synthetic ERA5.")
            return self._synthetic_era5(start_date, end_date)

        try:
            arctic = ee.Geometry.Polygon(self.ARCTIC_REGION_COORDS)
            collection = (
                ee.ImageCollection(self.cfg["gee"]["collection_era5"])
                .filterDate(start_date, end_date)
                .filterBounds(arctic)
                .select("mean_2m_air_temperature")
            )

            rows = []
            images = collection.toList(collection.size()).getInfo()
            for img_info in images:
                img = ee.Image(img_info["id"])
                ts  = img_info["properties"].get("system:time_start", 0)
                date = datetime.utcfromtimestamp(ts / 1000)
                mean_k = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=arctic,
                    scale=25000,
                    maxPixels=1e9,
                ).getInfo().get("mean_2m_air_temperature")
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "era5_t2m_celsius": round(mean_k - 273.15, 3) if mean_k else None,
                    "source": "ERA5_MONTHLY",
                })
            df = pd.DataFrame(rows)
            out = self.raw_dir / "era5_t2m.parquet"
            df.to_parquet(out, index=False)
            logger.info(f"ERA5 T2m saved → {out}  ({len(df)} records)")
            return df

        except Exception as e:
            logger.error(f"ERA5 fetch error: {e}")
            return self._synthetic_era5(start_date, end_date)

    # ── Synthetic fallbacks ───────────────────────────────────────────────────

    def _synthetic_lst(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return synthetic MODIS LST data for offline development."""
        logger.info("Generating synthetic MODIS LST data...")
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n = len(dates)
        month_idx = np.array([d.month for d in dates])
        # Arctic LST: peaks in July ~5°C, bottoms in Jan ~-32°C
        seasonal = 18.5 * np.cos(2 * np.pi * (month_idx - 7) / 12)
        years_el  = np.arange(n) / 12.0
        trend     = 0.06 * years_el          # +0.06°C / year warming trend
        noise     = np.random.normal(0, 1.2, n)
        lst = -13.5 + seasonal + trend + noise
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "lst_mean_celsius": np.round(lst, 3),
            "lst_std_celsius":  np.round(np.abs(noise) + 2.0, 3),
            "source": "synthetic_MODIS",
        })
        out = self.raw_dir / "modis_lst_synthetic.parquet"
        df.to_parquet(out, index=False)
        return df

    def _synthetic_era5(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return synthetic ERA5 2-m temperature data for offline development."""
        logger.info("Generating synthetic ERA5 T2m data...")
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n = len(dates)
        month_idx = np.array([d.month for d in dates])
        seasonal  = 15.0 * np.cos(2 * np.pi * (month_idx - 7) / 12)
        trend     = 0.05 * np.arange(n) / 12.0
        noise     = np.random.normal(0, 0.9, n)
        t2m = -11.0 + seasonal + trend + noise
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "era5_t2m_celsius": np.round(t2m, 3),
            "source": "synthetic_ERA5",
        })
        out = self.raw_dir / "era5_t2m_synthetic.parquet"
        df.to_parquet(out, index=False)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NOAA Sea Surface Temperature Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class NOAAFetcher:


    ERSST_URL = (
        "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
    )

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = _ensure_dir(raw_dir / "noaa_sst")
        self.nc_path = self.raw_dir / "noaa_ersst_v5.nc"

    def download_ersst(self, force: bool = False) -> Path:
        """
        Download the ERSSTv5 global monthly NetCDF file (~30 MB).

        Args:
            force: Re-download even if file already exists.

        Returns:
            Path to the downloaded NetCDF file.

        INSTRUCTIONS:
            The NOAA PSL server occasionally throttles downloads.
            If you get a connection error, retry after a few minutes.
        """
        if self.nc_path.exists() and not force:
            logger.info(f"ERSSTv5 already cached at {self.nc_path}")
            return self.nc_path

        logger.info(f"Downloading ERSSTv5 from NOAA PSL...")
        try:
            with requests.get(self.ERSST_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(self.nc_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True,
                    desc="ERSSTv5"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            logger.info(f"ERSSTv5 saved → {self.nc_path}")
        except requests.RequestException as e:
            logger.error(f"ERSSTv5 download failed: {e}")
            logger.info("Falling back to synthetic SST data.")
            return self._generate_synthetic_sst_nc()
        return self.nc_path

    def extract_arctic_sst(
        self,
        nc_path: Optional[Path] = None,
        lat_min: float = 60.0,
    ) -> pd.DataFrame:
   
        try:
            import xarray as xr
        except ImportError:
            logger.error("xarray not installed. Run: pip install xarray netCDF4")
            return self._synthetic_sst_dataframe("1979-01-01", "2024-12-31")

        nc_path = nc_path or self.nc_path
        if not nc_path.exists():
            logger.warning("NetCDF file not found. Using synthetic SST.")
            return self._synthetic_sst_dataframe("1979-01-01", "2024-12-31")

        logger.info(f"Extracting Arctic SST (lat ≥ {lat_min}°N) from {nc_path}")
        ds = xr.open_dataset(nc_path, mask_and_scale=True)

        # ERSSTv5 uses 'sst' variable, coordinates 'lat' and 'lon'
        arctic = ds["sst"].sel(lat=slice(lat_min, 90.0))
        monthly_mean = arctic.mean(dim=["lat", "lon"], skipna=True)

        df = pd.DataFrame({
            "date": pd.to_datetime(monthly_mean.time.values).strftime("%Y-%m-%d"),
            "arctic_sst_celsius": np.round(monthly_mean.values, 4),
            "source": "NOAA_ERSSTv5",
        })
        out = self.raw_dir / "arctic_sst.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Arctic SST saved → {out}  ({len(df)} records)")
        return df

    def _generate_synthetic_sst_nc(self) -> Path:
        """Return synthetic SST parquet path (no NetCDF written)."""
        return self.nc_path   # will trigger fallback in extract_arctic_sst

    def _synthetic_sst_dataframe(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic Arctic SST time series."""
        logger.info("Generating synthetic Arctic SST data...")
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n = len(dates)
        month_idx = np.array([d.month for d in dates])
        seasonal  = 2.0 * np.cos(2 * np.pi * (month_idx - 8) / 12)
        trend     = 0.018 * np.arange(n) / 12.0   # +0.018°C/year
        noise     = np.random.normal(0, 0.3, n)
        sst = -1.2 + seasonal + trend + noise
        sst = np.clip(sst, -1.9, 10.0)  # physical range

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "arctic_sst_celsius": np.round(sst, 4),
            "source": "synthetic_ERSSTv5",
        })
        out = self.raw_dir / "arctic_sst_synthetic.parquet"
        df.to_parquet(out, index=False)
        return df




class DataFetcher:
    """
    Master orchestrator for all ArcticVision data ingestion.

    Composes NASAFetcher, GEEFetcher, and NOAAFetcher and merges
    their outputs into a single raw joined dataset saved to
    data/raw/arctic_combined_raw.parquet

    Args:
        config_path: Path to configs/config.yaml
        env_path:    Path to .env file (default: '.env')

    Example:
        >>> fetcher = DataFetcher("configs/config.yaml")
        >>> combined = fetcher.run()
        >>> print(combined.head())
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        env_path: str | Path = ".env",
    ) -> None:
        load_dotenv(env_path)   # load NASA/GEE credentials from .env
        self.cfg = _load_config(config_path)

        # ── Root paths ────────────────────────────────────────────────────────
        self.root       = Path(config_path).parent.parent
        self.raw_dir    = self.root / self.cfg["paths"]["data_raw"]
        self.proc_dir   = self.root / self.cfg["paths"]["data_processed"]
        _ensure_dir(self.raw_dir)
        _ensure_dir(self.proc_dir)

        # ── Credentials ───────────────────────────────────────────────────────
        self._nasa_user  = os.getenv("NASA_EARTHDATA_USERNAME", "")
        self._nasa_pass  = os.getenv("NASA_EARTHDATA_PASSWORD", "")
        self._gee_proj   = os.getenv("GEE_PROJECT_ID", "")

        # ── Date range ────────────────────────────────────────────────────────
        self.start_date = self.cfg["data_pipeline"]["start_date"]
        self.end_date   = self.cfg["data_pipeline"]["end_date"]
        self.bbox       = self.cfg["data_pipeline"]["arctic_bbox"]

        # ── Sub-fetchers ──────────────────────────────────────────────────────
        self.nasa  = NASAFetcher(self._nasa_user, self._nasa_pass, self.raw_dir)
        self.gee   = GEEFetcher(self._gee_proj, self.raw_dir, self.cfg)
        self.noaa  = NOAAFetcher(self.raw_dir)

        logger.info(
            f"DataFetcher initialized | "
            f"{self.start_date} → {self.end_date} | "
            f"bbox: {self.bbox}"
        )

    # ── Credential helpers ────────────────────────────────────────────────────

    def _has_nasa_creds(self) -> bool:
        return bool(self._nasa_user and self._nasa_pass)

    def _has_gee_creds(self) -> bool:
        return bool(self._gee_proj)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        use_synthetic: bool = False,
        max_nasa_files: Optional[int] = None,
    ) -> pd.DataFrame:
    
        logger.info("=" * 60)
        logger.info("ArcticVision Data Ingestion Pipeline START")
        logger.info("=" * 60)

        np.random.seed(int(os.getenv("RANDOM_SEED", 42)))

        # ── Step 1: Sea Ice (NASA) ────────────────────────────────────────────
        logger.info("[1/4] Sea ice concentration (NASA NSIDC-0051)...")
        if not use_synthetic and self._has_nasa_creds():
            self.nasa.authenticate()
            granules = self.nasa.search_granules(
                self.start_date, self.end_date, self.bbox
            )
            self.nasa.download(granules, max_files=max_nasa_files)
            # After download, user should run preprocessing to parse HDF files.
            # For now, generate synthetic alongside for the combined dataset.
            df_ice = self.nasa.generate_synthetic_data(
                self.start_date, self.end_date
            )
        else:
            df_ice = self.nasa.generate_synthetic_data(
                self.start_date, self.end_date
            )

        # ── Step 2: LST (GEE / MODIS) ────────────────────────────────────────
        logger.info("[2/4] Land surface temperature (MODIS via GEE)...")
        if not use_synthetic and self._has_gee_creds():
            self.gee.authenticate()
            df_lst = self.gee.fetch_modis_lst(self.start_date, self.end_date)
        else:
            df_lst = self.gee._synthetic_lst(self.start_date, self.end_date)

        # ── Step 3: 2-m Air Temperature (GEE / ERA5) ──────────────────────────
        logger.info("[3/4] 2-m air temperature (ERA5 via GEE)...")
        if not use_synthetic and self._has_gee_creds() and self.gee._initialized:
            df_era5 = self.gee.fetch_era5_temperature(
                self.start_date, self.end_date
            )
        else:
            df_era5 = self.gee._synthetic_era5(self.start_date, self.end_date)

        # ── Step 4: SST (NOAA ERSSTv5) ───────────────────────────────────────
        logger.info("[4/4] Sea surface temperature (NOAA ERSSTv5)...")
        if not use_synthetic:
            nc = self.noaa.download_ersst()
            df_sst = self.noaa.extract_arctic_sst(nc)
        else:
            df_sst = self.noaa._synthetic_sst_dataframe(
                self.start_date, self.end_date
            )

        # ── Step 5: Merge all sources ─────────────────────────────────────────
        logger.info("Merging all data sources on monthly date key...")
        combined = self._merge_datasets(df_ice, df_lst, df_era5, df_sst)

        # ── Step 6: Save ──────────────────────────────────────────────────────
        out_path = self.raw_dir / "arctic_combined_raw.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info(f"Combined raw dataset saved → {out_path}")
        logger.info(
            f"Shape: {combined.shape} | "
            f"Date range: {combined['date'].min()} → {combined['date'].max()}"
        )
        logger.info("Data Ingestion Pipeline COMPLETE")
        return combined

    # ── Merge helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _merge_datasets(
        df_ice: pd.DataFrame,
        df_lst: pd.DataFrame,
        df_era5: pd.DataFrame,
        df_sst: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join all source DataFrames on a unified 'year_month' key.

        All inputs must have a 'date' column parseable by pd.to_datetime.
        The join key is 'YYYY-MM' to handle minor day offsets between sources.

        Returns:
            Merged DataFrame sorted by date with a clean DatetimeIndex.
        """
        def _add_key(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            d["year_month"] = d["date"].dt.to_period("M").astype(str)
            return d

        df_ice  = _add_key(df_ice)
        df_lst  = _add_key(df_lst) [["year_month", "lst_mean_celsius", "lst_std_celsius"]]
        df_era5 = _add_key(df_era5)[["year_month", "era5_t2m_celsius"]]
        df_sst  = _add_key(df_sst) [["year_month", "arctic_sst_celsius"]]

        merged = (
            df_ice
            .merge(df_lst,  on="year_month", how="left")
            .merge(df_era5, on="year_month", how="left")
            .merge(df_sst,  on="year_month", how="left")
        )

        merged = merged.sort_values("date").reset_index(drop=True)
        merged["year"]  = merged["date"].dt.year
        merged["month"] = merged["date"].dt.month

        # Drop redundant join key
        merged = merged.drop(columns=["year_month"])
        return merged

  

    def load_cached(self) -> Optional[pd.DataFrame]:
        """
        Load previously fetched combined dataset from parquet cache.

        Returns:
            DataFrame if cache exists, None otherwise.
        """
        cache = self.raw_dir / "arctic_combined_raw.parquet"
        if cache.exists():
            logger.info(f"Loading cached dataset from {cache}")
            return pd.read_parquet(cache)
        logger.info("No cached dataset found. Run fetcher.run() first.")
        return None