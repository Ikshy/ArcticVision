"""
climate_analysis/analyzer.py
=============================
Core climate analysis engine for ArcticVision.

Analyses performed:
  1. Long-term trend estimation  — OLS linear regression + Sen's slope
  2. Mann-Kendall trend test     — non-parametric significance test
  3. Monthly climatology         — mean seasonal cycle over reference period
  4. Temperature anomaly series  — deviation from baseline climatology
  5. Seasonal decomposition      — trend / seasonal / residual components
  6. Ice–temperature correlation — lagged cross-correlation analysis
  7. Record extremes detection   — yearly min/max with flagging
  8. Decade-by-decade summary    — aggregated statistics per decade

All methods return tidy DataFrames or named dicts so results can be
directly consumed by the visualization module or saved to reports/.

Entry point:
    from climate_analysis.analyzer import ClimateAnalyzer
    ca = ClimateAnalyzer("configs/config.yaml")
    results = ca.run()

Author  : ArcticVision Research Team
Version : 1.0.0
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Optional heavy imports — graceful degradation if not installed
try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    logger.warning("scipy not found. Some analyses will be skipped.")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels not found. Decomposition will be skipped.")


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
# 1. Trend Estimator
# ─────────────────────────────────────────────────────────────────────────────

class TrendEstimator:
    """
    Estimates long-term linear trends in climate time series.

    Methods:
      - OLS linear regression (numpy lstsq)
      - Sen's slope (robust, median-based estimator)
      - Mann-Kendall non-parametric trend test

    All methods operate on monthly or annual pandas Series with a
    DatetimeIndex or a numeric time index.
    """

    @staticmethod
    def ols_trend(series: pd.Series) -> dict:
        """
        Fit an OLS linear trend: y = slope × t + intercept.

        Args:
            series: Numeric pandas Series (NaNs dropped internally).

        Returns:
            dict with keys:
              slope_per_year, intercept, r_squared, p_value,
              trend_line (Series aligned to input index)
        """
        s = series.dropna()
        if len(s) < 3:
            return {}

        # Convert index to fractional years for interpretable slope units
        if hasattr(s.index, "to_timestamp"):
            dates = s.index.to_timestamp()
        elif isinstance(s.index, pd.DatetimeIndex):
            dates = s.index
        else:
            dates = pd.date_range("1979-01", periods=len(s), freq="MS")

        t0   = dates[0]
        t_yr = np.array([(d - t0).days / 365.25 for d in dates])
        y    = s.values

        # OLS via polyfit
        coeffs     = np.polyfit(t_yr, y, deg=1)
        slope      = coeffs[0]           # units per year
        intercept  = coeffs[1]
        trend_vals = np.polyval(coeffs, t_yr)

        # R² and p-value
        ss_res = np.sum((y - trend_vals) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        p_value = np.nan
        if SCIPY_OK:
            _, p_value = scipy_stats.pearsonr(t_yr, y)

        return {
            "slope_per_year": round(slope, 6),
            "slope_per_decade": round(slope * 10, 6),
            "intercept":       round(intercept, 4),
            "r_squared":       round(r2, 4),
            "p_value":         round(float(p_value), 6) if not np.isnan(p_value) else None,
            "trend_line":      pd.Series(trend_vals, index=s.index, name="trend"),
        }

    @staticmethod
    def sens_slope(series: pd.Series) -> dict:
        """
        Compute Sen's slope estimator — a robust, non-parametric
        alternative to OLS that is less sensitive to outliers.

        Sen's slope = median of all pairwise slopes (y_j - y_i) / (j - i)
        for all j > i.

        Args:
            series: Numeric pandas Series.

        Returns:
            dict with 'sens_slope_per_year', 'sens_slope_per_decade',
                       'median_slope', 'confidence_interval_95'
        """
        if not SCIPY_OK:
            logger.warning("scipy required for Sen's slope. Skipping.")
            return {}

        s = series.dropna().values
        n = len(s)
        if n < 3:
            return {}

        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                slopes.append((s[j] - s[i]) / (j - i))

        slopes = np.array(slopes)
        median_slope  = float(np.median(slopes))
        ci_lo, ci_hi  = float(np.percentile(slopes, 2.5)), \
                        float(np.percentile(slopes, 97.5))

        # Convert from per-month to per-year (assumes monthly data)
        per_year = median_slope * 12

        return {
            "sens_slope_per_year":   round(per_year, 6),
            "sens_slope_per_decade": round(per_year * 10, 6),
            "median_slope":          round(median_slope, 6),
            "ci_95_low":             round(ci_lo * 12, 6),
            "ci_95_high":            round(ci_hi * 12, 6),
        }

    @staticmethod
    def mann_kendall(series: pd.Series) -> dict:
        """
        Mann-Kendall non-parametric trend significance test.

        H₀: No monotonic trend.
        H₁: Monotonic trend (two-tailed).

        A p-value < 0.05 rejects H₀ — evidence of a significant trend.
        The MK statistic S > 0 indicates upward trend; S < 0 downward.

        Args:
            series: Numeric pandas Series.

        Returns:
            dict with 'mk_statistic', 'z_score', 'p_value',
                       'trend_direction', 'is_significant'
        """
        s = series.dropna().values
        n = len(s)

        # Compute MK statistic S
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = s[j] - s[i]
                if diff > 0:
                    S += 1
                elif diff < 0:
                    S -= 1

        # Variance of S (no ties assumed — acceptable for climate monthly data)
        var_S = n * (n - 1) * (2 * n + 5) / 18

        # Normalised test statistic Z
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0.0

        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(Z))) if SCIPY_OK else None

        return {
            "mk_statistic":    int(S),
            "z_score":         round(float(Z), 4),
            "p_value":         round(float(p_value), 6) if p_value else None,
            "trend_direction": "decreasing" if S < 0 else ("increasing" if S > 0 else "no trend"),
            "is_significant":  (p_value < 0.05) if p_value else None,
            "n_observations":  n,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Anomaly Calculator
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyCalculator:
    """
    Computes climate anomalies relative to a reference period climatology.

    The anomaly for month m in year y is:
        anomaly(m, y) = value(m, y) − mean(value(m) over baseline years)

    This removes the seasonal cycle so that the residual signal reflects
    only year-to-year and long-term variability.

    Attributes:
        baseline_start: Start of reference period (e.g. '1979-01-01')
        baseline_end:   End of reference period   (e.g. '2000-12-31')
        climatology_:   dict mapping month → baseline mean (fitted)
    """

    def __init__(
        self,
        baseline_start: str = "1979-01-01",
        baseline_end:   str = "2000-12-31",
    ) -> None:
        self.baseline_start = baseline_start
        self.baseline_end   = baseline_end
        self.climatology_: dict = {}

    def fit(self, df: pd.DataFrame, column: str) -> "AnomalyCalculator":
        """
        Compute monthly climatology from the baseline period.

        Args:
            df:     Feature DataFrame with 'date' and target column
            column: Column name to compute climatology for

        Returns:
            self (for chaining)
        """
        base = df[
            (df["date"] >= self.baseline_start) &
            (df["date"] <= self.baseline_end)
        ]
        if len(base) < 12:
            logger.warning("Baseline too short. Using full series climatology.")
            base = df

        self.climatology_[column] = (
            base.groupby(base["date"].dt.month)[column].mean().to_dict()
        )
        logger.info(
            f"Climatology fitted for '{column}' "
            f"({self.baseline_start} → {self.baseline_end})"
        )
        return self

    def transform(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Compute anomaly series for the given column.

        Args:
            df:     DataFrame with 'date' column
            column: Column name (must have been fitted)

        Returns:
            pandas Series of anomaly values, same index as df.
        """
        if column not in self.climatology_:
            self.fit(df, column)

        clim   = df["date"].dt.month.map(self.climatology_[column])
        anomaly = df[column] - clim
        return anomaly.rename(f"{column}_anomaly")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Seasonal Decomposer
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalDecomposer:
    """
    Performs seasonal decomposition of monthly time series using
    statsmodels STL-style additive decomposition.

    Decomposes:   observed = trend + seasonal + residual

    Args:
        period: Seasonal period in months (12 for annual cycle)
        model:  'additive' or 'multiplicative'
    """

    def __init__(self, period: int = 12, model: str = "additive") -> None:
        self.period = period
        self.model  = model

    def decompose(
        self,
        series: pd.Series,
        extrapolate_trend: int = 12,
    ) -> Optional[object]:
        """
        Decompose a monthly time series into trend/seasonal/residual.

        Args:
            series:             Monthly time series with DatetimeIndex or PeriodIndex.
                                Must have no NaN values.
            extrapolate_trend:  Extend trend at boundaries (months).

        Returns:
            statsmodels DecomposeResult object, or None if unavailable.
            Access components via .trend, .seasonal, .resid, .observed
        """
        if not STATSMODELS_OK:
            logger.warning(
                "statsmodels not installed. Seasonal decomposition skipped."
            )
            return None

        s = series.dropna()
        if len(s) < 2 * self.period:
            logger.warning(
                f"Series too short for decomposition (need ≥ {2*self.period} obs)."
            )
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(
                s,
                model=self.model,
                period=self.period,
                extrapolate_trend=extrapolate_trend,
            )
        return result

    @staticmethod
    def decomposition_to_df(result) -> pd.DataFrame:
        """
        Convert a DecomposeResult to a tidy DataFrame.

        Returns:
            DataFrame with columns [date, observed, trend, seasonal, residual]
        """
        df = pd.DataFrame({
            "date":     result.observed.index,
            "observed": result.observed.values,
            "trend":    result.trend.values,
            "seasonal": result.seasonal.values,
            "residual": result.resid.values,
        })
        return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation Analyst
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationAnalyst:
    """
    Analyses relationships between Arctic climate variables.

    Analyses:
      - Pearson correlation matrix for all numeric features
      - Lagged cross-correlations: temperature → sea ice (up to N months lag)
      - Granger causality test: does temperature Granger-cause ice loss?
    """

    @staticmethod
    def correlation_matrix(df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix for selected columns.

        Args:
            df:   Feature DataFrame
            cols: Columns to include (defaults to key climate variables)

        Returns:
            Symmetric correlation matrix as DataFrame.
        """
        DEFAULT_COLS = [
            "sea_ice_extent_mkm2",
            "era5_t2m_celsius",
            "lst_mean_celsius",
            "arctic_sst_celsius",
            "ice_anomaly",
            "ice_yoy_change",
            "temp_gradient",
        ]
        use_cols = [c for c in (cols or DEFAULT_COLS) if c in df.columns]
        corr = df[use_cols].corr(method="pearson").round(3)
        return corr

    @staticmethod
    def lagged_cross_correlation(
        series_x: pd.Series,
        series_y: pd.Series,
        max_lag: int = 24,
        name_x: str = "X",
        name_y: str = "Y",
    ) -> pd.DataFrame:
        """
        Compute cross-correlation between two series at lags 0 … max_lag.

        Useful for studying how temperature anomaly at lag k predicts
        future sea ice extent.

        Convention: positive lag means X leads Y.

        Args:
            series_x: Predictor series (e.g. temperature anomaly)
            series_y: Target series   (e.g. ice extent anomaly)
            max_lag:  Maximum lag months to examine
            name_x:   Label for series X in output
            name_y:   Label for series Y in output

        Returns:
            DataFrame with columns [lag_months, correlation, p_value]
        """
        x = series_x.dropna().values
        y = series_y.dropna().values
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        rows = []
        for lag in range(0, max_lag + 1):
            if lag == 0:
                xi, yi = x, y
            else:
                xi = x[:-lag]
                yi = y[lag:]
            if SCIPY_OK and len(xi) > 5:
                corr, pval = scipy_stats.pearsonr(xi, yi)
            else:
                corr = float(np.corrcoef(xi, yi)[0, 1])
                pval = np.nan
            rows.append({
                "lag_months":  lag,
                f"corr_{name_x}_leads_{name_y}": round(corr, 4),
                "p_value":     round(float(pval), 6) if not np.isnan(pval) else None,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def granger_causality(
        df: pd.DataFrame,
        cause_col: str,
        effect_col: str,
        max_lag: int = 12,
    ) -> pd.DataFrame:
        """
        Test whether `cause_col` Granger-causes `effect_col`.

        H₀: cause_col does NOT improve prediction of effect_col beyond
            its own past values.
        p < 0.05 at a given lag suggests Granger causality.

        Args:
            df:         Feature DataFrame
            cause_col:  Candidate cause variable (e.g. 'era5_t2m_celsius')
            effect_col: Effect variable (e.g. 'sea_ice_extent_mkm2')
            max_lag:    Maximum lag to test

        Returns:
            DataFrame with columns [lag, f_stat, p_value, granger_significant]
        """
        if not STATSMODELS_OK:
            logger.warning("statsmodels required for Granger causality.")
            return pd.DataFrame()

        data = df[[effect_col, cause_col]].dropna()
        if len(data) < max_lag * 5:
            logger.warning("Too few observations for Granger causality test.")
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        rows = []
        for lag, result in gc_res.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val  = result[0]["ssr_ftest"][1]
            rows.append({
                "lag":                    lag,
                "f_stat":                 round(f_stat, 4),
                "p_value":                round(p_val, 6),
                "granger_significant":    p_val < 0.05,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Extremes Detector
# ─────────────────────────────────────────────────────────────────────────────

class ExtremesDetector:
    """
    Identifies record extremes and anomalously low/high years.

    Tracks:
      - Annual minimum sea ice extent (September minimum — the key metric)
      - Annual maximum sea ice extent (March maximum)
      - Record-low years (running records broken)
      - Percentile exceedances (below 10th percentile = extreme low)
    """

    @staticmethod
    def annual_min_max(
        df: pd.DataFrame,
        column: str = "sea_ice_extent_mkm2",
    ) -> pd.DataFrame:
        """
        Compute annual minimum and maximum values.

        For sea ice, the minimum (Arctic sea ice minimum) occurs in
        September; the maximum in March. This method finds both.

        Args:
            df:     Feature DataFrame with 'date' column
            column: Target column

        Returns:
            DataFrame with columns [year, annual_min, min_month,
                                     annual_max, max_month,
                                     min_anomaly_pct, running_record_low]
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

        records = []
        current_record_low = np.inf

        for year, grp in df.groupby("year"):
            idx_min = grp[column].idxmin()
            idx_max = grp[column].idxmax()
            ann_min = grp.loc[idx_min, column]
            ann_max = grp.loc[idx_max, column]

            is_record = ann_min < current_record_low
            if is_record:
                current_record_low = ann_min

            records.append({
                "year":              int(year),
                "annual_min":        round(ann_min, 4),
                "min_month":         int(grp.loc[idx_min, "month"]),
                "annual_max":        round(ann_max, 4),
                "max_month":         int(grp.loc[idx_max, "month"]),
                "running_record_low": is_record,
            })

        result = pd.DataFrame(records)

        # Anomaly relative to the overall mean minimum
        mean_min = result["annual_min"].mean()
        result["min_anomaly_pct"] = (
            (result["annual_min"] - mean_min) / mean_min * 100
        ).round(2)

        # Flag extreme low years (below 10th percentile of annual minimums)
        p10 = result["annual_min"].quantile(0.10)
        result["extreme_low_year"] = result["annual_min"] <= p10

        return result

    @staticmethod
    def september_minimum_trend(
        df: pd.DataFrame,
        column: str = "sea_ice_extent_mkm2",
    ) -> dict:
        """
        Focused analysis on September sea ice minimum — the most
        widely cited indicator of Arctic sea ice loss.

        Args:
            df:     Feature DataFrame with 'date' column
            column: Sea ice extent column

        Returns:
            dict with OLS and Mann-Kendall results for September values only,
            plus the September-only time series as a DataFrame.
        """
        df = df.copy()
        df["date"]  = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month

        sept = df[df["month"] == 9][["date", column]].dropna()
        sept = sept.set_index("date")[column]

        te = TrendEstimator()
        ols = te.ols_trend(sept)
        mk  = te.mann_kendall(sept)
        ss  = te.sens_slope(sept)

        logger.info(
            f"September minimum trend: "
            f"{ols.get('slope_per_decade', 'N/A'):.4f} M km²/decade "
            f"(p={mk.get('p_value', 'N/A')})"
        )
        return {
            "series":      sept.reset_index(),
            "ols":         ols,
            "mann_kendall": mk,
            "sens_slope":  ss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decade Summariser
# ─────────────────────────────────────────────────────────────────────────────

class DecadeSummariser:
    """
    Aggregates climate indicators by decade for high-level reporting.

    Output columns per decade:
      mean_ice_extent, std_ice_extent, mean_t2m, ice_extent_change_pct
    """

    @staticmethod
    def summarise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce decade-by-decade summary statistics.

        Args:
            df: Feature DataFrame with 'date', 'sea_ice_extent_mkm2',
                and temperature columns.

        Returns:
            DataFrame indexed by decade label (e.g. '1980s', '1990s').
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["decade"] = (df["date"].dt.year // 10 * 10).astype(str) + "s"

        agg_cols = {}
        if "sea_ice_extent_mkm2" in df.columns:
            agg_cols["sea_ice_extent_mkm2"] = ["mean", "std", "min", "max"]
        if "era5_t2m_celsius" in df.columns:
            agg_cols["era5_t2m_celsius"] = ["mean", "std"]
        if "arctic_sst_celsius" in df.columns:
            agg_cols["arctic_sst_celsius"] = ["mean"]

        summary = df.groupby("decade").agg(agg_cols).round(3)
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]

        # Percentage change in mean ice extent relative to first decade
        if "sea_ice_extent_mkm2_mean" in summary.columns:
            ref = summary["sea_ice_extent_mkm2_mean"].iloc[0]
            summary["ice_extent_change_pct"] = (
                (summary["sea_ice_extent_mkm2_mean"] - ref) / ref * 100
            ).round(2)

        return summary.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Master ClimateAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ClimateAnalyzer:
    """
    Master orchestrator for all ArcticVision climate analyses.

    Chains all analysis components and produces a unified results
    dictionary saved to reports/climate_analysis_results.pkl and
    individual CSV summaries in reports/.

    Args:
        config_path: Path to configs/config.yaml

    Example:
        >>> ca = ClimateAnalyzer("configs/config.yaml")
        >>> results = ca.run()
        >>> print(results["september_trend"]["ols"]["slope_per_decade"])
        -0.847  # M km² per decade
    """

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.cfg        = _load_config(config_path)
        self.root       = Path(config_path).parent.parent
        self.proc_dir   = self.root / self.cfg["paths"]["data_processed"]
        self.reports_dir = _ensure_dir(self.root / self.cfg["paths"]["reports"])

        cl = self.cfg["climate_analysis"]
        self.baseline_start = cl["baseline_period"][0]
        self.baseline_end   = cl["baseline_period"][1]
        self.sig_level      = cl["trend_significance"]

        self.trend_est   = TrendEstimator()
        self.anomaly_calc = AnomalyCalculator(self.baseline_start, self.baseline_end)
        self.decomposer  = SeasonalDecomposer(period=cl["seasonal_periods"])
        self.corr_analyst = CorrelationAnalyst()
        self.extremes    = ExtremesDetector()
        self.decades     = DecadeSummariser()

    # ── Data loader ───────────────────────────────────────────────────────────

    def _load_features(self) -> pd.DataFrame:
        """Load feature-engineered (unscaled) DataFrame from preprocessing."""
        path = self.proc_dir / "arctic_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed features not found at {path}. "
                "Run DataPreprocessor().run() first."
            )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Features loaded: {df.shape} from {path}")
        return df

    # ── Individual analyses ───────────────────────────────────────────────────

    def analyse_ice_trend(self, df: pd.DataFrame) -> dict:
        """
        Full trend analysis on monthly sea ice extent.

        Returns:
            dict with 'ols', 'sens_slope', 'mann_kendall' sub-dicts.
        """
        logger.info("Analysing ice extent trend (OLS + MK + Sen)...")
        series = df.set_index("date")["sea_ice_extent_mkm2"]
        return {
            "ols":         self.trend_est.ols_trend(series),
            "sens_slope":  self.trend_est.sens_slope(series),
            "mann_kendall": self.trend_est.mann_kendall(series),
        }

    def compute_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute anomaly columns for ice extent and temperature variables.

        Returns:
            df with additional '*_anomaly' columns appended.
        """
        logger.info("Computing climate anomalies...")
        anomaly_targets = [
            "sea_ice_extent_mkm2",
            "era5_t2m_celsius",
            "arctic_sst_celsius",
            "lst_mean_celsius",
        ]
        df = df.copy()
        for col in anomaly_targets:
            if col in df.columns:
                df[f"{col}_anomaly"] = self.anomaly_calc.fit(df, col).transform(df, col)
                logger.debug(f"  Anomaly computed for '{col}'")
        return df

    def decompose_series(self, df: pd.DataFrame) -> dict:
        """
        Seasonal decomposition of ice extent and temperature.

        Returns:
            dict mapping column name → DecomposeResult (or None).
        """
        logger.info("Running seasonal decomposition...")
        targets = ["sea_ice_extent_mkm2", "era5_t2m_celsius"]
        results = {}
        ts = df.set_index("date")

        for col in targets:
            if col not in ts.columns:
                continue
            result = self.decomposer.decompose(ts[col])
            if result is not None:
                results[col] = {
                    "decompose_result": result,
                    "df":               SeasonalDecomposer.decomposition_to_df(result),
                }
                logger.info(f"  Decomposed '{col}' ✓")
        return results

    def analyse_correlations(self, df: pd.DataFrame) -> dict:
        """
        Full correlation analysis between all climate variables.

        Returns:
            dict with 'correlation_matrix', 'lagged_xcorr', 'granger'.
        """
        logger.info("Computing correlations and cross-correlations...")

        # ── Pearson correlation matrix ────────────────────────────────────────
        corr_matrix = self.corr_analyst.correlation_matrix(df)

        # ── Lagged cross-correlations: T2m → ice extent ───────────────────────
        xcorr = pd.DataFrame()
        if "era5_t2m_celsius_anomaly" in df.columns and \
           "sea_ice_extent_mkm2_anomaly" in df.columns:
            xcorr = self.corr_analyst.lagged_cross_correlation(
                df["era5_t2m_celsius_anomaly"],
                df["sea_ice_extent_mkm2_anomaly"],
                max_lag=24,
                name_x="T2m",
                name_y="Ice",
            )

        # ── Granger causality ─────────────────────────────────────────────────
        granger = pd.DataFrame()
        if "era5_t2m_celsius" in df.columns:
            granger = self.corr_analyst.granger_causality(
                df,
                cause_col="era5_t2m_celsius",
                effect_col="sea_ice_extent_mkm2",
                max_lag=12,
            )

        return {
            "correlation_matrix": corr_matrix,
            "lagged_xcorr":       xcorr,
            "granger":            granger,
        }

    def analyse_extremes(self, df: pd.DataFrame) -> dict:
        """
        Detect extreme years and compute September minimum trend.

        Returns:
            dict with 'annual_stats' and 'september_trend'.
        """
        logger.info("Analysing extreme events and September minimum trend...")
        return {
            "annual_stats":     self.extremes.annual_min_max(df),
            "september_trend":  self.extremes.september_minimum_trend(df),
        }

    def analyse_decades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute decade-by-decade summary statistics."""
        logger.info("Summarising by decade...")
        return self.decades.summarise(df)

    # ── Result reporting helpers ──────────────────────────────────────────────

    def _log_headline_findings(self, results: dict) -> None:
        """Print key findings to the logger for quick inspection."""
        logger.info("=" * 60)
        logger.info("CLIMATE ANALYSIS — HEADLINE FINDINGS")
        logger.info("=" * 60)

        # Overall ice trend
        ols = results["ice_trend"]["ols"]
        mk  = results["ice_trend"]["mann_kendall"]
        ss  = results["ice_trend"]["sens_slope"]
        logger.info(
            f"  Ice extent OLS trend    : "
            f"{ols.get('slope_per_decade', 'N/A'):.4f} M km²/decade  "
            f"(R²={ols.get('r_squared','N/A')}, p={ols.get('p_value','N/A')})"
        )
        logger.info(
            f"  Sen's slope             : "
            f"{ss.get('sens_slope_per_decade', 'N/A'):.4f} M km²/decade"
        )
        logger.info(
            f"  Mann-Kendall            : "
            f"{mk.get('trend_direction','N/A')}  "
            f"(Z={mk.get('z_score','N/A')}, p={mk.get('p_value','N/A')}, "
            f"significant={mk.get('is_significant','N/A')})"
        )

        # September minimum trend
        sept = results["extremes"]["september_trend"]["ols"]
        logger.info(
            f"  September min trend     : "
            f"{sept.get('slope_per_decade','N/A'):.4f} M km²/decade"
        )

        # Strongest T2m–ice correlation
        xcorr = results["correlations"]["lagged_xcorr"]
        if not xcorr.empty:
            col = [c for c in xcorr.columns if c.startswith("corr_")][0]
            best_lag = xcorr.loc[xcorr[col].abs().idxmax(), "lag_months"]
            best_corr = xcorr[col].abs().max()
            logger.info(
                f"  Peak T2m→Ice xcorr      : "
                f"r={best_corr:.3f} at lag {int(best_lag)} months"
            )

        # Decade summary
        dec = results["decade_summary"]
        if "ice_extent_change_pct" in dec.columns:
            last = dec.iloc[-1]
            logger.info(
                f"  Latest decade ice change: "
                f"{last['ice_extent_change_pct']:.1f}% vs 1st decade"
            )
        logger.info("=" * 60)

    def _save_reports(self, results: dict, df_with_anomalies: pd.DataFrame) -> None:
        """Save key result DataFrames as CSVs to reports/."""
        import pickle

        # Save full results dict
        with open(self.reports_dir / "climate_analysis_results.pkl", "wb") as f:
            pickle.dump(results, f)

        # Save individual CSVs for inspection / publication
        df_with_anomalies.to_csv(
            self.reports_dir / "arctic_features_with_anomalies.csv", index=False
        )
        results["decade_summary"].to_csv(
            self.reports_dir / "decade_summary.csv", index=False
        )
        results["extremes"]["annual_stats"].to_csv(
            self.reports_dir / "annual_extremes.csv", index=False
        )
        if not results["correlations"]["correlation_matrix"].empty:
            results["correlations"]["correlation_matrix"].to_csv(
                self.reports_dir / "correlation_matrix.csv"
            )
        if not results["correlations"]["lagged_xcorr"].empty:
            results["correlations"]["lagged_xcorr"].to_csv(
                self.reports_dir / "lagged_xcorr.csv", index=False
            )
        if "september_trend" in results["extremes"]:
            sept = results["extremes"]["september_trend"]["series"]
            if isinstance(sept, pd.DataFrame):
                sept.to_csv(self.reports_dir / "september_minimum.csv", index=False)

        logger.info(f"Reports saved to {self.reports_dir}")

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full climate analysis pipeline.

        Returns:
            dict with keys:
              'df'               : Feature DataFrame with anomaly columns added
              'ice_trend'        : OLS + MK + Sen trend results
              'anomaly_series'   : DataFrame of monthly anomaly columns
              'decomposition'    : Seasonal decomposition results per variable
              'correlations'     : Correlation matrix + lagged xcorr + Granger
              'extremes'         : Annual min/max stats + September trend
              'decade_summary'   : Decade-aggregated statistics
        """
        logger.info("=" * 60)
        logger.info("ArcticVision Climate Analysis Pipeline START")
        logger.info("=" * 60)

        # ── 1. Load data ──────────────────────────────────────────────────────
        df = self._load_features()

        # ── 2. Ice trend ──────────────────────────────────────────────────────
        ice_trend = self.analyse_ice_trend(df)

        # ── 3. Anomalies ──────────────────────────────────────────────────────
        df = self.compute_anomalies(df)

        # ── 4. Seasonal decomposition ─────────────────────────────────────────
        decomposition = self.decompose_series(df)

        # ── 5. Correlations ───────────────────────────────────────────────────
        correlations = self.analyse_correlations(df)

        # ── 6. Extremes ───────────────────────────────────────────────────────
        extremes = self.analyse_extremes(df)

        # ── 7. Decade summary ─────────────────────────────────────────────────
        decade_summary = self.analyse_decades(df)

        results = {
            "df":              df,
            "ice_trend":       ice_trend,
            "decomposition":   decomposition,
            "correlations":    correlations,
            "extremes":        extremes,
            "decade_summary":  decade_summary,
        }

        self._log_headline_findings(results)
        self._save_reports(results, df)

        logger.info("Climate Analysis Pipeline COMPLETE")
        return results

    # ── Convenience loader ────────────────────────────────────────────────────

    @staticmethod
    def load_results(
        reports_dir: str | Path = "reports",
    ) -> dict:
        """
        Load saved climate analysis results from pickle.

        Args:
            reports_dir: Path to reports/ directory.

        Returns:
            dict matching the structure returned by run().
        """
        import pickle
        path = Path(reports_dir) / "climate_analysis_results.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Results not found at {path}. Run ClimateAnalyzer().run() first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)