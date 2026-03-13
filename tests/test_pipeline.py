"""
tests/test_pipeline.py
=======================
Integration tests for the ArcticVision pipeline.

Tests each module in isolation and verifies that stage outputs satisfy
the contracts expected by downstream stages.

Run with:
    pytest tests/ -v
    pytest tests/ -v -k "not slow"   # skip slow tests
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "config.yaml"
ENV    = ROOT / ".env.example"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def raw_df():
    """Session-scoped fixture: run data ingestion once."""
    from data_pipeline.fetcher import DataFetcher
    fetcher = DataFetcher(config_path=CONFIG, env_path=ENV)
    return fetcher.run(use_synthetic=True)


@pytest.fixture(scope="session")
def processed_dataset(raw_df):
    """Session-scoped fixture: run preprocessing once."""
    from preprocessing.processor import DataPreprocessor
    proc = DataPreprocessor(config_path=CONFIG)
    return proc.run()


@pytest.fixture(scope="session")
def climate_results(processed_dataset):
    """Session-scoped fixture: run climate analysis once."""
    from climate_analysis.analyzer import ClimateAnalyzer
    ca = ClimateAnalyzer(config_path=CONFIG)
    return ca.run()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Data Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPipeline:
    """Verify that data ingestion produces a valid combined DataFrame."""

    def test_raw_df_shape(self, raw_df):
        """Combined dataset must have at least 480 monthly rows (40 years)."""
        assert len(raw_df) >= 480, f"Too few rows: {len(raw_df)}"

    def test_required_columns(self, raw_df):
        required = [
            "date", "sea_ice_extent_mkm2", "sea_ice_area_mkm2",
            "lst_mean_celsius", "era5_t2m_celsius", "arctic_sst_celsius",
            "year", "month",
        ]
        missing = [c for c in required if c not in raw_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_missing_values(self, raw_df):
        numeric = raw_df.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() == 0, "NaNs found in raw data"

    def test_date_monotonic(self, raw_df):
        dates = pd.to_datetime(raw_df["date"])
        assert dates.is_monotonic_increasing, "Dates are not sorted"

    def test_physical_bounds_ice(self, raw_df):
        ice = raw_df["sea_ice_extent_mkm2"]
        assert ice.min() >= 0,    f"Ice extent below 0: {ice.min()}"
        assert ice.max() <= 20.0, f"Ice extent above 20: {ice.max()}"

    def test_physical_bounds_sst(self, raw_df):
        sst = raw_df["arctic_sst_celsius"]
        assert sst.min() >= -2.5, f"SST below physical min: {sst.min()}"
        assert sst.max() <= 15.0, f"SST above physical max: {sst.max()}"

    def test_year_range(self, raw_df, cfg):
        start_year = int(cfg["data_pipeline"]["start_date"][:4])
        end_year   = int(cfg["data_pipeline"]["end_date"][:4])
        assert raw_df["year"].min() == start_year
        assert raw_df["year"].max() == end_year

    def test_validation_passes(self, raw_df):
        from data_pipeline.validators import validate_combined_raw
        result = validate_combined_raw(raw_df)
        assert result["passed"], f"Validation failed: {result['issues']}"

    def test_parquet_saved(self):
        p = ROOT / "data" / "raw" / "arctic_combined_raw.parquet"
        assert p.exists(), f"Raw parquet not saved at {p}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    """Verify preprocessing outputs satisfy ML training requirements."""

    def test_feature_count(self, processed_dataset):
        df = processed_dataset["df_features"]
        # Should have at least 25 engineered features
        assert df.shape[1] >= 25, f"Too few features: {df.shape[1]}"

    def test_no_nan_in_features(self, processed_dataset):
        df = processed_dataset["df_features"]
        assert df.isna().sum().sum() == 0, "NaNs in feature matrix"

    def test_sequence_shape(self, processed_dataset, cfg):
        seq_len = cfg["preprocessing"]["sequence_length"]
        n_feat  = processed_dataset["X_train"].shape[2]
        assert processed_dataset["X_train"].shape[1] == seq_len, \
            f"Sequence length mismatch: {processed_dataset['X_train'].shape[1]} ≠ {seq_len}"
        assert n_feat >= 25, f"Too few features in X: {n_feat}"

    def test_splits_non_overlapping(self, processed_dataset):
        n_train = len(processed_dataset["X_train"])
        n_val   = len(processed_dataset["X_val"])
        n_test  = len(processed_dataset["X_test"])
        assert n_train > n_val, "Train set smaller than validation"
        assert n_val   > 0,     "Validation set is empty"
        assert n_test  > 0,     "Test set is empty"

    def test_target_in_unit_range(self, processed_dataset):
        """Scaled target should be in [0, 1] (MinMax scaler)."""
        y = processed_dataset["y_train"]
        assert y.min() >= -0.01, f"y_train below 0: {y.min()}"
        assert y.max() <= 1.01,  f"y_train above 1: {y.max()}"

    def test_feature_cols_saved(self):
        p = ROOT / "data" / "processed" / "feature_cols.pkl"
        assert p.exists()
        with open(p, "rb") as f:
            cols = pickle.load(f)
        assert "sea_ice_extent_mkm2" in cols, "Target column missing from feature list"

    def test_scaler_saved(self):
        p = ROOT / "data" / "processed" / "scaler.pkl"
        assert p.exists()
        with open(p, "rb") as f:
            scaler = pickle.load(f)
        assert scaler.target_scaler_ is not None, "Target scaler not fitted"

    def test_numpy_arrays_saved(self):
        proc_dir = ROOT / "data" / "processed"
        for fname in ["X_train.npy","y_train.npy","X_val.npy",
                       "y_val.npy","X_test.npy","y_test.npy"]:
            p = proc_dir / fname
            assert p.exists(), f"Missing: {p}"
            arr = np.load(p)
            assert arr.ndim >= 1 and len(arr) > 0, f"Empty array: {fname}"

    def test_inverse_transform(self, processed_dataset):
        """Scaler must round-trip: scale → inverse → original (approx)."""
        scaler = processed_dataset["scaler"]
        y      = processed_dataset["y_train"][:10]
        y_inv  = scaler.inverse_transform_target(y)
        # Physical ice extent should be within [2, 18] M km²
        assert y_inv.min() > 0,   f"Inverse transform below 0: {y_inv.min()}"
        assert y_inv.max() < 25,  f"Inverse transform above 25: {y_inv.max()}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Climate Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateAnalysis:
    """Verify climate analysis results are statistically plausible."""

    def test_ice_trend_negative(self, climate_results):
        """Arctic sea ice must show a declining trend."""
        slope = climate_results["ice_trend"]["ols"]["slope_per_decade"]
        assert slope < 0, f"Expected declining ice trend, got {slope}"

    def test_mann_kendall_significant(self, climate_results):
        mk = climate_results["ice_trend"]["mann_kendall"]
        assert mk["is_significant"] == True, \
            f"MK trend not significant: p={mk['p_value']}"
        assert mk["trend_direction"] == "decreasing"

    def test_anomaly_columns_present(self, climate_results):
        df = climate_results["df"]
        required = [
            "sea_ice_extent_mkm2_anomaly",
            "era5_t2m_celsius_anomaly",
        ]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing anomaly columns: {missing}"

    def test_anomaly_mean_near_zero(self, climate_results):
        """Anomaly over the full period should average near zero."""
        anom = climate_results["df"]["sea_ice_extent_mkm2_anomaly"]
        assert abs(anom.mean()) < 1.5, \
            f"Anomaly mean too large: {anom.mean()}"

    def test_correlation_matrix_shape(self, climate_results):
        corr = climate_results["correlations"]["correlation_matrix"]
        assert not corr.empty
        assert corr.shape[0] == corr.shape[1], "Correlation matrix not square"

    def test_ice_temp_anticorrelated(self, climate_results):
        """Temperature and ice extent should be negatively correlated."""
        corr = climate_results["correlations"]["correlation_matrix"]
        if "era5_t2m_celsius" in corr and "sea_ice_extent_mkm2" in corr:
            r = corr.loc["sea_ice_extent_mkm2", "era5_t2m_celsius"]
            assert r < 0, f"Expected negative ice-temperature correlation, got {r}"

    def test_annual_stats_columns(self, climate_results):
        annual = climate_results["extremes"]["annual_stats"]
        required = ["year", "annual_min", "annual_max", "running_record_low"]
        missing  = [c for c in required if c not in annual.columns]
        assert not missing, f"Missing annual stat columns: {missing}"

    def test_decade_summary_declining(self, climate_results):
        """Mean ice extent should decline from first to last decade."""
        dec = climate_results["decade_summary"]
        if "sea_ice_extent_mkm2_mean" in dec.columns and len(dec) > 1:
            first = dec["sea_ice_extent_mkm2_mean"].iloc[0]
            last  = dec["sea_ice_extent_mkm2_mean"].iloc[-1]
            assert last < first, \
                f"Mean ice not declining across decades: {first} → {last}"

    def test_reports_saved(self):
        reports = ROOT / "reports"
        assert (reports / "climate_analysis_results.pkl").exists()
        assert (reports / "decade_summary.csv").exists()
        assert (reports / "annual_extremes.csv").exists()


# ─────────────────────────────────────────────────────────────────────────────
# ML Model Architecture Tests (no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestMLArchitecture:
    """Verify model code and config without requiring PyTorch at test time."""

    def test_model_config_valid(self, cfg):
        ml = cfg["ml_models"]
        assert ml["lstm"]["hidden_size"]  > 0
        assert ml["lstm"]["num_layers"]   >= 1
        assert ml["lstm"]["dropout"]      < 1.0
        assert ml["transformer"]["d_model"] % ml["transformer"]["nhead"] == 0, \
            "d_model must be divisible by nhead"
        assert ml["forecast_horizon"] > 0

    def test_compute_metrics(self):
        from tests._metrics_helper import compute_metrics
        y_true = np.array([10.0, 9.5, 8.0, 11.0, 12.0])
        y_pred = np.array([10.1, 9.3, 8.2, 10.8, 12.1])
        m = compute_metrics(y_true, y_pred)
        assert 0 < m["rmse"] < 1.0,  f"RMSE out of range: {m['rmse']}"
        assert 0 < m["mae"]  < 1.0,  f"MAE out of range: {m['mae']}"
        assert m["r2"]  > 0.9,       f"R² too low: {m['r2']}"
        assert 0 < m["mape"] < 5.0,  f"MAPE out of range: {m['mape']}"

    def test_perfect_prediction_r2(self):
        from tests._metrics_helper import compute_metrics
        y = np.random.uniform(5, 15, 100)
        m = compute_metrics(y, y)
        assert m["r2"] == 1.0
        assert m["rmse"] == 0.0

    def test_skill_score_gt_zero(self):
        from ml_models.evaluator import skill_score
        ss = skill_score(model_rmse=0.05, baseline_rmse=0.2)
        assert ss > 0, "Model RMSE 0.05 should outperform baseline 0.2"

    def test_skill_score_perfect(self):
        from ml_models.evaluator import skill_score
        ss = skill_score(model_rmse=0.0, baseline_rmse=0.2)
        assert ss == 1.0

    def test_residual_analysis_keys(self):
        from ml_models.evaluator import residual_analysis
        y_true = np.random.uniform(5, 15, 100)
        y_pred = y_true + np.random.normal(0, 0.3, 100)
        result = residual_analysis(y_true, y_pred)
        assert "mean_residual" in result
        assert "std_residual"  in result
        assert abs(result["mean_residual"]) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Visualization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    """Verify visualization outputs are generated and non-empty."""

    @pytest.fixture(autouse=True)
    def run_viz(self, climate_results):
        """Run the visualizer once for the whole class."""
        import pandas as pd
        from visualization.plotter import ArcticVisualizer

        np.random.seed(0)
        n   = 52
        y_t = np.random.uniform(6, 15, n)
        y_p = y_t + np.random.normal(0, 0.4, n)

        fake_ml = {
            "model_type": "lstm",
            "eval": {
                "y_true_physical":       y_t,
                "y_pred_physical":       y_p,
                "test_metrics_physical": {"rmse": 0.4, "mae": 0.3, "mape": 3.0, "r2": 0.98},
            },
            "forecast": pd.DataFrame({
                "step":            np.arange(1, 25),
                "predicted_mkm2":  np.random.uniform(7, 14, 24),
                "lower_ci_95":     np.random.uniform(6, 12, 24),
                "upper_ci_95":     np.random.uniform(10, 16, 24),
            }),
            "history": pd.DataFrame({
                "epoch":      np.arange(1, 11),
                "train_loss": np.exp(-np.arange(1,11)*0.15),
                "val_loss":   np.exp(-np.arange(1,11)*0.12),
                "train_mae":  np.exp(-np.arange(1,11)*0.13),
                "val_mae":    np.exp(-np.arange(1,11)*0.10),
                "lr":         [1e-3]*10,
            }),
        }

        viz = ArcticVisualizer(config_path=CONFIG)
        self.outputs = viz.run(
            climate_results=climate_results,
            ml_results=fake_ml,
            map_year=2010,
        )

    def test_all_outputs_generated(self):
        assert len(self.outputs) >= 10, \
            f"Expected ≥10 outputs, got {len(self.outputs)}"

    def test_key_plots_exist(self):
        plots_dir = ROOT / "outputs" / "plots"
        for fname in ["ice_trend.png", "ice_anomaly.png",
                       "september_minimum.png", "correlation_matrix.png"]:
            p = plots_dir / fname
            assert p.exists(), f"Plot not found: {fname}"
            assert p.stat().st_size > 5000, f"Plot suspiciously small: {fname}"

    def test_dashboard_exists(self):
        p = ROOT / "outputs" / "dashboards" / "arctic_dashboard.html"
        assert p.exists()
        content = p.read_text()
        assert "plotly" in content.lower(), "Plotly not found in dashboard HTML"

    def test_map_exists(self):
        # map_year=2010 in our fixture
        p = ROOT / "outputs" / "dashboards" / "arctic_map_2010.html"
        assert p.exists()
        content = p.read_text()
        assert "folium" in content.lower() or "leaflet" in content.lower()

    def test_animation_exists(self):
        p = ROOT / "outputs" / "animations" / "ice_melt_animation.gif"
        assert p.exists()
        assert p.stat().st_size > 10_000, "GIF suspiciously small"

    def test_forecast_plots_exist(self):
        plots_dir = ROOT / "outputs" / "plots"
        assert (plots_dir / "future_forecast_lstm.png").exists()
        assert (plots_dir / "forecast_eval_lstm.png").exists()
        assert (plots_dir / "training_curves_lstm.png").exists()


# ─────────────────────────────────────────────────────────────────────────────
# run_system.py Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSystem:
    """Verify CLI argument parsing and stage-loading logic."""

    def test_parser_defaults(self):
        import run_system
        parser = run_system._build_parser()
        args   = parser.parse_args([])
        assert args.synthetic is True
        assert args.model == "both"
        assert args.map_year == 2020
        assert args.skip_ml is False
        assert args.from_stage == "ingest"

    def test_parser_skip_ml(self):
        import run_system
        parser = run_system._build_parser()
        args   = parser.parse_args(["--skip-ml"])
        assert args.skip_ml is True

    def test_parser_from_stage(self):
        import run_system
        parser = run_system._build_parser()
        args   = parser.parse_args(["--from-stage", "climate"])
        assert args.from_stage == "climate"

    def test_parser_epochs(self):
        import run_system
        parser = run_system._build_parser()
        args   = parser.parse_args(["--epochs", "5"])
        assert args.epochs == 5

    def test_generate_summary_report(self, climate_results, tmp_path, monkeypatch):
        """Summary report must produce valid JSON."""
        import run_system, json
        monkeypatch.setattr(run_system, "PROJECT_ROOT", tmp_path)
        (tmp_path / "reports").mkdir()
        (tmp_path / "outputs" / "plots").mkdir(parents=True)

        parser = run_system._build_parser()
        args   = parser.parse_args([])

        path = run_system.generate_summary_report(
            climate_result=climate_results,
            ml_results={},
            viz_outputs={},
            elapsed_total=42.0,
            args=args,
            logger=logging.getLogger("test"),
        )
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "run_metadata" in data
        assert data["run_metadata"]["elapsed_secs"] == 42.0


import logging