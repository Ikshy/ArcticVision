#!/usr/bin/env python3
"""
run_system.py
=============
ArcticVision — End-to-End Pipeline Runner

This script sequentially executes all five ArcticVision modules and
produces a complete set of research outputs:

    ┌─────────────────┐
    │  data_pipeline  │  Ingest satellite + climate data (NASA/GEE/NOAA)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  preprocessing  │  Clean, engineer features, build ML sequences
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │climate_analysis │  Trend tests, anomaly, decomposition, correlations
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   ml_models     │  Train LSTM + Transformer, evaluate, forecast
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  visualization  │  14 plots, interactive map, animation, dashboard
    └─────────────────┘

Usage:
    # Full pipeline with synthetic data (no credentials required):
    python run_system.py

    # Full pipeline with real NASA / GEE data:
    python run_system.py --real-data

    # Skip ML training (fast mode for analysis + viz only):
    python run_system.py --skip-ml

    # Train only the Transformer model:
    python run_system.py --model transformer

    # Override epoch count (e.g. quick test):
    python run_system.py --epochs 5

    # Load cached intermediate results (skip re-running earlier stages):
    python run_system.py --from-stage climate

    # Choose map year for interactive Arctic map:
    python run_system.py --map-year 2023

    # Full help:
    python run_system.py --help

Output locations:
    data/raw/                  Raw ingested data
    data/processed/            Cleaned + feature-engineered datasets
    outputs/plots/             Static PNG visualizations (12 files)
    outputs/animations/        Ice melt GIF animation
    outputs/dashboards/        Interactive HTML map + Plotly dashboard
    outputs/models/            Trained model weights + metadata
    reports/                   Climate analysis CSVs + summary pickle

Author  : ArcticVision Research Team
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def _setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure root logger with console + file handlers."""
    log_dir = PROJECT_ROOT / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("arcticvision")
    logger.info(f"Log file: {log_file}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners — each is isolated so --from-stage can skip earlier stages
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(args, logger: logging.Logger) -> dict:
    """
    Stage 1 — Data Ingestion.

    Fetches or synthesises:
      - NASA NSIDC-0051 sea ice concentration
      - MODIS LST + ERA5 reanalysis (via GEE)
      - NOAA ERSSTv5 sea surface temperature

    Returns:
        dict with 'combined_df' (raw combined DataFrame)
    """
    logger.info("▶  STAGE 1: Data Ingestion")
    from data_pipeline.fetcher   import DataFetcher
    from data_pipeline.validators import validate_combined_raw

    fetcher = DataFetcher(
        config_path=args.config,
        env_path=args.env,
    )

    # If cache exists and --use-cache flag set, try loading it
    if args.use_cache:
        cached = fetcher.load_cached()
        if cached is not None:
            logger.info("Using cached raw dataset (--use-cache)")
            validation = validate_combined_raw(cached)
            if not validation["passed"]:
                logger.warning(f"Cache validation issues: {validation['issues']}")
            return {"combined_df": cached}

    # Run ingestion (synthetic if no credentials / --synthetic flag)
    use_synthetic = args.synthetic or not (
        os.getenv("NASA_EARTHDATA_USERNAME") and os.getenv("GEE_PROJECT_ID")
    )
    if use_synthetic:
        logger.info("Mode: SYNTHETIC data (no live API credentials detected)")
    else:
        logger.info("Mode: LIVE data (NASA + GEE credentials found)")

    combined_df = fetcher.run(
        use_synthetic=use_synthetic,
        max_nasa_files=args.max_nasa_files,
    )

    # Validate
    validation = validate_combined_raw(combined_df)
    if not validation["passed"]:
        logger.warning(f"Data validation issues: {validation['issues']}")
    else:
        logger.info("Data validation: PASSED ✓")

    logger.info(f"✅ Stage 1 complete — {len(combined_df)} monthly records ingested")
    return {"combined_df": combined_df}


def run_preprocessing(args, logger: logging.Logger) -> dict:
    """
    Stage 2 — Preprocessing.

    Cleans data, engineers 29+ features, normalises, builds
    LSTM-ready (X, y) sequence arrays and splits into train/val/test.

    Returns:
        dict with all preprocessed arrays + scaler + feature columns
    """
    logger.info("▶  STAGE 2: Preprocessing")
    from preprocessing.processor   import DataPreprocessor
    from preprocessing.diagnostics import describe_features, sequence_shape_report

    proc    = DataPreprocessor(config_path=args.config)
    dataset = proc.run()

    # Print feature statistics
    desc = describe_features(dataset["df_features"])
    logger.info(f"Feature matrix: {dataset['df_features'].shape[1]} cols, "
                f"{dataset['df_features'].shape[0]} rows")
    logger.info(f"Sequence shapes — "
                f"X_train: {dataset['X_train'].shape}  "
                f"X_val: {dataset['X_val'].shape}  "
                f"X_test: {dataset['X_test'].shape}")

    logger.info(f"✅ Stage 2 complete — preprocessed artefacts saved to data/processed/")
    return dataset


def run_climate_analysis(args, logger: logging.Logger) -> dict:
    """
    Stage 3 — Climate Analysis.

    Runs:
      - OLS + Sen slope + Mann-Kendall trend tests
      - Monthly anomaly computation (vs 1979-2000 baseline)
      - Seasonal decomposition (trend + seasonal + residual)
      - Pearson correlation matrix + lagged cross-correlation
      - Granger causality (T2m → ice extent)
      - Annual extremes detection + September minimum trend
      - Decade-by-decade aggregation

    Returns:
        dict with all analysis results (matches ClimateAnalyzer.run() output)
    """
    logger.info("▶  STAGE 3: Climate Analysis")
    from climate_analysis.analyzer import ClimateAnalyzer

    ca      = ClimateAnalyzer(config_path=args.config)
    results = ca.run()

    # Log key findings
    ols = results["ice_trend"]["ols"]
    mk  = results["ice_trend"]["mann_kendall"]
    logger.info(
        f"Key finding: Ice extent trend = "
        f"{ols.get('slope_per_decade', 'N/A'):.4f} M km²/decade  "
        f"[MK p={mk.get('p_value', 'N/A')}, "
        f"significant={mk.get('is_significant', 'N/A')}]"
    )

    logger.info(f"✅ Stage 3 complete — reports saved to reports/")
    return results


def run_ml(args, logger: logging.Logger) -> dict:
    """
    Stage 4 — Machine Learning.

    Trains LSTM and/or Transformer models, evaluates on the test set,
    generates 24-month forecasts with 95% MC-Dropout CIs, and saves
    model weights + metadata.

    Returns:
        dict with model results keyed by model type
        e.g. {'lstm': {...}, 'transformer': {...}}
    """
    logger.info("▶  STAGE 4: Machine Learning")

    try:
        import torch
        logger.info(f"PyTorch {torch.__version__} | device: "
                    f"{'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        logger.error(
            "PyTorch not installed!\n"
            "Install with:  pip install torch\n"
            "Then re-run:   python run_system.py --from-stage ml"
        )
        return {}

    from ml_models.trainer  import ModelTrainer
    from ml_models.evaluator import compare_models, persistence_baseline, skill_score

    trainer    = ModelTrainer(config_path=args.config)
    ml_results = {}

    models_to_train = (
        [args.model] if args.model != "both"
        else ["lstm", "transformer"]
    )

    for model_type in models_to_train:
        logger.info(f"--- Training {model_type.upper()} ---")
        result = trainer.run(
            model_type=model_type,
            epochs=args.epochs,
        )
        ml_results[model_type] = result

        # Quick skill score vs persistence baseline
        y_test  = result["eval"]["y_true_scaled"]
        base    = persistence_baseline(y_test)
        model_rmse = result["eval"]["test_metrics_scaled"]["rmse"]
        ss = skill_score(model_rmse, base["rmse"])
        logger.info(
            f"{model_type.upper()} skill score vs persistence: {ss:.4f} "
            f"({'✓ outperforms' if ss > 0 else '✗ underperforms'} baseline)"
        )

    # Model comparison table (if both trained)
    if len(models_to_train) > 1:
        cmp = compare_models(str(PROJECT_ROOT / "outputs" / "models"))
        if not cmp.empty:
            logger.info(f"\n=== Model Comparison ===\n{cmp.to_string()}")

    logger.info(f"✅ Stage 4 complete — model weights saved to outputs/models/")
    return ml_results


def run_visualization(
    args,
    logger: logging.Logger,
    climate_results: dict,
    ml_results: dict,
) -> dict:
    """
    Stage 5 — Visualization.

    Produces all 14 outputs:
      Static PNGs (12): trend, anomaly, September minimum, seasonal heatmaps,
                        correlation matrix, decompositions (×2), forecast eval,
                        future forecast, training curves, temperature heatmap
      Interactive (2): Folium Arctic map, Plotly 4-panel dashboard
      Animation  (1): Polar bar-chart GIF of ice melt by decade

    Returns:
        dict mapping plot name → output Path
    """
    logger.info("▶  STAGE 5: Visualization")
    from visualization.plotter import ArcticVisualizer

    viz = ArcticVisualizer(config_path=args.config)

    # Pass the first trained model's results if available
    ml_result_for_viz = None
    if ml_results:
        first_model = list(ml_results.keys())[0]
        ml_result_for_viz = ml_results[first_model]
        ml_result_for_viz["model_type"] = first_model

    outputs = viz.run(
        climate_results=climate_results,
        ml_results=ml_result_for_viz,
        map_year=args.map_year,
    )

    logger.info(f"✅ Stage 5 complete — {len(outputs)} outputs generated")
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Summary report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_report(
    climate_results: dict,
    ml_results:      dict,
    viz_outputs:     dict,
    elapsed_total:   float,
    args,
    logger: logging.Logger,
) -> Path:
    """
    Write a concise JSON + human-readable TXT summary of all pipeline outputs.

    Args:
        climate_results: Output from Stage 3
        ml_results:      Output from Stage 4
        viz_outputs:     Output from Stage 5
        elapsed_total:   Total wall-clock seconds for entire pipeline
        args:            Parsed CLI arguments
        logger:          Logger instance

    Returns:
        Path to saved JSON summary file.
    """
    report = {
        "run_metadata": {
            "timestamp":    datetime.now().isoformat(),
            "elapsed_secs": round(elapsed_total, 1),
            "config":       str(args.config),
            "synthetic":    args.synthetic,
            "models":       args.model,
        },
        "climate_analysis": {},
        "ml_results":        {},
        "outputs":           {},
    }

    # Climate findings
    if climate_results:
        ols  = climate_results.get("ice_trend", {}).get("ols", {})
        mk   = climate_results.get("ice_trend", {}).get("mann_kendall", {})
        ss   = climate_results.get("extremes", {}).get(
            "september_trend", {}
        ).get("ols", {})
        report["climate_analysis"] = {
            "ice_trend_slope_per_decade_mkm2":  ols.get("slope_per_decade"),
            "ice_trend_r2":                     ols.get("r_squared"),
            "ice_trend_p_value":                ols.get("p_value"),
            "mann_kendall_direction":           mk.get("trend_direction"),
            "mann_kendall_significant":         mk.get("is_significant"),
            "september_slope_per_decade_mkm2":  ss.get("slope_per_decade"),
        }
        dec = climate_results.get("decade_summary")
        if dec is not None and not dec.empty:
            try:
                last = dec.iloc[-1]
                report["climate_analysis"]["latest_decade_ice_change_pct"] = (
                    float(last.get("ice_extent_change_pct", "nan"))
                )
            except Exception:
                pass

    # ML findings
    for model_type, result in ml_results.items():
        eval_r  = result.get("eval", {})
        phys    = eval_r.get("test_metrics_physical")
        scaled  = eval_r.get("test_metrics_scaled")
        metrics = phys or scaled or {}
        report["ml_results"][model_type] = {
            "test_rmse":  metrics.get("rmse"),
            "test_mae":   metrics.get("mae"),
            "test_r2":    metrics.get("r2"),
            "test_mape":  metrics.get("mape"),
            "units":      "M_km2" if phys else "scaled",
            "epochs_trained": (
                len(result.get("history", [])) if result.get("history") is not None
                else None
            ),
        }

    # Output paths
    for name, path in viz_outputs.items():
        if path:
            from pathlib import Path as P
            p = P(str(path))
            if p.exists():
                report["outputs"][name] = {
                    "path":    str(p.relative_to(PROJECT_ROOT)),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }

    # Save JSON
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / "pipeline_summary.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save human-readable TXT
    txt_path = reports_dir / "pipeline_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  ARCTICVISION — PIPELINE RUN SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Run timestamp : {report['run_metadata']['timestamp']}\n")
        f.write(f"  Total elapsed : {elapsed_total:.1f} seconds\n")
        f.write(f"  Data mode     : {'Synthetic' if args.synthetic else 'Live'}\n")
        f.write(f"  Models trained: {args.model}\n")
        f.write("\n--- CLIMATE ANALYSIS ---\n")
        ca = report["climate_analysis"]
        f.write(f"  Ice trend (OLS)  : {ca.get('ice_trend_slope_per_decade_mkm2')} M km²/decade\n")
        f.write(f"  Trend R²         : {ca.get('ice_trend_r2')}\n")
        f.write(f"  Mann-Kendall     : {ca.get('mann_kendall_direction')}  "
                f"(significant={ca.get('mann_kendall_significant')})\n")
        f.write(f"  Sept. min trend  : {ca.get('september_slope_per_decade_mkm2')} M km²/decade\n")
        f.write(f"  Latest decade Δ  : {ca.get('latest_decade_ice_change_pct')}%\n")
        f.write("\n--- ML MODELS ---\n")
        for mt, mr in report["ml_results"].items():
            f.write(f"  {mt.upper()}\n")
            f.write(f"    RMSE  : {mr.get('test_rmse')}  ({mr.get('units')})\n")
            f.write(f"    MAE   : {mr.get('test_mae')}\n")
            f.write(f"    R²    : {mr.get('test_r2')}\n")
            f.write(f"    MAPE  : {mr.get('test_mape')}%\n")
            f.write(f"    Epochs: {mr.get('epochs_trained')}\n")
        f.write("\n--- OUTPUTS GENERATED ---\n")
        for name, info in report["outputs"].items():
            f.write(f"  {name:28s} {info['path']}  ({info['size_kb']} KB)\n")
        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"Pipeline summary saved → {json_path}")
    logger.info(f"Human-readable summary → {txt_path}")
    return json_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arcticvision",
        description="ArcticVision — AI-Driven Arctic Climate Change Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py                          # Full pipeline, synthetic data
  python run_system.py --real-data              # Full pipeline, live NASA/GEE
  python run_system.py --skip-ml               # Analysis + viz, no training
  python run_system.py --model transformer      # Train Transformer only
  python run_system.py --epochs 5              # Quick smoke test (5 epochs)
  python run_system.py --from-stage climate    # Resume from climate analysis
  python run_system.py --map-year 2023         # Use 2023 for Arctic map

Stages:
  ingest → preprocess → climate → ml → visualize

Note:
  Without NASA / GEE credentials in .env, synthetic data is used automatically.
  Run 'python setup_project.py' to verify your environment first.
        """,
    )

    # Config
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--env", type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to .env credentials file (default: .env)",
    )

    # Data options
    data_grp = parser.add_argument_group("Data options")
    data_grp.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic data even if credentials are available",
    )
    data_grp.add_argument(
        "--real-data", dest="synthetic", action="store_false",
        help="Use live NASA/GEE data (requires credentials in .env)",
    )
    data_grp.add_argument(
        "--use-cache", action="store_true", default=False,
        help="Load raw data from cache if available (skip re-fetching)",
    )
    data_grp.add_argument(
        "--max-nasa-files", type=int, default=None, metavar="N",
        help="Cap NASA granule downloads at N files (useful for testing)",
    )
    parser.set_defaults(synthetic=True)

    # Pipeline control
    pipe_grp = parser.add_argument_group("Pipeline control")
    pipe_grp.add_argument(
        "--from-stage",
        choices=["ingest", "preprocess", "climate", "ml", "visualize"],
        default="ingest",
        help=(
            "Resume pipeline from this stage (earlier stages must have "
            "already produced their outputs). Default: ingest"
        ),
    )
    pipe_grp.add_argument(
        "--skip-ml", action="store_true", default=False,
        help="Skip ML training entirely (run analysis + viz only)",
    )

    # ML options
    ml_grp = parser.add_argument_group("ML options")
    ml_grp.add_argument(
        "--model",
        choices=["lstm", "transformer", "both"],
        default="both",
        help="Which model(s) to train (default: both)",
    )
    ml_grp.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs (default: from config.yaml)",
    )

    # Visualization options
    viz_grp = parser.add_argument_group("Visualization options")
    viz_grp.add_argument(
        "--map-year", type=int, default=2020, metavar="YEAR",
        help="Year to visualise on the interactive Arctic map (default: 2020)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Stage loader — load already-saved results when --from-stage is used
# ─────────────────────────────────────────────────────────────────────────────

def _load_stage_cache(
    from_stage: str,
    args,
    logger: logging.Logger,
) -> tuple[dict, dict, dict]:
    """
    Load previously-computed results when resuming mid-pipeline.

    Returns:
        (ingestion_result, climate_result, ml_result)
        Any stage not yet computed is returned as an empty dict.
    """
    ingestion_result = {}
    climate_result   = {}
    ml_result        = {}

    STAGE_ORDER = ["ingest", "preprocess", "climate", "ml", "visualize"]
    from_idx    = STAGE_ORDER.index(from_stage)

    # If resuming at preprocess or later, preprocessing outputs must exist
    if from_idx >= 1:
        from preprocessing.processor import DataPreprocessor
        try:
            proc_dir = PROJECT_ROOT / "data" / "processed"
            dataset  = DataPreprocessor.load_processed(proc_dir)
            ingestion_result = {"combined_df": dataset.get("df_features")}
            logger.info("Loaded cached preprocessing artefacts ✓")
        except FileNotFoundError as e:
            logger.error(f"Cannot resume from '{from_stage}': {e}")
            sys.exit(1)

    # If resuming at climate or later, load climate results
    if from_idx >= 2:
        from climate_analysis.analyzer import ClimateAnalyzer
        try:
            climate_result = ClimateAnalyzer.load_results(
                PROJECT_ROOT / "reports"
            )
            logger.info("Loaded cached climate analysis results ✓")
        except FileNotFoundError as e:
            logger.error(f"Cannot resume from '{from_stage}': {e}")
            sys.exit(1)

    # If resuming at ml or later, try to load model metadata
    if from_idx >= 3:
        import json as _json
        models_dir = PROJECT_ROOT / "outputs" / "models"
        for model_type in ["lstm", "transformer"]:
            meta_path = models_dir / f"{model_type}_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = _json.load(f)
                # Re-create a minimal result dict for visualization
                ml_result[model_type] = {
                    "model_type": model_type,
                    "eval": {
                        "test_metrics_scaled":   meta.get("test_metrics_scaled"),
                        "test_metrics_physical": meta.get("test_metrics_physical"),
                        "y_true_scaled": None,
                        "y_pred_scaled": None,
                    },
                    "forecast": _load_forecast_csv(models_dir, model_type),
                    "history":  _load_history_csv(models_dir, model_type),
                }
                logger.info(f"Loaded cached {model_type} model metadata ✓")

    return ingestion_result, climate_result, ml_result


def _load_forecast_csv(models_dir: Path, model_type: str):
    """Load forecast CSV if it exists, else return empty DataFrame."""
    import pandas as pd
    p = models_dir / f"{model_type}_forecast.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_history_csv(models_dir: Path, model_type: str):
    """Load training history CSV if it exists, else return empty DataFrame."""
    import pandas as pd
    p = models_dir / f"{model_type}_history.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          █████╗ ██████╗  ██████╗████████╗██╗ ██████╗        ║
║         ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝        ║
║         ███████║██████╔╝██║        ██║   ██║██║              ║
║         ██╔══██║██╔══██╗██║        ██║   ██║██║              ║
║         ██║  ██║██║  ██║╚██████╗   ██║   ██║╚██████╗         ║
║         ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝        ║
║                                                              ║
║         VISION     AI-Driven Arctic Climate Prediction       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Main pipeline entry point.

    Returns:
        0 on success, 1 on error.
    """
    parser = _build_parser()
    args   = parser.parse_args()
    logger = _setup_logging(args.log_level)

    print(BANNER)
    logger.info(f"ArcticVision v1.0.0 — Pipeline starting")
    logger.info(f"Project root  : {PROJECT_ROOT}")
    logger.info(f"Config        : {args.config}")
    logger.info(f"From stage    : {args.from_stage}")
    logger.info(f"Synthetic data: {args.synthetic}")
    logger.info(f"Models        : {args.model}")
    logger.info(f"Skip ML       : {args.skip_ml}")
    logger.info(f"Map year      : {args.map_year}")

    t_start = time.time()
    STAGE_ORDER = ["ingest", "preprocess", "climate", "ml", "visualize"]
    from_idx    = STAGE_ORDER.index(args.from_stage)

    # ── Load cached intermediate results if resuming mid-pipeline ─────────────
    ingestion_result, climate_result, ml_result = _load_stage_cache(
        args.from_stage, args, logger
    )

    try:
        # ── Stage 1: Data ingestion ───────────────────────────────────────────
        if from_idx <= 0:
            t0 = time.time()
            ingestion_result = run_ingestion(args, logger)
            logger.info(f"Stage 1 elapsed: {time.time()-t0:.1f}s")

        # ── Stage 2: Preprocessing ────────────────────────────────────────────
        if from_idx <= 1:
            t0 = time.time()
            preproc_result = run_preprocessing(args, logger)
            logger.info(f"Stage 2 elapsed: {time.time()-t0:.1f}s")

        # ── Stage 3: Climate analysis ─────────────────────────────────────────
        if from_idx <= 2:
            t0 = time.time()
            climate_result = run_climate_analysis(args, logger)
            logger.info(f"Stage 3 elapsed: {time.time()-t0:.1f}s")

        # ── Stage 4: Machine learning ─────────────────────────────────────────
        if from_idx <= 3 and not args.skip_ml:
            t0 = time.time()
            ml_result = run_ml(args, logger)
            logger.info(f"Stage 4 elapsed: {time.time()-t0:.1f}s")
        elif args.skip_ml:
            logger.info("▶  STAGE 4: Machine Learning  [SKIPPED via --skip-ml]")

        # ── Stage 5: Visualization ────────────────────────────────────────────
        t0 = time.time()
        viz_outputs = run_visualization(args, logger, climate_result, ml_result)
        logger.info(f"Stage 5 elapsed: {time.time()-t0:.1f}s")

        # ── Summary report ────────────────────────────────────────────────────
        elapsed_total = time.time() - t_start
        summary_path  = generate_summary_report(
            climate_result, ml_result, viz_outputs, elapsed_total, args, logger
        )

        # ── Final banner ──────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("  ARCTICVISION PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Total elapsed  : {elapsed_total:.1f} seconds")
        logger.info(f"  Plots          : {PROJECT_ROOT / 'outputs' / 'plots'}")
        logger.info(f"  Dashboard      : {PROJECT_ROOT / 'outputs' / 'dashboards' / 'arctic_dashboard.html'}")
        logger.info(f"  Arctic map     : {PROJECT_ROOT / 'outputs' / 'dashboards' / 'arctic_map_2020.html'}")
        logger.info(f"  Animation      : {PROJECT_ROOT / 'outputs' / 'animations' / 'ice_melt_animation.gif'}")
        logger.info(f"  Models         : {PROJECT_ROOT / 'outputs' / 'models'}")
        logger.info(f"  Summary report : {summary_path}")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())