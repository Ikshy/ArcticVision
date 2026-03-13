"""
visualization/plotter.py
=========================
Master visualization orchestrator for ArcticVision.

Produces:
  1. Time-series trend plots  — ice extent with OLS trend + anomaly shading
  2. Seasonal cycle plots     — monthly climatology + anomaly heatmap
  3. Correlation heatmap      — variable cross-correlation matrix
  4. Decomposition plot       — trend / seasonal / residual components
  5. ML forecast plot         — actual vs predicted + confidence interval ribbon
  6. Interactive Arctic map   — Folium choropleth of sea ice anomaly
  7. Ice melt animation       — GIF / HTML5 of monthly ice extent over decades
  8. Temperature anomaly heat — 2D heatmap (year × month) of T2m anomaly
  9. Model training curves    — loss and MAE learning curves
 10. September minimum chart  — long-term decline with record-low annotations

All static plots are saved to outputs/plots/ as high-resolution PNG.
The interactive map is saved to outputs/dashboards/arctic_map.html.
The animation is saved to outputs/animations/ice_melt.gif.

Entry point:
    from visualization.plotter import ArcticVisualizer
    viz = ArcticVisualizer("configs/config.yaml")
    viz.run(climate_results, ml_results)

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
warnings.filterwarnings("ignore")

# ── Optional imports — graceful degradation ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server / CI use
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    MPL_OK = True
except ImportError:
    MPL_OK = False
    logger.warning("matplotlib not installed. Static plots will be skipped.")

try:
    import seaborn as sns
    SNS_OK = True
except ImportError:
    SNS_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    logger.warning("plotly not installed. Interactive plots will be skipped.")

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False
    logger.warning("folium not installed. Arctic map will be skipped.")

try:
    import imageio.v2 as imageio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

ARCTIC_PALETTE = {
    "ice_blue":    "#2196F3",
    "ice_light":   "#90CAF9",
    "ice_dark":    "#0D47A1",
    "melt_red":    "#EF5350",
    "melt_orange": "#FF9800",
    "land_green":  "#388E3C",
    "ocean":       "#1A237E",
    "anomaly_pos": "#E53935",
    "anomaly_neg": "#1E88E5",
    "trend":       "#FF6F00",
    "forecast":    "#9C27B0",
    "ci_band":     "#CE93D8",
    "bg_dark":     "#0A1628",
    "bg_light":    "#F0F4F8",
    "grid":        "#CFD8DC",
    "text_dark":   "#1A1A2E",
}

def _apply_arctic_style(ax, dark: bool = False) -> None:
    """Apply consistent Arctic theme to a matplotlib Axes."""
    bg = ARCTIC_PALETTE["bg_dark"] if dark else ARCTIC_PALETTE["bg_light"]
    ax.set_facecolor(bg)
    ax.tick_params(colors="#B0BEC5" if dark else "#455A64", labelsize=9)
    ax.xaxis.label.set_color("#CFD8DC" if dark else "#37474F")
    ax.yaxis.label.set_color("#CFD8DC" if dark else "#37474F")
    ax.title.set_color("#ECEFF1" if dark else "#1A237E")
    ax.grid(color="#263238" if dark else ARCTIC_PALETTE["grid"],
            linestyle="--", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474F" if dark else "#B0BEC5")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Time-series Trend Plots
# ─────────────────────────────────────────────────────────────────────────────

class TrendPlotter:
    """
    Produces ice extent trend and anomaly time-series plots.

    Plots generated:
      - ice_trend.png          : Monthly ice extent + OLS trend line
      - ice_anomaly.png        : Anomaly bar chart (positive/negative)
      - september_minimum.png  : September annual minimum with record lows flagged
    """

    def __init__(self, plots_dir: Path, cfg: dict) -> None:
        self.plots_dir = plots_dir
        self.dpi       = cfg["visualization"]["figure_dpi"]

    def plot_ice_trend(
        self,
        df: pd.DataFrame,
        trend_line: Optional[pd.Series] = None,
    ) -> Path:
        """
        Plot monthly sea ice extent with optional OLS trend overlay.

        Args:
            df:         Feature DataFrame with 'date' and 'sea_ice_extent_mkm2'
            trend_line: Optional Series of trend values (same index as df)

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                                  facecolor=ARCTIC_PALETTE["bg_dark"])
        fig.suptitle(
            "Arctic Sea Ice Extent (1979–2024)",
            fontsize=16, fontweight="bold",
            color="#ECEFF1", y=0.98
        )

        # ── Top: Monthly time series ──────────────────────────────────────────
        ax1 = axes[0]
        _apply_arctic_style(ax1, dark=True)
        dates  = pd.to_datetime(df["date"])
        extent = df["sea_ice_extent_mkm2"].values

        ax1.fill_between(dates, extent, alpha=0.25,
                          color=ARCTIC_PALETTE["ice_blue"], label="_fill")
        ax1.plot(dates, extent, linewidth=0.8,
                  color=ARCTIC_PALETTE["ice_light"], alpha=0.9, label="Monthly extent")

        if trend_line is not None and len(trend_line) == len(dates):
            ax1.plot(dates, trend_line.values, linewidth=2.5,
                      color=ARCTIC_PALETTE["trend"], linestyle="--",
                      label="OLS trend", zorder=5)

        ax1.set_ylabel("Extent (M km²)", fontsize=10)
        ax1.set_xlim(dates.iloc[0], dates.iloc[-1])
        ax1.legend(loc="upper right", framealpha=0.3,
                    labelcolor="#ECEFF1", fontsize=9)
        ax1.set_title("Monthly Sea Ice Extent", fontsize=11, pad=8)

        # ── Bottom: 12-month rolling mean ─────────────────────────────────────
        ax2 = axes[1]
        _apply_arctic_style(ax2, dark=True)
        roll12 = pd.Series(extent).rolling(12, center=True).mean()

        ax2.plot(dates, roll12, linewidth=2,
                  color=ARCTIC_PALETTE["ice_blue"], label="12-month rolling mean")

        # Shade below a key threshold (e.g., 10 M km² – Arctic minimum alert)
        ax2.axhline(10.0, linestyle=":", color=ARCTIC_PALETTE["melt_red"],
                     linewidth=1.2, alpha=0.7, label="10 M km² threshold")
        ax2.fill_between(dates, roll12.fillna(0), 10.0,
                          where=(roll12 < 10.0),
                          color=ARCTIC_PALETTE["melt_red"], alpha=0.2,
                          label="Below 10 M km²")

        ax2.set_ylabel("Extent (M km²)", fontsize=10)
        ax2.set_xlabel("Year", fontsize=10)
        ax2.set_xlim(dates.iloc[0], dates.iloc[-1])
        ax2.legend(loc="upper right", framealpha=0.3,
                    labelcolor="#ECEFF1", fontsize=9)
        ax2.set_title("12-Month Rolling Mean", fontsize=11, pad=8)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = self.plots_dir / "ice_trend.png"
        _save_fig(fig, out, self.dpi)
        return out

    def plot_anomaly_bars(self, df: pd.DataFrame) -> Path:
        """
        Bar chart of monthly sea ice anomaly (positive = more ice than normal,
        negative = less ice than normal). Bars coloured by sign.

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        anomaly_col = "sea_ice_extent_mkm2_anomaly"
        if anomaly_col not in df.columns:
            # Fall back to computing a simple anomaly
            monthly_mean = df.groupby("month")["sea_ice_extent_mkm2"].transform("mean")
            df = df.copy()
            df[anomaly_col] = df["sea_ice_extent_mkm2"] - monthly_mean

        fig, ax = plt.subplots(figsize=(14, 5),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        dates   = pd.to_datetime(df["date"])
        anomaly = df[anomaly_col].values
        colors  = [ARCTIC_PALETTE["anomaly_neg"] if v < 0
                   else ARCTIC_PALETTE["anomaly_pos"] for v in anomaly]

        ax.bar(dates, anomaly, width=20, color=colors, alpha=0.85)
        ax.axhline(0, color="#ECEFF1", linewidth=0.8, alpha=0.5)

        # 5-year running mean of anomaly
        roll60 = pd.Series(anomaly).rolling(60, center=True, min_periods=12).mean()
        ax.plot(dates, roll60, color=ARCTIC_PALETTE["melt_orange"],
                 linewidth=2.0, label="5-yr running mean")

        ax.set_title("Sea Ice Extent Anomaly vs 1979–2000 Baseline",
                      fontsize=13, color="#ECEFF1", pad=10)
        ax.set_ylabel("Anomaly (M km²)", fontsize=10)
        ax.set_xlabel("Year", fontsize=10)
        ax.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        out = self.plots_dir / "ice_anomaly.png"
        _save_fig(fig, out, self.dpi)
        return out

    def plot_september_minimum(
        self,
        annual_stats: pd.DataFrame,
        sept_ols: Optional[dict] = None,
    ) -> Path:
        """
        Long-term September minimum sea ice extent with record-low annotations.

        Args:
            annual_stats: DataFrame from ExtremesDetector.annual_min_max()
            sept_ols:     OLS results dict from ClimateAnalyzer for September series

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        fig, ax = plt.subplots(figsize=(12, 6),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        years = annual_stats["year"].values
        mins  = annual_stats["annual_min"].values

        # Bar chart of annual minimum
        bar_colors = [
            ARCTIC_PALETTE["melt_red"] if row["extreme_low_year"]
            else ARCTIC_PALETTE["ice_blue"]
            for _, row in annual_stats.iterrows()
        ]
        ax.bar(years, mins, color=bar_colors, alpha=0.8, width=0.8)

        # Trend line from OLS
        if sept_ols and "trend_line" in sept_ols:
            tl = sept_ols["trend_line"]
            # tl index is datetime — extract year to match x-axis
            tl_years = pd.to_datetime(tl.index).year
            ax.plot(tl_years, tl.values, color=ARCTIC_PALETTE["trend"],
                     linewidth=2.5, linestyle="--",
                     label=f"Trend: {sept_ols.get('slope_per_decade',0):.3f} M km²/decade",
                     zorder=6)

        # Annotate record-low years
        records = annual_stats[annual_stats["running_record_low"]]
        for _, row in records.iterrows():
            ax.annotate(
                f"⬇ {row['year']}", (row["year"], row["annual_min"]),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7, color=ARCTIC_PALETTE["melt_red"],
                arrowprops=dict(arrowstyle="-", color=ARCTIC_PALETTE["melt_red"],
                                lw=0.8),
            )

        ax.set_title("Arctic September Sea Ice Annual Minimum",
                      fontsize=13, color="#ECEFF1", pad=10)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Extent (M km²)", fontsize=10)
        ax.set_ylim(bottom=0)
        if sept_ols and "trend_line" in sept_ols:
            ax.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        # Custom legend for bar colours
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=ARCTIC_PALETTE["ice_blue"], label="Annual minimum"),
            Patch(facecolor=ARCTIC_PALETTE["melt_red"], label="Record low year"),
        ]
        ax.legend(handles=legend_handles + (
            [plt.Line2D([0], [0], color=ARCTIC_PALETTE["trend"],
                         linewidth=2, linestyle="--", label="OLS trend")]
            if sept_ols else []
        ), framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        out = self.plots_dir / "september_minimum.png"
        _save_fig(fig, out, self.dpi)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Seasonal Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalHeatmapPlotter:
    """
    Produces a 2-D heatmap of climate variables across year × month.

    The pattern reveals both the seasonal cycle (columns) and the
    long-term trend (rows declining from top to bottom).
    """

    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    def __init__(self, plots_dir: Path, cfg: dict) -> None:
        self.plots_dir = plots_dir
        self.dpi       = cfg["visualization"]["figure_dpi"]

    def plot_ice_heatmap(self, df: pd.DataFrame) -> Path:
        """
        Year × month heatmap of sea ice extent.
        Colours encode extent value — darker blue = more ice.

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        df = df.copy()
        df["date"]  = pd.to_datetime(df["date"])
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

        pivot = df.pivot_table(
            index="year", columns="month",
            values="sea_ice_extent_mkm2", aggfunc="mean"
        )
        pivot.columns = self.MONTH_LABELS

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.22)),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        cmap = plt.get_cmap("Blues")
        im   = ax.imshow(
            pivot.values,
            aspect="auto", cmap=cmap,
            interpolation="nearest",
            vmin=pivot.values[~np.isnan(pivot.values)].min(),
            vmax=pivot.values[~np.isnan(pivot.values)].max(),
        )

        ax.set_xticks(range(12))
        ax.set_xticklabels(self.MONTH_LABELS, fontsize=9, color="#ECEFF1")
        years = pivot.index.tolist()
        step = max(1, len(years) // 20)
        ax.set_yticks(range(0, len(years), step))
        ax.set_yticklabels([years[i] for i in range(0, len(years), step)],
                            fontsize=8, color="#ECEFF1")

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Sea Ice Extent (M km²)", color="#ECEFF1", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#ECEFF1")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ECEFF1")

        ax.set_title("Arctic Sea Ice Extent — Year × Month Heatmap",
                      fontsize=13, color="#ECEFF1", pad=10)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Year",  fontsize=10)

        out = self.plots_dir / "ice_seasonal_heatmap.png"
        _save_fig(fig, out, self.dpi)
        return out

    def plot_temperature_anomaly_heatmap(self, df: pd.DataFrame) -> Path:
        """
        Year × month heatmap of ERA5 2-m temperature anomaly.
        Diverging red/blue colour scheme: warm years red, cold years blue.

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        col = "era5_t2m_celsius_anomaly"
        if col not in df.columns:
            logger.warning(f"'{col}' not in df. Skipping temperature heatmap.")
            return Path()

        df = df.copy()
        df["date"]  = pd.to_datetime(df["date"])
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

        pivot = df.pivot_table(
            index="year", columns="month",
            values=col, aggfunc="mean"
        )
        pivot.columns = self.MONTH_LABELS

        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.1)

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.22)),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xticks(range(12))
        ax.set_xticklabels(self.MONTH_LABELS, fontsize=9, color="#ECEFF1")
        years = pivot.index.tolist()
        step  = max(1, len(years) // 20)
        ax.set_yticks(range(0, len(years), step))
        ax.set_yticklabels([years[i] for i in range(0, len(years), step)],
                            fontsize=8, color="#ECEFF1")

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("T2m Anomaly (°C)", color="#ECEFF1", fontsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ECEFF1")

        ax.set_title("Arctic 2-m Temperature Anomaly — Year × Month Heatmap",
                      fontsize=13, color="#ECEFF1", pad=10)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Year",  fontsize=10)

        out = self.plots_dir / "temperature_anomaly_heatmap.png"
        _save_fig(fig, out, self.dpi)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationPlotter:
    """Produces a styled correlation matrix heatmap."""

    def __init__(self, plots_dir: Path, cfg: dict) -> None:
        self.plots_dir = plots_dir
        self.dpi       = cfg["visualization"]["figure_dpi"]

    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> Path:
        """
        Annotated correlation heatmap with significance-based masking.

        Args:
            corr_matrix: Square correlation DataFrame from CorrelationAnalyst

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK or corr_matrix.empty:
            return Path()

        n = len(corr_matrix)
        fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(7, n * 1.0)),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        cmap = plt.get_cmap("coolwarm")
        mask = np.zeros_like(corr_matrix.values)
        mask[np.triu_indices_from(mask, k=1)] = True   # upper triangle

        im = ax.imshow(
            corr_matrix.values,
            cmap=cmap, vmin=-1, vmax=1,
            aspect="auto",
        )

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr_matrix.values[i, j]
                color = "white" if abs(val) > 0.5 else "#ECEFF1"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

        labels = [c.replace("_mkm2", "").replace("_celsius", "")
                   .replace("sea_ice_extent", "ice_ext") for c in corr_matrix.columns]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#ECEFF1")
        ax.set_yticklabels(labels, fontsize=8, color="#ECEFF1")

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Pearson r", color="#ECEFF1", fontsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ECEFF1")

        ax.set_title("Climate Variable Correlation Matrix",
                      fontsize=13, color="#ECEFF1", pad=10)
        plt.tight_layout()

        out = self.plots_dir / "correlation_matrix.png"
        _save_fig(fig, out, self.dpi)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Decomposition Plot
# ─────────────────────────────────────────────────────────────────────────────

class DecompositionPlotter:
    """Visualises seasonal decomposition (trend / seasonal / residual)."""

    def __init__(self, plots_dir: Path, cfg: dict) -> None:
        self.plots_dir = plots_dir
        self.dpi       = cfg["visualization"]["figure_dpi"]

    def plot_decomposition(
        self,
        decomp_df: pd.DataFrame,
        variable: str = "sea_ice_extent_mkm2",
    ) -> Path:
        """
        Four-panel decomposition plot: observed / trend / seasonal / residual.

        Args:
            decomp_df: DataFrame from SeasonalDecomposer.decomposition_to_df()
            variable:  Variable name for titling

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK or decomp_df.empty:
            return Path()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                                   facecolor=ARCTIC_PALETTE["bg_dark"],
                                   sharex=True)
        fig.suptitle(
            f"Seasonal Decomposition — {variable.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold", color="#ECEFF1", y=0.99
        )

        dates      = pd.to_datetime(decomp_df["date"])
        components = ["observed", "trend", "seasonal", "residual"]
        colors     = [ARCTIC_PALETTE["ice_light"],
                       ARCTIC_PALETTE["trend"],
                       ARCTIC_PALETTE["ice_blue"],
                       ARCTIC_PALETTE["melt_orange"]]

        for ax, comp, color in zip(axes, components, colors):
            _apply_arctic_style(ax, dark=True)
            vals = decomp_df[comp].values
            ax.plot(dates, vals, color=color, linewidth=1.0)
            if comp in ("observed", "trend"):
                ax.fill_between(dates, vals, alpha=0.15, color=color)
            if comp == "residual":
                ax.axhline(0, color="#607D8B", linewidth=0.8, linestyle=":")
            ax.set_ylabel(comp.capitalize(), fontsize=9, color="#ECEFF1")
            ax.set_xlim(dates.iloc[0], dates.iloc[-1])

        axes[-1].set_xlabel("Year", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        label = variable.replace("sea_ice_extent_mkm2", "ice").replace("era5_t2m_celsius", "t2m")
        out = self.plots_dir / f"decomposition_{label}.png"
        _save_fig(fig, out, self.dpi)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ML Forecast Plot
# ─────────────────────────────────────────────────────────────────────────────

class ForecastPlotter:
    """
    Produces actual vs predicted comparison plots and future forecast charts.
    """

    def __init__(self, plots_dir: Path, cfg: dict) -> None:
        self.plots_dir = plots_dir
        self.dpi       = cfg["visualization"]["figure_dpi"]

    def plot_actual_vs_predicted(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        model_name: str = "LSTM",
        units:      str = "M km²",
        metrics:    Optional[dict] = None,
    ) -> Path:
        """
        Two-panel plot: time-series overlay + scatter with regression line.

        Args:
            y_true:     Actual test values
            y_pred:     Model predictions
            model_name: Label for title and legend
            units:      Physical unit string for axis labels
            metrics:    Dict with rmse/mae/r2 for annotation

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK:
            return Path()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                         facecolor=ARCTIC_PALETTE["bg_dark"])
        fig.suptitle(f"{model_name} — Test Set Evaluation",
                      fontsize=14, color="#ECEFF1", fontweight="bold")

        # ── Left: Time-series overlay ─────────────────────────────────────────
        _apply_arctic_style(ax1, dark=True)
        idx = np.arange(len(y_true))
        ax1.plot(idx, y_true, color=ARCTIC_PALETTE["ice_light"],
                  linewidth=1.5, label="Actual", zorder=3)
        ax1.plot(idx, y_pred, color=ARCTIC_PALETTE["forecast"],
                  linewidth=1.5, linestyle="--", label="Predicted", zorder=4)
        ax1.fill_between(idx,
                          np.minimum(y_true, y_pred),
                          np.maximum(y_true, y_pred),
                          alpha=0.2, color=ARCTIC_PALETTE["melt_red"])
        ax1.set_title("Actual vs Predicted (Time Series)", color="#ECEFF1", fontsize=11)
        ax1.set_xlabel("Test Sample Index", fontsize=9)
        ax1.set_ylabel(f"Sea Ice Extent ({units})", fontsize=9)
        ax1.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        # Metrics annotation
        if metrics:
            txt = (f"RMSE: {metrics.get('rmse','N/A')}\n"
                   f"MAE:  {metrics.get('mae','N/A')}\n"
                   f"R²:   {metrics.get('r2','N/A')}")
            ax1.text(0.02, 0.05, txt, transform=ax1.transAxes,
                      fontsize=8, color="#ECEFF1",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A237E",
                                alpha=0.7, edgecolor="#3949AB"))

        # ── Right: Scatter ────────────────────────────────────────────────────
        _apply_arctic_style(ax2, dark=True)
        ax2.scatter(y_true, y_pred, alpha=0.55, s=20,
                     color=ARCTIC_PALETTE["ice_blue"], edgecolors="none")

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax2.plot([lo, hi], [lo, hi], color=ARCTIC_PALETTE["trend"],
                  linewidth=1.5, linestyle="--", label="Perfect fit")
        # Regression line
        if len(y_true) > 3:
            m, b = np.polyfit(y_true, y_pred, 1)
            xs = np.linspace(lo, hi, 100)
            ax2.plot(xs, m * xs + b, color=ARCTIC_PALETTE["melt_orange"],
                      linewidth=1.2, label=f"Fit (slope={m:.2f})")

        ax2.set_xlabel(f"Actual ({units})", fontsize=9)
        ax2.set_ylabel(f"Predicted ({units})", fontsize=9)
        ax2.set_title("Scatter: Actual vs Predicted", color="#ECEFF1", fontsize=11)
        ax2.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        plt.tight_layout()
        label = model_name.lower().replace(" ", "_")
        out = self.plots_dir / f"forecast_eval_{label}.png"
        _save_fig(fig, out, self.dpi)
        return out

    def plot_future_forecast(
        self,
        forecast_df:  pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        model_name:    str = "LSTM",
    ) -> Path:
        """
        Plot multi-step future forecast with 95% confidence interval ribbon.

        Args:
            forecast_df:   DataFrame from ModelTrainer.forecast_future()
            historical_df: Optional; last N months of historical data to anchor the plot
            model_name:    Label for title

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK or forecast_df.empty:
            return Path()

        fig, ax = plt.subplots(figsize=(13, 6),
                                 facecolor=ARCTIC_PALETTE["bg_dark"])
        _apply_arctic_style(ax, dark=True)

        steps = forecast_df["step"].values
        pred  = forecast_df["predicted_mkm2"].values
        lo    = forecast_df["lower_ci_95"].values
        hi    = forecast_df["upper_ci_95"].values

        # Optional historical anchor
        if historical_df is not None:
            hist = historical_df.tail(36)
            hist_dates = pd.to_datetime(hist["date"])
            ax.plot(np.arange(-len(hist), 0), hist["sea_ice_extent_mkm2"].values,
                     color=ARCTIC_PALETTE["ice_light"], linewidth=1.5,
                     label="Historical (last 3 years)")
            ax.axvline(0, color="#607D8B", linewidth=1, linestyle=":",
                        alpha=0.7, label="Forecast start")

        # CI ribbon
        ax.fill_between(steps, lo, hi,
                          color=ARCTIC_PALETTE["ci_band"], alpha=0.35,
                          label="95% CI (MC-Dropout)")
        ax.plot(steps, pred, color=ARCTIC_PALETTE["forecast"],
                 linewidth=2.2, marker="o", markersize=4, label=f"{model_name} forecast")

        ax.set_title(f"Arctic Sea Ice Extent — {len(steps)}-Month Forecast",
                      fontsize=13, color="#ECEFF1", pad=10)
        ax.set_xlabel("Months ahead", fontsize=10)
        ax.set_ylabel("Extent (M km²)", fontsize=10)
        ax.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        label = model_name.lower().replace(" ", "_")
        out = self.plots_dir / f"future_forecast_{label}.png"
        _save_fig(fig, out, self.dpi)
        return out

    def plot_training_curves(
        self,
        history_df:  pd.DataFrame,
        model_name:  str = "LSTM",
    ) -> Path:
        """
        Plot training and validation loss + MAE learning curves.

        Args:
            history_df: DataFrame from ModelTrainer.train() with columns
                        [epoch, train_loss, val_loss, train_mae, val_mae]
            model_name: Label for title

        Returns:
            Path to saved PNG.
        """
        if not MPL_OK or history_df.empty:
            return Path()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                         facecolor=ARCTIC_PALETTE["bg_dark"])
        fig.suptitle(f"{model_name} — Training Curves",
                      fontsize=13, color="#ECEFF1", fontweight="bold")

        for ax, train_col, val_col, ylabel in [
            (ax1, "train_loss", "val_loss", "Huber Loss"),
            (ax2, "train_mae",  "val_mae",  "MAE (scaled)"),
        ]:
            _apply_arctic_style(ax, dark=True)
            epochs = history_df["epoch"].values
            ax.plot(epochs, history_df[train_col].values,
                     color=ARCTIC_PALETTE["ice_blue"], linewidth=1.8, label="Train")
            ax.plot(epochs, history_df[val_col].values,
                     color=ARCTIC_PALETTE["melt_red"], linewidth=1.8,
                     linestyle="--", label="Validation")

            best_epoch = history_df.loc[history_df[val_col].idxmin(), "epoch"]
            best_val   = history_df[val_col].min()
            ax.axvline(best_epoch, color=ARCTIC_PALETTE["trend"],
                        linewidth=1.2, linestyle=":", alpha=0.8)
            ax.annotate(f"Best epoch {best_epoch}\n({best_val:.5f})",
                         (best_epoch, best_val),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=7, color=ARCTIC_PALETTE["trend"])

            ax.set_title(ylabel, color="#ECEFF1", fontsize=11)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(framealpha=0.3, labelcolor="#ECEFF1", fontsize=9)

        plt.tight_layout()
        label = model_name.lower().replace(" ", "_")
        out = self.plots_dir / f"training_curves_{label}.png"
        _save_fig(fig, out, self.dpi)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Interactive Arctic Map (Folium)
# ─────────────────────────────────────────────────────────────────────────────

class ArcticMapBuilder:
    """
    Builds an interactive Folium HTML map of Arctic sea ice anomaly.

    The map uses a lat/lon grid representation of ice concentration,
    overlaying a heatmap layer on an OpenStreetMap tile base.

    Since we work with scalar time-series (not raster grids), this
    visualisation uses a synthetic latitude-stratified heatmap derived
    from the monthly ice extent — showing which latitudes are most
    likely to be ice-covered given the observed extent.
    """

    def __init__(self, dashboards_dir: Path, cfg: dict) -> None:
        self.dashboards_dir = dashboards_dir
        self.cfg = cfg

    def build_map(self, df: pd.DataFrame, year: int = 2020) -> Path:
        """
        Build a Folium HTML map for a given year's mean ice extent.

        Methodology:
          - The Arctic sea ice edge (lat at which ~15% concentration occurs)
            is estimated from the extent value using empirical scaling.
          - A grid of synthetic lat/lon/intensity points is generated within
            the estimated ice-covered zone (ice_edge_lat to 90°N).
          - These are displayed as a heatmap overlay on a polar-projected map.

        Args:
            df:   Feature DataFrame with 'date' and 'sea_ice_extent_mkm2'
            year: Target year to visualise

        Returns:
            Path to saved HTML file.
        """
        if not FOLIUM_OK:
            logger.warning("folium not installed — map skipped.")
            return Path()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df_year = df[df["date"].dt.year == year]

        if df_year.empty:
            logger.warning(f"No data for year {year}.")
            return Path()

        annual_mean_extent = df_year["sea_ice_extent_mkm2"].mean()

        # Estimate ice edge latitude from extent (empirical formula)
        # Full Arctic (14 M km²) ≈ ice to ~60°N ; 5 M km² ≈ ice to ~78°N
        ice_edge_lat = max(60.0, min(85.0, 90.0 - annual_mean_extent * 2.1))

        logger.info(
            f"Building Arctic map for {year}: "
            f"extent={annual_mean_extent:.2f} M km², "
            f"estimated ice edge={ice_edge_lat:.1f}°N"
        )

        # Build Folium map (centred on North Pole)
        m = folium.Map(
            location=[90, 0],
            zoom_start=self.cfg["visualization"]["map_zoom"],
            tiles="CartoDB dark_matter",
            max_zoom=7, min_zoom=2,
        )

        # Generate synthetic lat/lon ice grid
        heat_data = self._generate_ice_grid(ice_edge_lat, annual_mean_extent)

        HeatMap(
            heat_data,
            radius=18,
            blur=25,
            gradient={0.0: "#1A237E", 0.4: "#1565C0",
                       0.7: "#90CAF9", 1.0: "#FFFFFF"},
            min_opacity=0.4,
        ).add_to(m)

        # Title + legend overlay
        title_html = f"""
        <div style="position:fixed; bottom:40px; left:50%; transform:translateX(-50%);
             background:rgba(10,22,40,0.85); padding:12px 24px; border-radius:8px;
             border:1px solid #1565C0; font-family:Arial; color:#ECEFF1; z-index:9999;">
          <h4 style="margin:0 0 6px 0; color:#90CAF9;">🧊 ArcticVision Sea Ice Map</h4>
          <p style="margin:0; font-size:12px;">
            Year: <b>{year}</b> &nbsp;|&nbsp;
            Mean extent: <b>{annual_mean_extent:.2f} M km²</b> &nbsp;|&nbsp;
            Est. ice edge: <b>{ice_edge_lat:.1f}°N</b>
          </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Add month selector info popup
        folium.Marker(
            location=[85, 0],
            popup=folium.Popup(
                f"<b>Arctic Sea Ice {year}</b><br>"
                f"Annual mean: {annual_mean_extent:.2f} M km²<br>"
                f"Ice edge: ~{ice_edge_lat:.0f}°N<br>"
                f"<i>Generated by ArcticVision</i>",
                max_width=200,
            ),
            icon=folium.Icon(color="blue", icon="snowflake", prefix="fa"),
        ).add_to(m)

        out = self.dashboards_dir / f"arctic_map_{year}.html"
        m.save(str(out))
        logger.info(f"Interactive map saved → {out}")
        return out

    @staticmethod
    def _generate_ice_grid(
        ice_edge_lat: float,
        extent_mkm2:  float,
        n_points:     int = 500,
    ) -> list[list[float]]:
        """
        Generate a synthetic lat/lon/intensity grid for the ice heatmap.

        Points are drawn from:
          - latitude: ice_edge_lat to 90°N (more points near centre)
          - longitude: uniform 0–360°
          - intensity: higher closer to pole (concentration proxy)

        Args:
            ice_edge_lat: Southern boundary of estimated ice cover
            extent_mkm2:  Annual mean extent (used to scale intensity)
            n_points:     Number of heatmap points to generate

        Returns:
            List of [lat, lon, intensity] for HeatMap plugin.
        """
        np.random.seed(42)
        lats = np.random.uniform(ice_edge_lat, 90.0, n_points)
        lons = np.random.uniform(-180.0, 180.0, n_points)
        # Intensity: higher near pole, scaled by extent fraction
        intensity = ((lats - ice_edge_lat) / (90.0 - ice_edge_lat)) ** 1.5
        intensity = intensity * min(1.0, extent_mkm2 / 14.0)
        return [[float(la), float(lo), float(it)]
                for la, lo, it in zip(lats, lons, intensity)]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Ice Melt Animation
# ─────────────────────────────────────────────────────────────────────────────

class IceMeltAnimator:
    """
    Creates a GIF animation showing Arctic sea ice monthly extent
    evolving over several decades — one frame per year.

    Each frame is a polar bar chart showing the 12-month seasonal cycle
    for that year, coloured by extent magnitude.
    """

    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    def __init__(self, anim_dir: Path, cfg: dict) -> None:
        self.anim_dir = anim_dir
        self.fps      = cfg["visualization"]["animation_fps"]
        self.dpi      = min(cfg["visualization"]["figure_dpi"], 100)

    def create_gif(
        self,
        df:         pd.DataFrame,
        decade_step: int = 5,
        max_frames:  int = 20,
    ) -> Path:
        """
        Create a GIF animation of Arctic sea ice seasonal cycles by year.

        Each frame shows the polar seasonal bar chart for a given year,
        with colour indicating whether extent is above/below the
        1979–2000 baseline mean.

        Args:
            df:           Feature DataFrame with 'date' and 'sea_ice_extent_mkm2'
            decade_step:  Sample every N years (reduces frame count for GIF size)
            max_frames:   Maximum number of animation frames

        Returns:
            Path to saved GIF file.
        """
        if not MPL_OK:
            return Path()

        df = df.copy()
        df["date"]  = pd.to_datetime(df["date"])
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Compute baseline (1979–2000) monthly means for reference
        base = df[df["year"] <= 2000].groupby("month")["sea_ice_extent_mkm2"].mean()

        years = sorted(df["year"].unique())
        selected = years[::decade_step][:max_frames]

        frame_paths = []
        tmp_dir = self.anim_dir / "_frames"
        _ensure_dir(tmp_dir)

        for year in selected:
            yr_data = df[df["year"] == year].sort_values("month")
            if len(yr_data) < 12:
                continue

            monthly = yr_data.groupby("month")["sea_ice_extent_mkm2"].mean().reindex(
                range(1, 13), fill_value=np.nan
            )

            fp = self._draw_polar_frame(year, monthly, base, tmp_dir)
            if fp:
                frame_paths.append(fp)

        if not frame_paths:
            logger.warning("No animation frames generated.")
            return Path()

        if not IMAGEIO_OK:
            logger.warning("imageio not installed — cannot create GIF.")
            return Path()

        # Assemble GIF
        frames = [imageio.imread(str(fp)) for fp in frame_paths]
        out = self.anim_dir / "ice_melt_animation.gif"
        imageio.mimsave(str(out), frames, fps=self.fps, loop=0)
        logger.info(
            f"Animation saved → {out}  ({len(frames)} frames @ {self.fps} fps)"
        )

        # Clean up temp frames
        for fp in frame_paths:
            fp.unlink(missing_ok=True)
        tmp_dir.rmdir()

        return out

    def _draw_polar_frame(
        self,
        year:     int,
        monthly:  pd.Series,
        baseline: pd.Series,
        tmp_dir:  Path,
    ) -> Optional[Path]:
        """
        Draw a single polar bar-chart frame for one year.

        The chart shows 12 months arranged in a clock-like layout.
        Bar colour: blue if extent ≥ baseline, red if below baseline.

        Args:
            year:     Year label for title
            monthly:  Series (index=month 1-12) of mean monthly ice extent
            baseline: Series of baseline monthly means (1979-2000)
            tmp_dir:  Directory to save the temp PNG frame

        Returns:
            Path to saved PNG frame, or None if matplotlib is unavailable.
        """
        if not MPL_OK:
            return None

        fig = plt.figure(figsize=(5, 5), facecolor=ARCTIC_PALETTE["bg_dark"])
        ax  = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(ARCTIC_PALETTE["bg_dark"])

        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        bar_width = 2 * np.pi / 12 * 0.85

        max_val = 16.0   # M km² (physical maximum)
        for i, (month, val) in enumerate(monthly.items()):
            if np.isnan(val):
                continue
            base_val = baseline.get(month, val)
            color = (ARCTIC_PALETTE["ice_blue"] if val >= base_val
                     else ARCTIC_PALETTE["melt_red"])
            ax.bar(
                angles[i], val / max_val,
                width=bar_width,
                bottom=0.05,
                color=color, alpha=0.85,
            )

        ax.set_xticks(angles)
        ax.set_xticklabels(self.MONTH_LABELS, fontsize=7,
                            color="#ECEFF1", fontweight="bold")
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["4", "8", "12", "16"], fontsize=6, color="#90A4AE")
        ax.tick_params(colors="#607D8B")
        ax.spines["polar"].set_color("#37474F")
        ax.grid(color="#263238", linewidth=0.5)

        mean_ext = monthly.dropna().mean()
        ax.set_title(
            f"{year}  |  Mean: {mean_ext:.1f} M km²",
            fontsize=10, color="#ECEFF1", pad=15, fontweight="bold"
        )
        fig.text(0.5, 0.02,
                  "🔵 Above baseline  🔴 Below baseline",
                  ha="center", fontsize=7, color="#90A4AE")

        fp = tmp_dir / f"frame_{year}.png"
        fig.savefig(str(fp), dpi=self.dpi, bbox_inches="tight",
                     facecolor=ARCTIC_PALETTE["bg_dark"])
        plt.close(fig)
        return fp


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Interactive Plotly Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class PlotlyDashboard:
    """
    Produces a single-file interactive HTML dashboard using Plotly.

    Contains:
      - Tab 1: Ice extent time series + trend
      - Tab 2: Anomaly bar chart
      - Tab 3: ML forecast with CI ribbon
      - Tab 4: Seasonal cycle comparison (1980s vs 2020s)
    """

    def __init__(self, dashboards_dir: Path, cfg: dict) -> None:
        self.dashboards_dir = dashboards_dir
        self.cfg = cfg

    def build(
        self,
        df:          pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        model_name:  str = "LSTM",
    ) -> Path:
        """
        Build and save the interactive dashboard HTML.

        Args:
            df:          Feature DataFrame with anomaly columns
            forecast_df: Future forecast DataFrame (optional)
            model_name:  Label for forecast tab

        Returns:
            Path to saved HTML file.
        """
        if not PLOTLY_OK:
            logger.warning("plotly not installed — dashboard skipped.")
            return Path()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sea Ice Extent + OLS Trend",
                "Monthly Anomaly (vs 1979-2000 Baseline)",
                f"{model_name} Future Forecast + 95% CI",
                "Seasonal Cycle: 1980s vs 2020s",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # ── Tab 1: Ice extent + 5-yr rolling mean ────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["sea_ice_extent_mkm2"],
                mode="lines", name="Monthly extent",
                line=dict(color="#90CAF9", width=0.8),
                opacity=0.7,
            ), row=1, col=1
        )
        roll60 = df["sea_ice_extent_mkm2"].rolling(60, center=True, min_periods=12).mean()
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=roll60,
                mode="lines", name="5-yr rolling mean",
                line=dict(color="#FF6F00", width=2.0),
            ), row=1, col=1
        )

        # ── Tab 2: Anomaly bars ───────────────────────────────────────────────
        anomaly_col = "sea_ice_extent_mkm2_anomaly"
        if anomaly_col not in df.columns:
            monthly_mean = df.groupby("month")["sea_ice_extent_mkm2"].transform("mean")
            df[anomaly_col] = df["sea_ice_extent_mkm2"] - monthly_mean

        anom = df[anomaly_col]
        bar_colors = ["#1E88E5" if v >= 0 else "#E53935" for v in anom]
        fig.add_trace(
            go.Bar(
                x=df["date"], y=anom,
                name="Ice anomaly",
                marker=dict(color=bar_colors),
                opacity=0.8,
            ), row=1, col=2
        )

        # ── Tab 3: Future forecast ────────────────────────────────────────────
        if forecast_df is not None and not forecast_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["step"],
                    y=forecast_df["upper_ci_95"],
                    mode="lines", line=dict(width=0),
                    name="95% CI upper", showlegend=False,
                    fillcolor="rgba(156,39,176,0.2)",
                ), row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["step"],
                    y=forecast_df["lower_ci_95"],
                    fill="tonexty", mode="lines",
                    line=dict(width=0),
                    name="95% CI",
                    fillcolor="rgba(156,39,176,0.2)",
                ), row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["step"],
                    y=forecast_df["predicted_mkm2"],
                    mode="lines+markers",
                    name=f"{model_name} forecast",
                    line=dict(color="#CE93D8", width=2),
                    marker=dict(size=4),
                ), row=2, col=1
            )

        # ── Tab 4: Seasonal cycle comparison ─────────────────────────────────
        df["year"] = df["date"].dt.year
        for decade, label, color in [
            ((1980, 1989), "1980s", "#90CAF9"),
            ((2010, 2019), "2010s", "#EF9A9A"),
            ((2020, 2024), "2020s", "#EF5350"),
        ]:
            decade_df = df[(df["year"] >= decade[0]) & (df["year"] <= decade[1])]
            if not decade_df.empty:
                seasonal = decade_df.groupby("month")["sea_ice_extent_mkm2"].mean()
                fig.add_trace(
                    go.Scatter(
                        x=seasonal.index, y=seasonal.values,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    ), row=2, col=2
                )

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            height=800,
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0F2030",
            font=dict(family="Arial", size=11, color="#ECEFF1"),
            title=dict(
                text="🧊 ArcticVision — Interactive Climate Dashboard",
                x=0.5, font=dict(size=18, color="#90CAF9"),
            ),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(10,22,40,0.8)",
                bordercolor="#1565C0",
                font=dict(size=9),
            ),
            margin=dict(t=80, b=40, l=40, r=40),
        )
        fig.update_xaxes(gridcolor="#1A2744", zerolinecolor="#263238")
        fig.update_yaxes(gridcolor="#1A2744", zerolinecolor="#263238")
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="M km²", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Anomaly (M km²)", row=1, col=2)
        fig.update_xaxes(title_text="Months ahead", row=2, col=1)
        fig.update_yaxes(title_text="M km²", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2, tickvals=list(range(1,13)),
                          ticktext=["J","F","M","A","M","J","J","A","S","O","N","D"])
        fig.update_yaxes(title_text="M km²", row=2, col=2)

        out = self.dashboards_dir / "arctic_dashboard.html"
        fig.write_html(
            str(out),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True},
        )
        logger.info(f"Interactive dashboard saved → {out}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Master ArcticVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class ArcticVisualizer:
    """
    Master orchestrator for all ArcticVision visualisations.

    Chains all individual plotters and produces a complete set of
    publication-ready static plots, an interactive HTML dashboard,
    an Arctic map, and an ice melt animation.

    Args:
        config_path: Path to configs/config.yaml

    Example:
        >>> viz = ArcticVisualizer("configs/config.yaml")
        >>> viz.run(climate_results=ca_results, ml_results=ml_results)
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
    ) -> None:
        self.cfg          = _load_config(config_path)
        self.root         = Path(config_path).parent.parent
        self.plots_dir    = _ensure_dir(
            self.root / self.cfg["paths"]["outputs_plots"]
        )
        self.anim_dir     = _ensure_dir(
            self.root / self.cfg["paths"]["outputs_anim"]
        )
        self.dash_dir     = _ensure_dir(
            self.root / self.cfg["paths"]["outputs_dash"]
        )
        self.proc_dir     = self.root / self.cfg["paths"]["data_processed"]

        viz = self.cfg["visualization"]
        self.trend_plt    = TrendPlotter(self.plots_dir, self.cfg)
        self.heatmap_plt  = SeasonalHeatmapPlotter(self.plots_dir, self.cfg)
        self.corr_plt     = CorrelationPlotter(self.plots_dir, self.cfg)
        self.decomp_plt   = DecompositionPlotter(self.plots_dir, self.cfg)
        self.forecast_plt = ForecastPlotter(self.plots_dir, self.cfg)
        self.map_builder  = ArcticMapBuilder(self.dash_dir, self.cfg)
        self.animator     = IceMeltAnimator(self.anim_dir, self.cfg)
        self.dashboard    = PlotlyDashboard(self.dash_dir, self.cfg)

    def run(
        self,
        climate_results: Optional[dict] = None,
        ml_results:      Optional[dict] = None,
        map_year:        int = 2020,
    ) -> dict:
        """
        Produce all visualisations.

        Args:
            climate_results: Output dict from ClimateAnalyzer.run()
            ml_results:      Output dict from ModelTrainer.run()
            map_year:        Year for the interactive Arctic map

        Returns:
            dict mapping plot name → output Path
        """
        logger.info("=" * 60)
        logger.info("ArcticVision Visualization Pipeline START")
        logger.info("=" * 60)

        outputs = {}

        # ── Load data ─────────────────────────────────────────────────────────
        df = self._resolve_dataframe(climate_results)
        if df is None:
            logger.error("No data available. Pass climate_results or ensure "
                         "data/processed/arctic_features.parquet exists.")
            return outputs

        # ── 1. Ice trend + anomaly ────────────────────────────────────────────
        logger.info("[1/9] Trend and anomaly plots...")
        trend_line = None
        if climate_results and "ice_trend" in climate_results:
            trend_line = climate_results["ice_trend"]["ols"].get("trend_line")
        outputs["ice_trend"]  = self.trend_plt.plot_ice_trend(df, trend_line)
        outputs["ice_anomaly"] = self.trend_plt.plot_anomaly_bars(df)

        # ── 2. September minimum ──────────────────────────────────────────────
        logger.info("[2/9] September minimum chart...")
        annual_stats = None
        sept_ols     = None
        if climate_results and "extremes" in climate_results:
            annual_stats = climate_results["extremes"]["annual_stats"]
            sept_ols     = climate_results["extremes"]["september_trend"]["ols"]
        if annual_stats is None:
            from climate_analysis.analyzer import ExtremesDetector
            annual_stats = ExtremesDetector.annual_min_max(df)
        outputs["september_minimum"] = self.trend_plt.plot_september_minimum(
            annual_stats, sept_ols
        )

        # ── 3. Seasonal heatmaps ──────────────────────────────────────────────
        logger.info("[3/9] Seasonal heatmaps...")
        outputs["ice_heatmap"] = self.heatmap_plt.plot_ice_heatmap(df)
        outputs["temp_heatmap"] = self.heatmap_plt.plot_temperature_anomaly_heatmap(df)

        # ── 4. Correlation matrix ─────────────────────────────────────────────
        logger.info("[4/9] Correlation matrix...")
        if climate_results and "correlations" in climate_results:
            corr_matrix = climate_results["correlations"]["correlation_matrix"]
            outputs["correlation"] = self.corr_plt.plot_correlation_matrix(corr_matrix)

        # ── 5. Seasonal decomposition ─────────────────────────────────────────
        logger.info("[5/9] Decomposition plots...")
        if climate_results and "decomposition" in climate_results:
            for varname, decomp in climate_results["decomposition"].items():
                if isinstance(decomp, dict) and "df" in decomp:
                    outputs[f"decomp_{varname}"] = self.decomp_plt.plot_decomposition(
                        decomp["df"], varname
                    )

        # ── 6. ML forecast plots ──────────────────────────────────────────────
        logger.info("[6/9] ML forecast plots...")
        if ml_results:
            model_name = ml_results.get("model_type", "LSTM").upper()
            eval_r     = ml_results.get("eval", {})

            phys_metrics = eval_r.get("test_metrics_physical")
            scal_metrics = eval_r.get("test_metrics_scaled")
            metrics_to_plot = phys_metrics or scal_metrics
            units = "M km²" if phys_metrics else "scaled"

            y_true_key = "y_true_physical" if eval_r.get("y_true_physical") is not None else "y_true_scaled"
            y_true = eval_r.get(y_true_key)
            y_pred_key = "y_pred_physical" if eval_r.get("y_pred_physical") is not None else "y_pred_scaled"
            y_pred = eval_r.get(y_pred_key)

            if y_true is not None and y_pred is not None:
                outputs["forecast_eval"] = self.forecast_plt.plot_actual_vs_predicted(
                    y_true, y_pred, model_name, units, metrics_to_plot
                )

            forecast_df = ml_results.get("forecast", pd.DataFrame())
            if not (isinstance(forecast_df, pd.DataFrame) and forecast_df.empty):
                outputs["future_forecast"] = self.forecast_plt.plot_future_forecast(
                    forecast_df, df, model_name
                )

            history = ml_results.get("history", pd.DataFrame())
            if not history.empty:
                outputs["training_curves"] = self.forecast_plt.plot_training_curves(
                    history, model_name
                )

        # ── 7. Interactive Arctic map ─────────────────────────────────────────
        logger.info("[7/9] Interactive Arctic map...")
        outputs["arctic_map"] = self.map_builder.build_map(df, year=map_year)

        # ── 8. Ice melt animation ─────────────────────────────────────────────
        logger.info("[8/9] Ice melt animation...")
        outputs["ice_animation"] = self.animator.create_gif(df)

        # ── 9. Plotly interactive dashboard ───────────────────────────────────
        logger.info("[9/9] Interactive Plotly dashboard...")
        forecast_df = (
            ml_results.get("forecast", pd.DataFrame())
            if ml_results else pd.DataFrame()
        )
        model_name = (ml_results.get("model_type", "LSTM").upper()
                      if ml_results else "LSTM")
        outputs["dashboard"] = self.dashboard.build(df, forecast_df, model_name)

        # ── Summary ───────────────────────────────────────────────────────────
        self._log_summary(outputs)
        logger.info("Visualization Pipeline COMPLETE")
        return outputs

    def _resolve_dataframe(
        self, climate_results: Optional[dict]
    ) -> Optional[pd.DataFrame]:
        """Get the best available DataFrame for plotting."""
        if climate_results and "df" in climate_results:
            return climate_results["df"]
        parquet = self.proc_dir / "arctic_features.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return None

    @staticmethod
    def _log_summary(outputs: dict) -> None:
        logger.info("=" * 60)
        logger.info("VISUALIZATION OUTPUTS")
        logger.info("=" * 60)
        for name, path in outputs.items():
            if path and Path(path).exists():
                size_kb = Path(path).stat().st_size / 1024
                logger.info(f"  ✓ {name:25s} → {Path(path).name} ({size_kb:.0f} KB)")
            elif path:
                logger.warning(f"  ✗ {name:25s} → MISSING")
        logger.info("=" * 60)
