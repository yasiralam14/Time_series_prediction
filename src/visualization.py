"""Reusable plotting utilities for EDA and model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STYLE_DEFAULTS = {
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}


def apply_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(STYLE_DEFAULTS)


def save_fig(fig: plt.Figure, path: str | Path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Distribution plots ──────────────────────────────────────────────


def plot_target_distribution(
    targets: np.ndarray,
    weights: Optional[np.ndarray] = None,
    title: str = "Target (responder_6) Distribution",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(targets, bins=100, edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[0].set_xlabel("responder_6")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Unweighted Distribution")
    axes[0].axvline(np.mean(targets), color="red", linestyle="--", label=f"Mean: {np.mean(targets):.4f}")
    axes[0].axvline(np.median(targets), color="orange", linestyle="--", label=f"Median: {np.median(targets):.4f}")
    axes[0].legend()

    if weights is not None:
        axes[1].hist(targets, bins=100, weights=weights, edgecolor="black", alpha=0.7, color="#55A868")
        axes[1].set_xlabel("responder_6")
        axes[1].set_ylabel("Weighted Frequency")
        axes[1].set_title("Weighted Distribution")
        w_mean = np.average(targets, weights=weights)
        axes[1].axvline(w_mean, color="red", linestyle="--", label=f"Weighted Mean: {w_mean:.4f}")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No weights provided", ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title, fontsize=15, y=1.02)
    return fig


def plot_feature_distributions(
    df: pd.DataFrame, features: list[str], ncols: int = 4
) -> plt.Figure:
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        axes[i].hist(df[feat].dropna(), bins=50, alpha=0.7, edgecolor="black")
        axes[i].set_title(feat, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, y=1.01)
    return fig


# ── Missing data ─────────────────────────────────────────────────────


def plot_missing_data(
    missing_pct: pd.Series, title: str = "Missing Values by Feature"
) -> plt.Figure:
    non_zero = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if len(non_zero) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    fig, ax = plt.subplots(figsize=(14, max(4, len(non_zero) * 0.3)))
    colors = ["#C44E52" if v > 20 else "#DD8452" if v > 5 else "#55A868" for v in non_zero.values]
    ax.barh(non_zero.index, non_zero.values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.invert_yaxis()

    for i, v in enumerate(non_zero.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    return fig


# ── Correlation ──────────────────────────────────────────────────────


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (16, 14),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=6)
    return fig


def plot_target_correlations(
    correlations: pd.Series, top_n: int = 20, title: str = "Top Feature Correlations with Target"
) -> plt.Figure:
    top = correlations.abs().nlargest(top_n)
    values = correlations.loc[top.index]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in values.values]
    ax.barh(values.index, values.values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    return fig


# ── Time series ──────────────────────────────────────────────────────


def plot_temporal_patterns(
    df: pd.DataFrame,
    date_col: str = "date_id",
    target_col: str = "responder_6",
    weight_col: str = "weight",
) -> plt.Figure:
    daily = df.groupby(date_col).agg(
        mean_target=(target_col, "mean"),
        std_target=(target_col, "std"),
        mean_weight=(weight_col, "mean"),
        count=("weight", "size"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(daily[date_col], daily["mean_target"], linewidth=0.8, color="#4C72B0")
    axes[0, 0].fill_between(
        daily[date_col],
        daily["mean_target"] - daily["std_target"],
        daily["mean_target"] + daily["std_target"],
        alpha=0.2,
    )
    axes[0, 0].set_title("Daily Mean Target with Std Band")
    axes[0, 0].set_xlabel("Date ID")
    axes[0, 0].set_ylabel("responder_6")

    axes[0, 1].plot(daily[date_col], daily["std_target"], linewidth=0.8, color="#C44E52")
    axes[0, 1].set_title("Daily Target Volatility")
    axes[0, 1].set_xlabel("Date ID")
    axes[0, 1].set_ylabel("Std Dev")

    axes[1, 0].plot(daily[date_col], daily["mean_weight"], linewidth=0.8, color="#55A868")
    axes[1, 0].set_title("Daily Mean Weight")
    axes[1, 0].set_xlabel("Date ID")
    axes[1, 0].set_ylabel("Weight")

    axes[1, 1].plot(daily[date_col], daily["count"], linewidth=0.8, color="#8172B2")
    axes[1, 1].set_title("Daily Sample Count")
    axes[1, 1].set_xlabel("Date ID")
    axes[1, 1].set_ylabel("Count")

    fig.suptitle("Temporal Patterns", fontsize=15, y=1.02)
    return fig


def plot_symbol_analysis(
    df: pd.DataFrame,
    symbol_col: str = "symbol_id",
    target_col: str = "responder_6",
    weight_col: str = "weight",
) -> plt.Figure:
    sym_stats = df.groupby(symbol_col).agg(
        mean_target=(target_col, "mean"),
        std_target=(target_col, "std"),
        mean_weight=(weight_col, "mean"),
        count=(weight_col, "size"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(sym_stats[symbol_col], sym_stats["mean_target"], color="#4C72B0", edgecolor="black")
    axes[0, 0].set_title("Mean Target by Symbol")
    axes[0, 0].set_xlabel("Symbol ID")
    axes[0, 0].axhline(0, color="red", linewidth=0.8, linestyle="--")

    axes[0, 1].bar(sym_stats[symbol_col], sym_stats["std_target"], color="#C44E52", edgecolor="black")
    axes[0, 1].set_title("Target Volatility by Symbol")
    axes[0, 1].set_xlabel("Symbol ID")

    axes[1, 0].bar(sym_stats[symbol_col], sym_stats["mean_weight"], color="#55A868", edgecolor="black")
    axes[1, 0].set_title("Mean Weight by Symbol")
    axes[1, 0].set_xlabel("Symbol ID")

    axes[1, 1].bar(sym_stats[symbol_col], sym_stats["count"], color="#8172B2", edgecolor="black")
    axes[1, 1].set_title("Sample Count by Symbol")
    axes[1, 1].set_xlabel("Symbol ID")

    fig.suptitle("Per-Symbol Analysis", fontsize=15, y=1.02)
    return fig


# ── Model comparison ─────────────────────────────────────────────────


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric_key: str = "weighted_r2",
    title: str = "Model Comparison: Weighted R²",
) -> plt.Figure:
    models = list(results.keys())
    offline = [results[m].get(f"offline_{metric_key}", 0) for m in models]
    online = [results[m].get(f"online_{metric_key}", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, offline, width, label="Offline", color="#4C72B0", edgecolor="black")
    bars2 = ax.bar(x + width / 2, online, width, label="Online", color="#55A868", edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Weighted R²")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    return fig


def plot_residual_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> plt.Figure:
    residuals = targets - predictions

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(predictions, residuals, alpha=0.1, s=1, color="#4C72B0")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=100, edgecolor="black", alpha=0.7, color="#55A868")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].axvline(0, color="red", linewidth=1)

    if weights is not None:
        axes[2].scatter(weights, np.abs(residuals), alpha=0.1, s=1, color="#C44E52")
        axes[2].set_xlabel("Weight")
        axes[2].set_ylabel("|Residual|")
        axes[2].set_title("Error vs Weight")
    else:
        from scipy import stats
        stats.probplot(residuals, plot=axes[2])
        axes[2].set_title("Q-Q Plot")

    fig.suptitle(f"Residual Analysis: {model_name}", fontsize=15, y=1.02)
    return fig


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.05, s=1, color="#4C72B0")

    lims = [
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual: {model_name}")
    ax.legend()
    ax.set_aspect("equal")
    return fig


def plot_online_learning_curve(
    r2_over_time: list[float],
    window: int = 50,
    title: str = "Online Learning Curve",
) -> plt.Figure:
    rolling = pd.Series(r2_over_time).rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(r2_over_time, alpha=0.2, color="#4C72B0", label="Per-batch R²")
    ax.plot(rolling, color="#C44E52", linewidth=2, label=f"Rolling mean (w={window})")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Weighted R²")
    ax.set_title(title)
    ax.legend()
    return fig
