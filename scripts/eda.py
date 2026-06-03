#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for the Jane Street market data.

Generates a comprehensive set of visualizations and statistics:
  - Data overview and summary statistics
  - Missing value analysis
  - Target distribution (weighted and unweighted)
  - Feature distributions and correlations
  - Temporal patterns (daily trends, volatility)
  - Per-symbol analysis
  - Highly correlated feature detection

All plots are saved to the specified output directory.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.feature_analysis import (
    compute_correlations,
    compute_summary_statistics,
    find_highly_correlated,
    missing_data_profile,
    per_symbol_summary,
    weight_analysis,
)
from src.visualization import (
    apply_style,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_missing_data,
    plot_symbol_analysis,
    plot_target_correlations,
    plot_target_distribution,
    plot_temporal_patterns,
    save_fig,
)


def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis")
    parser.add_argument("--data-root", type=str, required=True, help="Path to train.parquet root")
    parser.add_argument("--partition", type=int, default=0, help="Partition to analyze")
    parser.add_argument("--output-dir", type=str, default="outputs/eda", help="Directory for plots and reports")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fraction of data to sample for speed")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.data_root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    apply_style()

    # ── Load data ───────────────────────────────────────────────────
    print("Loading data...")
    path = cfg.data.partition_path(args.partition)
    df_pl = pl.read_parquet(str(path))

    if args.sample_frac and args.sample_frac < 1.0:
        df_pl = df_pl.sample(fraction=args.sample_frac, seed=42)
        print(f"  Sampled {len(df_pl):,} rows ({args.sample_frac:.0%})")

    df = df_pl.to_pandas()
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    feature_cols = [f"feature_{i:02d}" for i in range(79)]
    target_col = "responder_6"

    # ── 1. Data overview ────────────────────────────────────────────
    print("\n1. Summary statistics...")
    summary = compute_summary_statistics(df)
    summary.to_csv(output / "summary_statistics.csv")
    print(f"   Saved to {output / 'summary_statistics.csv'}")

    # ── 2. Missing data ─────────────────────────────────────────────
    print("2. Missing data analysis...")
    profile = missing_data_profile(df)
    profile.to_csv(output / "missing_data_profile.csv")

    missing_pct = profile["missing_pct"]
    fig = plot_missing_data(missing_pct)
    save_fig(fig, output / "missing_values.png")
    print(f"   Features with missing data: {(missing_pct > 0).sum()}")
    print(f"   Max missing: {missing_pct.max():.1f}%")

    # ── 3. Target distribution ──────────────────────────────────────
    print("3. Target distribution...")
    targets = df[target_col].fillna(0).values
    weights = df["weight"].fillna(0).values
    fig = plot_target_distribution(targets, weights)
    save_fig(fig, output / "target_distribution.png")
    print(f"   Mean: {targets.mean():.6f}  Std: {targets.std():.6f}")

    # ── 4. Feature distributions ────────────────────────────────────
    print("4. Feature distributions...")
    top_features = feature_cols[:20]
    fig = plot_feature_distributions(df, top_features, ncols=5)
    save_fig(fig, output / "feature_distributions_top20.png")

    # ── 5. Correlations ─────────────────────────────────────────────
    print("5. Correlation analysis...")
    corr_matrix, target_corr = compute_correlations(df, feature_cols, target_col)

    fig = plot_correlation_heatmap(corr_matrix)
    save_fig(fig, output / "correlation_heatmap.png")

    fig = plot_target_correlations(target_corr, top_n=25)
    save_fig(fig, output / "target_correlations_top25.png")
    target_corr.to_csv(output / "target_correlations.csv")

    highly_corr = find_highly_correlated(corr_matrix, threshold=0.90)
    highly_corr.to_csv(output / "highly_correlated_pairs.csv", index=False)
    print(f"   Highly correlated pairs (|r| >= 0.90): {len(highly_corr)}")

    # ── 6. Temporal patterns ────────────────────────────────────────
    print("6. Temporal patterns...")
    fig = plot_temporal_patterns(df)
    save_fig(fig, output / "temporal_patterns.png")

    # ── 7. Per-symbol analysis ──────────────────────────────────────
    print("7. Per-symbol analysis...")
    fig = plot_symbol_analysis(df)
    save_fig(fig, output / "symbol_analysis.png")

    sym_summary = per_symbol_summary(df)
    sym_summary.to_csv(output / "per_symbol_summary.csv")
    print(f"   Unique symbols: {df['symbol_id'].nunique()}")

    # ── 8. Weight analysis ──────────────────────────────────────────
    print("8. Weight analysis...")
    w_stats = weight_analysis(df)
    with open(output / "weight_analysis.json", "w") as f:
        json.dump(w_stats, f, indent=2)
    print(f"   Weight mean: {w_stats['mean']:.4f}  Zero pct: {w_stats['zero_pct']:.2f}%")

    # ── Report ──────────────────────────────────────────────────────
    report = {
        "partition": args.partition,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": len(feature_cols),
        "symbols": int(df["symbol_id"].nunique()),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "missing_features": int((missing_pct > 0).sum()),
        "highly_correlated_pairs": len(highly_corr),
    }
    with open(output / "eda_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEDA complete. All outputs saved to: {output}/")


if __name__ == "__main__":
    main()
