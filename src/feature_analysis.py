"""Feature analysis utilities: missing data profiling, correlation, importance ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missing_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column missing data statistics."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total) * 100
    dtypes = df.dtypes

    profile = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct.round(2),
        "dtype": dtypes,
        "nunique": df.nunique(),
        "mean": df.select_dtypes(include=[np.number]).mean(),
        "std": df.select_dtypes(include=[np.number]).std(),
    })
    return profile.sort_values("missing_pct", ascending=False)


def compute_correlations(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "responder_6"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the full feature correlation matrix and per-feature correlation with target."""
    features = df[feature_cols].select_dtypes(include=[np.number])
    corr_matrix = features.corr()

    if target_col in df.columns:
        target_corr = features.corrwith(df[target_col]).sort_values(ascending=False)
    else:
        target_corr = pd.Series(dtype=float)

    return corr_matrix, target_corr


def find_highly_correlated(
    corr_matrix: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    """Find feature pairs with absolute correlation above threshold."""
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({"feature_1": idx, "feature_2": col, "correlation": val})
    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values("correlation", key=abs, ascending=False)
    return result


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Extended descriptive statistics for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["skew"] = numeric.skew()
    stats["kurtosis"] = numeric.kurtosis()
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["range"] = stats["max"] - stats["min"]
    return stats


def weight_analysis(df: pd.DataFrame, weight_col: str = "weight") -> dict:
    """Analyze the weight distribution and its relationship to other columns."""
    w = df[weight_col].dropna()
    return {
        "mean": float(w.mean()),
        "median": float(w.median()),
        "std": float(w.std()),
        "min": float(w.min()),
        "max": float(w.max()),
        "zero_pct": float((w == 0).mean() * 100),
        "negative_pct": float((w < 0).mean() * 100),
        "skew": float(w.skew()),
        "kurtosis": float(w.kurtosis()),
    }


def per_symbol_summary(
    df: pd.DataFrame,
    symbol_col: str = "symbol_id",
    target_col: str = "responder_6",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Aggregate statistics per symbol."""
    return df.groupby(symbol_col).agg(
        count=(target_col, "size"),
        target_mean=(target_col, "mean"),
        target_std=(target_col, "std"),
        target_median=(target_col, "median"),
        target_skew=(target_col, lambda x: x.skew()),
        weight_mean=(weight_col, "mean"),
        weight_std=(weight_col, "std"),
    ).round(6)
