"""Data preparation utilities."""

from __future__ import annotations

import pandas as pd


def prepare_ats_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    require_spread: bool = True,
    require_kenpom: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter and prepare data for ATS training/evaluation.

    Returns:
        (filtered_df, available_feature_cols)
    """
    working = df.copy()

    if require_spread:
        working = working[working["spread_a"].notna()].copy()

    if require_kenpom:
        working = working[working["kp_matched"] == True].copy()

    available = [c for c in feature_cols if c in working.columns]
    missing = sorted(set(feature_cols) - set(available))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(f"Warning: Missing {len(missing)} features: {preview}{suffix}")

    working = working.dropna(subset=available + ["final_margin_a", "cover_a"])

    return working, available
