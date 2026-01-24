"""
Enhanced feature engineering for improved ATS prediction.

This module provides comprehensive feature extraction including:
1. Extended KenPom features (rankings, all metrics)
2. Situational features (rest days, conference play)
3. Rolling performance metrics with recency weighting
4. Market-derived features

Performance optimizations:
- Vectorized operations instead of row-by-row iteration
- Pre-indexed KenPom lookups
- Efficient pandas merges

Target: Get above 52.4% breakeven threshold for -110 odds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.cbb.utils.team_names import TeamNameMapper, build_team_mapping


def load_all_kenpom_snapshots(kenpom_dir: Path) -> pd.DataFrame:
    """
    Load all KenPom daily snapshots into a single DataFrame with date index.

    Returns DataFrame with columns:
    - snapshot_date: Date of the KenPom snapshot
    - TeamName: Team name in KenPom format
    - All KenPom metrics (AdjEM, AdjOE, AdjDE, etc.)
    """
    daily_files = sorted(kenpom_dir.glob("kenpom_????-??-??.csv"))

    if not daily_files:
        raise FileNotFoundError(f"No KenPom daily files found in {kenpom_dir}")

    dfs = []
    for f in daily_files:
        df = pd.read_csv(f)
        date_str = f.stem.replace("kenpom_", "")
        df["snapshot_date"] = pd.to_datetime(date_str)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "AdjEM",
        "AdjOE",
        "AdjDE",
        "AdjTempo",
        "Tempo",
        "OE",
        "DE",
        "RankAdjEM",
        "RankAdjOE",
        "RankAdjDE",
        "RankAdjTempo",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def compute_rest_days_vectorized(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for each team using vectorized operations.

    10-50x faster than row-by-row iteration.

    Adds columns:
    - rest_days_a: Days since team A's last game
    - rest_days_b: Days since team B's last game
    - rest_diff: rest_days_a - rest_days_b
    - b2b_a, b2b_b: Back-to-back indicators
    """
    df = games_df.sort_values("date").copy()

    # Create team-level game records (melt to long format)
    team_a_games = df[["date", "team_a", "game_key"]].rename(columns={"team_a": "team"})
    team_b_games = df[["date", "team_b", "game_key"]].rename(columns={"team_b": "team"})
    all_team_games = pd.concat([team_a_games, team_b_games], ignore_index=True)
    all_team_games = all_team_games.sort_values(["team", "date"])

    # Compute previous game date per team using shift within groups
    all_team_games["prev_date"] = all_team_games.groupby("team")["date"].shift(1)
    all_team_games["rest_days"] = (
        all_team_games["date"] - all_team_games["prev_date"]
    ).dt.days
    all_team_games["rest_days"] = all_team_games["rest_days"].fillna(7).clip(upper=14)

    rest_df = (
        all_team_games[["team", "date", "rest_days"]]
        .groupby(["team", "date"], as_index=False)
        .first()
    )

    df = df.merge(
        rest_df.rename(columns={"team": "team_a", "rest_days": "rest_days_a"}),
        on=["team_a", "date"],
        how="left",
    )
    df = df.merge(
        rest_df.rename(columns={"team": "team_b", "rest_days": "rest_days_b"}),
        on=["team_b", "date"],
        how="left",
    )
    df["rest_days_a"] = df["rest_days_a"].fillna(7)
    df["rest_days_b"] = df["rest_days_b"].fillna(7)
    df["rest_diff"] = df["rest_days_a"] - df["rest_days_b"]

    # Back-to-back indicators
    df["b2b_a"] = (df["rest_days_a"] == 1).astype(int)
    df["b2b_b"] = (df["rest_days_b"] == 1).astype(int)

    return df


def compute_rolling_ats_vectorized(
    games_df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling ATS record using vectorized operations.

    Captures "hot" and "cold" teams from a betting perspective.
    """
    df = games_df.sort_values("date").copy()

    # Create team-level ATS records (need to process chronologically to avoid lookahead)
    # Team A: cover_a, Team B: 1 - cover_a
    team_a_ats = df[["date", "team_a", "cover_a"]].rename(
        columns={"team_a": "team", "cover_a": "ats_result"}
    )
    team_b_ats = df[["date", "team_b", "cover_a"]].copy()
    team_b_ats["ats_result"] = 1 - team_b_ats["cover_a"]
    team_b_ats = team_b_ats.rename(columns={"team_b": "team"})[
        ["date", "team", "ats_result"]
    ]

    all_ats = pd.concat([team_a_ats, team_b_ats], ignore_index=True)
    all_ats = all_ats.sort_values(["team", "date"])

    # Filter to valid ATS results (no pushes)
    all_ats = all_ats[all_ats["ats_result"].isin([0, 1])]

    # Compute rolling mean using shift to avoid lookahead
    all_ats["rolling_ats"] = all_ats.groupby("team")["ats_result"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    all_ats["rolling_ats"] = all_ats["rolling_ats"].fillna(0.5)

    ats_df = all_ats.groupby(["team", "date"], as_index=False)["rolling_ats"].first()

    df = df.merge(
        ats_df.rename(columns={"team": "team_a", "rolling_ats": "rolling_ats_a"}),
        on=["team_a", "date"],
        how="left",
    )
    df = df.merge(
        ats_df.rename(columns={"team": "team_b", "rolling_ats": "rolling_ats_b"}),
        on=["team_b", "date"],
        how="left",
    )
    df["rolling_ats_a"] = df["rolling_ats_a"].fillna(0.5)
    df["rolling_ats_b"] = df["rolling_ats_b"].fillna(0.5)
    df["rolling_ats_diff"] = df["rolling_ats_a"] - df["rolling_ats_b"]

    return df


def compute_recency_weighted_stats_vectorized(
    games_df: pd.DataFrame,
    half_life: int = 5,
) -> pd.DataFrame:
    """
    Compute exponentially weighted rolling stats using vectorized operations.

    Args:
        games_df: Games DataFrame
        half_life: Number of games for weight to decay by half
    """
    df = games_df.sort_values("date").copy()

    # Create team-level stats (margin from each team's perspective)
    team_a_stats = df[["date", "team_a", "final_margin_a", "points_a"]].rename(
        columns={"team_a": "team", "final_margin_a": "margin", "points_a": "ppg"}
    )
    team_b_stats = df[["date", "team_b", "final_margin_a", "points_b"]].copy()
    team_b_stats["margin"] = -team_b_stats["final_margin_a"]
    team_b_stats = team_b_stats.rename(columns={"team_b": "team", "points_b": "ppg"})[
        ["date", "team", "margin", "ppg"]
    ]

    all_stats = pd.concat([team_a_stats, team_b_stats], ignore_index=True)
    all_stats = all_stats.sort_values(["team", "date"])

    # Compute EWM using shift to avoid lookahead
    all_stats["ew_margin"] = all_stats.groupby("team")["margin"].transform(
        lambda x: x.shift(1).ewm(halflife=half_life, min_periods=3).mean()
    )
    all_stats["ew_ppg"] = all_stats.groupby("team")["ppg"].transform(
        lambda x: x.shift(1).ewm(halflife=half_life, min_periods=3).mean()
    )

    ew_stats = all_stats.groupby(["team", "date"], as_index=False)[
        ["ew_margin", "ew_ppg"]
    ].first()

    df = df.merge(
        ew_stats.rename(
            columns={"team": "team_a", "ew_margin": "ew_margin_a", "ew_ppg": "ew_ppg_a"}
        ),
        on=["team_a", "date"],
        how="left",
    )
    df = df.merge(
        ew_stats.rename(
            columns={"team": "team_b", "ew_margin": "ew_margin_b", "ew_ppg": "ew_ppg_b"}
        ),
        on=["team_b", "date"],
        how="left",
    )
    df["ew_margin_diff"] = df["ew_margin_a"] - df["ew_margin_b"]

    return df


def merge_kenpom_features_vectorized(
    games_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    """
    Merge KenPom features using efficient pandas operations.

    Instead of looking up KenPom for each game individually, this:
    1. Pre-computes the most recent snapshot date for each game date
    2. Maps team names in bulk
    3. Uses pandas merge for efficient joining
    """
    df = games_df.copy()

    # Map team names to KenPom names
    df["team_a_kp"] = df["team_a"].map(team_map)
    df["team_b_kp"] = df["team_b"].map(team_map)

    # Get unique game dates
    game_dates = df["date"].unique()
    snapshot_dates = sorted(kenpom_df["snapshot_date"].unique())

    # Build mapping: game_date -> most recent snapshot_date
    date_to_snapshot: dict = {}
    for game_date in game_dates:
        # Find most recent snapshot before game date (within 14 days for daily, 365 for seasonal)
        valid_snapshots = [s for s in snapshot_dates if s < game_date]
        if valid_snapshots:
            # Prefer recent snapshots (within 14 days)
            recent = [s for s in valid_snapshots if (game_date - s).days <= 14]
            if recent:
                date_to_snapshot[game_date] = max(recent)
            else:
                # Fall back to seasonal (up to 365 days)
                seasonal = [s for s in valid_snapshots if (game_date - s).days <= 365]
                if seasonal:
                    date_to_snapshot[game_date] = max(seasonal)

    # Add snapshot date to games
    df["kp_snapshot_date"] = df["date"].map(date_to_snapshot)

    # Prepare KenPom data for merge
    team_col = "TeamName" if "TeamName" in kenpom_df.columns else "Team"
    kp_cols = [team_col, "snapshot_date", "AdjEM", "AdjOE", "AdjDE", "AdjTempo"]
    rank_cols = ["RankAdjEM", "RankAdjOE", "RankAdjDE", "RankAdjTempo"]
    kp_cols_available = kp_cols + [c for c in rank_cols if c in kenpom_df.columns]

    kp_subset = kenpom_df[kp_cols_available].copy()
    kp_subset = kp_subset.rename(columns={team_col: "kp_team"})

    # Merge for Team A
    kp_a = kp_subset.copy()
    kp_a.columns = [
        f"{c}_a" if c not in ["kp_team", "snapshot_date"] else c for c in kp_a.columns
    ]
    df = df.merge(
        kp_a,
        left_on=["team_a_kp", "kp_snapshot_date"],
        right_on=["kp_team", "snapshot_date"],
        how="left",
    )
    df = df.drop(columns=["kp_team", "snapshot_date"], errors="ignore")

    # Merge for Team B
    kp_b = kp_subset.copy()
    kp_b.columns = [
        f"{c}_b" if c not in ["kp_team", "snapshot_date"] else c for c in kp_b.columns
    ]
    df = df.merge(
        kp_b,
        left_on=["team_b_kp", "kp_snapshot_date"],
        right_on=["kp_team", "snapshot_date"],
        how="left",
    )
    df = df.drop(columns=["kp_team", "snapshot_date"], errors="ignore")

    # Rename columns to standard format
    rename_map = {
        "AdjEM_a": "kp_adj_em_a",
        "AdjEM_b": "kp_adj_em_b",
        "AdjOE_a": "kp_adj_o_a",
        "AdjOE_b": "kp_adj_o_b",
        "AdjDE_a": "kp_adj_d_a",
        "AdjDE_b": "kp_adj_d_b",
        "AdjTempo_a": "kp_tempo_a",
        "AdjTempo_b": "kp_tempo_b",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Rename rank columns
    n_teams = 365
    for side in ["a", "b"]:
        for col in ["RankAdjEM", "RankAdjOE", "RankAdjDE"]:
            old_col = f"{col}_{side}"
            if old_col in df.columns:
                new_col = f"kp_rank_{col.replace('RankAdj', '').lower()}_{side}"
                df[new_col] = df[old_col] / n_teams
                df = df.drop(columns=[old_col])

    # Mark KenPom matched
    df["kp_matched"] = df["kp_adj_em_a"].notna() & df["kp_adj_em_b"].notna()

    # Compute derived features vectorized
    df["kp_adj_em_diff"] = df["kp_adj_em_a"] - df["kp_adj_em_b"]
    df["kp_tempo_avg"] = (df["kp_tempo_a"] + df["kp_tempo_b"]) / 2
    df["kp_tempo_diff"] = df["kp_tempo_a"] - df["kp_tempo_b"]
    df["kp_o_vs_d_a"] = df["kp_adj_o_a"] - df["kp_adj_d_b"]
    df["kp_o_vs_d_b"] = df["kp_adj_o_b"] - df["kp_adj_d_a"]

    # Predicted margin (raw, before HCA)
    df["kp_pred_margin_raw"] = df["kp_adj_em_diff"] * df["kp_tempo_avg"] / 100

    return df


def build_enhanced_features(
    games_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    """
    Build comprehensive feature set for ATS prediction.

    Features included:
    1. Extended KenPom metrics (efficiency, rankings, matchup-specific)
    2. Rest days and back-to-back indicators
    3. Rolling ATS record
    4. Recency-weighted performance stats
    5. Context features (home/neutral)

    Uses vectorized operations for efficiency.
    """
    print("Computing rest days (vectorized)...")
    df = compute_rest_days_vectorized(games_df)

    print("Computing rolling ATS records (vectorized)...")
    df = compute_rolling_ats_vectorized(df)

    print("Computing recency-weighted stats (vectorized)...")
    df = compute_recency_weighted_stats_vectorized(df)

    print("Merging KenPom features (vectorized)...")
    df = merge_kenpom_features_vectorized(df, kenpom_df, team_map)

    # Add context features
    df["is_home_a"] = (~df["is_neutral"]).astype(int)
    df["is_neutral"] = df["is_neutral"].astype(int)

    # Add home court advantage adjusted prediction
    HCA = 3.5  # Home court advantage in points
    if "kp_pred_margin_raw" in df.columns:
        df["kp_pred_margin"] = df["kp_pred_margin_raw"] + (
            df["is_home_a"] * HCA - (1 - df["is_home_a"] - df["is_neutral"]) * HCA
        )

    # Select and order columns for output
    core_cols = [
        "game_key",
        "season",
        "date",
        "team_a",
        "team_b",
        "is_home_a",
        "is_neutral",
        "spread_a",
        "final_margin_a",
        "cover_a",
    ]
    kenpom_cols = [c for c in df.columns if c.startswith("kp_")]
    rest_cols = ["rest_days_a", "rest_days_b", "rest_diff", "b2b_a", "b2b_b"]
    rolling_cols = ["rolling_ats_a", "rolling_ats_b", "rolling_ats_diff"]
    ew_cols = ["ew_margin_a", "ew_margin_b", "ew_ppg_a", "ew_ppg_b", "ew_margin_diff"]

    # Ensure all columns exist
    all_cols = core_cols + kenpom_cols + rest_cols + rolling_cols + ew_cols
    available_cols = [c for c in all_cols if c in df.columns]

    # Add any extra columns not in our list
    extra_cols = [c for c in df.columns if c not in available_cols]
    output_cols = available_cols + extra_cols

    return df[output_cols]


def main() -> pd.DataFrame:
    """Build enhanced feature dataset."""
    project_root = Path(__file__).parent.parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    kenpom_dir = project_root / "data" / "kenpom"
    features_dir = project_root / "data" / "features"

    features_dir.mkdir(parents=True, exist_ok=True)

    # Load games
    games_df = pd.read_parquet(processed_dir / "games_base.parquet")
    print(f"Loaded {len(games_df):,} games")

    # Load KenPom data
    kenpom_df = load_all_kenpom_snapshots(kenpom_dir)
    print(
        f"Loaded {len(kenpom_df):,} KenPom records from "
        f"{kenpom_df['snapshot_date'].nunique()} snapshots"
    )

    # Build team name mapping
    game_teams = set(games_df["team_a"].unique()) | set(games_df["team_b"].unique())
    team_col = "TeamName" if "TeamName" in kenpom_df.columns else "Team"
    kenpom_teams = set(kenpom_df[team_col].unique())
    team_map = build_team_mapping(game_teams, kenpom_teams)
    print(f"Mapped {len(team_map)} teams")

    # Save improved mapping
    mapping_df = pd.DataFrame(
        [
            {"raw_name": k, "kenpom_name": v, "notes": "auto-matched"}
            for k, v in team_map.items()
        ]
    )
    mapping_df.to_csv(processed_dir / "team_name_map_v2.csv", index=False)

    # Build features
    features_df = build_enhanced_features(games_df, kenpom_df, team_map)

    # Report stats
    kp_matched = features_df["kp_matched"].sum()
    with_spread = features_df["spread_a"].notna().sum()
    print(f"\nFeature dataset stats:")
    print(f"  Total games: {len(features_df):,}")
    print(f"  KenPom matched: {kp_matched:,} ({kp_matched / len(features_df):.1%})")
    print(f"  With spread: {with_spread:,} ({with_spread / len(features_df):.1%})")

    # Save
    features_df.to_parquet(
        features_dir / "games_features_enhanced.parquet", index=False
    )
    print(f"\n✓ Saved to {features_dir / 'games_features_enhanced.parquet'}")

    return features_df


if __name__ == "__main__":
    main()
