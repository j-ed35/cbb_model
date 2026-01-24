"""
KenPom feature integration.

Loads KenPom exports and joins to game dataset.
Supports two modes:
1. Daily snapshots: data/kenpom/kenpom_YYYY-MM-DD.csv (time-safe)
2. Season files: data/kenpom/kenpom_YYYY.csv (LEAKY - end-of-season data)

Output:
- data/features/games_features.parquet
- data/processed/team_name_map.csv (team name mapping)
- reports/quarantine/kenpom_unmatched.csv
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.cbb.utils.team_names import TeamNameMapper, build_team_mapping


def load_kenpom_data(kenpom_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Load KenPom data, preferring daily snapshots over season files.

    Returns:
        kenpom_df: Combined KenPom data
        mode: 'daily' or 'season' (season mode is LEAKY)
    """
    daily_files = sorted(kenpom_dir.glob("kenpom_????-??-??.csv"))
    season_files = sorted(kenpom_dir.glob("kenpom_????.csv"))

    if daily_files:
        # Load daily snapshots
        dfs = []
        for f in daily_files:
            df = pd.read_csv(f)
            date_str = f.stem.replace("kenpom_", "")
            df["snapshot_date"] = pd.to_datetime(date_str)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True), "daily"

    elif season_files:
        # Load season files (LEAKY)
        warnings.warn(
            "Using season-level KenPom files. Results are LEAKY and should "
            "only be used for prototyping. Replace with daily snapshots for production."
        )
        dfs = []
        for f in season_files:
            df = pd.read_csv(f)
            season = f.stem.replace("kenpom_", "")
            df["season"] = season
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True), "season"

    else:
        raise FileNotFoundError(
            f"No KenPom files found in {kenpom_dir}. "
            "Expected: kenpom_YYYY-MM-DD.csv (daily) or kenpom_YYYY.csv (season)"
        )


def get_kenpom_features_for_game(
    game_date: datetime,
    team_a: str,
    team_b: str,
    kenpom_df: pd.DataFrame,
    team_map: dict[str, str],
    mode: str,
) -> dict:
    """
    Get KenPom features for a single game.

    For daily mode: uses most recent snapshot before game date.
    For season mode: uses season data (LEAKY).
    """
    team_a_kp = team_map.get(team_a)
    team_b_kp = team_map.get(team_b)

    if not team_a_kp or not team_b_kp:
        return {"kp_matched": False}

    # Determine team column name
    team_col = "TeamName" if "TeamName" in kenpom_df.columns else "Team"

    if mode == "daily":
        # Get most recent snapshot before game date
        valid_snapshots = kenpom_df[kenpom_df["snapshot_date"] < game_date]
        if len(valid_snapshots) == 0:
            return {"kp_matched": False}
        latest_date = valid_snapshots["snapshot_date"].max()
        snapshot = valid_snapshots[valid_snapshots["snapshot_date"] == latest_date]
    else:
        # Season mode - just use all data (LEAKY)
        snapshot = kenpom_df

    # Find team rows
    team_a_row = snapshot[snapshot[team_col] == team_a_kp]
    team_b_row = snapshot[snapshot[team_col] == team_b_kp]

    if len(team_a_row) == 0 or len(team_b_row) == 0:
        return {"kp_matched": False}

    team_a_row = team_a_row.iloc[0]
    team_b_row = team_b_row.iloc[0]

    # Extract features using actual KenPom export column names:
    # AdjEM, AdjOE (offense), AdjDE (defense), AdjTempo
    features = {
        "kp_matched": True,
        # Adjusted Efficiency Margin
        "kp_adj_em_a": team_a_row.get("AdjEM", np.nan),
        "kp_adj_em_b": team_b_row.get("AdjEM", np.nan),
        # Adjusted Offense (column is AdjOE in exports)
        "kp_adj_o_a": team_a_row.get("AdjOE", team_a_row.get("AdjO", np.nan)),
        "kp_adj_o_b": team_b_row.get("AdjOE", team_b_row.get("AdjO", np.nan)),
        # Adjusted Defense (column is AdjDE in exports)
        "kp_adj_d_a": team_a_row.get("AdjDE", team_a_row.get("AdjD", np.nan)),
        "kp_adj_d_b": team_b_row.get("AdjDE", team_b_row.get("AdjD", np.nan)),
        # Tempo (column is AdjTempo in exports)
        "kp_tempo_a": team_a_row.get("AdjTempo", team_a_row.get("AdjT", np.nan)),
        "kp_tempo_b": team_b_row.get("AdjTempo", team_b_row.get("AdjT", np.nan)),
    }

    # Derived features
    if not np.isnan(features["kp_adj_em_a"]) and not np.isnan(features["kp_adj_em_b"]):
        features["kp_adj_em_diff"] = features["kp_adj_em_a"] - features["kp_adj_em_b"]

    if not np.isnan(features["kp_tempo_a"]) and not np.isnan(features["kp_tempo_b"]):
        features["kp_tempo_avg"] = (features["kp_tempo_a"] + features["kp_tempo_b"]) / 2

    if not np.isnan(features["kp_adj_o_a"]) and not np.isnan(features["kp_adj_d_b"]):
        features["kp_o_vs_d"] = features["kp_adj_o_a"] - features["kp_adj_d_b"]

    if not np.isnan(features["kp_adj_o_b"]) and not np.isnan(features["kp_adj_d_a"]):
        features["kp_o_vs_d_opp"] = features["kp_adj_o_b"] - features["kp_adj_d_a"]

    return features


def build_features(
    games_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    team_map: dict[str, str],
    mode: str,
) -> pd.DataFrame:
    """
    Build feature dataset by joining KenPom features to games.
    """
    features_list = []

    for idx, row in games_df.iterrows():
        game_features = {
            "game_key": row["game_key"],
            "season": row["season"],
            "date": row["date"],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "is_home_a": 1 if not row["is_neutral"] else 0,
            "is_neutral": 1 if row["is_neutral"] else 0,
            "spread_a": row["spread_a"],
            "final_margin_a": row["final_margin_a"],
            "cover_a": row["cover_a"],
        }

        kp_features = get_kenpom_features_for_game(
            row["date"], row["team_a"], row["team_b"],
            kenpom_df, team_map, mode
        )
        game_features.update(kp_features)
        features_list.append(game_features)

    features_df = pd.DataFrame(features_list)

    # Add LEAKY flag if using season mode
    features_df["LEAKY_KENPOM"] = mode == "season"

    return features_df


def main() -> Optional[pd.DataFrame]:
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent.parent
    kenpom_dir = project_root / "data" / "kenpom"
    processed_dir = project_root / "data" / "processed"
    features_dir = project_root / "data" / "features"
    quarantine_dir = project_root / "reports" / "quarantine"

    features_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Load games
    games_df = pd.read_parquet(processed_dir / "games_base.parquet")
    print(f"Loaded {len(games_df):,} games")

    # Load KenPom data
    try:
        kenpom_df, mode = load_kenpom_data(kenpom_dir)
        print(f"Loaded KenPom data in {mode} mode")
        if mode == "season":
            print("WARNING: Using season-level KenPom data. Results are LEAKY!")
    except FileNotFoundError as e:
        print(f"No KenPom data found: {e}")
        print("Creating base features without KenPom...")

        # Create features without KenPom
        features_df = games_df[[
            "game_key", "season", "date", "team_a", "team_b",
            "is_neutral", "spread_a", "final_margin_a", "cover_a"
        ]].copy()
        features_df["is_home_a"] = (~features_df["is_neutral"]).astype(int)
        features_df["is_neutral"] = features_df["is_neutral"].astype(int)
        features_df["kp_matched"] = False
        features_df["LEAKY_KENPOM"] = False

        features_df.to_parquet(features_dir / "games_features.parquet", index=False)
        print(f" Wrote {features_dir / 'games_features.parquet'}")
        print("  Note: No KenPom features included. Add KenPom data to data/kenpom/")
        return features_df

    # Build team name mapping using shared utility
    game_teams = set(games_df["team_a"].unique()) | set(games_df["team_b"].unique())
    team_col = "TeamName" if "TeamName" in kenpom_df.columns else "Team"
    kenpom_teams = set(kenpom_df[team_col].unique())

    team_map = build_team_mapping(game_teams, kenpom_teams)

    # Save mapping
    team_map_df = pd.DataFrame([
        {"raw_name": k, "kenpom_name": v, "notes": "auto-matched"}
        for k, v in team_map.items()
    ])

    # Add unmapped teams
    unmapped = game_teams - set(team_map.keys())
    for team in sorted(unmapped):
        team_map_df = pd.concat([
            team_map_df,
            pd.DataFrame([{
                "raw_name": team,
                "kenpom_name": None,
                "notes": "NEEDS MANUAL MAPPING"
            }])
        ], ignore_index=True)

    team_map_df.to_csv(processed_dir / "team_name_map.csv", index=False)
    print(" Wrote team name mapping")

    # Check mapping coverage
    mapped_count = len(team_map)
    total_count = len(game_teams)
    print(f"  Mapped {mapped_count}/{total_count} teams ({mapped_count/total_count:.1%})")

    # Build features
    features_df = build_features(games_df, kenpom_df, team_map, mode)
    print(f"Built features for {len(features_df):,} games")

    # Report matching stats
    matched = features_df["kp_matched"].sum()
    print(f"  KenPom matched: {matched:,} ({matched/len(features_df):.1%})")

    # Save unmatched games to quarantine
    unmatched = features_df[~features_df["kp_matched"]]
    if len(unmatched) > 0:
        unmatched.to_csv(quarantine_dir / "kenpom_unmatched.csv", index=False)
        print(f"  Wrote {len(unmatched):,} unmatched games to quarantine")

    # Save features
    features_df.to_parquet(features_dir / "games_features.parquet", index=False)
    print(f" Wrote {features_dir / 'games_features.parquet'}")

    return features_df


if __name__ == "__main__":
    main()
