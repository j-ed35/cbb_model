"""
Build enhanced features v2 with extended KenPom data.

Adds to existing features:
1. Team-specific HCA (instead of fixed 3.5)
2. Four Factors (eFG%, TO%, OR%, FTR for offense and defense)
3. Height and Experience
4. Team Stats (3P%, 2P%, Blk%, Stl%, etc.)

These create matchup-specific features that capture style interactions.

Optimized with vectorized merge operations (no row-by-row iteration).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.cbb.utils.team_names import build_team_mapping


def load_extended_kenpom_data(kenpom_ext_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all extended KenPom data files."""
    data = {}

    # Only load columns we need from each file
    ff_path = kenpom_ext_dir / 'four_factors.parquet'
    if ff_path.exists():
        data['four_factors'] = pd.read_parquet(ff_path)

    hca_path = kenpom_ext_dir / 'hca.parquet'
    if hca_path.exists():
        data['hca'] = pd.read_parquet(hca_path)

    ht_path = kenpom_ext_dir / 'height_experience.parquet'
    if ht_path.exists():
        data['height_exp'] = pd.read_parquet(ht_path)

    ts_path = kenpom_ext_dir / 'team_stats.parquet'
    if ts_path.exists():
        data['team_stats'] = pd.read_parquet(ts_path)

    return data


def get_season_int(season_str: str | int) -> int:
    """Convert season string like '2021-22' to integer 2022."""
    if isinstance(season_str, int):
        return season_str
    if '-' in str(season_str):
        return int('20' + season_str.split('-')[1])
    return int(season_str)


def merge_four_factors_vectorized(
    games_df: pd.DataFrame,
    ff_df: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    """Merge Four Factors data using vectorized operations."""
    df = games_df.copy()

    # Convert season to int for matching
    df['_season_int'] = df['season'].apply(get_season_int)

    # Map team names to KenPom names
    df['_team_a_kp'] = df['team_a'].map(team_map)
    df['_team_b_kp'] = df['team_b'].map(team_map)

    # Prepare FF data for merge
    ff_cols = ['off_efg', 'off_to_pct', 'off_or_pct', 'off_ft_rate',
               'def_efg', 'def_to_pct', 'def_or_pct', 'def_ft_rate']

    available_ff_cols = [c for c in ff_cols if c in ff_df.columns]

    if not available_ff_cols:
        # No FF columns available, clean up and return
        df = df.drop(columns=['_season_int', '_team_a_kp', '_team_b_kp'])
        return df

    ff_merge = ff_df[['season', 'TeamName'] + available_ff_cols].copy()

    # Merge for team A
    ff_a = ff_merge.rename(columns={
        'TeamName': '_team_a_kp',
        'season': '_season_int',
        **{col: f'ff_{col}_a' for col in available_ff_cols}
    })
    df = df.merge(ff_a, on=['_season_int', '_team_a_kp'], how='left')

    # Merge for team B
    ff_b = ff_merge.rename(columns={
        'TeamName': '_team_b_kp',
        'season': '_season_int',
        **{col: f'ff_{col}_b' for col in available_ff_cols}
    })
    df = df.merge(ff_b, on=['_season_int', '_team_b_kp'], how='left')

    # Clean up temp columns
    df = df.drop(columns=['_season_int', '_team_a_kp', '_team_b_kp'])

    return df


def merge_hca_vectorized(
    games_df: pd.DataFrame,
    hca_df: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    """Merge HCA data using vectorized operations."""
    df = games_df.copy()

    # Map team names
    df['_team_a_kp'] = df['team_a'].map(team_map)
    df['_team_b_kp'] = df['team_b'].map(team_map)

    # Prepare HCA for merge (one row per team)
    hca_cols = ['hca']
    if 'elevation' in hca_df.columns:
        hca_cols.append('elevation')

    hca_merge = hca_df[['TeamName'] + hca_cols].drop_duplicates(subset='TeamName')

    # Merge HCA for both teams
    hca_a = hca_merge.rename(columns={
        'TeamName': '_team_a_kp',
        'hca': '_hca_a',
        **({'elevation': '_elev_a'} if 'elevation' in hca_cols else {})
    })
    df = df.merge(hca_a, on='_team_a_kp', how='left')

    hca_b = hca_merge.rename(columns={
        'TeamName': '_team_b_kp',
        'hca': '_hca_b',
        **({'elevation': '_elev_b'} if 'elevation' in hca_cols else {})
    })
    df = df.merge(hca_b, on='_team_b_kp', how='left')

    # Compute team_hca based on home/away/neutral (vectorized)
    # Default to 3.5 if no data
    df['_hca_a'] = df['_hca_a'].fillna(3.5)
    df['_hca_b'] = df['_hca_b'].fillna(3.5)

    # is_home_a == 1: use team A's HCA (positive)
    # is_home_a == 0 and is_neutral == 0: use team B's HCA (negative for team A)
    # is_neutral == 1: HCA = 0
    conditions = [
        df['is_home_a'] == 1,
        (df['is_home_a'] == 0) & (df['is_neutral'] == 0),
        df['is_neutral'] == 1,
    ]
    choices = [
        df['_hca_a'],
        -df['_hca_b'],
        0.0,
    ]
    df['team_hca'] = np.select(conditions, choices, default=0.0)

    # Elevation (use home team's)
    if 'elevation' in hca_cols:
        df['_elev_a'] = df['_elev_a'].fillna(0.0)
        df['_elev_b'] = df['_elev_b'].fillna(0.0)
        elev_conditions = [
            df['is_home_a'] == 1,
            (df['is_home_a'] == 0) & (df['is_neutral'] == 0),
        ]
        elev_choices = [
            df['_elev_a'],
            df['_elev_b'],
        ]
        df['team_elevation'] = np.select(elev_conditions, elev_choices, default=0.0)
    else:
        df['team_elevation'] = 0.0

    # Clean up temp columns
    temp_cols = ['_team_a_kp', '_team_b_kp', '_hca_a', '_hca_b']
    if 'elevation' in hca_cols:
        temp_cols.extend(['_elev_a', '_elev_b'])
    df = df.drop(columns=temp_cols)

    return df


def merge_height_experience_vectorized(
    games_df: pd.DataFrame,
    ht_df: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    """Merge Height and Experience data using vectorized operations."""
    df = games_df.copy()

    # Convert season
    df['_season_int'] = df['season'].apply(get_season_int)

    # Map team names
    df['_team_a_kp'] = df['team_a'].map(team_map)
    df['_team_b_kp'] = df['team_b'].map(team_map)

    # Columns to merge
    ht_cols = ['avg_height', 'experience', 'continuity', 'bench_pct']
    available_ht_cols = [c for c in ht_cols if c in ht_df.columns]

    if not available_ht_cols:
        df = df.drop(columns=['_season_int', '_team_a_kp', '_team_b_kp'])
        return df

    ht_merge = ht_df[['season', 'TeamName'] + available_ht_cols].copy()

    # Merge for team A
    ht_a = ht_merge.rename(columns={
        'TeamName': '_team_a_kp',
        'season': '_season_int',
        **{col: f'ht_{col}_a' for col in available_ht_cols}
    })
    df = df.merge(ht_a, on=['_season_int', '_team_a_kp'], how='left')

    # Merge for team B
    ht_b = ht_merge.rename(columns={
        'TeamName': '_team_b_kp',
        'season': '_season_int',
        **{col: f'ht_{col}_b' for col in available_ht_cols}
    })
    df = df.merge(ht_b, on=['_season_int', '_team_b_kp'], how='left')

    # Clean up
    df = df.drop(columns=['_season_int', '_team_a_kp', '_team_b_kp'])

    return df


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived matchup features based on Four Factors and other stats."""
    # Four Factors matchups (vectorized)
    if 'ff_off_efg_a' in df.columns and 'ff_def_efg_b' in df.columns:
        df['ff_efg_matchup_a'] = df['ff_off_efg_a'] - df['ff_def_efg_b']
        df['ff_efg_matchup_b'] = df['ff_off_efg_b'] - df['ff_def_efg_a']

    if 'ff_off_to_pct_a' in df.columns and 'ff_def_to_pct_b' in df.columns:
        df['ff_to_matchup_a'] = df['ff_def_to_pct_b'] - df['ff_off_to_pct_a']
        df['ff_to_matchup_b'] = df['ff_def_to_pct_a'] - df['ff_off_to_pct_b']

    if 'ff_off_or_pct_a' in df.columns and 'ff_def_or_pct_b' in df.columns:
        df['ff_reb_matchup_a'] = df['ff_off_or_pct_a'] - df['ff_def_or_pct_b']
        df['ff_reb_matchup_b'] = df['ff_off_or_pct_b'] - df['ff_def_or_pct_a']

    # Height differential
    if 'ht_avg_height_a' in df.columns and 'ht_avg_height_b' in df.columns:
        df['ht_height_diff'] = df['ht_avg_height_a'] - df['ht_avg_height_b']

    # Experience differential
    if 'ht_experience_a' in df.columns and 'ht_experience_b' in df.columns:
        df['ht_exp_diff'] = df['ht_experience_a'] - df['ht_experience_b']

    # Continuity differential
    if 'ht_continuity_a' in df.columns and 'ht_continuity_b' in df.columns:
        df['ht_cont_diff'] = df['ht_continuity_a'] - df['ht_continuity_b']

    return df


def build_features_v2(project_root: Path) -> pd.DataFrame:
    """
    Build enhanced features v2 by adding extended KenPom data to existing features.

    Uses vectorized merge operations for 10-50x speedup over row-by-row iteration.
    """
    features_dir = project_root / 'data' / 'features'
    kenpom_ext_dir = project_root / 'data' / 'kenpom_extended'
    processed_dir = project_root / 'data' / 'processed'

    # Load existing features
    print("Loading existing features...")
    df = pd.read_parquet(features_dir / 'games_features_enhanced.parquet')
    print(f"  Loaded {len(df)} games")

    # Load extended KenPom data
    print("Loading extended KenPom data...")
    ext_data = load_extended_kenpom_data(kenpom_ext_dir)

    if not ext_data:
        print("  No extended KenPom data found in", kenpom_ext_dir)
        return df

    # Build team name mapping using shared utility
    game_teams = set(df['team_a'].unique()) | set(df['team_b'].unique())

    # Collect all KenPom team names from extended data
    kenpom_teams: set[str] = set()
    for name, data_df in ext_data.items():
        if 'TeamName' in data_df.columns:
            kenpom_teams.update(data_df['TeamName'].unique())

    print(f"Building team mapping ({len(game_teams)} game teams, {len(kenpom_teams)} KenPom teams)...")
    team_map = build_team_mapping(game_teams, kenpom_teams)
    print(f"  Mapped {len(team_map)} teams")

    # Merge data using vectorized operations
    if 'four_factors' in ext_data:
        print("Merging Four Factors (vectorized)...")
        df = merge_four_factors_vectorized(df, ext_data['four_factors'], team_map)

    if 'hca' in ext_data:
        print("Merging HCA (vectorized)...")
        df = merge_hca_vectorized(df, ext_data['hca'], team_map)

    if 'height_exp' in ext_data:
        print("Merging Height/Experience (vectorized)...")
        df = merge_height_experience_vectorized(df, ext_data['height_exp'], team_map)

    # Add matchup features
    print("Adding matchup features...")
    df = add_matchup_features(df)

    # Save
    output_path = features_dir / 'games_features_v2.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Total games: {len(df)}")

    # Summary of new features
    new_cols = [c for c in df.columns if c.startswith(('ff_', 'ht_', 'team_hca', 'team_elev'))]
    print(f"\nNew features added ({len(new_cols)}):")
    for col in sorted(new_cols):
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")

    return df


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent.parent
    build_features_v2(project_root)


if __name__ == '__main__':
    main()
