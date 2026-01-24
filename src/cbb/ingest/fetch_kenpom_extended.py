"""
Fetch extended KenPom data for model training.

This module fetches historical Four Factors, HCA, Height/Experience, and Team Stats
data from kenpompy for all training seasons.

These features are season-level (not daily snapshots), which is appropriate for:
- HCA: Stable throughout season (arena doesn't change)
- Height/Experience: Fixed at season start
- Four Factors: Use end-of-season values (with awareness of data leakage)

Output: data/kenpom/kenpom_extended_{season}.parquet
"""

import pandas as pd
from pathlib import Path
import os
from typing import Optional


def get_browser():
    """Get authenticated kenpompy browser."""
    from kenpompy.utils import login

    email = os.environ.get('KENPOM_EMAIL', '')
    pw = os.environ.get('KENPOM_PW', '')

    if not email:
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if 'KENPOM_EMAIL' in line:
                        email = line.split('=')[1].strip().strip('"')
                    if 'KENPOM_PW' in line:
                        pw = line.split('=')[1].strip().strip('"')

    if not email or not pw:
        raise ValueError("KENPOM_EMAIL and KENPOM_PW required")

    return login(email, pw)


def fetch_four_factors(browser, season: str) -> pd.DataFrame:
    """Fetch Four Factors data for a season."""
    import kenpompy.summary as kp

    df = kp.get_fourfactors(browser, season=season)

    # Rename columns for clarity
    rename_map = {
        'Team': 'TeamName',
        'Off-eFG%': 'off_efg',
        'Off-TO%': 'off_to_pct',
        'Off-OR%': 'off_or_pct',
        'Off-FTRate': 'off_ft_rate',
        'Def-eFG%': 'def_efg',
        'Def-TO%': 'def_to_pct',
        'Def-OR%': 'def_or_pct',  # This is opponent's OR%, i.e., our DRB%
        'Def-FTRate': 'def_ft_rate',
    }

    # Only rename columns that exist
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Keep only relevant columns
    keep_cols = ['TeamName', 'off_efg', 'off_to_pct', 'off_or_pct', 'off_ft_rate',
                 'def_efg', 'def_to_pct', 'def_or_pct', 'def_ft_rate']
    df = df[[c for c in keep_cols if c in df.columns]]

    # Convert to numeric
    for col in df.columns:
        if col != 'TeamName':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['season'] = int(season)
    return df


def fetch_hca(browser) -> pd.DataFrame:
    """
    Fetch Home Court Advantage data.

    Note: HCA data is current season only, but HCA values are relatively stable
    year-over-year for most teams (based on arena, elevation, crowd).
    """
    import kenpompy.misc as kp

    df = kp.get_hca(browser)

    rename_map = {
        'Team': 'TeamName',
        'HCA': 'hca',
        'Elev': 'elevation',
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    keep_cols = ['TeamName', 'hca', 'elevation']
    df = df[[c for c in keep_cols if c in df.columns]]

    for col in ['hca', 'elevation']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def fetch_height_experience(browser, season: str) -> pd.DataFrame:
    """Fetch Height and Experience data for a season."""
    import kenpompy.summary as kp

    df = kp.get_height(browser, season=season)

    rename_map = {
        'Team': 'TeamName',
        'AvgHgt': 'avg_height',
        'EffHgt': 'eff_height',
        'Experience': 'experience',
        'Continuity': 'continuity',
        'Bench': 'bench_pct',
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    keep_cols = ['TeamName', 'avg_height', 'eff_height', 'experience', 'continuity', 'bench_pct']
    df = df[[c for c in keep_cols if c in df.columns]]

    for col in df.columns:
        if col != 'TeamName':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['season'] = int(season)
    return df


def fetch_team_stats(browser, season: str) -> pd.DataFrame:
    """Fetch detailed team stats (shooting percentages, etc.)."""
    import kenpompy.summary as kp

    # Offense stats
    df_off = kp.get_teamstats(browser, season=season, defense=False)
    df_off = df_off.rename(columns={'Team': 'TeamName'})

    off_rename = {
        '3P%': 'off_3p_pct',
        '2P%': 'off_2p_pct',
        'FT%': 'off_ft_pct',
        'Blk%': 'off_blk_pct',
        'Stl%': 'off_stl_pct',
        'A%': 'off_ast_pct',
        '3PA%': 'off_3pa_rate',
    }
    for old, new in off_rename.items():
        if old in df_off.columns:
            df_off = df_off.rename(columns={old: new})

    off_cols = ['TeamName'] + [c for c in off_rename.values() if c in df_off.columns]
    df_off = df_off[off_cols]

    # Defense stats
    df_def = kp.get_teamstats(browser, season=season, defense=True)
    df_def = df_def.rename(columns={'Team': 'TeamName'})

    def_rename = {
        '3P%': 'def_3p_pct',
        '2P%': 'def_2p_pct',
        'Blk%': 'def_blk_pct',
        'Stl%': 'def_stl_pct',
        '3PA%': 'def_3pa_rate',
    }
    for old, new in def_rename.items():
        if old in df_def.columns:
            df_def = df_def.rename(columns={old: new})

    def_cols = ['TeamName'] + [c for c in def_rename.values() if c in df_def.columns]
    df_def = df_def[def_cols]

    # Merge offense and defense
    df = df_off.merge(df_def, on='TeamName', how='outer')

    for col in df.columns:
        if col != 'TeamName':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['season'] = int(season)
    return df


def fetch_all_extended_data(seasons: list[str], output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Fetch all extended KenPom data for multiple seasons.

    Returns dict with keys: 'four_factors', 'hca', 'height_exp', 'team_stats'
    """
    print("Authenticating with KenPom...")
    browser = get_browser()

    output_dir.mkdir(parents=True, exist_ok=True)

    all_ff = []
    all_ht = []
    all_ts = []

    for season in seasons:
        print(f"\nFetching season {season}...")

        # Four Factors
        print(f"  Four Factors...", end=" ")
        ff = fetch_four_factors(browser, season)
        all_ff.append(ff)
        print(f"{len(ff)} teams")

        # Height/Experience
        print(f"  Height/Experience...", end=" ")
        ht = fetch_height_experience(browser, season)
        all_ht.append(ht)
        print(f"{len(ht)} teams")

        # Team Stats
        print(f"  Team Stats...", end=" ")
        ts = fetch_team_stats(browser, season)
        all_ts.append(ts)
        print(f"{len(ts)} teams")

    # HCA (current season only, but we'll use for all)
    print(f"\nFetching HCA data...")
    hca = fetch_hca(browser)
    print(f"  {len(hca)} teams")

    # Combine all seasons
    df_ff = pd.concat(all_ff, ignore_index=True)
    df_ht = pd.concat(all_ht, ignore_index=True)
    df_ts = pd.concat(all_ts, ignore_index=True)

    # Save to parquet
    df_ff.to_parquet(output_dir / 'four_factors.parquet', index=False)
    df_ht.to_parquet(output_dir / 'height_experience.parquet', index=False)
    df_ts.to_parquet(output_dir / 'team_stats.parquet', index=False)
    hca.to_parquet(output_dir / 'hca.parquet', index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  four_factors.parquet: {len(df_ff)} rows")
    print(f"  height_experience.parquet: {len(df_ht)} rows")
    print(f"  team_stats.parquet: {len(df_ts)} rows")
    print(f"  hca.parquet: {len(hca)} rows")

    return {
        'four_factors': df_ff,
        'height_experience': df_ht,
        'team_stats': df_ts,
        'hca': hca,
    }


def main():
    """Fetch extended data for all training seasons."""
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'data' / 'kenpom_extended'

    seasons = ['2022', '2023', '2024', '2025', '2026']

    fetch_all_extended_data(seasons, output_dir)


if __name__ == '__main__':
    main()
