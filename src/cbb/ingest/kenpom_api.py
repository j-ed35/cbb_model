"""
KenPom data fetching via kenpompy.

Fetches live KenPom ratings directly into memory - no CSV persistence needed.
"""

import pandas as pd
from typing import Optional
import os
from pathlib import Path


def load_kenpom_credentials(project_root: Optional[Path] = None) -> tuple[str, str]:
    """
    Load KenPom credentials from environment or .env file.

    Returns: (email, password)
    """
    # Try environment first
    email = os.environ.get('KENPOM_EMAIL', '')
    password = os.environ.get('KENPOM_PW', '')

    if email and password:
        # Strip quotes if present (from .env file)
        return email.strip('"'), password.strip('"')

    # Fall back to .env file
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent

    env_path = project_root / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key == 'KENPOM_EMAIL':
                        email = value
                    elif key == 'KENPOM_PW':
                        password = value

    if not email or not password:
        raise ValueError("KENPOM_EMAIL and KENPOM_PW must be set in environment or .env file")

    return email, password


def fetch_kenpom_ratings(season: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch current KenPom Pomeroy ratings.

    Args:
        season: Optional season year (e.g., "2026"). Defaults to current season.

    Returns:
        DataFrame with columns matching existing pipeline expectations:
        - Team (or TeamName)
        - AdjEM, AdjOE (AdjO), AdjDE (AdjD), AdjTempo (AdjT)
        - RankAdjEM, RankAdjOE, RankAdjDE, etc.
    """
    from kenpompy.utils import login
    from kenpompy.misc import get_pomeroy_ratings

    email, password = load_kenpom_credentials()

    # Login and get authenticated browser
    browser = login(email, password)

    # Fetch ratings
    if season:
        df = get_pomeroy_ratings(browser, season=season)
    else:
        df = get_pomeroy_ratings(browser)

    # Normalize column names to match existing pipeline
    # kenpompy returns: Rk, Team, Conf, W-L, AdjEM, AdjO, AdjO.Rank, AdjD, AdjD.Rank, AdjT, AdjT.Rank, ...
    column_mapping = {
        'Team': 'TeamName',
        'Conf': 'Conference',
        'AdjO': 'AdjOE',
        'AdjD': 'AdjDE',
        'AdjT': 'AdjTempo',
        'Rk': 'RankAdjEM',
        'AdjO.Rank': 'RankAdjOE',
        'AdjD.Rank': 'RankAdjDE',
        'AdjT.Rank': 'RankAdjTempo',
    }

    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Ensure numeric columns
    numeric_cols = ['AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo',
                    'RankAdjEM', 'RankAdjOE', 'RankAdjDE', 'RankAdjTempo']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def fetch_fanmatch(date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch FanMatch data for a specific date.

    FanMatch contains KenPom's game predictions for a given day,
    which can be used as an alternative/complement to Odds API.

    Args:
        date: Date string in "YYYY-MM-DD" format. Defaults to today.

    Returns:
        DataFrame with game predictions, or None if no games.
    """
    from kenpompy.utils import login
    from kenpompy.FanMatch import FanMatch

    email, password = load_kenpom_credentials()
    browser = login(email, password)

    fm = FanMatch(browser, date=date)

    return fm.fm_df


def fetch_efficiency_stats(season: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch efficiency and tempo statistics.

    Args:
        season: Optional season year.

    Returns:
        DataFrame with efficiency stats.
    """
    from kenpompy.utils import login
    import kenpompy.summary as kp

    email, password = load_kenpom_credentials()
    browser = login(email, password)

    # get_efficiency doesn't take season param in current version
    df = kp.get_efficiency(browser)

    return df
