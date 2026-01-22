"""
Enhanced feature engineering for improved ATS prediction.

This module provides comprehensive feature extraction including:
1. Extended KenPom features (rankings, all metrics)
2. Situational features (rest days, conference play)
3. Rolling performance metrics with recency weighting
4. Market-derived features

Target: Get above 52.4% breakeven threshold for -110 odds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import warnings


def load_all_kenpom_snapshots(kenpom_dir: Path) -> pd.DataFrame:
    """
    Load all KenPom daily snapshots into a single DataFrame with date index.

    Returns DataFrame with columns:
    - snapshot_date: Date of the KenPom snapshot
    - TeamName: Team name in KenPom format
    - All KenPom metrics (AdjEM, AdjOE, AdjDE, etc.)
    """
    daily_files = sorted(kenpom_dir.glob('kenpom_????-??-??.csv'))

    if not daily_files:
        raise FileNotFoundError(f"No KenPom daily files found in {kenpom_dir}")

    dfs = []
    for f in daily_files:
        df = pd.read_csv(f)
        date_str = f.stem.replace('kenpom_', '')
        df['snapshot_date'] = pd.to_datetime(date_str)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure numeric columns are properly typed
    numeric_cols = ['AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo', 'Tempo', 'OE', 'DE',
                    'RankAdjEM', 'RankAdjOE', 'RankAdjDE', 'RankAdjTempo']
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    return combined


def build_improved_team_name_mapping(
    games_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    existing_map_path: Optional[Path] = None
) -> dict:
    """
    Build improved team name mapping with better fuzzy matching.

    Key improvements:
    - Handle state abbreviations (St. vs State)
    - Handle common variations (UTEP, TCU, etc.)
    - Manual overrides for known problem cases
    """
    # Known manual mappings for problem cases
    manual_mappings = {
        'Alabama Crimson Tide': 'Alabama',
        'Alabama State Hornets': 'Alabama St.',
        'Arkansas-Little Rock Trojans': 'Little Rock',
        'Arkansas-Pine Bluff Golden Lions': 'Arkansas Pine Bluff',
        'Bethune-Cookman Wildcats': 'Bethune Cookman',
        'Cal State-Fullerton Titans': 'Cal St. Fullerton',
        'Cal State-Northridge Matadors': 'CSUN',
        'Central Connecticut State Blue Devils': 'Central Connecticut',
        'Central Florida Knights': 'UCF',
        'East Texas A&M Lions': 'East Texas A&M',
        'FIU Panthers': 'FIU',
        'Florida International Golden Panthers': 'FIU',
        'Florida Atlantic Owls': 'Florida Atlantic',
        'Hawaii Rainbow Warriors': 'Hawaii',
        "Hawai'i Rainbow Warriors": 'Hawaii',
        'Illinois-Chicago Flames': 'Illinois Chicago',
        'UIC Flames': 'Illinois Chicago',
        'IUPUI Jaguars': 'IU Indy',
        'Indiana-Purdue-Indianapolis Jaguars': 'IU Indy',
        'LIU Sharks': 'LIU',
        'Long Island University Sharks': 'LIU',
        'Louisiana Ragin Cajuns': 'Louisiana',
        'Louisiana-Lafayette Ragin Cajuns': 'Louisiana',
        'Louisiana-Monroe Warhawks': 'Louisiana Monroe',
        'Loyola (IL) Ramblers': 'Loyola Chicago',
        'Loyola Chicago Ramblers': 'Loyola Chicago',
        'Loyola (MD) Greyhounds': 'Loyola MD',
        'Loyola Marymount Lions': 'Loyola Marymount',
        'Miami (FL) Hurricanes': 'Miami FL',
        'Miami (OH) RedHawks': 'Miami OH',
        'Mississippi Rebels': 'Mississippi',
        'Ole Miss Rebels': 'Mississippi',
        'Mississippi State Bulldogs': 'Mississippi St.',
        'N.C. State Wolfpack': 'N.C. State',
        'NC State Wolfpack': 'N.C. State',
        'Nebraska-Omaha Mavericks': 'Nebraska Omaha',
        'NJIT Highlanders': 'NJIT',
        'North Carolina A&T Aggies': 'North Carolina A&T',
        'North Carolina Central Eagles': 'North Carolina Central',
        'North Carolina Tar Heels': 'North Carolina',
        'Northern Kentucky Norse': 'Northern Kentucky',
        'Penn Quakers': 'Penn',
        'Pittsburgh Panthers': 'Pittsburgh',
        'Purdue Fort Wayne Mastodons': 'Purdue Fort Wayne',
        'Saint Francis Red Flash': 'Saint Francis',
        "Saint Joseph's Hawks": 'Saint Joseph\'s',
        "Saint Mary's Gaels": 'Saint Mary\'s',
        "Saint Peter's Peacocks": 'Saint Peter\'s',
        'Sam Houston Bearkats': 'Sam Houston St.',
        'Sam Houston State Bearkats': 'Sam Houston St.',
        'San Jose State Spartans': 'San Jose St.',
        'SIU-Edwardsville Cougars': 'SIUE',
        'SMU Mustangs': 'SMU',
        'South Carolina State Bulldogs': 'South Carolina St.',
        'South Florida Bulls': 'South Florida',
        'Southeast Missouri State Redhawks': 'Southeast Missouri',
        'Southeastern Louisiana Lions': 'Southeastern Louisiana',
        'Southern Illinois Salukis': 'Southern Illinois',
        'Southern Miss Golden Eagles': 'Southern Miss',
        'Southern Mississippi Golden Eagles': 'Southern Miss',
        'Stephen F. Austin Lumberjacks': 'Stephen F. Austin',
        "St. Bonaventure Bonnies": 'St. Bonaventure',
        "St. John's Red Storm": 'St. John\'s',
        'St. Thomas Tommies': 'St. Thomas',
        'TCU Horned Frogs': 'TCU',
        'Texas A&M Aggies': 'Texas A&M',
        'Texas A&M-Corpus Christi Islanders': 'Texas A&M Corpus Chris',
        'Texas-Arlington Mavericks': 'UT Arlington',
        'Texas-Rio Grande Valley Vaqueros': 'UT Rio Grande Valley',
        'Texas-San Antonio Roadrunners': 'UTSA',
        'Toledo Rockets': 'Toledo',
        'Troy Trojans': 'Troy',
        'Tulane Green Wave': 'Tulane',
        'Tulsa Golden Hurricane': 'Tulsa',
        'UAB Blazers': 'UAB',
        'UC-Davis Aggies': 'UC Davis',
        'UC-Irvine Anteaters': 'UC Irvine',
        'UC-Riverside Highlanders': 'UC Riverside',
        'UC-San Diego Tritons': 'UC San Diego',
        'UC-Santa Barbara Gauchos': 'UC Santa Barbara',
        'UCF Knights': 'UCF',
        'UCLA Bruins': 'UCLA',
        'UMass Minutemen': 'Massachusetts',
        'UMass-Lowell River Hawks': 'UMass Lowell',
        'UMBC Retrievers': 'UMBC',
        'UNC-Asheville Bulldogs': 'UNC Asheville',
        'UNC-Greensboro Spartans': 'UNC Greensboro',
        'UNC-Wilmington Seahawks': 'UNC Wilmington',
        'UNLV Rebels': 'UNLV',
        'USC Trojans': 'USC',
        'USC-Upstate Spartans': 'USC Upstate',
        'UT-Martin Skyhawks': 'Tennessee Martin',
        'UTEP Miners': 'UTEP',
        'VCU Rams': 'VCU',
        'VMI Keydets': 'VMI',
        'William & Mary Tribe': 'William & Mary',
    }

    # Get unique team names
    game_teams = set(games_df['team_a'].unique()) | set(games_df['team_b'].unique())
    team_col = 'TeamName' if 'TeamName' in kenpom_df.columns else 'Team'
    kenpom_teams = set(kenpom_df[team_col].unique())

    # Build mapping
    mapping = {}

    # First, apply manual mappings
    for raw_name, kp_name in manual_mappings.items():
        if raw_name in game_teams and kp_name in kenpom_teams:
            mapping[raw_name] = kp_name

    # Build lookup dictionaries for KenPom names
    kenpom_lower = {t.lower(): t for t in kenpom_teams}
    kenpom_no_st = {t.replace(' St.', ' State').lower(): t for t in kenpom_teams}
    kenpom_no_state = {t.replace(' State', ' St.').lower(): t for t in kenpom_teams}

    # Try to match remaining teams
    for team in game_teams:
        if team in mapping:
            continue

        team_lower = team.lower()

        # Remove mascot (last word typically)
        parts = team.split()
        if len(parts) >= 2:
            school = ' '.join(parts[:-1]).lower()
        else:
            school = team_lower

        # Try exact match on school name
        if school in kenpom_lower:
            mapping[team] = kenpom_lower[school]
            continue

        # Try St. <-> State variations
        if school in kenpom_no_st:
            mapping[team] = kenpom_no_st[school]
            continue
        if school in kenpom_no_state:
            mapping[team] = kenpom_no_state[school]
            continue

        # Try first word match (for unique schools)
        first_word = team_lower.split()[0]
        matches = [kp for kp in kenpom_teams if kp.lower().startswith(first_word)]
        if len(matches) == 1:
            mapping[team] = matches[0]

    return mapping


def get_kenpom_for_date(
    kenpom_df: pd.DataFrame,
    game_date: datetime,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Get the most recent KenPom snapshot before the game date.

    For end-of-season snapshots (April), we use them for the entire following season
    since they represent the final ratings from the previous season.

    Args:
        kenpom_df: Combined KenPom data with snapshot_date column
        game_date: Date of the game
        lookback_days: Maximum days to look back for a snapshot (default 365 for season-level)

    Returns:
        DataFrame with KenPom data for the most recent snapshot
    """
    # First try to find a snapshot within a reasonable window (for daily data)
    cutoff_short = game_date - timedelta(days=14)
    valid = kenpom_df[
        (kenpom_df['snapshot_date'] < game_date) &
        (kenpom_df['snapshot_date'] >= cutoff_short)
    ]

    if len(valid) == 0:
        # Fall back to end-of-season snapshots (up to 1 year lookback)
        cutoff_long = game_date - timedelta(days=lookback_days)
        valid = kenpom_df[
            (kenpom_df['snapshot_date'] < game_date) &
            (kenpom_df['snapshot_date'] >= cutoff_long)
        ]

    if len(valid) == 0:
        return pd.DataFrame()

    latest = valid['snapshot_date'].max()
    return valid[valid['snapshot_date'] == latest]


def extract_extended_kenpom_features(
    team_a: str,
    team_b: str,
    kenpom_snapshot: pd.DataFrame,
    team_map: dict
) -> dict:
    """
    Extract extended KenPom features for a game.

    Features extracted:
    - AdjEM (Adjusted Efficiency Margin)
    - AdjOE (Adjusted Offensive Efficiency)
    - AdjDE (Adjusted Defensive Efficiency)
    - AdjTempo (Adjusted Tempo)
    - Rankings for each metric
    - Derived matchup features
    """
    features = {'kp_matched': False}

    team_a_kp = team_map.get(team_a)
    team_b_kp = team_map.get(team_b)

    if not team_a_kp or not team_b_kp:
        return features

    team_col = 'TeamName' if 'TeamName' in kenpom_snapshot.columns else 'Team'

    row_a = kenpom_snapshot[kenpom_snapshot[team_col] == team_a_kp]
    row_b = kenpom_snapshot[kenpom_snapshot[team_col] == team_b_kp]

    if len(row_a) == 0 or len(row_b) == 0:
        return features

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    features['kp_matched'] = True

    # Core efficiency metrics
    features['kp_adj_em_a'] = row_a.get('AdjEM', np.nan)
    features['kp_adj_em_b'] = row_b.get('AdjEM', np.nan)
    features['kp_adj_o_a'] = row_a.get('AdjOE', np.nan)
    features['kp_adj_o_b'] = row_b.get('AdjOE', np.nan)
    features['kp_adj_d_a'] = row_a.get('AdjDE', np.nan)
    features['kp_adj_d_b'] = row_b.get('AdjDE', np.nan)
    features['kp_tempo_a'] = row_a.get('AdjTempo', np.nan)
    features['kp_tempo_b'] = row_b.get('AdjTempo', np.nan)

    # Rankings (normalized to 0-1, lower is better)
    n_teams = 365  # Approximate number of D1 teams
    features['kp_rank_em_a'] = row_a.get('RankAdjEM', np.nan) / n_teams
    features['kp_rank_em_b'] = row_b.get('RankAdjEM', np.nan) / n_teams
    features['kp_rank_o_a'] = row_a.get('RankAdjOE', np.nan) / n_teams
    features['kp_rank_o_b'] = row_b.get('RankAdjOE', np.nan) / n_teams
    features['kp_rank_d_a'] = row_a.get('RankAdjDE', np.nan) / n_teams
    features['kp_rank_d_b'] = row_b.get('RankAdjDE', np.nan) / n_teams

    # Derived features
    if not np.isnan(features['kp_adj_em_a']) and not np.isnan(features['kp_adj_em_b']):
        features['kp_adj_em_diff'] = features['kp_adj_em_a'] - features['kp_adj_em_b']

    if not np.isnan(features['kp_tempo_a']) and not np.isnan(features['kp_tempo_b']):
        features['kp_tempo_avg'] = (features['kp_tempo_a'] + features['kp_tempo_b']) / 2
        features['kp_tempo_diff'] = features['kp_tempo_a'] - features['kp_tempo_b']

    # Matchup-specific: Team A offense vs Team B defense
    if not np.isnan(features['kp_adj_o_a']) and not np.isnan(features['kp_adj_d_b']):
        features['kp_o_vs_d_a'] = features['kp_adj_o_a'] - features['kp_adj_d_b']

    # Matchup-specific: Team B offense vs Team A defense
    if not np.isnan(features['kp_adj_o_b']) and not np.isnan(features['kp_adj_d_a']):
        features['kp_o_vs_d_b'] = features['kp_adj_o_b'] - features['kp_adj_d_a']

    # Predicted game total (rough estimate)
    if all(not np.isnan(features.get(k, np.nan)) for k in ['kp_adj_o_a', 'kp_adj_o_b', 'kp_adj_d_a', 'kp_adj_d_b', 'kp_tempo_avg']):
        # Possessions ≈ tempo * 0.4 (games are ~40 min, tempo is per 40 min)
        avg_ppp_a = (features['kp_adj_o_a'] + features['kp_adj_d_b']) / 2 / 100
        avg_ppp_b = (features['kp_adj_o_b'] + features['kp_adj_d_a']) / 2 / 100
        features['kp_pred_total'] = features['kp_tempo_avg'] * (avg_ppp_a + avg_ppp_b)

    # KenPom predicted margin (simplified)
    if 'kp_adj_em_diff' in features:
        # Rough HCA adjustment will be added externally
        features['kp_pred_margin_raw'] = features['kp_adj_em_diff'] * features.get('kp_tempo_avg', 67.5) / 100

    return features


def compute_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for each team in each game.

    Adds columns:
    - rest_days_a: Days since team A's last game
    - rest_days_b: Days since team B's last game
    - rest_diff: rest_days_a - rest_days_b
    """
    games_df = games_df.sort_values('date').copy()

    # Build last game date for each team
    team_last_game = {}
    rest_a = []
    rest_b = []

    for idx, row in games_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']
        game_date = row['date']

        # Get rest days for each team
        if team_a in team_last_game:
            days_a = (game_date - team_last_game[team_a]).days
        else:
            days_a = 7  # Default for first game

        if team_b in team_last_game:
            days_b = (game_date - team_last_game[team_b]).days
        else:
            days_b = 7

        rest_a.append(min(days_a, 14))  # Cap at 14 days
        rest_b.append(min(days_b, 14))

        # Update last game dates
        team_last_game[team_a] = game_date
        team_last_game[team_b] = game_date

    games_df['rest_days_a'] = rest_a
    games_df['rest_days_b'] = rest_b
    games_df['rest_diff'] = games_df['rest_days_a'] - games_df['rest_days_b']

    # Back-to-back indicator (played yesterday)
    games_df['b2b_a'] = (games_df['rest_days_a'] == 1).astype(int)
    games_df['b2b_b'] = (games_df['rest_days_b'] == 1).astype(int)

    return games_df


def compute_rolling_ats_record(games_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling ATS (against the spread) record for each team.

    This captures "hot" and "cold" teams from a betting perspective.
    """
    games_df = games_df.sort_values('date').copy()

    # Build team ATS history
    team_ats_history = {}
    rolling_ats_a = []
    rolling_ats_b = []

    for idx, row in games_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']
        cover_a = row.get('cover_a', 0.5)

        # Get rolling ATS for each team (before this game)
        if team_a in team_ats_history and len(team_ats_history[team_a]) >= 3:
            recent = team_ats_history[team_a][-window:]
            rolling_ats_a.append(np.mean(recent))
        else:
            rolling_ats_a.append(0.5)  # Default

        if team_b in team_ats_history and len(team_ats_history[team_b]) >= 3:
            recent = team_ats_history[team_b][-window:]
            rolling_ats_b.append(np.mean(recent))
        else:
            rolling_ats_b.append(0.5)

        # Update history after using it (avoid lookahead)
        if cover_a in [0, 1]:
            if team_a not in team_ats_history:
                team_ats_history[team_a] = []
            if team_b not in team_ats_history:
                team_ats_history[team_b] = []

            team_ats_history[team_a].append(cover_a)
            team_ats_history[team_b].append(1 - cover_a)

    games_df['rolling_ats_a'] = rolling_ats_a
    games_df['rolling_ats_b'] = rolling_ats_b
    games_df['rolling_ats_diff'] = games_df['rolling_ats_a'] - games_df['rolling_ats_b']

    return games_df


def compute_recency_weighted_stats(games_df: pd.DataFrame, half_life: int = 5) -> pd.DataFrame:
    """
    Compute exponentially weighted rolling stats (more recent games weighted higher).

    Args:
        games_df: Games DataFrame
        half_life: Number of games for weight to decay by half
    """
    games_df = games_df.sort_values('date').copy()

    # Build weighted stats for each team
    team_stats = {}  # team -> list of (margin, ppg, papg, weight_remaining)

    decay = 0.5 ** (1 / half_life)

    ew_margin_a = []
    ew_margin_b = []
    ew_ppg_a = []
    ew_ppg_b = []

    for idx, row in games_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']

        # Get weighted averages before this game
        for team, ew_margin_list, ew_ppg_list in [
            (team_a, ew_margin_a, ew_ppg_a),
            (team_b, ew_margin_b, ew_ppg_b)
        ]:
            if team in team_stats and len(team_stats[team]) >= 3:
                stats = team_stats[team]
                weights = [decay ** i for i in range(len(stats))]
                weights = weights[::-1]  # Most recent first
                total_w = sum(weights)

                ew_margin = sum(s[0] * w for s, w in zip(reversed(stats), weights)) / total_w
                ew_ppg = sum(s[1] * w for s, w in zip(reversed(stats), weights)) / total_w

                ew_margin_list.append(ew_margin)
                ew_ppg_list.append(ew_ppg)
            else:
                ew_margin_list.append(np.nan)
                ew_ppg_list.append(np.nan)

        # Update team stats after using them
        margin_a = row['final_margin_a']
        points_a = row['points_a']
        points_b = row['points_b']

        if team_a not in team_stats:
            team_stats[team_a] = []
        if team_b not in team_stats:
            team_stats[team_b] = []

        team_stats[team_a].append((margin_a, points_a, points_b))
        team_stats[team_b].append((-margin_a, points_b, points_a))

    games_df['ew_margin_a'] = ew_margin_a
    games_df['ew_margin_b'] = ew_margin_b
    games_df['ew_ppg_a'] = ew_ppg_a
    games_df['ew_ppg_b'] = ew_ppg_b
    games_df['ew_margin_diff'] = games_df['ew_margin_a'] - games_df['ew_margin_b']

    return games_df


def build_enhanced_features(
    games_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    team_map: dict
) -> pd.DataFrame:
    """
    Build comprehensive feature set for ATS prediction.

    Features included:
    1. Extended KenPom metrics (efficiency, rankings, matchup-specific)
    2. Rest days and back-to-back indicators
    3. Rolling ATS record
    4. Recency-weighted performance stats
    5. Context features (home/neutral)
    """
    print("Computing rest days...")
    games_df = compute_rest_days(games_df)

    print("Computing rolling ATS records...")
    games_df = compute_rolling_ats_record(games_df)

    print("Computing recency-weighted stats...")
    games_df = compute_recency_weighted_stats(games_df)

    print("Extracting KenPom features...")
    features_list = []

    for idx, row in games_df.iterrows():
        game_features = {
            'game_key': row['game_key'],
            'season': row['season'],
            'date': row['date'],
            'team_a': row['team_a'],
            'team_b': row['team_b'],
            # Context
            'is_home_a': 1 if not row.get('is_neutral', True) else 0,
            'is_neutral': 1 if row.get('is_neutral', False) else 0,
            # Target and market
            'spread_a': row.get('spread_a', np.nan),
            'final_margin_a': row['final_margin_a'],
            'cover_a': row.get('cover_a', np.nan),
            # Rest features
            'rest_days_a': row.get('rest_days_a', 7),
            'rest_days_b': row.get('rest_days_b', 7),
            'rest_diff': row.get('rest_diff', 0),
            'b2b_a': row.get('b2b_a', 0),
            'b2b_b': row.get('b2b_b', 0),
            # Rolling ATS
            'rolling_ats_a': row.get('rolling_ats_a', 0.5),
            'rolling_ats_b': row.get('rolling_ats_b', 0.5),
            'rolling_ats_diff': row.get('rolling_ats_diff', 0),
            # Recency-weighted stats
            'ew_margin_a': row.get('ew_margin_a', np.nan),
            'ew_margin_b': row.get('ew_margin_b', np.nan),
            'ew_margin_diff': row.get('ew_margin_diff', np.nan),
        }

        # Get KenPom features
        kp_snapshot = get_kenpom_for_date(kenpom_df, row['date'])
        if len(kp_snapshot) > 0:
            kp_features = extract_extended_kenpom_features(
                row['team_a'], row['team_b'], kp_snapshot, team_map
            )
            game_features.update(kp_features)
        else:
            game_features['kp_matched'] = False

        features_list.append(game_features)

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} games...")

    features_df = pd.DataFrame(features_list)

    # Add home court advantage adjusted prediction
    HCA = 3.5  # Home court advantage in points
    if 'kp_pred_margin_raw' in features_df.columns:
        features_df['kp_pred_margin'] = features_df['kp_pred_margin_raw'] + (
            features_df['is_home_a'] * HCA - (1 - features_df['is_home_a'] - features_df['is_neutral']) * HCA
        )

    return features_df


def main():
    """Build enhanced feature dataset."""
    project_root = Path(__file__).parent.parent.parent.parent
    processed_dir = project_root / 'data' / 'processed'
    kenpom_dir = project_root / 'data' / 'kenpom'
    features_dir = project_root / 'data' / 'features'

    features_dir.mkdir(parents=True, exist_ok=True)

    # Load games
    games_df = pd.read_parquet(processed_dir / 'games_base.parquet')
    print(f"Loaded {len(games_df):,} games")

    # Load KenPom data
    kenpom_df = load_all_kenpom_snapshots(kenpom_dir)
    print(f"Loaded {len(kenpom_df):,} KenPom records from {kenpom_df['snapshot_date'].nunique()} snapshots")

    # Build team name mapping
    team_map = build_improved_team_name_mapping(games_df, kenpom_df)
    print(f"Mapped {len(team_map)} teams")

    # Save improved mapping
    mapping_df = pd.DataFrame([
        {'raw_name': k, 'kenpom_name': v, 'notes': 'auto-matched'}
        for k, v in team_map.items()
    ])
    mapping_df.to_csv(processed_dir / 'team_name_map_v2.csv', index=False)

    # Build features
    features_df = build_enhanced_features(games_df, kenpom_df, team_map)

    # Report stats
    kp_matched = features_df['kp_matched'].sum()
    with_spread = features_df['spread_a'].notna().sum()
    print(f"\nFeature dataset stats:")
    print(f"  Total games: {len(features_df):,}")
    print(f"  KenPom matched: {kp_matched:,} ({kp_matched/len(features_df):.1%})")
    print(f"  With spread: {with_spread:,} ({with_spread/len(features_df):.1%})")

    # Save
    features_df.to_parquet(features_dir / 'games_features_enhanced.parquet', index=False)
    print(f"\n✓ Saved to {features_dir / 'games_features_enhanced.parquet'}")

    return features_df


if __name__ == '__main__':
    main()
