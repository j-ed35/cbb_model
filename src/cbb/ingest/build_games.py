"""
Build game-level dataset from team-row CSV files.

Transforms 2 rows per game (one per team) into 1 row per game.

Output:
- data/processed/games_base.parquet (combined)
- data/processed/games_base_<season>.parquet (per-season)
- reports/quarantine/unpaired_rows_<season>.csv
- reports/metrics/games_base_audit.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_and_clean_csv(file_path: Path) -> pd.DataFrame:
    """Load CSV and apply basic cleaning."""
    df = pd.read_csv(file_path)

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')

    # Normalize team names for matching
    df['team_norm'] = df['team_name'].str.strip().str.lower()
    df['opponent_norm'] = df['opponent'].str.strip().str.lower()

    # Calculate team's actual score from winner/loser scores
    df['team_score'] = np.where(df['result'] == 'W', df['winner_score'], df['loser_score'])
    df['opp_score'] = np.where(df['result'] == 'W', df['loser_score'], df['winner_score'])

    # Ensure scores are numeric
    df['team_score'] = pd.to_numeric(df['team_score'], errors='coerce')
    df['opp_score'] = pd.to_numeric(df['opp_score'], errors='coerce')

    return df


def build_opponent_to_team_id_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from opponent short names to team_ids.

    Uses the fact that each team appears in the dataset with their full team_name,
    and when they're an opponent, they appear with a short name.
    """
    # Get unique team_id -> team_name mappings
    team_lookup = df.groupby('team_id')['team_name'].first().to_dict()

    # Create reverse mapping: extract school name (without mascot) -> team_id
    # Team names are like "Texas Longhorns", "Texas State Bobcats"
    school_to_id = {}
    for team_id, team_name in team_lookup.items():
        # Try to extract school name by removing last word (mascot)
        parts = team_name.split()
        if len(parts) >= 2:
            # School name is everything except the last word (mascot)
            school_name = ' '.join(parts[:-1]).lower()
            if school_name not in school_to_id:
                school_to_id[school_name] = team_id
            # Also add full lowercase name
            school_to_id[team_name.lower()] = team_id

    # Manual overrides for known tricky names
    # These handle cases where opponent name doesn't match the pattern
    MANUAL_OVERRIDES = {
        # State schools that drop "state"
        'alabama': 'alabama crimson tide',
        'illinois': 'illinois fighting illini',
        'utah': "utah runnin' utes",
        'delaware': "delaware fightin' blue hens",
        'north dakota': 'north dakota fighting hawks',
        # Schools with alternate/abbreviated names
        'mississippi - ole miss': 'ole miss rebels',
        'ole miss': 'ole miss rebels',
        'north carolina': 'north carolina tar heels',
        'north carolina state': 'nc state wolfpack',
        'nc state': 'nc state wolfpack',
        "saint mary's-california": "st. mary's gaels",
        "saint mary's": "st. mary's gaels",
        "st. mary's-california": "st. mary's gaels",
        'southern methodist': 'smu mustangs',
        'smu': 'smu mustangs',
        'st. francis-brooklyn': 'st. francis-brooklyn terriers',
        'hartford': 'hartford hawks',
        # Other common variations
        'usc': 'southern california trojans',
        'lsu': 'louisiana state tigers',
        'uconn': 'connecticut huskies',
        'unlv': 'nevada-las vegas rebels',
        'vcu': 'virginia commonwealth rams',
        'ucf': 'central florida knights',
        'pitt': 'pittsburgh panthers',
        'penn': 'pennsylvania quakers',
        'cal': 'california golden bears',
    }

    # Apply manual overrides to school_to_id
    for override_key, override_val in MANUAL_OVERRIDES.items():
        override_val_lower = override_val.lower()
        if override_val_lower in school_to_id:
            school_to_id[override_key] = school_to_id[override_val_lower]

    # Get unique opponent names
    opponent_names = df['opponent'].str.strip().unique()

    # Build mapping from opponent_name -> team_id
    opp_to_id = {}

    for opp_name in opponent_names:
        opp_lower = opp_name.lower()

        # Try exact match first (includes manual overrides)
        if opp_lower in school_to_id:
            opp_to_id[opp_name] = school_to_id[opp_lower]
            continue

        # Try to find matching team_id via containment
        matches = []
        for team_id, team_name in team_lookup.items():
            team_lower = team_name.lower()
            # Extract school name (without mascot)
            parts = team_name.split()
            school_name = ' '.join(parts[:-1]).lower() if len(parts) >= 2 else team_lower

            # Check various matching strategies
            if opp_lower == school_name:
                # Exact match to school name (without mascot)
                matches = [(team_id, team_name, 100)]  # Perfect score
                break
            elif school_name.startswith(opp_lower + ' ') or school_name == opp_lower:
                # School name starts with opponent name
                matches.append((team_id, team_name, 90))
            elif opp_lower in school_name:
                # Opponent name is contained in school name
                matches.append((team_id, team_name, 50))

        if matches:
            # Sort by score and take best match
            matches.sort(key=lambda x: -x[2])
            if len(matches) == 1 or matches[0][2] > matches[1][2]:
                opp_to_id[opp_name] = matches[0][0]
            elif matches[0][2] == 100:
                # Perfect match
                opp_to_id[opp_name] = matches[0][0]

    return opp_to_id


def create_game_key(row: pd.Series, opp_team_id: int) -> str:
    """
    Create a unique game key from date + sorted team_id pair.

    This ensures both rows for the same game get the same key.
    """
    date_str = row['date_parsed'].strftime('%Y-%m-%d')
    team_ids = sorted([row['team_id'], opp_team_id])
    return f"{date_str}|{team_ids[0]}|{team_ids[1]}"


def pair_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pair team-rows into game-rows.

    Returns:
        games_df: Successfully paired games (1 row per game)
        quarantine_df: Rows that couldn't be paired properly
    """
    # Build opponent name to team_id mapping
    opp_to_id = build_opponent_to_team_id_map(df)

    # Add opponent_team_id column
    df['opponent_team_id'] = df['opponent'].map(opp_to_id)

    # Quarantine rows where we couldn't map opponent
    unmapped = df[df['opponent_team_id'].isna()].copy()
    unmapped['quarantine_reason'] = 'could not map opponent to team_id'
    quarantine_unmapped = unmapped.to_dict('records')

    # Keep only rows with mapped opponents
    df_mapped = df[df['opponent_team_id'].notna()].copy()
    df_mapped['opponent_team_id'] = df_mapped['opponent_team_id'].astype(int)

    # Create game keys using team_ids
    df_mapped['game_key'] = df_mapped.apply(
        lambda row: create_game_key(row, int(row['opponent_team_id'])), axis=1
    )

    # Group by game key
    grouped = df_mapped.groupby('game_key')

    games = []
    quarantine = list(quarantine_unmapped)

    for game_key, group in grouped:
        if len(group) != 2:
            # Quarantine: wrong number of rows
            for _, row in group.iterrows():
                quarantine.append({
                    **row.to_dict(),
                    'quarantine_reason': f'expected 2 rows, got {len(group)}'
                })
            continue

        rows = group.to_dict('records')
        row1, row2 = rows[0], rows[1]

        # Verify they're actually playing each other (using team_ids now)
        if row1['opponent_team_id'] != row2['team_id'] or row2['opponent_team_id'] != row1['team_id']:
            for row in rows:
                quarantine.append({
                    **row,
                    'quarantine_reason': 'opponent team_id mismatch'
                })
            continue

        # Determine home/away/neutral
        # Convention: Team A = home team (or first alphabetically if neutral)
        if row1['location'] == 'Home' and row2['location'] == 'Away':
            home_row, away_row = row1, row2
        elif row1['location'] == 'Away' and row2['location'] == 'Home':
            home_row, away_row = row2, row1
        elif row1['location'] == 'Neutral' and row2['location'] == 'Neutral':
            # Both neutral: use alphabetical order by team_norm
            if row1['team_norm'] < row2['team_norm']:
                home_row, away_row = row1, row2
            else:
                home_row, away_row = row2, row1
        else:
            # Unexpected location combination
            for row in rows:
                quarantine.append({
                    **row,
                    'quarantine_reason': f"unexpected location combo: {row1['location']}/{row2['location']}"
                })
            continue

        # Build game record
        # Team A = home (or first alphabetically if neutral)
        is_neutral = home_row['location'] == 'Neutral'

        # Compute season from actual date (not the incorrect season column in raw data)
        # Season is Nov of year N to Apr of year N+1, labeled as "N-N+1"
        game_date = home_row['date_parsed']
        if game_date.month >= 10:  # Oct/Nov/Dec = start of season
            season = f"{game_date.year}-{str(game_date.year + 1)[-2:]}"
        else:  # Jan-Apr = end of season
            season = f"{game_date.year - 1}-{str(game_date.year)[-2:]}"

        game = {
            'game_key': game_key,
            'season': season,
            'date': home_row['date_parsed'],
            'game_type': home_row['game'],

            # Teams
            'team_a': home_row['team_name'],
            'team_b': away_row['team_name'],
            'team_a_id': home_row['team_id'],
            'team_b_id': away_row['team_id'],

            # Home/Away
            'home_team': home_row['team_name'] if not is_neutral else None,
            'away_team': away_row['team_name'] if not is_neutral else None,
            'is_neutral': is_neutral,

            # Scores
            'points_a': home_row['team_score'],
            'points_b': away_row['team_score'],

            # Spread from Team A perspective
            'spread_a': home_row['spread'],
            'total_line': home_row['total'],

            # ATS results (from data)
            'ats_a_raw': home_row['ats'],
            'ats_b_raw': away_row['ats'],
            'ou_raw': home_row['ou'],
        }

        # Calculate derived fields
        game['final_margin_a'] = game['points_a'] - game['points_b']
        game['game_total'] = game['points_a'] + game['points_b']

        # ATS label
        if pd.isna(game['spread_a']):
            game['cover_a'] = None
        else:
            margin_vs_spread = game['final_margin_a'] + game['spread_a']
            if margin_vs_spread > 0:
                game['cover_a'] = 1
            elif margin_vs_spread < 0:
                game['cover_a'] = 0
            else:
                game['cover_a'] = 0.5  # Push

        # Win/Loss labels
        if game['final_margin_a'] > 0:
            game['win_a'] = 1
        elif game['final_margin_a'] < 0:
            game['win_a'] = 0
        else:
            game['win_a'] = 0.5  # Tie (rare)

        # Over/Under label
        if pd.isna(game['total_line']):
            game['over'] = None
        else:
            if game['game_total'] > game['total_line']:
                game['over'] = 1
            elif game['game_total'] < game['total_line']:
                game['over'] = 0
            else:
                game['over'] = 0.5  # Push

        games.append(game)

    games_df = pd.DataFrame(games)
    quarantine_df = pd.DataFrame(quarantine) if quarantine else pd.DataFrame()

    return games_df, quarantine_df


def validate_ats_labels(games_df: pd.DataFrame) -> dict:
    """
    Validate our calculated ATS labels against raw data.
    """
    # Filter to games with spread data and non-push ATS
    valid = games_df[
        games_df['spread_a'].notna() &
        games_df['ats_a_raw'].isin(['W', 'L'])
    ].copy()

    if len(valid) == 0:
        return {'sample_size': 0, 'match_rate': None}

    # Our cover_a == 1 should match ats_a_raw == 'W'
    valid['expected'] = (valid['cover_a'] == 1)
    valid['actual'] = (valid['ats_a_raw'] == 'W')

    match_rate = (valid['expected'] == valid['actual']).mean()

    return {
        'sample_size': len(valid),
        'match_rate': match_rate,
        'mismatches': len(valid) - (valid['expected'] == valid['actual']).sum()
    }


def generate_audit_report(games_by_season: dict, quarantine_counts: dict) -> pd.DataFrame:
    """Generate audit metrics."""
    records = []

    for season, df in games_by_season.items():
        # Filter to games with spread for ATS stats
        with_spread = df[df['spread_a'].notna()]

        # Cover distribution (excluding pushes)
        covers = with_spread[with_spread['cover_a'].isin([0, 1])]

        record = {
            'season': season,
            'total_games': len(df),
            'games_with_spread': len(with_spread),
            'games_without_spread': len(df) - len(with_spread),
            'cover_a_wins': (covers['cover_a'] == 1).sum() if len(covers) > 0 else 0,
            'cover_a_losses': (covers['cover_a'] == 0).sum() if len(covers) > 0 else 0,
            'pushes': (with_spread['cover_a'] == 0.5).sum(),
            'cover_rate': (covers['cover_a'] == 1).mean() if len(covers) > 0 else None,
            'quarantine_rows': quarantine_counts.get(season, 0),
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    quarantine_dir = project_root / 'reports' / 'quarantine'
    metrics_dir = project_root / 'reports' / 'metrics'

    processed_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")

    all_games = []
    games_by_season = {}
    quarantine_counts = {}

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")

        df = load_and_clean_csv(csv_file)
        print(f"  Loaded {len(df):,} team-rows")

        games_df, quarantine_df = pair_rows(df)
        season = games_df['season'].iloc[0] if len(games_df) > 0 else csv_file.stem

        print(f"  Created {len(games_df):,} game records")
        print(f"  Quarantined {len(quarantine_df):,} rows")

        # Validate ATS labels
        validation = validate_ats_labels(games_df)
        if validation['match_rate'] is not None:
            print(f"  ATS validation: {validation['match_rate']:.1%} match rate ({validation['sample_size']:,} games)")
            if validation['match_rate'] < 0.99:
                print(f"  WARNING: {validation['mismatches']} ATS mismatches!")

        # Store
        all_games.append(games_df)
        games_by_season[season] = games_df
        quarantine_counts[season] = len(quarantine_df)

        # Write per-season parquet
        season_file = processed_dir / f"games_base_{season.replace('-', '_')}.parquet"
        games_df.to_parquet(season_file, index=False)
        print(f"  Wrote {season_file.name}")

        # Write quarantine if any
        if len(quarantine_df) > 0:
            quarantine_file = quarantine_dir / f"unpaired_rows_{season.replace('-', '_')}.csv"
            quarantine_df.to_csv(quarantine_file, index=False)
            print(f"  Wrote {quarantine_file.name}")

    # Combine all games
    combined_df = pd.concat(all_games, ignore_index=True)
    combined_df = combined_df.sort_values(['date', 'team_a'])

    combined_file = processed_dir / 'games_base.parquet'
    combined_df.to_parquet(combined_file, index=False)
    print(f"\n✓ Wrote combined dataset: {combined_file}")
    print(f"  Total games: {len(combined_df):,}")

    # Generate audit report
    audit_df = generate_audit_report(games_by_season, quarantine_counts)
    audit_file = metrics_dir / 'games_base_audit.csv'
    audit_df.to_csv(audit_file, index=False)
    print(f"✓ Wrote audit report: {audit_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(audit_df.to_string(index=False))

    # Calculate overall stats
    total_rows_in = sum(len(load_and_clean_csv(f)) for f in csv_files)
    total_games_out = len(combined_df)
    total_quarantined = sum(quarantine_counts.values())

    pairing_rate = (total_games_out * 2) / total_rows_in
    print(f"\n✓ Pairing rate: {pairing_rate:.1%} of rows successfully paired")
    print(f"  ({total_games_out * 2:,} rows paired into {total_games_out:,} games)")
    print(f"  ({total_quarantined:,} rows quarantined)")

    # Check spread sign convention
    sample = combined_df[combined_df['spread_a'].notna()].sample(min(10, len(combined_df)))
    print("\nSpread convention verification (sample of 10):")
    print(sample[['team_a', 'team_b', 'points_a', 'points_b', 'final_margin_a', 'spread_a', 'cover_a']].to_string())

    return combined_df


if __name__ == '__main__':
    main()
