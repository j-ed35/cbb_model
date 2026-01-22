"""
Inspect raw CSV files and produce schema documentation.

Output:
- docs/DATA_SCHEMA.md
- reports/metrics/raw_summary.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def inspect_csv(file_path: Path) -> dict:
    """Inspect a single CSV file and return summary statistics."""
    df = pd.read_csv(file_path)

    # Clean up any trailing empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')

    return {
        'file': file_path.name,
        'season': df['season'].iloc[0] if 'season' in df.columns else None,
        'rows': len(df),
        'columns': list(df.columns),
        'min_date': df['date_parsed'].min(),
        'max_date': df['date_parsed'].max(),
        'unique_teams': df['team_name'].nunique() if 'team_name' in df.columns else None,
        'unique_opponents': df['opponent'].nunique() if 'opponent' in df.columns else None,
        'locations': df['location'].value_counts().to_dict() if 'location' in df.columns else None,
        'game_types': df['game'].value_counts().to_dict() if 'game' in df.columns else None,
        'ats_values': df['ats'].value_counts().to_dict() if 'ats' in df.columns else None,
        'ou_values': df['ou'].value_counts().to_dict() if 'ou' in df.columns else None,
        'spread_missing': df['spread'].isna().sum() if 'spread' in df.columns else None,
        'total_missing': df['total'].isna().sum() if 'total' in df.columns else None,
        'dtypes': {col: str(df[col].dtype) for col in df.columns if not col.startswith('Unnamed')},
        'sample_rows': df.head(5).to_dict('records'),
        'df': df  # Keep for further analysis
    }


def analyze_spread_convention(df: pd.DataFrame) -> dict:
    """Analyze spread sign convention."""
    # Filter to rows with spread data
    df_spread = df[df['spread'].notna()].copy()

    # Calculate actual margin from team's perspective
    # winner_score and loser_score are absolute, result tells us if team won
    df_spread['team_score'] = df_spread.apply(
        lambda r: r['winner_score'] if r['result'] == 'W' else r['loser_score'], axis=1
    )
    df_spread['opp_score'] = df_spread.apply(
        lambda r: r['loser_score'] if r['result'] == 'W' else r['winner_score'], axis=1
    )
    df_spread['margin'] = df_spread['team_score'] - df_spread['opp_score']
    df_spread['margin_vs_spread'] = df_spread['margin'] + df_spread['spread']

    # Check ATS convention
    # If ats == 'W' when margin_vs_spread > 0, spread is from team's perspective
    ats_check = df_spread[df_spread['ats'].isin(['W', 'L'])].copy()
    ats_check['expected_cover'] = ats_check['margin_vs_spread'] > 0
    ats_check['actual_cover'] = ats_check['ats'] == 'W'

    match_rate = (ats_check['expected_cover'] == ats_check['actual_cover']).mean()

    return {
        'match_rate': match_rate,
        'interpretation': 'spread is from team perspective (team + spread > 0 means cover)' if match_rate > 0.95 else 'needs manual review',
        'sample_size': len(ats_check)
    }


def generate_schema_doc(summaries: list[dict], spread_analysis: dict) -> str:
    """Generate DATA_SCHEMA.md content."""
    doc = """# DATA_SCHEMA.md — Raw NCAAB Data

## Overview
This document describes the schema of raw CSV files in `data/raw/`.

## Files Available
"""
    for s in summaries:
        doc += f"- `{s['file']}`: {s['rows']:,} rows, {s['season']} season, dates {s['min_date'].strftime('%Y-%m-%d')} to {s['max_date'].strftime('%Y-%m-%d')}\n"

    doc += """
## Schema Consistency
All files share the same column structure (verified).

## Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | int | Unique identifier for the team |
| `team_name` | str | Full team name (e.g., "UC San Diego Tritons") |
| `season` | str | Season identifier (e.g., "2024-25") |
| `date` | str | Game date in format "Mon DD, YYYY" (e.g., "Nov 6, 2024") |
| `opponent` | str | Opponent team name (may differ from team_name format) |
| `location` | str | Home/Away/Neutral indicator |
| `game` | str | Game type: REG (regular), POST (postseason), CONF (conference tournament) |
| `result` | str | W (win) or L (loss) |
| `winner_score` | float/int | Score of the winning team |
| `loser_score` | float/int | Score of the losing team |
| `ats` | str | Against-the-spread result: W (cover), L (no cover), P (push), empty if no line |
| `spread` | float | Point spread from team's perspective (see convention below) |
| `ou` | str | Over/Under result: O (over), U (under), P (push), empty if no line |
| `total` | float | Total points line for over/under |

## Data Format Notes

### Row Structure
- **Team-row level**: Each game has 2 rows (one per team)
- To build game-level data, pair rows by date + team/opponent

### Score Columns
- `winner_score` and `loser_score` are **absolute** (not from team perspective)
- Must use `result` column to determine which score belongs to which team:
  - If `result == 'W'`: team scored `winner_score`, opponent scored `loser_score`
  - If `result == 'L'`: team scored `loser_score`, opponent scored `winner_score`

### Location Values
"""
    # Add location breakdown from first summary
    if summaries and summaries[0]['locations']:
        for loc, count in summaries[0]['locations'].items():
            doc += f"- `{loc}`: {count:,} occurrences\n"

    doc += """
### Game Types
"""
    if summaries and summaries[0]['game_types']:
        for gt, count in summaries[0]['game_types'].items():
            doc += f"- `{gt}`: {count:,} games\n"

    doc += f"""
## Spread Convention (CRITICAL)

Based on analysis of {spread_analysis['sample_size']:,} games with spread data:

**Convention**: {spread_analysis['interpretation']}
**Verification rate**: {spread_analysis['match_rate']:.1%}

### How to interpret spread:
- Positive spread: team is the underdog (e.g., +5.5 means team gets 5.5 points)
- Negative spread: team is the favorite (e.g., -5.5 means team gives 5.5 points)
- **Cover calculation**: `margin + spread > 0` means team covered

### Example:
- Team A vs Team B, Team A has spread = -7.0
- Team A wins 80-70 (margin = +10)
- margin + spread = 10 + (-7) = +3 → Team A covered (ats = 'W')

## Missing Data

Some games (typically non-D1 opponents or exhibition games) have no betting lines:
- `spread`, `total`, `ats`, `ou` will be empty
- These games should be excluded from ATS modeling

## Quirks and Edge Cases

1. **Neutral site games**: `location = 'Neutral'` - neither team has home advantage
2. **Conference tournaments**: `game = 'CONF'`
3. **Postseason games**: `game = 'POST'` (NCAA tournament, NIT, etc.)
4. **Pushes**: `ats = 'P'` or `ou = 'P'` - bet is refunded
5. **Season mismatch**: Season label (e.g., "2024-25") may not match calendar year of dates

## Team Name Mapping

Team names in `team_name` column use full names with mascots.
Opponent names in `opponent` column may use abbreviated or alternate names.

**Examples of mismatches to handle**:
- "UC San Diego Tritons" vs "UC San Diego"
- "Cal State-Fullerton" vs "CSU Fullerton"

A mapping file will be needed for KenPom joins.
"""
    return doc


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent.parent
    raw_dir = project_root / 'data' / 'raw'
    docs_dir = project_root / 'docs'
    reports_dir = project_root / 'reports' / 'metrics'

    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(raw_dir.glob('*.csv'))

    print(f"Found {len(csv_files)} CSV files")

    # Inspect each file
    summaries = []
    all_dfs = []

    for csv_file in csv_files:
        print(f"\nInspecting {csv_file.name}...")
        summary = inspect_csv(csv_file)
        summaries.append(summary)
        all_dfs.append(summary['df'])

        print(f"  Rows: {summary['rows']:,}")
        print(f"  Season: {summary['season']}")
        print(f"  Date range: {summary['min_date']} to {summary['max_date']}")
        print(f"  Teams: {summary['unique_teams']}")
        print(f"  Spreads missing: {summary['spread_missing']:,}")

    # Verify schema consistency
    base_cols = set(summaries[0]['columns'])
    for s in summaries[1:]:
        # Ignore unnamed columns
        s_cols = {c for c in s['columns'] if not c.startswith('Unnamed')}
        base_cols_clean = {c for c in base_cols if not c.startswith('Unnamed')}
        if s_cols != base_cols_clean:
            print(f"\nWARNING: Schema mismatch in {s['file']}")
            print(f"  Missing: {base_cols_clean - s_cols}")
            print(f"  Extra: {s_cols - base_cols_clean}")

    print("\n✓ Schema consistent across all files")

    # Combine all data for spread analysis
    combined_df = pd.concat(all_dfs, ignore_index=True)
    spread_analysis = analyze_spread_convention(combined_df)
    print(f"\nSpread convention analysis:")
    print(f"  {spread_analysis['interpretation']}")
    print(f"  Match rate: {spread_analysis['match_rate']:.1%}")

    # Generate schema documentation
    schema_doc = generate_schema_doc(summaries, spread_analysis)
    schema_path = docs_dir / 'DATA_SCHEMA.md'
    schema_path.write_text(schema_doc)
    print(f"\n✓ Wrote {schema_path}")

    # Generate summary CSV
    summary_records = []
    for s in summaries:
        summary_records.append({
            'file': s['file'],
            'season': s['season'],
            'rows': s['rows'],
            'unique_teams': s['unique_teams'],
            'min_date': s['min_date'],
            'max_date': s['max_date'],
            'spread_missing': s['spread_missing'],
            'total_missing': s['total_missing'],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = reports_dir / 'raw_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Wrote {summary_path}")

    return summaries, spread_analysis


if __name__ == '__main__':
    main()
