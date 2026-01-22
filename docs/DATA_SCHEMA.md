# DATA_SCHEMA.md — Raw NCAAB Data

## Overview
This document describes the schema of raw CSV files in `data/raw/`.

## Files Available
- `2022_ncaab.csv`: 10,936 rows, 2021-22 season, dates 2025-11-03 to 2026-03-03
- `2023_ncaab.csv`: 11,339 rows, 2022-23 season, dates 2021-11-09 to 2022-04-04
- `2024_ncaab.csv`: 11,826 rows, 2023-24 season, dates 2022-11-07 to 2023-04-03
- `2025_ncaab.csv`: 11,911 rows, 2024-25 season, dates 2023-11-06 to 2024-04-08
- `2026_ncaab.csv`: 12,009 rows, 2025-26 season, dates 2024-11-04 to 2025-04-07

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
- `Home`: 5,743 occurrences
- `Away`: 5,193 occurrences

### Game Types
- `REG`: 10,936 games

## Spread Convention (CRITICAL)

Based on analysis of 50,714 games with spread data:

**Convention**: spread is from team perspective (team + spread > 0 means cover)
**Verification rate**: 100.0%

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
