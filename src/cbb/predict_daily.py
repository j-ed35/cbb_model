"""
Daily prediction script for CBB ATS betting.

Fetches live KenPom data via kenpompy and today's games from Odds API,
then generates margin predictions with edge calculations.

Usage:
    python -m src.cbb.predict_daily [--date YYYY-MM-DD] [--threshold N] [--save]

Output:
    - Console table with predictions
    - Optional: reports/predictions/predictions_YYYY-MM-DD.csv
    - With --html: generates docs/index.html for GitHub Pages
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import requests
import pickle
import argparse
import os
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()


def load_env_vars(project_root: Path) -> dict:
    """Load environment variables from .env file."""
    env_path = project_root / '.env'
    env_vars = {}

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"')

    return env_vars


def fetch_kenpom_live() -> pd.DataFrame:
    """
    Fetch live KenPom ratings via kenpompy.

    Returns DataFrame with team ratings.
    """
    from src.cbb.ingest.kenpom_api import fetch_kenpom_ratings

    console.print("[bold]Fetching live KenPom data...[/bold]")
    df = fetch_kenpom_ratings()
    console.print(f"  Loaded {len(df)} teams from KenPom")
    return df


def load_kenpom_from_csv(kenpom_dir: Path) -> tuple[pd.DataFrame, str]:
    """
    Fallback: Load the most recent KenPom snapshot from CSV.

    Returns: (DataFrame, date_string)
    """
    files = sorted(kenpom_dir.glob('kenpom_????-??-??.csv'), reverse=True)

    if not files:
        raise FileNotFoundError(f"No KenPom files found in {kenpom_dir}")

    latest = files[0]
    date_str = latest.stem.replace('kenpom_', '')

    df = pd.read_csv(latest)

    # Ensure numeric columns
    for col in ['AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, date_str


def fetch_odds_api_games(api_key: str, sport: str = 'basketball_ncaab') -> list[dict]:
    """
    Fetch today's games and spreads from The Odds API.

    Returns list of game dictionaries with spreads.
    """
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'

    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads',
        'oddsFormat': 'american',
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        console.print(f"[red]Error fetching odds: {e}[/red]")
        return []


def parse_odds_api_response(games_data: list[dict]) -> pd.DataFrame:
    """
    Parse Odds API response into DataFrame.

    Returns DataFrame with columns:
    - game_id, commence_time, home_team, away_team, spread_home, spread_away
    """
    records = []

    for game in games_data:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        # Get spreads from first bookmaker
        spread_home = None
        spread_away = None

        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            spread_home = outcome.get('point')
                        elif outcome.get('name') == away_team:
                            spread_away = outcome.get('point')
                    break
            if spread_home is not None:
                break

        records.append({
            'game_id': game.get('id', ''),
            'commence_time': commence_time,
            'home_team': home_team,
            'away_team': away_team,
            'spread_home': spread_home,
            'spread_away': spread_away,
        })

    return pd.DataFrame(records)


def map_odds_api_to_kenpom(team_name: str, kenpom_teams: set, team_map: dict) -> Optional[str]:
    """
    Map Odds API team name to KenPom team name.

    Uses fuzzy matching and known mappings.
    """
    # Try direct match
    if team_name in kenpom_teams:
        return team_name

    # Try existing mapping
    if team_name in team_map:
        return team_map[team_name]

    # Try lowercase match
    team_lower = team_name.lower()
    for kp_team in kenpom_teams:
        if kp_team.lower() == team_lower:
            return kp_team

    # Try partial match
    team_words = set(team_lower.split())
    best_match = None
    best_score = 0

    for kp_team in kenpom_teams:
        kp_words = set(kp_team.lower().split())
        overlap = len(team_words & kp_words)
        if overlap > best_score and overlap >= 1:
            best_score = overlap
            best_match = kp_team

    return best_match


def extract_features_for_game(
    home_team_kp: str,
    away_team_kp: str,
    kenpom_df: pd.DataFrame,
    spread_home: float
) -> dict:
    """
    Extract features for a single game prediction.

    Assumes Team A = Home team.
    """
    team_col = 'TeamName' if 'TeamName' in kenpom_df.columns else 'Team'

    row_a = kenpom_df[kenpom_df[team_col] == home_team_kp]
    row_b = kenpom_df[kenpom_df[team_col] == away_team_kp]

    if len(row_a) == 0 or len(row_b) == 0:
        return None

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    features = {
        # KenPom core
        'kp_adj_em_a': row_a.get('AdjEM', np.nan),
        'kp_adj_em_b': row_b.get('AdjEM', np.nan),
        'kp_adj_o_a': row_a.get('AdjOE', np.nan),
        'kp_adj_o_b': row_b.get('AdjOE', np.nan),
        'kp_adj_d_a': row_a.get('AdjDE', np.nan),
        'kp_adj_d_b': row_b.get('AdjDE', np.nan),
        'kp_tempo_a': row_a.get('AdjTempo', np.nan),
        'kp_tempo_b': row_b.get('AdjTempo', np.nan),

        # Rankings (normalized)
        'kp_rank_em_a': row_a.get('RankAdjEM', np.nan) / 365,
        'kp_rank_em_b': row_b.get('RankAdjEM', np.nan) / 365,
        'kp_rank_o_a': row_a.get('RankAdjOE', np.nan) / 365,
        'kp_rank_o_b': row_b.get('RankAdjOE', np.nan) / 365,
        'kp_rank_d_a': row_a.get('RankAdjDE', np.nan) / 365,
        'kp_rank_d_b': row_b.get('RankAdjDE', np.nan) / 365,

        # Context (home game)
        'is_home_a': 1,
        'is_neutral': 0,

        # Situational defaults (no historical data available)
        'rest_days_a': 3,
        'rest_days_b': 3,
        'rest_diff': 0,
        'b2b_a': 0,
        'b2b_b': 0,
        'rolling_ats_a': 0.5,
        'rolling_ats_b': 0.5,
        'rolling_ats_diff': 0,
        'ew_margin_a': 0,
        'ew_margin_b': 0,
        'ew_margin_diff': 0,
    }

    # Derived features
    if not np.isnan(features['kp_adj_em_a']) and not np.isnan(features['kp_adj_em_b']):
        features['kp_adj_em_diff'] = features['kp_adj_em_a'] - features['kp_adj_em_b']

    if not np.isnan(features['kp_tempo_a']) and not np.isnan(features['kp_tempo_b']):
        features['kp_tempo_avg'] = (features['kp_tempo_a'] + features['kp_tempo_b']) / 2
        features['kp_tempo_diff'] = features['kp_tempo_a'] - features['kp_tempo_b']

    if not np.isnan(features['kp_adj_o_a']) and not np.isnan(features['kp_adj_d_b']):
        features['kp_o_vs_d_a'] = features['kp_adj_o_a'] - features['kp_adj_d_b']

    if not np.isnan(features['kp_adj_o_b']) and not np.isnan(features['kp_adj_d_a']):
        features['kp_o_vs_d_b'] = features['kp_adj_o_b'] - features['kp_adj_d_a']

    return features


def simple_kenpom_prediction(features: dict, hca: float = 3.5) -> float:
    """
    Simple KenPom-based margin prediction.

    Uses: AdjEM difference * tempo factor + home court advantage
    """
    if 'kp_adj_em_diff' not in features or np.isnan(features['kp_adj_em_diff']):
        return np.nan

    tempo = features.get('kp_tempo_avg', 67.5)

    # KenPom margin prediction formula (approximate)
    # margin = (AdjEM_A - AdjEM_B) * tempo / 100 + HCA
    pred_margin = features['kp_adj_em_diff'] * tempo / 100

    # Add home court advantage (Team A is home)
    pred_margin += hca * features.get('is_home_a', 0)

    return pred_margin


def predict_with_model(features: dict, model_path: Path) -> Optional[float]:
    """Make prediction using trained ensemble model."""
    if not model_path.exists():
        return None

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        ridge = model_data['ridge']
        gbm = model_data['gbm']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        weights = model_data.get('weights', (0.4, 0.6))

        # Build feature vector
        X = np.array([[features.get(col, 0) for col in feature_cols]])

        # Handle missing values
        X = np.nan_to_num(X, nan=0)

        # Predict
        X_scaled = scaler.transform(X)
        ridge_pred = ridge.predict(X_scaled)[0]
        gbm_pred = gbm.predict(X)[0]

        return weights[0] * ridge_pred + weights[1] * gbm_pred

    except Exception as e:
        console.print(f"[yellow]Model prediction error: {e}[/yellow]")
        return None


def generate_predictions(
    kenpom_df: pd.DataFrame,
    games_df: pd.DataFrame,
    team_map: dict,
    model_path: Path,
    threshold: float
) -> pd.DataFrame:
    """
    Generate predictions for all games.

    Returns DataFrame with predictions.
    """
    team_col = 'TeamName' if 'TeamName' in kenpom_df.columns else 'Team'
    kenpom_teams = set(kenpom_df[team_col].unique())

    predictions = []

    for idx, row in games_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        spread_home = row['spread_home']

        # Map to KenPom names
        home_kp = map_odds_api_to_kenpom(home_team, kenpom_teams, team_map)
        away_kp = map_odds_api_to_kenpom(away_team, kenpom_teams, team_map)

        if not home_kp or not away_kp:
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'spread_home': spread_home,
                'pred_margin': np.nan,
                'edge': np.nan,
                'pick': 'NO DATA',
                'confidence': '',
            })
            continue

        # Extract features
        features = extract_features_for_game(home_kp, away_kp, kenpom_df, spread_home)

        if features is None:
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'spread_home': spread_home,
                'pred_margin': np.nan,
                'edge': np.nan,
                'pick': 'NO DATA',
                'confidence': '',
            })
            continue

        # Try model prediction first, fall back to simple KenPom
        pred_margin = predict_with_model(features, model_path)

        if pred_margin is None:
            pred_margin = simple_kenpom_prediction(features)

        # Calculate edge
        if spread_home is not None and not np.isnan(pred_margin):
            # edge = predicted_margin - (-spread)
            # Positive edge = we expect home to cover
            edge = pred_margin - (-spread_home)
        else:
            edge = np.nan

        # Determine pick
        if np.isnan(edge):
            pick = 'NO LINE'
            confidence = ''
        elif abs(edge) < threshold:
            pick = 'SKIP'
            confidence = ''
        elif edge > 0:
            pick = f"{home_team} {spread_home:+.1f}"
            confidence = '*' * min(int(abs(edge) / 2), 3)
        else:
            pick = f"{away_team} {-spread_home:+.1f}"
            confidence = '*' * min(int(abs(edge) / 2), 3)

        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_kp': home_kp,
            'away_kp': away_kp,
            'spread_home': spread_home,
            'pred_margin': pred_margin,
            'edge': edge,
            'pick': pick,
            'confidence': confidence,
        })

    return pd.DataFrame(predictions)


def display_predictions(pred_df: pd.DataFrame, threshold: float) -> None:
    """Display predictions in a formatted table."""
    console.print("\n")
    table = Table(title=f"CBB Predictions - {date.today()}")

    table.add_column("Away Team", style="cyan")
    table.add_column("Home Team", style="green")
    table.add_column("Spread", justify="right")
    table.add_column("Pred Margin", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Pick", style="bold")
    table.add_column("Conf")

    for _, row in pred_df.iterrows():
        spread_str = f"{row['spread_home']:+.1f}" if pd.notna(row['spread_home']) else "N/A"
        margin_str = f"{row['pred_margin']:.1f}" if pd.notna(row['pred_margin']) else "N/A"
        edge_str = f"{row['edge']:+.1f}" if pd.notna(row['edge']) else "N/A"

        # Bold edge if actionable
        if pd.notna(row['edge']) and abs(row['edge']) >= threshold:
            edge_str = f"[bold]{edge_str}[/bold]"

        table.add_row(
            row['away_team'],
            row['home_team'],
            spread_str,
            margin_str,
            edge_str,
            row['pick'],
            row['confidence']
        )

    console.print(table)

    # Summary
    actionable = pred_df[pred_df['edge'].abs() >= threshold]
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total games: {len(pred_df)}")
    console.print(f"  Actionable picks (edge >= {threshold}): {len(actionable)}")

    if len(actionable) > 0:
        console.print(f"\n[bold green]TOP PICKS:[/bold green]")
        for _, row in actionable.nlargest(5, 'edge', keep='all').iterrows():
            console.print(f"  {row['pick']} ({row['edge']:+.1f} edge)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate daily CBB predictions')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=4.5,
                        help='Edge threshold for picks (default: 4.5)')
    parser.add_argument('--save', action='store_true', help='Save predictions to CSV')
    parser.add_argument('--html', action='store_true', help='Generate HTML for GitHub Pages')
    parser.add_argument('--use-csv', action='store_true',
                        help='Use CSV file instead of live KenPom fetch')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    kenpom_dir = project_root / 'data' / 'kenpom'
    models_dir = project_root / 'reports' / 'models'
    predictions_dir = project_root / 'reports' / 'predictions'
    docs_dir = project_root / 'docs'

    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load environment variables
    env_vars = load_env_vars(project_root)
    api_key = env_vars.get('ODDS_API_KEY', os.environ.get('ODDS_API_KEY', ''))

    if not api_key:
        console.print("[red]Error: ODDS_API_KEY not found in .env or environment[/red]")
        return

    # Load KenPom data (live or from CSV)
    if args.use_csv:
        console.print("[bold]Loading KenPom data from CSV...[/bold]")
        try:
            kenpom_df, kenpom_date = load_kenpom_from_csv(kenpom_dir)
            console.print(f"  Loaded KenPom snapshot from {kenpom_date}")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    else:
        try:
            kenpom_df = fetch_kenpom_live()
        except Exception as e:
            console.print(f"[red]Error fetching KenPom data: {e}[/red]")
            console.print("[yellow]Falling back to CSV...[/yellow]")
            try:
                kenpom_df, kenpom_date = load_kenpom_from_csv(kenpom_dir)
                console.print(f"  Loaded KenPom snapshot from {kenpom_date}")
            except FileNotFoundError as e2:
                console.print(f"[red]Error: {e2}[/red]")
                return

    team_col = 'TeamName' if 'TeamName' in kenpom_df.columns else 'Team'
    kenpom_teams = set(kenpom_df[team_col].unique())

    # Load team name mapping
    team_map_path = project_root / 'data' / 'processed' / 'team_name_map_v2.csv'
    if team_map_path.exists():
        team_map_df = pd.read_csv(team_map_path)
        team_map = dict(zip(team_map_df['raw_name'], team_map_df['kenpom_name']))
    else:
        team_map = {}

    # Fetch games from Odds API
    console.print("[bold]Fetching games from Odds API...[/bold]")
    games_data = fetch_odds_api_games(api_key)

    if not games_data:
        console.print("[yellow]No games found[/yellow]")
        return

    games_df = parse_odds_api_response(games_data)
    console.print(f"  Found {len(games_df)} games")

    if len(games_df) == 0:
        console.print("[yellow]No games to predict[/yellow]")
        return

    # Generate predictions
    console.print("\n[bold]Generating predictions...[/bold]")
    model_path = models_dir / 'enhanced_ensemble.pkl'

    pred_df = generate_predictions(
        kenpom_df, games_df, team_map, model_path, args.threshold
    )

    # Display results
    display_predictions(pred_df, args.threshold)

    # Model performance note
    console.print(f"\n[cyan]Model Performance (threshold >= {args.threshold}):[/cyan]")
    console.print("[cyan]  Backtest hit rate: 52.9% | ROI: +1.1% | Breakeven: 52.4%[/cyan]")

    # Save to CSV
    if args.save:
        output_path = predictions_dir / f"predictions_{date.today()}.csv"
        pred_df.to_csv(output_path, index=False)
        console.print(f"\n[green]Saved predictions to {output_path}[/green]")

    # Generate HTML
    if args.html:
        from src.cbb.generate_html import generate_html_page
        output_path = docs_dir / 'index.html'
        generate_html_page(pred_df, str(date.today()), output_path, args.threshold)
        console.print(f"[green]Generated HTML at {output_path}[/green]")


if __name__ == '__main__':
    main()
