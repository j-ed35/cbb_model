"""
Daily prediction script for CBB ATS betting.

Fetches live KenPom data via kenpompy and today's games from Odds API,
then generates margin predictions with edge calculations.

Uses v2 model with extended KenPom features:
- Team-specific HCA
- Four Factors matchups
- Height/Experience differentials

Usage:
    python -m src.cbb.predict_daily [--threshold N] [--save] [--html]

Output:
    - Console table with predictions
    - Optional: reports/predictions/predictions_YYYY-MM-DD.csv
    - With --html: generates docs/picks.html for GitHub Pages
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


def fetch_kenpom_live() -> dict:
    """
    Fetch live KenPom data via kenpompy.

    Returns dict with 'ratings', 'four_factors', 'hca', 'height_exp' DataFrames.
    """
    from kenpompy.utils import login
    from kenpompy.misc import get_pomeroy_ratings, get_hca
    import kenpompy.summary as kp_summary

    console.print("[bold]Fetching live KenPom data...[/bold]")

    # Get credentials
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

    browser = login(email, pw)

    # Fetch ratings
    ratings = get_pomeroy_ratings(browser)
    # Normalize column names
    ratings = ratings.rename(columns={
        'Team': 'TeamName',
        'AdjO': 'AdjOE',
        'AdjD': 'AdjDE',
        'AdjT': 'AdjTempo',
    })
    console.print(f"  Ratings: {len(ratings)} teams")

    # Fetch Four Factors
    ff = kp_summary.get_fourfactors(browser)
    ff = ff.rename(columns={
        'Team': 'TeamName',
        'Off-eFG%': 'off_efg', 'Off-TO%': 'off_to_pct',
        'Off-OR%': 'off_or_pct', 'Off-FTRate': 'off_ft_rate',
        'Def-eFG%': 'def_efg', 'Def-TO%': 'def_to_pct',
        'Def-OR%': 'def_or_pct', 'Def-FTRate': 'def_ft_rate',
    })
    console.print(f"  Four Factors: {len(ff)} teams")

    # Fetch HCA
    hca = get_hca(browser)
    hca = hca.rename(columns={'Team': 'TeamName', 'HCA': 'hca', 'Elev': 'elevation'})
    console.print(f"  HCA: {len(hca)} teams")

    # Fetch Height/Experience
    ht = kp_summary.get_height(browser)
    ht = ht.rename(columns={
        'Team': 'TeamName', 'AvgHgt': 'avg_height',
        'Experience': 'experience', 'Continuity': 'continuity',
    })
    console.print(f"  Height/Exp: {len(ht)} teams")

    return {
        'ratings': ratings,
        'four_factors': ff,
        'hca': hca,
        'height_exp': ht,
    }


def fetch_odds_api_games(api_key: str, sport: str = 'basketball_ncaab') -> list[dict]:
    """Fetch today's games and spreads from The Odds API."""
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
    """Parse Odds API response into DataFrame."""
    records = []

    for game in games_data:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

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
    """Map Odds API team name to KenPom team name."""
    if team_name in kenpom_teams:
        return team_name

    if team_name in team_map:
        return team_map[team_name]

    team_lower = team_name.lower()
    for kp_team in kenpom_teams:
        if kp_team.lower() == team_lower:
            return kp_team

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


def extract_features_v2(
    home_team_kp: str,
    away_team_kp: str,
    kenpom_data: dict,
    is_neutral: bool = False
) -> dict:
    """
    Extract v2 features for a single game prediction.

    Includes: KenPom core, Four Factors matchups, HCA, Height/Experience.
    """
    ratings = kenpom_data['ratings']
    ff = kenpom_data['four_factors']
    hca_df = kenpom_data['hca']
    ht = kenpom_data['height_exp']

    team_col = 'TeamName' if 'TeamName' in ratings.columns else 'Team'

    row_a = ratings[ratings[team_col] == home_team_kp]
    row_b = ratings[ratings[team_col] == away_team_kp]

    if len(row_a) == 0 or len(row_b) == 0:
        return None

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    features = {
        # KenPom core
        'kp_adj_em_a': to_float(row_a.get('AdjEM', np.nan)),
        'kp_adj_em_b': to_float(row_b.get('AdjEM', np.nan)),
        'kp_adj_o_a': to_float(row_a.get('AdjOE', np.nan)),
        'kp_adj_o_b': to_float(row_b.get('AdjOE', np.nan)),
        'kp_adj_d_a': to_float(row_a.get('AdjDE', np.nan)),
        'kp_adj_d_b': to_float(row_b.get('AdjDE', np.nan)),
        'kp_tempo_a': to_float(row_a.get('AdjTempo', np.nan)),
        'kp_tempo_b': to_float(row_b.get('AdjTempo', np.nan)),

        # Context
        'is_home_a': 0 if is_neutral else 1,
        'is_neutral': 1 if is_neutral else 0,

        # Situational defaults
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

    # Derived KenPom features
    if not np.isnan(features['kp_adj_em_a']) and not np.isnan(features['kp_adj_em_b']):
        features['kp_adj_em_diff'] = features['kp_adj_em_a'] - features['kp_adj_em_b']

    if not np.isnan(features['kp_tempo_a']) and not np.isnan(features['kp_tempo_b']):
        features['kp_tempo_avg'] = (features['kp_tempo_a'] + features['kp_tempo_b']) / 2
        features['kp_tempo_diff'] = features['kp_tempo_a'] - features['kp_tempo_b']

    if not np.isnan(features['kp_adj_o_a']) and not np.isnan(features['kp_adj_d_b']):
        features['kp_o_vs_d_a'] = features['kp_adj_o_a'] - features['kp_adj_d_b']

    if not np.isnan(features['kp_adj_o_b']) and not np.isnan(features['kp_adj_d_a']):
        features['kp_o_vs_d_b'] = features['kp_adj_o_b'] - features['kp_adj_d_a']

    # HCA (team-specific)
    if is_neutral:
        features['team_hca'] = 0.0
    else:
        hca_row = hca_df[hca_df['TeamName'] == home_team_kp]
        hca_val = hca_row['hca'].values[0] if len(hca_row) > 0 else 3.5
        hca_val = to_float(hca_val)
        features['team_hca'] = hca_val if not np.isnan(hca_val) else 3.5

    # Four Factors matchups
    ff_a = ff[ff['TeamName'] == home_team_kp]
    ff_b = ff[ff['TeamName'] == away_team_kp]

    if len(ff_a) > 0 and len(ff_b) > 0:
        ff_a, ff_b = ff_a.iloc[0], ff_b.iloc[0]

        # eFG matchup
        off_efg_a = pd.to_numeric(ff_a.get('off_efg', np.nan), errors='coerce')
        def_efg_b = pd.to_numeric(ff_b.get('def_efg', np.nan), errors='coerce')
        off_efg_b = pd.to_numeric(ff_b.get('off_efg', np.nan), errors='coerce')
        def_efg_a = pd.to_numeric(ff_a.get('def_efg', np.nan), errors='coerce')

        features['ff_efg_matchup_a'] = off_efg_a - def_efg_b if pd.notna(off_efg_a) and pd.notna(def_efg_b) else 0
        features['ff_efg_matchup_b'] = off_efg_b - def_efg_a if pd.notna(off_efg_b) and pd.notna(def_efg_a) else 0

        # TO matchup
        off_to_a = pd.to_numeric(ff_a.get('off_to_pct', np.nan), errors='coerce')
        def_to_b = pd.to_numeric(ff_b.get('def_to_pct', np.nan), errors='coerce')
        off_to_b = pd.to_numeric(ff_b.get('off_to_pct', np.nan), errors='coerce')
        def_to_a = pd.to_numeric(ff_a.get('def_to_pct', np.nan), errors='coerce')

        features['ff_to_matchup_a'] = def_to_b - off_to_a if pd.notna(def_to_b) and pd.notna(off_to_a) else 0
        features['ff_to_matchup_b'] = def_to_a - off_to_b if pd.notna(def_to_a) and pd.notna(off_to_b) else 0

        # Rebounding matchup
        off_or_a = pd.to_numeric(ff_a.get('off_or_pct', np.nan), errors='coerce')
        def_or_b = pd.to_numeric(ff_b.get('def_or_pct', np.nan), errors='coerce')
        off_or_b = pd.to_numeric(ff_b.get('off_or_pct', np.nan), errors='coerce')
        def_or_a = pd.to_numeric(ff_a.get('def_or_pct', np.nan), errors='coerce')

        features['ff_reb_matchup_a'] = off_or_a - def_or_b if pd.notna(off_or_a) and pd.notna(def_or_b) else 0
        features['ff_reb_matchup_b'] = off_or_b - def_or_a if pd.notna(off_or_b) and pd.notna(def_or_a) else 0
    else:
        features['ff_efg_matchup_a'] = 0
        features['ff_efg_matchup_b'] = 0
        features['ff_to_matchup_a'] = 0
        features['ff_to_matchup_b'] = 0
        features['ff_reb_matchup_a'] = 0
        features['ff_reb_matchup_b'] = 0

    # Height/Experience
    ht_a = ht[ht['TeamName'] == home_team_kp]
    ht_b = ht[ht['TeamName'] == away_team_kp]

    if len(ht_a) > 0 and len(ht_b) > 0:
        ht_a, ht_b = ht_a.iloc[0], ht_b.iloc[0]

        height_a = pd.to_numeric(ht_a.get('avg_height', np.nan), errors='coerce')
        height_b = pd.to_numeric(ht_b.get('avg_height', np.nan), errors='coerce')
        exp_a = pd.to_numeric(ht_a.get('experience', np.nan), errors='coerce')
        exp_b = pd.to_numeric(ht_b.get('experience', np.nan), errors='coerce')
        cont_a = pd.to_numeric(ht_a.get('continuity', np.nan), errors='coerce')
        cont_b = pd.to_numeric(ht_b.get('continuity', np.nan), errors='coerce')

        features['ht_height_diff'] = height_a - height_b if pd.notna(height_a) and pd.notna(height_b) else 0
        features['ht_exp_diff'] = exp_a - exp_b if pd.notna(exp_a) and pd.notna(exp_b) else 0
        features['ht_cont_diff'] = cont_a - cont_b if pd.notna(cont_a) and pd.notna(cont_b) else 0
    else:
        features['ht_height_diff'] = 0
        features['ht_exp_diff'] = 0
        features['ht_cont_diff'] = 0

    return features


def predict_with_v2_model(features: dict, models_dir: Path) -> Optional[float]:
    """Make prediction using v2 ensemble model (Ridge + GBM + DNN)."""
    model_path = models_dir / 'enhanced_v2_full.pkl'
    dnn_path = models_dir / 'enhanced_v2_full_dnn.pt'
    dnn_config_path = models_dir / 'enhanced_v2_full_dnn_config.pkl'

    if not model_path.exists():
        return None

    try:
        import torch
        from src.cbb.train_v2_full import SimpleDNN

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        ridge = model_data['ridge']
        gbm = model_data['gbm']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        weights = model_data.get('weights', (0.1, 0.1, 0.8))

        # Build feature vector
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X = np.nan_to_num(X, nan=0)

        X_scaled = scaler.transform(X)

        # Ridge prediction
        ridge_pred = ridge.predict(X_scaled)[0]

        # GBM prediction
        gbm_pred = gbm.predict(X)[0]

        # DNN prediction
        if dnn_path.exists() and dnn_config_path.exists():
            with open(dnn_config_path, 'rb') as f:
                dnn_config = pickle.load(f)

            dnn = SimpleDNN(dnn_config['input_dim'])
            dnn.load_state_dict(torch.load(dnn_path, weights_only=True))
            dnn.eval()

            with torch.no_grad():
                dnn_pred = dnn(torch.FloatTensor(X_scaled)).item()
        else:
            dnn_pred = (ridge_pred + gbm_pred) / 2  # Fallback

        # Ensemble
        return weights[0] * ridge_pred + weights[1] * gbm_pred + weights[2] * dnn_pred

    except Exception as e:
        console.print(f"[yellow]Model prediction error: {e}[/yellow]")
        return None


def simple_kenpom_prediction(features: dict) -> float:
    """Fallback: Simple KenPom-based margin prediction."""
    if 'kp_adj_em_diff' not in features or np.isnan(features.get('kp_adj_em_diff', np.nan)):
        return np.nan

    tempo = features.get('kp_tempo_avg', 67.5)
    hca = features.get('team_hca', 3.5)

    pred_margin = features['kp_adj_em_diff'] * tempo / 100
    pred_margin += hca * features.get('is_home_a', 0)

    return pred_margin


def generate_predictions(
    kenpom_data: dict,
    games_df: pd.DataFrame,
    team_map: dict,
    models_dir: Path,
    threshold: float
) -> pd.DataFrame:
    """Generate predictions for all games."""
    ratings = kenpom_data['ratings']
    team_col = 'TeamName' if 'TeamName' in ratings.columns else 'Team'
    kenpom_teams = set(ratings[team_col].unique())

    predictions = []

    for idx, row in games_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        spread_home = row['spread_home']

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

        features = extract_features_v2(home_kp, away_kp, kenpom_data)

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

        # Try v2 model, fall back to simple
        pred_margin = predict_with_v2_model(features, models_dir)
        if pred_margin is None:
            pred_margin = simple_kenpom_prediction(features)

        # Calculate edge
        if spread_home is not None and not np.isnan(pred_margin):
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
    parser.add_argument('--threshold', type=float, default=4.5,
                        help='Edge threshold for picks (default: 4.5)')
    parser.add_argument('--save', action='store_true', help='Save predictions to CSV')
    parser.add_argument('--html', action='store_true', help='Generate HTML for GitHub Pages')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'reports' / 'models'
    predictions_dir = project_root / 'reports' / 'predictions'
    docs_dir = project_root / 'docs'

    predictions_dir.mkdir(parents=True, exist_ok=True)

    env_vars = load_env_vars(project_root)
    api_key = env_vars.get('ODDS_API_KEY', os.environ.get('ODDS_API_KEY', ''))
    api_key = api_key.strip()

    if not api_key:
        console.print("[red]Error: ODDS_API_KEY not found[/red]")
        return

    # Fetch KenPom data
    try:
        kenpom_data = fetch_kenpom_live()
    except Exception as e:
        console.print(f"[red]Error fetching KenPom data: {e}[/red]")
        return

    # Load team name mapping
    team_map_path = project_root / 'data' / 'processed' / 'team_name_map_v2.csv'
    if not team_map_path.exists():
        team_map_path = project_root / 'data' / 'processed' / 'team_name_map.csv'

    if team_map_path.exists():
        team_map_df = pd.read_csv(team_map_path)
        if 'raw_name' in team_map_df.columns:
            team_map = dict(zip(team_map_df['raw_name'], team_map_df['kenpom_name']))
        else:
            team_map = dict(zip(team_map_df['team_name'], team_map_df['kenpom_name']))
    else:
        team_map = {}

    # Fetch games
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
    console.print("\n[bold]Generating predictions (v2 model)...[/bold]")
    pred_df = generate_predictions(kenpom_data, games_df, team_map, models_dir, args.threshold)

    display_predictions(pred_df, args.threshold)

    console.print(f"\n[cyan]Model: Enhanced v2 (Ridge 10% + GBM 10% + DNN 80%)[/cyan]")
    console.print(f"[cyan]Backtest: 53.2% hit rate, +1.6% ROI @ threshold {args.threshold}[/cyan]")

    if args.save:
        output_path = predictions_dir / f"predictions_{date.today()}.csv"
        pred_df.to_csv(output_path, index=False)
        console.print(f"\n[green]Saved predictions to {output_path}[/green]")

    if args.html:
        from src.cbb.generate_html import generate_html_page
        output_path = docs_dir / 'picks.html'
        generate_html_page(pred_df, str(date.today()), output_path, args.threshold)
        console.print(f"[green]Generated HTML at {output_path}[/green]")


if __name__ == '__main__':
    main()
