"""
Daily prediction script for CBB ATS betting.

Fetches live KenPom data and today's games from Odds API,
runs a single GBM model, and outputs predictions.

Usage:
    python -m src.cbb.predict_daily [--threshold N] [--save] [--html] [--json]
"""

import json
import os
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.table import Table

console = Console()

# Same features used in training (expanded set)
FEATURES = [
    # KenPom core (8)
    "kp_adj_em_a", "kp_adj_em_b",
    "kp_adj_o_a", "kp_adj_o_b",
    "kp_adj_d_a", "kp_adj_d_b",
    "kp_tempo_a", "kp_tempo_b",
    # KenPom derived (5)
    "kp_adj_em_diff",
    "kp_tempo_avg", "kp_tempo_diff",
    "kp_o_vs_d_a", "kp_o_vs_d_b",
    # Context (2)
    "is_home_a", "is_neutral",
    # Game context (2)
    "is_conference_tourney", "is_postseason",
    # Rest/situational (3)
    "rest_days_a", "rest_days_b", "rest_diff",
    # Recency (3)
    "ew_margin_a", "ew_margin_b", "ew_margin_diff",
    # Rolling ATS (2)
    "rolling_ats_a", "rolling_ats_b",
]

# Neutral defaults for features not available at live inference time
INFERENCE_DEFAULTS = {
    "is_conference_tourney": 0,
    "is_postseason": 0,
    "rest_days_a": 7.0,
    "rest_days_b": 7.0,
    "rest_diff": 0.0,
    "ew_margin_a": 0.0,
    "ew_margin_b": 0.0,
    "ew_margin_diff": 0.0,
    "rolling_ats_a": 0.5,
    "rolling_ats_b": 0.5,
}


def load_model(models_dir: Path) -> dict:
    """Load the GBM model."""
    path = models_dir / "gbm.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train_enhanced.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_kenpom_live() -> pd.DataFrame:
    """Fetch live KenPom ratings. Uses kenpompy (cloudscraper) to bypass Cloudflare."""
    from kenpompy.misc import get_pomeroy_ratings
    from kenpompy.utils import login

    console.print("[bold]Fetching KenPom data...[/bold]")

    email = os.environ.get("KENPOM_EMAIL", "")
    pw = os.environ.get("KENPOM_PW", "")

    if not email:
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "KENPOM_EMAIL" in line:
                        email = line.split("=")[1].strip().strip('"')
                    if "KENPOM_PW" in line:
                        pw = line.split("=")[1].strip().strip('"')

    browser = login(email, pw)
    ratings = get_pomeroy_ratings(browser)
    ratings = ratings.rename(columns={
        "Team": "TeamName",
        "AdjO": "AdjOE",
        "AdjD": "AdjDE",
        "AdjT": "AdjTempo",
    })
    console.print(f"  {len(ratings)} teams loaded")
    return ratings


def fetch_odds_api_games(api_key: str) -> list[dict]:
    """Fetch today's NCAAB games and spreads from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        console.print(f"[red]Odds API error: {e}[/red]")
        return []


def parse_games(games_data: list[dict]) -> pd.DataFrame:
    """Parse Odds API response into a DataFrame."""
    records = []
    for game in games_data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")

        spread_home = total = None
        for bk in game.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home:
                            spread_home = o.get("point")
                if mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Over":
                            total = o.get("point")
            if spread_home is not None:
                break

        records.append({
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "spread_home": spread_home,
            "total": total,
        })
    return pd.DataFrame(records)


# Hardcoded overrides for Odds API names that the CSV/fuzzy matching can't handle
_ODDS_API_OVERRIDES: dict[str, str] = {
    "UNLV Rebels": "UNLV",
    "Omaha Mavericks": "Nebraska Omaha",
    "Queens University Royals": "Queens",
    "Arizona St Sun Devils": "Arizona St.",
    "North Dakota Fighting Hawks": "North Dakota",
}


def map_team_name(name: str, kenpom_teams: set, team_map: dict) -> Optional[str]:
    """Map Odds API team name to KenPom name."""
    if name in kenpom_teams:
        return name
    if name in team_map:
        return team_map[name]
    if name in _ODDS_API_OVERRIDES:
        return _ODDS_API_OVERRIDES[name]

    # Try common abbreviation expansions then check the map again
    # Apply all expansions: "St" -> "State", "SE" -> "Southeast", etc.
    expanded = name
    if " St " in expanded:
        expanded = expanded.replace(" St ", " State ")
    if expanded.startswith("SE "):
        expanded = "Southeast " + expanded[3:]

    if expanded != name:
        if expanded in team_map:
            return team_map[expanded]
        if expanded in kenpom_teams:
            return expanded
        # Also try the expanded form in case-insensitive matching
        exp_lower = expanded.lower()
        for kp in kenpom_teams:
            if kp.lower() == exp_lower:
                return kp

    # Try case-insensitive exact match
    lower = name.lower()
    for kp in kenpom_teams:
        if kp.lower() == lower:
            return kp

    # Try word overlap on school name (strip mascot = last word)
    name_words = lower.split()
    name_core = set(name_words[:-1]) if len(name_words) > 1 else set(name_words)

    # First pass: require 2+ word overlap to avoid false positives
    best, best_score = None, 0
    for kp in kenpom_teams:
        kp_lower = kp.lower()
        kp_words = kp_lower.split()
        kp_core = set(kp_words) if len(kp_words) == 1 else set(kp_words[:-1])
        overlap = len(name_core & kp_core)
        if overlap > best_score and overlap >= 2:
            best_score = overlap
            best = kp

    if best:
        return best

    # Second pass: 1-word overlap (only if name core is a single word like "UNLV")
    if len(name_core) == 1:
        for kp in kenpom_teams:
            kp_lower = kp.lower()
            if kp_lower in name_core or name_core.issubset({kp_lower}):
                return kp

    return None


def extract_features(home_kp: str, away_kp: str, ratings: pd.DataFrame, is_neutral: bool = False) -> Optional[dict]:
    """Extract model features for a single game."""
    col = "TeamName" if "TeamName" in ratings.columns else "Team"
    row_a = ratings[ratings[col] == home_kp]
    row_b = ratings[ratings[col] == away_kp]
    if len(row_a) == 0 or len(row_b) == 0:
        return None

    row_a, row_b = row_a.iloc[0], row_b.iloc[0]

    def f(row, key):
        try:
            return float(row.get(key, np.nan))
        except (ValueError, TypeError):
            return np.nan

    em_a = f(row_a, "AdjEM")
    em_b = f(row_b, "AdjEM")
    o_a = f(row_a, "AdjOE")
    o_b = f(row_b, "AdjOE")
    d_a = f(row_a, "AdjDE")
    d_b = f(row_b, "AdjDE")
    t_a = f(row_a, "AdjTempo")
    t_b = f(row_b, "AdjTempo")

    features = {
        "kp_adj_em_a": em_a, "kp_adj_em_b": em_b,
        "kp_adj_o_a": o_a, "kp_adj_o_b": o_b,
        "kp_adj_d_a": d_a, "kp_adj_d_b": d_b,
        "kp_tempo_a": t_a, "kp_tempo_b": t_b,
        "kp_adj_em_diff": em_a - em_b if not (np.isnan(em_a) or np.isnan(em_b)) else 0,
        "kp_tempo_avg": (t_a + t_b) / 2 if not (np.isnan(t_a) or np.isnan(t_b)) else 67.5,
        "kp_tempo_diff": t_a - t_b if not (np.isnan(t_a) or np.isnan(t_b)) else 0,
        "kp_o_vs_d_a": o_a - d_b if not (np.isnan(o_a) or np.isnan(d_b)) else 0,
        "kp_o_vs_d_b": o_b - d_a if not (np.isnan(o_b) or np.isnan(d_a)) else 0,
        "is_home_a": 0 if is_neutral else 1,
        "is_neutral": 1 if is_neutral else 0,
    }

    # Add defaults for features not available at live inference
    features.update(INFERENCE_DEFAULTS)

    # Override tournament context: neutral site games during March are postseason
    if is_neutral:
        features["is_postseason"] = 1

    return features


def generate_predictions(
    ratings: pd.DataFrame,
    games_df: pd.DataFrame,
    team_map: dict,
    model_data: dict,
    threshold: float,
) -> pd.DataFrame:
    """Generate predictions for all games."""
    gbm = model_data["model"]
    feature_cols = model_data["feature_cols"]
    shrinkage = model_data.get("shrinkage", 1.0)
    kenpom_teams = set(ratings["TeamName" if "TeamName" in ratings.columns else "Team"].unique())

    results = []
    for _, row in games_df.iterrows():
        home_kp = map_team_name(row["home_team"], kenpom_teams, team_map)
        away_kp = map_team_name(row["away_team"], kenpom_teams, team_map)

        base = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "commence_time": row.get("commence_time", ""),
            "spread_home": row["spread_home"],
            "total": row.get("total"),
        }

        if not home_kp or not away_kp:
            base.update({"pred_margin": np.nan, "edge": np.nan, "pick": "NO DATA"})
            results.append(base)
            continue

        features = extract_features(home_kp, away_kp, ratings)
        if features is None:
            base.update({"pred_margin": np.nan, "edge": np.nan, "pick": "NO DATA"})
            results.append(base)
            continue

        X = np.array([[features.get(c, 0) for c in feature_cols]])
        X = np.nan_to_num(X, nan=0)
        pred_margin = gbm.predict(X)[0] * shrinkage

        spread = pd.to_numeric(row["spread_home"], errors="coerce")
        edge = pred_margin - (-spread) if pd.notna(spread) else np.nan

        # Cap absurd edges — usually indicates bad data from the Odds API
        MAX_EDGE = 15.0
        if pd.notna(edge) and abs(edge) > MAX_EDGE:
            pick = "SKIP"
            base.update({
                "pred_margin": pred_margin, "edge": edge,
                "pick": pick, "home_kp": home_kp, "away_kp": away_kp,
            })
            results.append(base)
            continue

        if pd.isna(edge):
            pick = "NO LINE"
        elif abs(edge) < threshold:
            pick = "SKIP"
        elif edge > 0:
            pick = f"{row['home_team']} {spread:+.1f}"
        else:
            pick = f"{row['away_team']} {-spread:+.1f}"

        base.update({
            "pred_margin": pred_margin,
            "edge": edge,
            "pick": pick,
            "home_kp": home_kp,
            "away_kp": away_kp,
        })
        results.append(base)

    return pd.DataFrame(results)


def _kenpom_profile(team_name: str, kp_name: str, ratings: pd.DataFrame) -> dict | None:
    """Extract a KenPom profile dict for embedding in JSON output."""
    if not kp_name:
        return None
    col = "TeamName" if "TeamName" in ratings.columns else "Team"
    rows = ratings[ratings[col] == kp_name]
    if len(rows) == 0:
        return None
    r = rows.iloc[0]

    def safe_float(val):
        try:
            return float(str(val).replace("+", ""))
        except (ValueError, TypeError):
            return None

    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    return {
        "rank": safe_int(r.get("Rk")),
        "adj_em": safe_float(r.get("AdjEM")),
        "adj_oe": safe_float(r.get("AdjOE")),
        "adj_oe_rank": safe_int(r.get("AdjO.Rank")),
        "adj_de": safe_float(r.get("AdjDE")),
        "adj_de_rank": safe_int(r.get("AdjD.Rank")),
        "adj_tempo": safe_float(r.get("AdjTempo")),
        "adj_tempo_rank": safe_int(r.get("AdjT.Rank")),
        "record": str(r.get("W-L", "")),
        "conference": str(r.get("Conf", "")),
    }


def predictions_to_json(pred_df: pd.DataFrame, threshold: float, ratings: pd.DataFrame = None) -> dict:
    """Convert predictions DataFrame to JSON-serializable dict for the API."""
    from src.cbb.utils.espn_ids import get_logo_url, get_abbrev, get_espn_id

    MAX_EDGE = 15.0
    games = []
    for _, row in pred_df.iterrows():
        edge = row.get("edge")
        if pd.isna(edge) or abs(edge) < threshold or abs(edge) > MAX_EDGE:
            continue

        spread = row.get("spread_home")
        if edge > 0:
            pick_team = row["home_team"]
            pick_spread = f"{spread:+.1f}" if pd.notna(spread) else ""
        else:
            pick_team = row["away_team"]
            pick_spread = f"{-spread:+.1f}" if pd.notna(spread) else ""

        game = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_abbrev": get_abbrev(row["home_team"]),
            "away_abbrev": get_abbrev(row["away_team"]),
            "home_logo": get_logo_url(row["home_team"]),
            "away_logo": get_logo_url(row["away_team"]),
            "home_espn_id": get_espn_id(row["home_team"]),
            "away_espn_id": get_espn_id(row["away_team"]),
            "commence_time": row.get("commence_time", ""),
            "spread_home": float(spread) if pd.notna(spread) else None,
            "total": float(row["total"]) if pd.notna(row.get("total")) else None,
            "pred_margin": round(float(row["pred_margin"]), 1),
            "edge": round(float(edge), 1),
            "pick_team": pick_team,
            "pick_abbrev": get_abbrev(pick_team),
            "pick_spread": pick_spread,
            "pick_logo": get_logo_url(pick_team),
        }

        # Embed KenPom profiles if ratings DataFrame is available
        if ratings is not None:
            home_kp = row.get("home_kp", "")
            away_kp = row.get("away_kp", "")
            game["home_kenpom"] = _kenpom_profile(row["home_team"], home_kp, ratings)
            game["away_kenpom"] = _kenpom_profile(row["away_team"], away_kp, ratings)

        games.append(game)

    # Sort by absolute edge descending
    games.sort(key=lambda g: abs(g["edge"]), reverse=True)

    # Build all_games list (every game, including those without model data)
    all_games = []
    for _, row in pred_df.iterrows():
        spread = row.get("spread_home")
        edge = row.get("edge")
        has_pred = pd.notna(edge)

        game = {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_abbrev": get_abbrev(row["home_team"]),
            "away_abbrev": get_abbrev(row["away_team"]),
            "home_logo": get_logo_url(row["home_team"]),
            "away_logo": get_logo_url(row["away_team"]),
            "commence_time": row.get("commence_time", ""),
            "spread_home": float(spread) if pd.notna(spread) else None,
            "total": float(row["total"]) if pd.notna(row.get("total")) else None,
            "pred_margin": round(float(row["pred_margin"]), 1) if has_pred else None,
            "edge": round(float(edge), 1) if has_pred else None,
        }

        # Add pick info if this game qualifies
        if has_pred and abs(edge) >= threshold and abs(edge) <= MAX_EDGE:
            if edge > 0:
                pick_team = row["home_team"]
                pick_spread = f"{spread:+.1f}" if pd.notna(spread) else ""
            else:
                pick_team = row["away_team"]
                pick_spread = f"{-spread:+.1f}" if pd.notna(spread) else ""
            game["pick_team"] = pick_team
            game["pick_abbrev"] = get_abbrev(pick_team)
            game["pick_spread"] = pick_spread
            game["pick_logo"] = get_logo_url(pick_team)

        if ratings is not None:
            home_kp = row.get("home_kp", "")
            away_kp = row.get("away_kp", "")
            game["home_kenpom"] = _kenpom_profile(row["home_team"], home_kp, ratings)
            game["away_kenpom"] = _kenpom_profile(row["away_team"], away_kp, ratings)

        all_games.append(game)

    return {
        "date": str(date.today()),
        "generated": datetime.now().isoformat(),
        "threshold": threshold,
        "total_games": len(pred_df),
        "picks_count": len(games),
        "games": games,
        "all_games": all_games,
    }


def display_predictions(pred_df: pd.DataFrame, threshold: float) -> None:
    """Display predictions in a formatted table."""
    table = Table(title=f"CBB Predictions - {date.today()}")
    table.add_column("Away", style="cyan")
    table.add_column("Home", style="green")
    table.add_column("Spread", justify="right")
    table.add_column("Pred", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Pick", style="bold")

    for _, row in pred_df.iterrows():
        spread_str = f"{row['spread_home']:+.1f}" if pd.notna(row["spread_home"]) else "N/A"
        margin_str = f"{row['pred_margin']:.1f}" if pd.notna(row["pred_margin"]) else "N/A"
        edge_str = f"{row['edge']:+.1f}" if pd.notna(row["edge"]) else "N/A"

        if pd.notna(row["edge"]) and abs(row["edge"]) >= threshold:
            edge_str = f"[bold]{edge_str}[/bold]"

        table.add_row(row["away_team"], row["home_team"], spread_str, margin_str, edge_str, row["pick"])

    console.print(table)

    actionable = pred_df[pred_df["edge"].abs() >= threshold] if "edge" in pred_df else pd.DataFrame()
    console.print(f"\n[bold]Total games:[/bold] {len(pred_df)}")
    console.print(f"[bold]Picks (|edge| >= {threshold}):[/bold] {len(actionable)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate daily CBB predictions")
    parser.add_argument("--threshold", type=float, default=None, help="Edge threshold (default: from model)")
    parser.add_argument("--save", action="store_true", help="Save predictions CSV")
    parser.add_argument("--html", action="store_true", help="Generate HTML page")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "reports" / "models"
    predictions_dir = project_root / "reports" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load .env
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        console.print("[red]Error: ODDS_API_KEY not found[/red]")
        return

    # Load model
    model_data = load_model(models_dir)
    threshold = args.threshold or model_data.get("threshold", 4.5)

    # Fetch data
    ratings = fetch_kenpom_live()

    console.print("[bold]Fetching games...[/bold]")
    games_data = fetch_odds_api_games(api_key)
    if not games_data:
        console.print("[yellow]No games found[/yellow]")
        return

    games_df = parse_games(games_data)
    console.print(f"  {len(games_df)} games")

    # Load team name map
    team_map = {}
    for name in ["team_name_map_v2.csv", "team_name_map.csv"]:
        path = project_root / "data" / "processed" / name
        if path.exists():
            map_df = pd.read_csv(path)
            col = "raw_name" if "raw_name" in map_df.columns else "team_name"
            team_map = dict(zip(map_df[col], map_df["kenpom_name"]))
            break

    # Predict
    pred_df = generate_predictions(ratings, games_df, team_map, model_data, threshold)
    display_predictions(pred_df, threshold)

    if args.save:
        out = predictions_dir / f"predictions_{date.today()}.csv"
        pred_df.to_csv(out, index=False)
        console.print(f"[green]Saved: {out}[/green]")

    if args.json:
        result = predictions_to_json(pred_df, threshold, ratings)
        # Save to docs/data for GitHub Pages
        json_dir = project_root / "docs" / "data"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / "picks.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]Saved JSON: {json_path}[/green]")

    if args.html:
        from src.cbb.generate_html import generate_html_page
        docs_dir = project_root / "docs"
        output_path = docs_dir / "picks.html"
        generate_html_page(pred_df, str(date.today()), output_path, threshold)
        console.print(f"[green]Generated: {output_path}[/green]")


if __name__ == "__main__":
    main()
