"""
Vercel serverless function for CBB predictions.

Endpoint: GET /api/predictions
Returns: JSON with today's model picks

Environment variables (set in Vercel dashboard):
    KENPOM_EMAIL - KenPom login email
    KENPOM_PW - KenPom login password
    ODDS_API_KEY - The Odds API key
"""

import json
import os
import pickle
import sys
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Features must match training
FEATURES = [
    "kp_adj_em_a", "kp_adj_em_b",
    "kp_adj_o_a", "kp_adj_o_b",
    "kp_adj_d_a", "kp_adj_d_b",
    "kp_tempo_a", "kp_tempo_b",
    "kp_adj_em_diff",
    "kp_tempo_avg", "kp_tempo_diff",
    "kp_o_vs_d_a", "kp_o_vs_d_b",
    "is_home_a", "is_neutral",
]


def load_model():
    model_path = PROJECT_ROOT / "reports" / "models" / "gbm.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def fetch_kenpom():
    from io import StringIO

    email = os.environ.get("KENPOM_EMAIL", "")
    pw = os.environ.get("KENPOM_PW", "")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    # Login
    session.get("https://kenpom.com/index.php", timeout=15)
    session.post(
        "https://kenpom.com/handlers/login_handler.php",
        data={"email": email, "password": pw, "submit": "Login!"},
        allow_redirects=True,
        timeout=15,
    )

    # Verify login
    home = session.get("https://kenpom.com/", timeout=15)
    if "Logged in as" not in home.text:
        raise Exception("KenPom login failed - check credentials")

    # Parse ratings table
    from bs4 import BeautifulSoup
    page = BeautifulSoup(home.text, "html.parser")
    table = page.find_all("table")[0]
    ratings_df = pd.read_html(StringIO(str(table)))[0]
    ratings_df.columns = ratings_df.columns.map(lambda x: x[1])
    ratings_df.dropna(inplace=True)
    ratings_df = ratings_df[ratings_df["Rk"] != "Rk"]
    ratings_df.reset_index(drop=True, inplace=True)

    # Parse team name (strip seed)
    tmp = ratings_df["Team"].str.extract(r"(?P<Team>[a-zA-Z.&'\s]+(?<!\s))\s*(?P<Seed>\d*)")
    ratings_df["Team"] = tmp["Team"]

    ratings_df.columns = [
        "Rk", "Team", "Conf", "W-L", "AdjEM", "AdjO", "AdjO.Rank",
        "AdjD", "AdjD.Rank", "AdjT", "AdjT.Rank", "Luck", "Luck.Rank",
        "SOS-AdjEM", "SOS-AdjEM.Rank", "SOS-OppO", "SOS-OppO.Rank",
        "SOS-OppD", "SOS-OppD.Rank", "NCSOS-AdjEM", "NCSOS-AdjEM.Rank", "Seed",
    ]

    ratings_df = ratings_df.rename(columns={
        "Team": "TeamName", "AdjO": "AdjOE", "AdjD": "AdjDE", "AdjT": "AdjTempo",
    })
    return ratings_df


def fetch_games():
    api_key = os.environ.get("ODDS_API_KEY", "")
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    records = []
    for game in resp.json():
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")
        spread_home = total = None
        for bk in game.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        if o["name"] == home:
                            spread_home = o.get("point")
                if mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            total = o.get("point")
            if spread_home is not None:
                break
        records.append({
            "home_team": home, "away_team": away,
            "commence_time": commence,
            "spread_home": spread_home, "total": total,
        })
    return pd.DataFrame(records)


def map_team(name, kenpom_teams, team_map):
    if name in kenpom_teams:
        return name
    if name in team_map:
        return team_map[name]
    lower = name.lower()
    for kp in kenpom_teams:
        if kp.lower() == lower:
            return kp
    words = set(lower.split())
    best, best_score = None, 0
    for kp in kenpom_teams:
        overlap = len(words & set(kp.lower().split()))
        if overlap > best_score and overlap >= 1:
            best_score = overlap
            best = kp
    return best


def extract_features(home_kp, away_kp, ratings):
    col = "TeamName" if "TeamName" in ratings.columns else "Team"
    ra = ratings[ratings[col] == home_kp]
    rb = ratings[ratings[col] == away_kp]
    if len(ra) == 0 or len(rb) == 0:
        return None
    ra, rb = ra.iloc[0], rb.iloc[0]

    def f(row, key):
        try:
            return float(row.get(key, np.nan))
        except (ValueError, TypeError):
            return np.nan

    em_a, em_b = f(ra, "AdjEM"), f(rb, "AdjEM")
    o_a, o_b = f(ra, "AdjOE"), f(rb, "AdjOE")
    d_a, d_b = f(ra, "AdjDE"), f(rb, "AdjDE")
    t_a, t_b = f(ra, "AdjTempo"), f(rb, "AdjTempo")

    return {
        "kp_adj_em_a": em_a, "kp_adj_em_b": em_b,
        "kp_adj_o_a": o_a, "kp_adj_o_b": o_b,
        "kp_adj_d_a": d_a, "kp_adj_d_b": d_b,
        "kp_tempo_a": t_a, "kp_tempo_b": t_b,
        "kp_adj_em_diff": em_a - em_b if not (np.isnan(em_a) or np.isnan(em_b)) else 0,
        "kp_tempo_avg": (t_a + t_b) / 2 if not (np.isnan(t_a) or np.isnan(t_b)) else 67.5,
        "kp_tempo_diff": t_a - t_b if not (np.isnan(t_a) or np.isnan(t_b)) else 0,
        "kp_o_vs_d_a": o_a - d_b if not (np.isnan(o_a) or np.isnan(d_b)) else 0,
        "kp_o_vs_d_b": o_b - d_a if not (np.isnan(o_b) or np.isnan(d_a)) else 0,
        "is_home_a": 1, "is_neutral": 0,
    }


def get_predictions():
    from src.cbb.utils.espn_ids import get_logo_url, get_abbrev

    model_data = load_model()
    if model_data is None:
        return {"error": "Model not found"}

    gbm = model_data["model"]
    feature_cols = model_data["feature_cols"]
    threshold = model_data.get("threshold", 4.5)

    ratings = fetch_kenpom()
    games_df = fetch_games()

    # Load team map
    team_map = {}
    map_path = PROJECT_ROOT / "data" / "processed" / "team_name_map_v2.csv"
    if map_path.exists():
        m = pd.read_csv(map_path)
        col = "raw_name" if "raw_name" in m.columns else "team_name"
        team_map = dict(zip(m[col], m["kenpom_name"]))

    kenpom_teams = set(ratings["TeamName"].unique())

    games = []
    for _, row in games_df.iterrows():
        home_kp = map_team(row["home_team"], kenpom_teams, team_map)
        away_kp = map_team(row["away_team"], kenpom_teams, team_map)
        if not home_kp or not away_kp:
            continue

        feats = extract_features(home_kp, away_kp, ratings)
        if feats is None:
            continue

        X = np.array([[feats.get(c, 0) for c in feature_cols]])
        X = np.nan_to_num(X, nan=0)
        pred = gbm.predict(X)[0]

        spread = pd.to_numeric(row["spread_home"], errors="coerce")
        if pd.isna(spread):
            continue

        edge = pred - (-spread)
        if abs(edge) < threshold:
            continue

        if edge > 0:
            pick_team = row["home_team"]
            pick_spread = f"{spread:+.1f}"
        else:
            pick_team = row["away_team"]
            pick_spread = f"{-spread:+.1f}"

        games.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_abbrev": get_abbrev(row["home_team"]),
            "away_abbrev": get_abbrev(row["away_team"]),
            "home_logo": get_logo_url(row["home_team"]),
            "away_logo": get_logo_url(row["away_team"]),
            "commence_time": row.get("commence_time", ""),
            "spread_home": float(spread),
            "total": float(row["total"]) if pd.notna(row.get("total")) else None,
            "pred_margin": round(float(pred), 1),
            "edge": round(float(edge), 1),
            "pick_team": pick_team,
            "pick_abbrev": get_abbrev(pick_team),
            "pick_spread": pick_spread,
            "pick_logo": get_logo_url(pick_team),
        })

    games.sort(key=lambda g: abs(g["edge"]), reverse=True)

    return {
        "date": str(date.today()),
        "generated": datetime.now().isoformat(),
        "threshold": threshold,
        "total_games": len(games_df),
        "picks_count": len(games),
        "games": games,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        import traceback
        try:
            result = get_predictions()
            self.send_response(200)
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()}
            self.send_response(500)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
