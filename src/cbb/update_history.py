"""
update_history.py
─────────────────
Appends today's model picks to docs/data/picks_history.json and resolves
ATS results for pending picks by fetching final scores from ESPN's public
scoreboard API.  Falls back to raw season CSV files if ESPN is unavailable.

Usage:
    python -m src.cbb.update_history
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from difflib import get_close_matches
from pathlib import Path

import requests

from src.cbb.utils.espn_ids import get_espn_id

PICKS_JSON = Path("docs/data/picks.json")
HISTORY_JSON = Path("docs/data/picks_history.json")
RAW_DIR = Path("data/raw")

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates={date_str}&limit=400&groups=50"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def parse_csv_date(date_str: str) -> str | None:
    """Convert 'Nov 6, 2024' → '2024-11-06', or return None on failure."""
    date_str = date_str.strip().strip('"')
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def commence_to_et_date(iso_str: str) -> str:
    """
    Convert a UTC ISO commence time to the Eastern-time calendar date.
    Eastern is UTC-5 (EST) or UTC-4 (EDT). We use UTC-5 as a safe default
    for the college basketball season (Nov–Apr).
    """
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_et = dt - timedelta(hours=5)  # approximate ET offset
        return dt_et.strftime("%Y-%m-%d")
    except Exception:
        return iso_str[:10]  # fall back to raw date prefix


def normalize(name: str) -> str:
    """Lower-case, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


# ── ESPN score fetching ──────────────────────────────────────────────────────

def fetch_espn_scores(date_str: str) -> dict[int, dict]:
    """
    Fetch final scores from ESPN scoreboard for a given date.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        Dictionary mapping ESPN team IDs to score info:
        {espn_team_id: {team_id, team_name, score, opponent_id,
                        opponent_score, home_away, completed}}
    """
    espn_date = date_str.replace("-", "")
    url = ESPN_SCOREBOARD_URL.format(date_str=espn_date)

    try:
        resp = requests.get(url, headers={"User-Agent": "CBBModel/1.0"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        print(f"  Warning: ESPN fetch failed for {date_str}: {exc}", file=sys.stderr)
        return {}

    scores: dict[int, dict] = {}

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        parsed = []
        for c in competitors:
            try:
                team_id = int(c["team"]["id"])
                score = int(c.get("score", "0"))
                home_away = c.get("homeAway", "")
                team_name = c.get("team", {}).get("displayName", "")
                parsed.append({
                    "team_id": team_id,
                    "team_name": team_name,
                    "score": score,
                    "home_away": home_away,
                })
            except (KeyError, ValueError):
                continue

        if len(parsed) != 2:
            continue

        for i, p in enumerate(parsed):
            opp = parsed[1 - i]
            scores[p["team_id"]] = {
                "team_id": p["team_id"],
                "team_name": p["team_name"],
                "score": p["score"],
                "opponent_id": opp["team_id"],
                "opponent_score": opp["score"],
                "home_away": p["home_away"],
                "completed": True,
            }

    return scores


# ── ESPN-based result resolution ─────────────────────────────────────────────

def resolve_via_espn(history: dict) -> int:
    """
    Resolve pending picks by fetching scores from ESPN's scoreboard API.
    Returns number of results newly resolved.
    """
    pending_picks = [
        (day, pick)
        for day in history["days"]
        for pick in day["picks"]
        if pick.get("result") is None
    ]

    if not pending_picks:
        print("  No pending picks to resolve.")
        return 0

    print(f"  Resolving {len(pending_picks)} pending picks via ESPN…")

    # Collect unique dates we need to fetch
    dates_needed: set[str] = set()
    for day, pick in pending_picks:
        commence = pick.get("commence_time", "")
        if commence:
            dates_needed.add(commence_to_et_date(commence))
        dates_needed.add(day["date"])

    # Fetch ESPN scoreboards for all needed dates
    all_scores: dict[int, dict] = {}
    for d in sorted(dates_needed):
        print(f"    Fetching ESPN scores for {d}…")
        day_scores = fetch_espn_scores(d)
        all_scores.update(day_scores)

    if not all_scores:
        print("  No ESPN scores available.")
        return 0

    print(f"    {len(all_scores) // 2} completed games found.")

    resolved = 0
    for day, pick in pending_picks:
        # Try to match using ESPN IDs stored in the pick
        home_espn_id = pick.get("home_espn_id")
        away_espn_id = pick.get("away_espn_id")

        score_entry = None
        if home_espn_id and home_espn_id in all_scores:
            score_entry = all_scores[home_espn_id]
        elif away_espn_id and away_espn_id in all_scores:
            score_entry = all_scores[away_espn_id]
        else:
            # Fallback: look up ESPN IDs from team names
            home_id = get_espn_id(pick.get("home_team", ""))
            away_id = get_espn_id(pick.get("away_team", ""))
            if home_id and home_id in all_scores:
                score_entry = all_scores[home_id]
                pick["home_espn_id"] = home_id
            elif away_id and away_id in all_scores:
                score_entry = all_scores[away_id]
                pick["away_espn_id"] = away_id

        if not score_entry:
            continue

        # Determine home and away scores
        if score_entry["home_away"] == "home":
            home_score = score_entry["score"]
            away_score = score_entry["opponent_score"]
        else:
            home_score = score_entry["opponent_score"]
            away_score = score_entry["score"]

        # Pick team's margin
        pick_team = pick.get("pick_team", "")
        home_team = pick.get("home_team", "")
        if pick_team == home_team:
            pick_team_margin = home_score - away_score
        else:
            pick_team_margin = away_score - home_score

        # Cover margin: pick_team_margin + spread_value
        # pick_spread is already from the picked team's perspective
        try:
            spread_value = float(pick.get("pick_spread", "0"))
        except (ValueError, TypeError):
            spread_value = 0.0

        cover_margin = pick_team_margin + spread_value

        if cover_margin > 0:
            result = "W"
        elif cover_margin < 0:
            result = "L"
        else:
            result = "P"

        # Store results
        pick["home_score"] = home_score
        pick["away_score"] = away_score
        pick["cover_margin"] = round(cover_margin, 1)
        pick["result"] = result
        resolved += 1

    print(f"  Resolved {resolved} results via ESPN.")
    return resolved


# ── CSV result lookup (fallback) ─────────────────────────────────────────────

def build_csv_lookup() -> tuple[dict, list]:
    """
    Returns:
        lookup  : {(csv_team_name, 'YYYY-MM-DD'): 'W'|'L'}
        teams   : list of unique team names (for fuzzy matching)
    """
    lookup: dict[tuple[str, str], str] = {}

    for csv_path in sorted(RAW_DIR.glob("*_ncaab.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ats = row.get("ats", "").strip()
                    if ats not in ("W", "L"):
                        continue
                    team = row.get("team_name", "").strip()
                    date_key = parse_csv_date(row.get("date", ""))
                    if team and date_key:
                        lookup[(team, date_key)] = ats
        except Exception as exc:
            print(f"  Warning: could not read {csv_path}: {exc}", file=sys.stderr)

    teams = sorted({t for t, _ in lookup})
    return lookup, teams


def resolve_via_csv(history: dict) -> int:
    """
    Fallback: resolve remaining pending picks from raw CSV data.
    Returns number of results newly resolved.
    """
    pending_picks = [
        (day, pick)
        for day in history["days"]
        for pick in day["picks"]
        if pick.get("result") is None
    ]

    if not pending_picks:
        return 0

    print(f"  Resolving {len(pending_picks)} pending picks from CSV data…")
    lookup, all_csv_teams = build_csv_lookup()

    if not lookup:
        print("  No CSV data found — cannot resolve results.")
        return 0

    resolved = 0
    for day, pick in pending_picks:
        pick_team = pick.get("pick_team", "")
        commence = pick.get("commence_time", "")
        dates_to_try = list({
            day["date"],
            commence_to_et_date(commence) if commence else day["date"],
        })

        matches = get_close_matches(pick_team, all_csv_teams, n=1, cutoff=0.72)
        if not matches:
            norm_pick = normalize(pick_team)
            norm_teams = {normalize(t): t for t in all_csv_teams}
            norm_match = get_close_matches(norm_pick, list(norm_teams.keys()), n=1, cutoff=0.72)
            if norm_match:
                matches = [norm_teams[norm_match[0]]]

        if not matches:
            continue

        csv_team = matches[0]
        ats = None
        for date_key in dates_to_try:
            ats = lookup.get((csv_team, date_key))
            if ats:
                break

        if ats:
            pick["result"] = ats
            resolved += 1

    print(f"  Resolved {resolved} results from CSV.")
    return resolved


# ── history management ─────────────────────────────────────────────────────────

def add_today_picks(history: dict, picks_data: dict) -> bool:
    """Add today's picks to history as pending. Returns True if new picks added."""
    date = picks_data.get("date")
    if not date:
        print("  picks.json has no 'date' field — skipping.", file=sys.stderr)
        return False

    existing_dates = {d["date"] for d in history["days"]}
    if date in existing_dates:
        print(f"  {date} already in history — skipping add.")
        return False

    games = picks_data.get("games", [])
    if not games:
        print(f"  No picks for {date}.")
        return False

    day_entry = {
        "date": date,
        "picks": [
            {
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "home_abbrev": g.get("home_abbrev", ""),
                "away_abbrev": g.get("away_abbrev", ""),
                "home_logo": g.get("home_logo", ""),
                "away_logo": g.get("away_logo", ""),
                "home_espn_id": g.get("home_espn_id"),
                "away_espn_id": g.get("away_espn_id"),
                "pick_team": g.get("pick_team", ""),
                "pick_abbrev": g.get("pick_abbrev", ""),
                "pick_spread": g.get("pick_spread", ""),
                "spread_home": g.get("spread_home"),
                "pred_margin": g.get("pred_margin"),
                "edge": g.get("edge"),
                "commence_time": g.get("commence_time", ""),
                "result": None,
                "home_score": None,
                "away_score": None,
                "cover_margin": None,
            }
            for g in games
        ],
    }

    history["days"].append(day_entry)
    history["days"].sort(key=lambda d: d["date"], reverse=True)
    print(f"  Added {len(day_entry['picks'])} picks for {date}.")
    return True


def recalculate_overall(history: dict):
    wins = losses = pushes = pending = 0
    for day in history["days"]:
        for pick in day["picks"]:
            r = pick.get("result")
            if r == "W":
                wins += 1
            elif r == "L":
                losses += 1
            elif r == "P":
                pushes += 1
            else:
                pending += 1
    history["overall"] = {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending,
    }
    history["last_updated"] = datetime.now(timezone.utc).isoformat()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not PICKS_JSON.exists():
        print(f"ERROR: {PICKS_JSON} not found. Run predict_daily first.", file=sys.stderr)
        sys.exit(1)

    picks_data = load_json(PICKS_JSON, {})
    history = load_json(HISTORY_JSON, {
        "overall": {"wins": 0, "losses": 0, "pushes": 0, "pending": 0},
        "days": [],
    })

    print("Adding today's picks to history…")
    add_today_picks(history, picks_data)

    print("Resolving ATS results via ESPN…")
    espn_resolved = resolve_via_espn(history)

    # Fallback: try CSV for any remaining pending
    remaining = sum(
        1 for day in history["days"]
        for pick in day["picks"]
        if pick.get("result") is None
    )
    if remaining > 0:
        print(f"  {remaining} still pending, trying CSV fallback…")
        resolve_via_csv(history)

    recalculate_overall(history)
    save_json(HISTORY_JSON, history)

    o = history["overall"]
    total = o["wins"] + o["losses"]
    pct = o["wins"] / total if total > 0 else 0.0
    pushes_str = f"-{o['pushes']}" if o["pushes"] else ""
    print(
        f"\nHistory saved → "
        f"{o['wins']}-{o['losses']}{pushes_str} ({pct:.1%}) | "
        f"{o['pending']} pending | "
        f"{len(history['days'])} days tracked"
    )


if __name__ == "__main__":
    main()
