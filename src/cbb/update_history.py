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

def _game_date(pick: dict, day_date: str) -> str:
    """Return the ET calendar date the game is actually played on."""
    commence = pick.get("commence_time", "")
    if commence:
        return commence_to_et_date(commence)
    return day_date


def resolve_via_espn(history: dict) -> int:
    """
    Resolve pending picks by fetching scores from ESPN's scoreboard API.
    Returns number of results newly resolved.
    """
    now = datetime.now(timezone.utc)
    pending_picks = []
    skipped_future = 0
    for day in history["days"]:
        for pick in day["picks"]:
            if pick.get("result") is not None:
                continue
            # Skip games that haven't started yet
            commence = pick.get("commence_time", "")
            if commence:
                try:
                    game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if game_dt > now:
                        skipped_future += 1
                        continue
                except Exception:
                    pass
            pending_picks.append((day, pick))

    if not pending_picks:
        if skipped_future:
            print(f"  {skipped_future} picks skipped (games not started yet).")
        print("  No completed picks to resolve.")
        return 0

    print(f"  Resolving {len(pending_picks)} pending picks via ESPN…")
    if skipped_future:
        print(f"  ({skipped_future} future games skipped)")

    # Collect unique game dates (from commence_time, not day["date"])
    dates_needed: set[str] = set()
    for day, pick in pending_picks:
        dates_needed.add(_game_date(pick, day["date"]))

    # Fetch ESPN scoreboards keyed by (date, team_id)
    scores_by_date: dict[str, dict[int, dict]] = {}
    for d in sorted(dates_needed):
        print(f"    Fetching ESPN scores for {d}…")
        scores_by_date[d] = fetch_espn_scores(d)

    total_games = sum(len(s) // 2 for s in scores_by_date.values())
    if total_games == 0:
        print("  No ESPN scores available.")
        return 0

    print(f"    {total_games} completed games found.")

    resolved = 0
    for day, pick in pending_picks:
        game_dt = _game_date(pick, day["date"])
        day_scores = scores_by_date.get(game_dt, {})
        if not day_scores:
            continue

        # Try to match using ESPN IDs stored in the pick
        home_espn_id = pick.get("home_espn_id")
        away_espn_id = pick.get("away_espn_id")

        score_entry = None
        if home_espn_id and home_espn_id in day_scores:
            score_entry = day_scores[home_espn_id]
        elif away_espn_id and away_espn_id in day_scores:
            score_entry = day_scores[away_espn_id]
        else:
            # Fallback: look up ESPN IDs from team names
            home_id = get_espn_id(pick.get("home_team", ""))
            away_id = get_espn_id(pick.get("away_team", ""))
            if home_id and home_id in day_scores:
                score_entry = day_scores[home_id]
                pick["home_espn_id"] = home_id
            elif away_id and away_id in day_scores:
                score_entry = day_scores[away_id]
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

def _existing_game_keys(history: dict) -> set[tuple[str, str, str]]:
    """Return set of (commence_time, home_team, away_team) already in history."""
    keys: set[tuple[str, str, str]] = set()
    for day in history["days"]:
        for p in day["picks"]:
            keys.add((
                p.get("commence_time", ""),
                p.get("home_team", ""),
                p.get("away_team", ""),
            ))
    return keys


def _today_et() -> str:
    """Return today's date string in Eastern Time."""
    return (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")


def _make_pick_entry(g: dict) -> dict:
    return {
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


def add_past_picks(history: dict, picks_data: dict) -> bool:
    """
    Add picks from picks_data to history, filed under their actual game date
    (ET).  Only adds games whose game date is strictly before today so that
    picks don't appear in History until the day after they're played.
    Returns True if any new picks were added.
    """
    today = _today_et()
    games = picks_data.get("games", [])
    if not games:
        print("  No games in picks.json.")
        return False

    existing_keys = _existing_game_keys(history)
    skipped_future = 0
    skipped_dup = 0
    games_by_date: dict[str, list[dict]] = {}

    for g in games:
        commence = g.get("commence_time", "")
        game_date = commence_to_et_date(commence) if commence else picks_data.get("date", "")

        # Only add games that have already been played (game date < today)
        if game_date >= today:
            skipped_future += 1
            continue

        key = (commence, g.get("home_team", ""), g.get("away_team", ""))
        if key in existing_keys:
            skipped_dup += 1
            continue

        games_by_date.setdefault(game_date, []).append(g)

    if skipped_future:
        print(f"  Skipped {skipped_future} picks for today/future dates.")
    if skipped_dup:
        print(f"  Skipped {skipped_dup} picks already in history.")

    if not games_by_date:
        print("  No new past picks to add.")
        return False

    # Find or create day entries and add picks
    added = 0
    for game_date in sorted(games_by_date):
        day_entry = next((d for d in history["days"] if d["date"] == game_date), None)
        if day_entry is None:
            day_entry = {"date": game_date, "picks": []}
            history["days"].append(day_entry)

        for g in games_by_date[game_date]:
            day_entry["picks"].append(_make_pick_entry(g))
            added += 1

    history["days"].sort(key=lambda d: d["date"], reverse=True)
    print(f"  Added {added} past picks across {len(games_by_date)} day(s).")
    return True


def deduplicate_history(history: dict) -> int:
    """
    Remove duplicate picks across days.  A pick is a duplicate if the same
    (commence_time, home_team, away_team) appears in multiple day entries.

    Also removes picks from a day whose commence_time doesn't match that
    day's date (i.e. future-day games that snuck into history).

    Keeps the entry with a resolved result if one exists, otherwise keeps
    the one from the day closest to the actual game date.  Returns count
    of picks removed.
    """
    seen: dict[tuple[str, str, str], tuple[str, dict]] = {}  # key → (day_date, pick)
    removed = 0

    for day in history["days"]:
        keep = []
        for pick in day["picks"]:
            commence = pick.get("commence_time", "")
            game_date = commence_to_et_date(commence) if commence else day["date"]

            # Drop picks whose game date doesn't match the day they're filed under
            if game_date != day["date"]:
                removed += 1
                continue

            key = (commence, pick.get("home_team", ""), pick.get("away_team", ""))
            if key in seen:
                prev_day_date, prev_pick = seen[key]
                # Prefer the entry that has a resolved result
                if pick.get("result") is not None and prev_pick.get("result") is None:
                    # Replace previous with this one — remove previous from its day
                    for d in history["days"]:
                        if d["date"] == prev_day_date:
                            d["picks"] = [p for p in d["picks"] if p is not prev_pick]
                            break
                    seen[key] = (day["date"], pick)
                    keep.append(pick)
                else:
                    removed += 1
                    continue
            else:
                seen[key] = (day["date"], pick)
                keep.append(pick)
        day["picks"] = keep

    # Remove empty days
    history["days"] = [d for d in history["days"] if d["picks"]]

    if removed:
        print(f"  Deduplicated history: removed {removed} duplicate/misplaced picks.")
    return removed


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

    print("Adding past picks to history…")
    add_past_picks(history, picks_data)

    print("Deduplicating history…")
    deduplicate_history(history)

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
