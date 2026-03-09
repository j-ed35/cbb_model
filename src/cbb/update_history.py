"""
update_history.py
─────────────────
Appends today's model picks to docs/data/picks_history.json and attempts
to fill in ATS results for any pending picks by looking them up in the
raw season CSV files (data/raw/*_ncaab.csv).

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

PICKS_JSON = Path("docs/data/picks.json")
HISTORY_JSON = Path("docs/data/picks_history.json")
RAW_DIR = Path("data/raw")


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


# ── CSV result lookup ──────────────────────────────────────────────────────────

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


# ── history management ─────────────────────────────────────────────────────────

def add_today_picks(history: dict, picks_data: dict) -> bool:
    """Add today's picks to history as pending. Returns True if new picks added."""
    date = picks_data.get("date")
    if not date:
        print("  picks.json has no 'date' field — skipping.", file=sys.stderr)
        return False

    # Skip if this date is already in history
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
                "pick_team": g.get("pick_team", ""),
                "pick_abbrev": g.get("pick_abbrev", ""),
                "pick_spread": g.get("pick_spread", ""),
                "edge": g.get("edge"),
                "commence_time": g.get("commence_time", ""),
                "result": None,  # pending
            }
            for g in games
        ],
    }

    history["days"].append(day_entry)
    history["days"].sort(key=lambda d: d["date"], reverse=True)
    print(f"  Added {len(day_entry['picks'])} picks for {date}.")
    return True


def lookup_results(history: dict) -> int:
    """
    Try to fill in results for pending picks from raw CSV data.
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

    print(f"  Resolving {len(pending_picks)} pending picks from CSV data…")
    lookup, all_csv_teams = build_csv_lookup()

    if not lookup:
        print("  No CSV data found — cannot resolve results.")
        return 0

    resolved = 0
    for day, pick in pending_picks:
        pick_team = pick.get("pick_team", "")
        # Try the ET calendar date first, then the stored day date
        commence = pick.get("commence_time", "")
        dates_to_try = list({
            day["date"],
            commence_to_et_date(commence) if commence else day["date"],
        })

        # Fuzzy-match pick_team against known CSV team names
        matches = get_close_matches(pick_team, all_csv_teams, n=1, cutoff=0.72)
        if not matches:
            # Try normalized form
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
            pick["result"] = ats  # 'W' means the pick team covered
            resolved += 1

    print(f"  Resolved {resolved} results.")
    return resolved


def recalculate_overall(history: dict):
    wins = losses = pending = 0
    for day in history["days"]:
        for pick in day["picks"]:
            r = pick.get("result")
            if r == "W":
                wins += 1
            elif r == "L":
                losses += 1
            else:
                pending += 1
    history["overall"] = {"wins": wins, "losses": losses, "pending": pending}
    history["last_updated"] = datetime.now(timezone.utc).isoformat()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not PICKS_JSON.exists():
        print(f"ERROR: {PICKS_JSON} not found. Run predict_daily first.", file=sys.stderr)
        sys.exit(1)

    picks_data = load_json(PICKS_JSON, {})
    history = load_json(HISTORY_JSON, {"overall": {"wins": 0, "losses": 0, "pending": 0}, "days": []})

    print("Adding today's picks to history…")
    add_today_picks(history, picks_data)

    print("Looking up ATS results for pending picks…")
    lookup_results(history)

    recalculate_overall(history)
    save_json(HISTORY_JSON, history)

    o = history["overall"]
    total = o["wins"] + o["losses"]
    pct = o["wins"] / total if total > 0 else 0.0
    print(
        f"\nHistory saved → "
        f"{o['wins']}-{o['losses']} ({pct:.1%}) | "
        f"{o['pending']} pending | "
        f"{len(history['days'])} days tracked"
    )


if __name__ == "__main__":
    main()
