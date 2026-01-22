# oddsshark_ncaab_all_teams_multi_season.py
import argparse
import re
import sys
import time
from typing import List, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.oddsshark.com"
INDEX_URL = BASE + "/ncaab/game-logs"  # contains all team gamelog links in page source
GAMELOG_URL = BASE + "/stats/gamelog/basketball/ncaab/{team_id}"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

EXPECTED_COLS = [
    "team_id",
    "team_name",
    "season",
    "date",
    "opponent",
    "location",
    "game",
    "result",
    "winner_score",
    "loser_score",
    "ats",
    "spread",
    "ou",
    "total",
]


def fetch(url: str, params: dict | None = None) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def extract_team_ids_from_index(html: str) -> List[int]:
    """
    Pull team ids from any link that matches:
    /stats/gamelog/basketball/ncaab/{id}
    """
    # Use regex against the raw html; it's robust even if markup changes.
    ids = set(
        int(x) for x in re.findall(r"/stats/gamelog/basketball/ncaab/(\d+)", html)
    )
    return sorted(ids)


def parse_team_name_from_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    # "Villanova Wildcats - ..." -> "Villanova Wildcats"
    name = title.split(" - ")[0].strip() if " - " in title else title.strip()
    return name or "Unknown Team"


def parse_location_and_opponent(raw: str) -> Tuple[str, str]:
    s = (raw or "").strip()
    if s.startswith("@"):
        return s.lstrip("@").strip(), "Away"
    if s.lower().startswith("vs "):
        return s[3:].strip(), "Home"
    return s, "Neutral"


def parse_winner_loser(score: str) -> Tuple[int | None, int | None]:
    if not isinstance(score, str):
        return None, None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", score)
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a >= b else (b, a)


def parse_game_log_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    if not tables:
        raise ValueError("No tables found on gamelog page.")
    df = tables[0].copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def scrape_one_team(team_id: int, season_label: str, season_param: int) -> pd.DataFrame:
    url = GAMELOG_URL.format(team_id=team_id)
    html = fetch(url, params={"season": season_param})

    team_name = parse_team_name_from_title(html)
    df = parse_game_log_table(html)

    required = [
        "date",
        "opponent",
        "game",
        "result",
        "score",
        "ats",
        "spread",
        "ou",
        "total",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # opponent + location
    opp_loc = df["opponent"].apply(parse_location_and_opponent)
    df["opponent"] = opp_loc.apply(lambda x: x[0])
    df.insert(
        df.columns.get_loc("opponent") + 1, "location", opp_loc.apply(lambda x: x[1])
    )

    # winner/loser scores
    wl = df["score"].apply(parse_winner_loser)
    df.insert(
        df.columns.get_loc("result") + 1, "winner_score", wl.apply(lambda x: x[0])
    )
    df.insert(df.columns.get_loc("result") + 2, "loser_score", wl.apply(lambda x: x[1]))

    # identifiers
    df.insert(0, "team_id", team_id)
    df.insert(1, "team_name", team_name)
    df.insert(2, "season", season_label)

    # drop original score
    df = df.drop(columns=["score"])

    return df[EXPECTED_COLS]


def season_label_from_param(season_param: int) -> str:
    # season=2025 -> "2025-26"
    return f"{season_param}-{str(season_param + 1)[-2:]}"


def output_year_from_param(season_param: int) -> int:
    # season=2025 -> 2026 (your file naming preference)
    return season_param + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seasons",
        type=str,
        required=True,
        help="Comma-separated season params, e.g. '2025,2024,2023'",
    )
    ap.add_argument(
        "--sleep", type=float, default=0.6, help="Delay between requests (seconds)"
    )
    ap.add_argument(
        "--max-teams", type=int, default=0, help="0=all teams; set small for testing"
    )
    ap.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = ap.parse_args()

    season_params = [int(x) for x in re.split(r"[,\s]+", args.seasons.strip()) if x]
    if not season_params:
        raise SystemExit("No seasons provided.")

    # 1) discover all team ids
    index_html = fetch(INDEX_URL)
    team_ids = extract_team_ids_from_index(index_html)

    if args.max_teams and args.max_teams > 0:
        team_ids = team_ids[: args.max_teams]

    print(f"Found {len(team_ids)} team ids from {INDEX_URL}")

    # 2) loop seasons -> write 1 CSV per season
    for sp in season_params:
        season_label = season_label_from_param(sp)
        out_year = output_year_from_param(sp)
        out_path = f"{args.out_dir.rstrip('/')}/{out_year}_ncaab.csv"

        frames = []
        failures = 0

        print(f"\n=== Season {season_label} (season={sp}) ===")

        for i, team_id in enumerate(team_ids, start=1):
            try:
                df_team = scrape_one_team(team_id, season_label, sp)
                frames.append(df_team)
                print(f"[{i}/{len(team_ids)}] OK team_id={team_id} rows={len(df_team)}")
            except Exception as e:
                failures += 1
                print(
                    f"[{i}/{len(team_ids)}] FAIL team_id={team_id}: {e}",
                    file=sys.stderr,
                )
            time.sleep(args.sleep)

        if not frames:
            print(f"WARNING: no data scraped for season={sp}", file=sys.stderr)
            continue

        out_df = pd.concat(frames, ignore_index=True)
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {len(out_df)} rows to {out_path} (failures={failures})")


if __name__ == "__main__":
    main()
