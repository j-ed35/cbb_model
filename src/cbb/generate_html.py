"""
Generate static HTML picks page (fallback for local/offline use).

The primary picks page (docs/picks.html) fetches from the Vercel API.
This generates a self-contained static version with embedded data.

Usage:
    python -m src.cbb.generate_html [--date YYYY-MM-DD]
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.cbb.utils.espn_ids import get_logo_url, get_abbrev


def generate_html_page(
    pred_df: pd.DataFrame,
    pred_date: str,
    output_path: Path,
    threshold: float = 4.5,
) -> None:
    """Generate a static HTML page with embedded prediction data."""

    games = []
    for _, row in pred_df.iterrows():
        edge = row.get("edge")
        if pd.isna(edge) or abs(edge) < threshold:
            continue

        spread = row.get("spread_home")
        if edge > 0:
            pick_team = row["home_team"]
            pick_spread = f"{spread:+.1f}" if pd.notna(spread) else ""
        else:
            pick_team = row["away_team"]
            pick_spread = f"{-spread:+.1f}" if pd.notna(spread) else ""

        games.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_abbrev": get_abbrev(row["home_team"]),
            "away_abbrev": get_abbrev(row["away_team"]),
            "home_logo": get_logo_url(row["home_team"]),
            "away_logo": get_logo_url(row["away_team"]),
            "commence_time": row.get("commence_time", ""),
            "spread_home": float(spread) if pd.notna(spread) else None,
            "total": float(row["total"]) if pd.notna(row.get("total")) else None,
            "pred_margin": round(float(row["pred_margin"]), 1),
            "edge": round(float(edge), 1),
            "pick_team": pick_team,
            "pick_abbrev": get_abbrev(pick_team),
            "pick_spread": pick_spread,
            "pick_logo": get_logo_url(pick_team),
        })

    games.sort(key=lambda g: abs(g["edge"]), reverse=True)

    data = {
        "date": pred_date,
        "generated": datetime.now().isoformat(),
        "threshold": threshold,
        "total_games": len(pred_df),
        "picks_count": len(games),
        "games": games,
    }

    data_json = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CBB Model Picks - {pred_date}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f2f5; color: #333;
    }}
    .header {{ background: #1a1a2e; color: white; padding: 20px 0; text-align: center; }}
    .header h1 {{ font-size: 24px; font-weight: 700; letter-spacing: 1px; }}
    .header .subtitle {{ font-size: 13px; color: #8899aa; margin-top: 4px; }}
    .container {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
    .summary-bar {{
      display: flex; justify-content: space-between; align-items: center;
      background: white; border-radius: 8px; padding: 12px 20px;
      margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 14px;
    }}
    .summary-bar .count {{ font-weight: 700; color: #1a1a2e; }}
    .summary-bar .date {{ color: #666; }}
    .table-header {{
      display: grid; grid-template-columns: 1fr 200px 220px;
      background: #e8ecf0; border-radius: 8px 8px 0 0;
      padding: 10px 16px; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 1px; color: #667;
    }}
    .table-header .col-odds {{ text-align: center; }}
    .table-header .col-pick {{ text-align: center; }}
    .game-card {{
      display: grid; grid-template-columns: 1fr 200px 220px;
      background: white; border-bottom: 1px solid #e8ecf0;
      padding: 12px 16px; align-items: center;
    }}
    .game-card:last-child {{ border-radius: 0 0 8px 8px; border-bottom: none; }}
    .game-card:hover {{ background: #f8f9fa; }}
    .game-info {{ display: flex; flex-direction: column; gap: 6px; }}
    .team-row {{ display: flex; align-items: center; gap: 10px; }}
    .team-logo {{ width: 32px; height: 32px; border-radius: 4px; object-fit: contain; background: #f5f5f5; }}
    .team-name {{ font-size: 15px; font-weight: 600; }}
    .game-time {{ font-size: 12px; color: #888; margin-left: 42px; }}
    .odds-col {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
    .odds-total, .odds-spread {{
      font-size: 14px; font-weight: 600; color: #444;
      background: #f0f2f5; border-radius: 4px; padding: 2px 10px; min-width: 70px; text-align: center;
    }}
    .pick-col {{
      display: flex; align-items: center; justify-content: center; gap: 10px;
      background: #f0f7ff; border-radius: 8px; padding: 8px 12px; border: 2px solid #c4ddff;
    }}
    .pick-col.strong {{ background: #e8f5e8; border-color: #86d486; }}
    .pick-logo {{ width: 28px; height: 28px; object-fit: contain; }}
    .pick-text {{ font-size: 16px; font-weight: 700; color: #1a1a2e; }}
    .no-picks {{ text-align: center; padding: 40px; color: #888; font-size: 16px; }}
    .footer {{ text-align: center; padding: 24px; font-size: 12px; color: #999; }}
    @media (max-width: 700px) {{
      .table-header, .game-card {{ grid-template-columns: 1fr 100px 140px; }}
      .team-name {{ max-width: 120px; font-size: 13px; }}
      .pick-text {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>CBB MODEL PICKS</h1>
    <div class="subtitle">Against the Spread</div>
  </div>
  <div class="container">
    <div class="summary-bar">
      <span class="count" id="summary-count"></span>
      <span class="date" id="summary-date"></span>
    </div>
    <div class="table-header">
      <div>Game Info</div>
      <div class="col-odds">Current Odds</div>
      <div class="col-pick">Model Pick</div>
    </div>
    <div id="games-container"></div>
  </div>
  <div class="footer">
    Model: GBM trained on KenPom ratings (2022-2026) | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>
  <script>
    const DATA = {data_json};

    function fallbackLogo(img) {{ img.style.display = 'none'; }}

    function formatTime(isoStr) {{
      if (!isoStr) return '';
      const d = new Date(isoStr);
      return d.toLocaleTimeString('en-US', {{ hour: 'numeric', minute: '2-digit' }});
    }}

    document.getElementById('summary-date').textContent = DATA.date;
    document.getElementById('summary-count').textContent =
      DATA.picks_count + ' picks from ' + DATA.total_games + ' games';

    const container = document.getElementById('games-container');
    if (!DATA.games.length) {{
      container.innerHTML = '<div class="no-picks">No model picks today.</div>';
    }} else {{
      let html = '';
      for (const g of DATA.games) {{
        const isStrong = Math.abs(g.edge) >= 7;
        const pickClass = isStrong ? 'pick-col strong' : 'pick-col';
        const spreadStr = g.spread_home != null ? (g.spread_home > 0 ? '+' + g.spread_home : '' + g.spread_home) : '';
        const totalStr = g.total != null ? 'o' + g.total : '';
        html += `
          <div class="game-card">
            <div class="game-info">
              <div class="team-row">
                <img class="team-logo" src="${{g.away_logo}}" alt="" onerror="fallbackLogo(this)">
                <span class="team-name">${{g.away_team.toUpperCase()}}</span>
              </div>
              <div class="team-row">
                <img class="team-logo" src="${{g.home_logo}}" alt="" onerror="fallbackLogo(this)">
                <span class="team-name">${{g.home_team.toUpperCase()}}</span>
              </div>
              <div class="game-time">${{formatTime(g.commence_time)}}</div>
            </div>
            <div class="odds-col">
              ${{totalStr ? `<div class="odds-total">${{totalStr}}</div>` : ''}}
              ${{spreadStr ? `<div class="odds-spread">${{spreadStr}}</div>` : ''}}
            </div>
            <div class="${{pickClass}}">
              <img class="pick-logo" src="${{g.pick_logo}}" alt="" onerror="fallbackLogo(this)">
              <span class="pick-text">${{g.pick_abbrev}} ${{g.pick_spread}}</span>
            </div>
          </div>`;
      }}
      container.innerHTML = html;
    }}
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Generated: {output_path}")
