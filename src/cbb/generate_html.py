"""
Generate static HTML page for CBB picks.

Creates a Web 1.0 style HTML table with today's picks
and historical performance tracking.

Usage:
    python -m src.cbb.generate_html [--date YYYY-MM-DD]

Output:
    docs/picks.html - Daily picks page for GitHub Pages
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date
import argparse
from typing import Optional
import glob


def load_predictions(predictions_dir: Path, target_date: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Load predictions for a specific date or the most recent."""
    if target_date:
        pred_file = predictions_dir / f"predictions_{target_date}.csv"
        if not pred_file.exists():
            raise FileNotFoundError(f"No predictions found for {target_date}")
        return pd.read_csv(pred_file), target_date

    # Find most recent
    files = sorted(predictions_dir.glob("predictions_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError("No prediction files found")

    latest = files[0]
    date_str = latest.stem.replace("predictions_", "")
    return pd.read_csv(latest), date_str


def load_historical_predictions(predictions_dir: Path, limit: int = 7) -> list[tuple[str, pd.DataFrame]]:
    """Load historical predictions for the past N days."""
    files = sorted(predictions_dir.glob("predictions_*.csv"), reverse=True)

    historical = []
    for f in files[:limit]:
        date_str = f.stem.replace("predictions_", "")
        df = pd.read_csv(f)
        historical.append((date_str, df))

    return historical


def generate_picks_table(df: pd.DataFrame, threshold: float = 4.5) -> str:
    """Generate HTML table for actionable picks."""
    # Filter to actionable picks only
    df = df.copy()
    df['edge'] = pd.to_numeric(df['edge'], errors='coerce')
    actionable = df[df['edge'].abs() >= threshold].copy()

    if len(actionable) == 0:
        return "<p>No actionable picks today (edge threshold: {:.1f} pts)</p>".format(threshold)

    # Sort by absolute edge descending
    actionable['abs_edge'] = actionable['edge'].abs()
    actionable = actionable.sort_values('abs_edge', ascending=False)

    rows = []
    for _, row in actionable.iterrows():
        edge = row['edge']
        pick = row['pick']
        spread = row.get('spread_home', '')
        pred_margin = row.get('pred_margin', '')

        # Determine confidence stars (text-based for Web 1.0)
        stars = '*' * min(int(abs(edge) / 2), 3) if pd.notna(edge) else ''

        # Format values
        edge_str = f"{edge:+.1f}" if pd.notna(edge) else "N/A"
        margin_str = f"{pred_margin:.1f}" if pd.notna(pred_margin) else "N/A"
        spread_str = f"{spread:+.1f}" if pd.notna(spread) else "N/A"

        away = row.get('away_team', '')
        home = row.get('home_team', '')

        rows.append(f"""    <tr>
      <td>{away}</td>
      <td>@</td>
      <td>{home}</td>
      <td align="right">{spread_str}</td>
      <td align="right">{margin_str}</td>
      <td align="right"><b>{edge_str}</b></td>
      <td><b>{pick}</b></td>
      <td>{stars}</td>
    </tr>""")

    return """<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>Away</th>
      <th></th>
      <th>Home</th>
      <th>Spread</th>
      <th>Pred</th>
      <th>Edge</th>
      <th>Pick</th>
      <th>Conf</th>
    </tr>
  </thead>
  <tbody>
""" + "\n".join(rows) + """
  </tbody>
</table>"""


def generate_all_games_table(df: pd.DataFrame) -> str:
    """Generate HTML table showing all games."""
    df = df.copy()
    df['edge'] = pd.to_numeric(df['edge'], errors='coerce')

    rows = []
    for _, row in df.iterrows():
        edge = row['edge']
        pick = row['pick']
        spread = row.get('spread_home', '')
        pred_margin = row.get('pred_margin', '')

        # Format values
        edge_str = f"{edge:+.1f}" if pd.notna(edge) else "N/A"
        margin_str = f"{pred_margin:.1f}" if pd.notna(pred_margin) else "N/A"
        spread_str = f"{spread:+.1f}" if pd.notna(spread) else "N/A"

        away = row.get('away_team', '')
        home = row.get('home_team', '')

        # Highlight actionable picks
        if pd.notna(edge) and abs(edge) >= 4.5:
            style = ' style="background-color: #ffffcc;"'
        else:
            style = ''

        rows.append(f"""    <tr{style}>
      <td>{away}</td>
      <td>@</td>
      <td>{home}</td>
      <td align="right">{spread_str}</td>
      <td align="right">{margin_str}</td>
      <td align="right">{edge_str}</td>
      <td>{pick}</td>
    </tr>""")

    return """<table border="1" cellpadding="4" cellspacing="0" style="font-size: 12px;">
  <thead>
    <tr>
      <th>Away</th>
      <th></th>
      <th>Home</th>
      <th>Spread</th>
      <th>Pred</th>
      <th>Edge</th>
      <th>Pick</th>
    </tr>
  </thead>
  <tbody>
""" + "\n".join(rows) + """
  </tbody>
</table>"""


def generate_html_page(
    pred_df: pd.DataFrame,
    pred_date: str,
    output_path: Path,
    threshold: float = 4.5
) -> None:
    """Generate the complete HTML page."""

    # Count stats
    total_games = len(pred_df)
    pred_df['edge'] = pd.to_numeric(pred_df['edge'], errors='coerce')
    actionable = pred_df[pred_df['edge'].abs() >= threshold]
    num_picks = len(actionable)

    picks_table = generate_picks_table(pred_df, threshold)
    all_games_table = generate_all_games_table(pred_df)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CBB ATS Picks - {pred_date}</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      max-width: 900px;
      margin: 20px auto;
      padding: 0 15px;
      background-color: #f5f5f5;
    }}
    h1 {{
      color: #333;
      border-bottom: 2px solid #333;
      padding-bottom: 10px;
    }}
    h2 {{
      color: #555;
      margin-top: 30px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background-color: white;
    }}
    th {{
      background-color: #333;
      color: white;
      padding: 8px;
      text-align: left;
    }}
    td {{
      padding: 6px 8px;
      border-bottom: 1px solid #ddd;
    }}
    tr:hover {{
      background-color: #f0f0f0;
    }}
    .summary {{
      background-color: #e8e8e8;
      padding: 15px;
      margin: 20px 0;
      border-left: 4px solid #333;
    }}
    .model-info {{
      font-size: 12px;
      color: #666;
      margin-top: 30px;
      padding: 10px;
      background-color: #e8e8e8;
    }}
    .updated {{
      font-size: 11px;
      color: #888;
    }}
    hr {{
      border: none;
      border-top: 1px solid #ccc;
      margin: 30px 0;
    }}
  </style>
</head>
<body>
  <h1>CBB ATS Picks</h1>
  <p class="updated">Predictions for: <b>{pred_date}</b> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

  <div class="summary">
    <b>Summary:</b> {num_picks} actionable picks from {total_games} games (edge >= {threshold} pts)<br>
    <b>Model:</b> Ridge + GBM + DNN ensemble | Backtest: 52.9% hit rate, +1.1% ROI
  </div>

  <h2>Today's Picks</h2>
  <p>Only showing games with edge >= {threshold} points. Sorted by edge strength.</p>
  {picks_table}

  <hr>

  <h2>All Games</h2>
  <p>Full game list. Yellow = actionable pick.</p>
  {all_games_table}

  <div class="model-info">
    <b>How to read:</b><br>
    - <b>Spread:</b> Vegas line (+ means home team is underdog)<br>
    - <b>Pred:</b> Model's predicted margin (home team perspective)<br>
    - <b>Edge:</b> Pred minus implied spread. Positive = bet home cover, Negative = bet away cover<br>
    - <b>Conf:</b> * = 4.5-6 pts, ** = 6-8 pts, *** = 8+ pts edge<br>
    <br>
    <b>Betting strategy:</b> Only bet when |edge| >= 4.5 points. Backtest shows 52.9% win rate (breakeven is 52.4% at -110 odds).
  </div>

  <p class="updated">
    <br>
    Source: KenPom ratings + The Odds API spreads<br>
    Model trained on 10,830 games (2022-2026 seasons)
  </p>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML picks page')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=4.5, help='Edge threshold')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    predictions_dir = project_root / 'reports' / 'predictions'
    docs_dir = project_root / 'docs'

    # Load predictions
    try:
        pred_df, pred_date = load_predictions(predictions_dir, args.date)
        print(f"Loaded predictions for {pred_date}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Generate HTML
    output_path = docs_dir / 'picks.html'
    generate_html_page(pred_df, pred_date, output_path, args.threshold)

    print(f"\nDone! Open {output_path} in a browser or push to GitHub Pages.")


if __name__ == '__main__':
    main()
