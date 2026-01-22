from pathlib import Path
from io import StringIO
import pandas as pd

# INPUT: saved KenPom archive HTML files
HTML_DIR = Path("/Users/jacobederer/Repositories/cbb/data/raw/kenpom_html")

# OUTPUT: where you want the CSVs saved
OUT_DIR = Path("/Users/jacobederer/Repositories/cbb/data/kenpom")

CF_MARKERS = (
    "Just a moment",
    "cf-chl",
    "Enable JavaScript and cookies",
)


def read_ratings_table(html: str, html_path: Path) -> pd.DataFrame:
    if any(marker in html for marker in CF_MARKERS):
        raise ValueError(f"{html_path.name} looks like a Cloudflare challenge page.")

    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ImportError:
        tables = None
    except ValueError as exc:
        raise ValueError(f"No tables found in {html_path.name}.") from exc

    if tables:
        return tables[0]

    try:
        tables = pd.read_html(StringIO(html), flavor="bs4")
    except ImportError:
        tables = None
    except ValueError as exc:
        raise ValueError(f"No tables found in {html_path.name}.") from exc

    if tables:
        return tables[0]

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Install lxml, html5lib, or beautifulsoup4 to parse KenPom HTML."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="ratings-table")
    if table is None:
        raise ValueError(f"No ratings table found in {html_path.name}.")

    rows = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) >= 10:
            rows.append(cells)

    if not rows:
        raise ValueError(f"No data rows found in {html_path.name}.")

    return pd.DataFrame(rows)


def scrape_one(html_path: Path) -> Path:
    # Expect filenames like: kenpom_archive_YYYY-MM-DD.html
    date = html_path.stem.replace("kenpom_archive_", "")
    out_path = OUT_DIR / f"kenpom_{date}.csv"

    html = html_path.read_text(encoding="utf-8", errors="ignore")

    # Read the archive ratings table
    df = read_ratings_table(html, html_path)

    # Left panel only (archive-day ratings)
    df = df.iloc[:, :10].copy()
    df.columns = [
        "_Rank",
        "TeamName",
        "_Conf",
        "AdjEM",
        "AdjOE",
        "_AdjOE_Rk",
        "AdjDE",
        "_AdjDE_Rk",
        "AdjTempo",
        "_AdjTempo_Rk",
    ]

    # Keep exactly what you asked for, with Data-export names
    df = df[["TeamName", "AdjEM", "AdjOE", "AdjDE", "AdjTempo"]]

    # Remove separator rows (NaN TeamName or rows containing date markers)
    df = df.dropna(subset=["TeamName"])
    df = df[~df["TeamName"].astype(str).str.contains("Ratings on", na=False)]

    # Convert numeric columns (remove '+' prefix from AdjEM if present)
    df["AdjEM"] = pd.to_numeric(df["AdjEM"].astype(str).str.replace("+", ""), errors="coerce")
    df["AdjOE"] = pd.to_numeric(df["AdjOE"], errors="coerce")
    df["AdjDE"] = pd.to_numeric(df["AdjDE"], errors="coerce")
    df["AdjTempo"] = pd.to_numeric(df["AdjTempo"], errors="coerce")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return out_path


def main():
    html_files = sorted(HTML_DIR.glob("kenpom_archive_*.html"))
    if not html_files:
        raise RuntimeError(f"No kenpom_archive_*.html files found in {HTML_DIR}")

    wrote = 0
    skipped = 0
    for html_path in html_files:
        try:
            out_path = scrape_one(html_path)
        except ValueError as exc:
            print(f"Skipping {html_path.name}: {exc}")
            skipped += 1
            continue

        print(f"Wrote {out_path}")
        wrote += 1

    print(f"Done. Wrote {wrote} file(s) to {OUT_DIR}")
    if skipped:
        print(f"Skipped {skipped} file(s) due to parse errors.")


if __name__ == "__main__":
    main()
