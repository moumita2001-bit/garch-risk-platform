"""Download daily series from FRED (Federal Reserve Economic Data).

Used for VIX (series VIXCLS): the market's 30-day implied volatility for the
S&P 500, quoted in annualized percent.

Same safeguards as the Tiingo downloader: retries with exponential backoff,
skip if already downloaded today, and a UTC audit log. The API key is read
from the FRED_API_KEY environment variable (.env) and never written to logs
or error messages.

Usage:
    python -m volrisk.data.fred            # downloads VIXCLS
    python -m volrisk.data.fred --force    # download even if done today
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = RAW_DIR / "fred_log.csv"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FredError(RuntimeError):
    """Raised when FRED cannot return the series."""


def get_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise FredError("FRED_API_KEY is not set (add it to .env)")
    return key


def parse_observations(payload: dict, series_id: str) -> pd.DataFrame:
    """Turn FRED's JSON into a DataFrame with columns date and <series_id>.

    FRED marks missing days (e.g. holidays) with "."; those rows are dropped.
    """
    obs = payload.get("observations")
    if obs is None:
        raise FredError(f"unexpected FRED response for {series_id}: no 'observations'")
    df = pd.DataFrame(obs, columns=["date", "value"])
    df = df[df["value"] != "."]
    df = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        series_id.lower(): pd.to_numeric(df["value"]),
    })
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_series(
    series_id: str,
    api_key: str,
    start: str = "2008-01-01",
    session=None,
    retries: int = 3,
    backoff: float = 1.0,
) -> pd.DataFrame:
    """Fetch one series, retrying network errors and temporary server errors."""
    session = session or requests.Session()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    for attempt in range(retries + 1):
        try:
            resp = session.get(FRED_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            problem = type(exc).__name__  # no URL: it contains the API key
        else:
            if resp.status_code == 200:
                return parse_observations(resp.json(), series_id)
            if resp.status_code not in RETRYABLE_STATUS:
                raise FredError(f"FRED returned HTTP {resp.status_code} for {series_id}")
            problem = f"HTTP {resp.status_code}"
        if attempt < retries:
            time.sleep(backoff * 2**attempt)
    raise FredError(f"FRED request for {series_id} failed after {retries + 1} attempts ({problem})")


def downloaded_today(path: Path) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
    return modified == datetime.now(timezone.utc).date()


def append_log(series_id: str, df: pd.DataFrame, log_file: Path = LOG_FILE) -> None:
    new_file = not log_file.exists()
    with log_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp_utc", "series_id", "rows", "first_date", "last_date"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            series_id,
            len(df),
            df["date"].iloc[0].date(),
            df["date"].iloc[-1].date(),
        ])


def download_series(
    series_id: str = "VIXCLS",
    raw_dir: Path = RAW_DIR,
    force: bool = False,
    session=None,
) -> Path:
    """Download a series to data/raw/<series>.csv unless already done today."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{series_id.lower()}.csv"
    if downloaded_today(out) and not force:
        print(f"{series_id}: already downloaded today, skipping ({out.name})")
        return out
    df = fetch_series(series_id, get_api_key(), session=session)
    df.to_csv(out, index=False)
    append_log(series_id, df, raw_dir / LOG_FILE.name)
    print(f"{series_id}: {len(df):,} rows, {df['date'].iloc[0].date()} to "
          f"{df['date'].iloc[-1].date()} -> {out.name}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", default="VIXCLS")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_series(args.series, force=args.force)
