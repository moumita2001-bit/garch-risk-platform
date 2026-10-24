"""Download daily price history from the Tiingo end-of-day API.

Each run saves one raw CSV per ticker per day to data/raw/tiingo/ and records
the download in data/raw/download_log.csv (the audit trail). Raw files are
never edited afterwards; cleaning happens in clean.py.

Usage:
    python -m volrisk.data.download --tickers SPY --start 2008-01-01
    python -m volrisk.data.download --tickers SPY QQQ IWM --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

# src/volrisk/data/download.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tiingo"
LOG_FILE = PROJECT_ROOT / "data" / "raw" / "download_log.csv"

# Temporary problems worth retrying: rate limiting and server errors.
RETRY_STATUS = {429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """Raised when a download fails and should not be retried further."""


def get_api_key() -> str:
    """Read the Tiingo key from .env without ever printing it."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("TIINGO_API_KEY")
    if not key or key.startswith("your_"):
        raise DownloadError(
            "TIINGO_API_KEY is missing. Add it to .env (see .env.example)."
        )
    return key


def fetch_tiingo_daily(
    ticker: str,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
    backoff: float = 2.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch daily prices for one ticker, retrying temporary failures.

    Waits backoff**attempt seconds between attempts (2s, 4s, 8s by default).
    """
    api_key = api_key or get_api_key()
    session = session or requests.Session()
    url = TIINGO_URL.format(ticker=ticker.lower())
    headers = {"Authorization": f"Token {api_key}"}
    params = {"startDate": start, "format": "json"}
    if end:
        params["endDate"] = end

    error = "unknown error"
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            error = f"network error ({exc.__class__.__name__})"
        else:
            if response.status_code == 200:
                records = response.json()
                if not records:
                    raise DownloadError(
                        f"{ticker}: no data returned for {start} to {end or 'today'}"
                    )
                return pd.DataFrame.from_records(records)
            if response.status_code not in RETRY_STATUS:
                # 401 (bad key), 404 (unknown ticker) etc. won't fix themselves.
                raise DownloadError(
                    f"{ticker}: HTTP {response.status_code}: {response.text[:200]}"
                )
            error = f"HTTP {response.status_code}"

        if attempt < max_retries:
            wait = backoff**attempt
            logger.warning(
                "%s: %s (attempt %d/%d), retrying in %.0fs",
                ticker, error, attempt, max_retries, wait,
            )
            time.sleep(wait)

    raise DownloadError(f"{ticker}: failed after {max_retries} attempts ({error})")


def raw_path(ticker: str, on: date | None = None) -> Path:
    """Path of the raw file for a ticker on a given day, e.g. spy_20261024.csv."""
    on = on or date.today()
    return RAW_DIR / f"{ticker.lower()}_{on:%Y%m%d}.csv"


def append_log(ticker: str, path: Path, rows: int, start: str, end: str | None) -> None:
    """Record each download: when, from where, what range and how many rows."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["downloaded_at_utc", "source", "ticker", "start", "end", "rows", "file"]
            )
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tiingo",
            ticker.upper(),
            start,
            end or "",
            rows,
            path.name,
        ])


def download_ticker(
    ticker: str, start: str, end: str | None = None, force: bool = False
) -> Path:
    """Download one ticker unless today's file already exists."""
    path = raw_path(ticker)
    if path.exists() and not force:
        logger.info("%s: already downloaded today (%s), skipping", ticker, path.name)
        return path

    df = fetch_tiingo_daily(ticker, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    append_log(ticker, path, len(df), start, end)
    logger.info(
        "%s: saved %d rows to %s", ticker, len(df), path.relative_to(PROJECT_ROOT)
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily prices from Tiingo.")
    parser.add_argument("--tickers", nargs="+", default=["SPY"], help="e.g. SPY QQQ")
    parser.add_argument("--start", default="2008-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if today's file exists"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    failures = []
    for ticker in args.tickers:
        try:
            download_ticker(ticker, args.start, args.end, force=args.force)
        except DownloadError as exc:
            logger.error("%s", exc)
            failures.append(ticker)
        time.sleep(1)  # be polite to the API between tickers

    if failures:
        logger.error("Failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())