"""Turn raw Tiingo price files into clean data the models can use.

Reads the newest raw file for each ticker from data/raw/tiingo/, checks it,
tidies it, adds daily log returns and Garman-Klass realized variance, and
saves the result to data/processed/<ticker>.csv. Raw files are never changed.

Usage:
    python -m volrisk.data.clean --tickers SPY
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tiingo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Tiingo columns we rely on. Adjusted prices account for splits and dividends.
REQUIRED_COLUMNS = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]

# More than 7 calendar days between rows means at least a full trading week
# is missing. Normal weekends and holidays never exceed 4-5 calendar days.
MAX_GAP_DAYS = 7

# Garman-Klass constant: 2*ln(2) - 1
GK_CONSTANT = 2 * np.log(2) - 1


class DataValidationError(ValueError):
    """Raised when raw data is too broken to clean safely."""


def latest_raw_file(ticker: str) -> Path:
    """Newest raw file for a ticker; names sort by date, e.g. spy_20261024.csv."""
    files = sorted(RAW_DIR.glob(f"{ticker.lower()}_*.csv"))
    if not files:
        raise DataValidationError(
            f"{ticker}: no raw files in {RAW_DIR}. Run volrisk.data.download first."
        )
    return files[-1]


def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw Tiingo CSV and turn its timestamps into plain dates."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise DataValidationError(f"{path.name}: no 'date' column")
    # Tiingo dates look like 2026-10-01T00:00:00.000Z; keep just the day.
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df


def validate(df: pd.DataFrame, name: str = "data") -> None:
    """Stop on problems that would make every later number wrong."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")

    prices = df[["adjOpen", "adjHigh", "adjLow", "adjClose"]]
    if prices.isna().any().any():
        raise DataValidationError(f"{name}: {int(prices.isna().sum().sum())} missing prices")
    if (prices <= 0).any().any():
        raise DataValidationError(f"{name}: zero or negative prices found")
    bad_range = int((df["adjHigh"] < df["adjLow"]).sum())
    if bad_range:
        raise DataValidationError(f"{name}: {bad_range} days where high < low")

    # Open or close slightly outside the day's range is usually rounding in
    # the adjustment, so warn instead of failing.
    outside = int(
        ((df["adjOpen"] > df["adjHigh"]) | (df["adjOpen"] < df["adjLow"])
         | (df["adjClose"] > df["adjHigh"]) | (df["adjClose"] < df["adjLow"])).sum()
    )
    if outside:
        logger.warning("%s: %d days with open/close outside high-low range", name, outside)


def tidy(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """Sort by date, drop duplicate days and flag suspicious gaps."""
    df = df.sort_values("date")
    before = len(df)
    df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if before - len(df):
        logger.warning("%s: removed %d duplicate days", name, before - len(df))

    gap_days = df["date"].diff().dt.days
    df["gap_flag"] = gap_days > MAX_GAP_DAYS
    if df["gap_flag"].any():
        logger.warning(
            "%s: %d gaps longer than %d days, ending on %s",
            name, int(df["gap_flag"].sum()), MAX_GAP_DAYS,
            ", ".join(df.loc[df["gap_flag"], "date"].dt.strftime("%Y-%m-%d")),
        )
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily log return: ln(today's close / yesterday's close), in decimals."""
    df["log_return"] = np.log(df["adjClose"] / df["adjClose"].shift(1))
    return df


def add_garman_klass(df: pd.DataFrame) -> pd.DataFrame:
    """Garman-Klass realized variance for each day.

    rv_gk = 0.5 * ln(High/Low)^2 - (2 ln 2 - 1) * ln(Close/Open)^2

    Uses the day's whole price range, so it is far less noisy than squared
    returns. It measures open-to-close movement and leaves out the overnight
    gap, which is a known limitation to mention in the model documentation.
    """
    hl = np.log(df["adjHigh"] / df["adjLow"])
    co = np.log(df["adjClose"] / df["adjOpen"])
    df["rv_gk"] = (0.5 * hl**2 - GK_CONSTANT * co**2).clip(lower=0)
    return df


def clean_ticker(ticker: str) -> Path:
    """Run every cleaning step for one ticker and save the result."""
    raw_file = latest_raw_file(ticker)
    df = load_raw(raw_file)
    validate(df, ticker)
    df = tidy(df, ticker)
    df = add_log_returns(df)
    df = add_garman_klass(df)

    # Keep only the adjusted prices, then give them simple names. Selecting
    # first matters: the raw file also has unadjusted open/high/low/close.
    out = (
        df[["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
            "log_return", "rv_gk", "gap_flag"]]
        .rename(columns={
            "adjOpen": "open", "adjHigh": "high", "adjLow": "low",
            "adjClose": "close", "adjVolume": "volume",
        })
        .iloc[1:]  # first day has no previous close, so no return
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{ticker.lower()}.csv"
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    logger.info(
        "%s: %d rows from %s to %s (source %s) saved to %s",
        ticker, len(out),
        out["date"].iloc[0].date(), out["date"].iloc[-1].date(),
        raw_file.name, path.relative_to(PROJECT_ROOT),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean raw Tiingo price files.")
    parser.add_argument("--tickers", nargs="+", default=["SPY"], help="e.g. SPY QQQ")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    failures = []
    for ticker in args.tickers:
        try:
            clean_ticker(ticker)
        except DataValidationError as exc:
            logger.error("%s", exc)
            failures.append(ticker)

    if failures:
        logger.error("Failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())