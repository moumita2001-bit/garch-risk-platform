"""Tests for volrisk.data.clean.

Each test builds a small, hand-made price table where the right answer is
known in advance, so a failure points straight at the broken step.

Run with:  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from volrisk.data import clean
from volrisk.data.clean import (
    DataValidationError,
    add_garman_klass,
    add_log_returns,
    tidy,
    validate,
)


def make_prices(dates, closes=None):
    """A tiny valid price table in Tiingo's adjusted-column format."""
    dates = pd.to_datetime(dates)
    closes = np.array(closes if closes is not None else [100.0 + i for i in range(len(dates))])
    return pd.DataFrame({
        "date": dates,
        "adjOpen": closes,
        "adjHigh": closes * 1.01,
        "adjLow": closes * 0.99,
        "adjClose": closes,
        "adjVolume": 1_000_000,
    })


# ---------- validate ----------

def test_validate_accepts_good_data():
    validate(make_prices(["2026-01-05", "2026-01-06"]))  # no error raised


def test_validate_rejects_missing_column():
    df = make_prices(["2026-01-05"]).drop(columns="adjClose")
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(df)


def test_validate_rejects_negative_price():
    df = make_prices(["2026-01-05", "2026-01-06"])
    df.loc[1, "adjLow"] = -5.0
    with pytest.raises(DataValidationError, match="negative"):
        validate(df)


def test_validate_rejects_high_below_low():
    df = make_prices(["2026-01-05", "2026-01-06"])
    df.loc[0, "adjHigh"] = 50.0  # below that day's low of 99
    with pytest.raises(DataValidationError, match="high < low"):
        validate(df)


def test_validate_rejects_missing_price():
    df = make_prices(["2026-01-05", "2026-01-06"])
    df.loc[1, "adjClose"] = np.nan
    with pytest.raises(DataValidationError, match="missing prices"):
        validate(df)


# ---------- tidy ----------

def test_tidy_sorts_and_removes_duplicate_days():
    df = make_prices(["2026-01-07", "2026-01-05", "2026-01-06", "2026-01-06"])
    out = tidy(df)
    assert out["date"].is_monotonic_increasing
    assert out["date"].is_unique
    assert len(out) == 3


def test_tidy_flags_long_gap_but_not_weekends():
    # Fri -> Mon is a normal 3-day weekend; Jan 12 -> Jan 26 is a 14-day gap.
    df = make_prices(["2026-01-09", "2026-01-12", "2026-01-26"])
    out = tidy(df)
    assert out["gap_flag"].tolist() == [False, False, True]


# ---------- returns and realized variance ----------

def test_log_returns_match_hand_calculation():
    df = add_log_returns(make_prices(["2026-01-05", "2026-01-06"], closes=[100.0, 110.0]))
    assert np.isnan(df["log_return"].iloc[0])  # no previous close on day one
    assert df["log_return"].iloc[1] == pytest.approx(np.log(110 / 100))


def test_garman_klass_matches_formula():
    df = pd.DataFrame({"adjOpen": [100.0], "adjHigh": [105.0], "adjLow": [98.0], "adjClose": [103.0]})
    expected = 0.5 * np.log(105 / 98) ** 2 - (2 * np.log(2) - 1) * np.log(103 / 100) ** 2
    assert add_garman_klass(df)["rv_gk"].iloc[0] == pytest.approx(expected)


def test_garman_klass_is_zero_on_a_flat_day():
    df = pd.DataFrame({"adjOpen": [100.0], "adjHigh": [100.0], "adjLow": [100.0], "adjClose": [100.0]})
    assert add_garman_klass(df)["rv_gk"].iloc[0] == 0


def test_garman_klass_is_unbiased_on_simulated_prices():
    """Average GK should equal the true daily variance (the point of 2 ln 2 - 1)."""
    rng = np.random.default_rng(42)
    # Fine steps (5,000 per day) so the simulated high and low are close to the
    # true extremes; coarse sampling would miss some of the range.
    days, steps, sigma = 2000, 5000, 0.01
    paths = np.cumsum(rng.normal(0, sigma / np.sqrt(steps), (days, steps)), axis=1)
    paths = np.hstack([np.zeros((days, 1)), paths])  # each day opens at log price 0
    df = pd.DataFrame({
        "adjOpen": np.exp(paths[:, 0]),
        "adjHigh": np.exp(paths.max(axis=1)),
        "adjLow": np.exp(paths.min(axis=1)),
        "adjClose": np.exp(paths[:, -1]),
    })
    mean_gk = add_garman_klass(df)["rv_gk"].mean()
    # Allow 5% for sampling noise and the small remaining discretization bias.
    assert mean_gk == pytest.approx(sigma**2, rel=0.05)


# ---------- end to end ----------

def test_clean_ticker_end_to_end(tmp_path, monkeypatch):
    """Raw file in, processed file out, using a temporary folder."""
    raw_dir = tmp_path / "data" / "raw" / "tiingo"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(clean, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(clean, "RAW_DIR", raw_dir)
    monkeypatch.setattr(clean, "PROCESSED_DIR", tmp_path / "data" / "processed")

    raw = make_prices(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-07"])
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%dT00:00:00.000Z")  # Tiingo format
    raw["close"] = 1.0  # unadjusted columns exist in real files too
    raw.to_csv(raw_dir / "test_20260108.csv", index=False)

    out = pd.read_csv(clean.clean_ticker("TEST"))

    assert list(out.columns) == [
        "date", "open", "high", "low", "close", "volume", "log_return", "rv_gk", "gap_flag",
    ]
    assert len(out) == 2  # 3 unique days, minus the first (no return)
    assert out["date"].tolist() == ["2026-01-06", "2026-01-07"]
    # Adjusted closes, not the raw 1.0; for the duplicate Jan 7 the last row (103) wins.
    assert out["close"].tolist() == [101.0, 103.0]
    assert out["log_return"].notna().all()


def test_clean_ticker_without_raw_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "RAW_DIR", tmp_path)
    with pytest.raises(DataValidationError, match="no raw files"):
        clean.clean_ticker("NOPE")