"""Tests for the rolling VaR backtest and the Kupiec / Christoffersen tests."""

import numpy as np
import pandas as pd
import pytest

from volrisk.models import GARCH, GarchParams
from volrisk.models.simulate import simulate_garch
from volrisk.risk import basel_zone, christoffersen_test, kupiec_test, rolling_var

TRUE = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)


def as_series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2010-01-01", periods=len(values)))


@pytest.fixture(scope="module")
def simulated_backtest():
    returns, _ = simulate_garch(TRUE, n=6_000, seed=7)
    return rolling_var(GARCH(), as_series(returns), p=0.01, window=1000, refit_every=250)


# --- Rolling VaR ---------------------------------------------------------------

def test_output_covers_every_day_after_the_window(simulated_backtest):
    assert len(simulated_backtest) == 5_000
    assert simulated_backtest["var"].notna().all()
    assert (simulated_backtest["var"] > 0).all()


def test_breach_rate_is_close_to_one_percent_when_model_is_right(simulated_backtest):
    rate = simulated_backtest["breach"].mean()
    assert 0.005 < rate < 0.02


def test_no_look_ahead():
    # Changing returns AFTER a day must not change that day's VaR forecast.
    returns, _ = simulate_garch(TRUE, n=1_600, seed=3)
    original = rolling_var(GARCH(), as_series(returns), window=1000, refit_every=250)
    altered = returns.copy()
    altered[1300:] = 10.0  # wild future values
    changed = rolling_var(GARCH(), as_series(altered), window=1000, refit_every=250)
    first = original.index[:300]  # days 1000-1299 are unaffected by the change
    np.testing.assert_allclose(original.loc[first, "var"], changed.loc[first, "var"])


def test_too_little_data_is_refused():
    with pytest.raises(ValueError):
        rolling_var(GARCH(), as_series(np.zeros(50)), window=100)


# --- Kupiec ----------------------------------------------------------------------

def test_kupiec_statistic_is_zero_at_the_exact_rate():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:10] = True  # exactly 1%
    result = kupiec_test(breaches, p=0.01)
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert not result.rejected


def test_kupiec_rejects_far_too_many_breaches():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:40] = True  # 4% when 1% is promised
    assert kupiec_test(breaches, p=0.01).rejected


def test_kupiec_handles_zero_breaches():
    result = kupiec_test(np.zeros(1000, dtype=bool), p=0.01)
    assert np.isfinite(result.statistic)


# --- Christoffersen --------------------------------------------------------------

def test_christoffersen_detects_clustered_breaches():
    breaches = np.zeros(1000, dtype=bool)
    breaches[500:510] = True  # ten breaches in a row
    assert christoffersen_test(breaches).rejected


def test_christoffersen_accepts_spread_out_breaches():
    breaches = np.zeros(1000, dtype=bool)
    breaches[::100] = True  # one every 100 days
    assert not christoffersen_test(breaches).rejected


# --- Basel ----------------------------------------------------------------------

@pytest.mark.parametrize("count, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_traffic_light(count, zone):
    assert basel_zone(count) == zone