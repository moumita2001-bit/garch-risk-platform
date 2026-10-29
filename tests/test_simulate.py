"""Tests that simulated GARCH data has the properties theory predicts."""

import numpy as np
import pytest
from scipy import stats

from volrisk.models import GarchParams
from volrisk.models.simulate import simulate_garch

TYPICAL = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)


def lag1_autocorr(x: np.ndarray) -> float:
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


@pytest.fixture(scope="module")
def long_sample():
    returns, sigma2 = simulate_garch(TYPICAL, n=100_000, seed=42)
    return returns, sigma2


def test_output_shapes(long_sample):
    returns, sigma2 = long_sample
    assert len(returns) == len(sigma2) == 100_000


def test_same_seed_gives_same_data():
    a, _ = simulate_garch(TYPICAL, n=50, seed=7)
    b, _ = simulate_garch(TYPICAL, n=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_sample_variance_matches_long_run_variance(long_sample):
    returns, _ = long_sample
    assert returns.var() == pytest.approx(TYPICAL.long_run_variance, rel=0.10)


def test_mean_matches_mu(long_sample):
    returns, _ = long_sample
    assert returns.mean() == pytest.approx(TYPICAL.mu, abs=0.02)


def test_garch_creates_volatility_clustering(long_sample):
    returns, _ = long_sample
    eps = returns - TYPICAL.mu
    assert lag1_autocorr(eps**2) > 0.1      # big moves follow big moves
    assert abs(lag1_autocorr(eps)) < 0.02    # direction stays unpredictable


def test_garch_creates_fat_tails_even_with_normal_shocks(long_sample):
    returns, _ = long_sample
    assert stats.kurtosis(returns) > 1.0     # a normal distribution has 0


def test_no_garch_effects_gives_constant_variance_and_no_clustering():
    flat = GarchParams(mu=0.0, omega=1.0, alpha=0.0, beta=0.0)
    returns, sigma2 = simulate_garch(flat, n=100_000, seed=42)
    np.testing.assert_allclose(sigma2, 1.0)
    assert abs(lag1_autocorr(returns**2)) < 0.02


def test_explosive_parameters_are_refused():
    with pytest.raises(ValueError):
        simulate_garch(GarchParams(0, omega=0.1, alpha=0.2, beta=0.8), n=10)