"""Tests for the GARCH(1,1) variance recursion and log-likelihood."""

import numpy as np
import pytest
from scipy import stats

from volrisk.models import GARCH, GarchParams, VolatilityModel
from volrisk.models.garch import garch_variance

TYPICAL = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)


def test_recursion_matches_hand_calculation():
    eps = np.array([1.0, -2.0, 0.5])
    sigma2 = garch_variance(eps, omega=0.1, alpha=0.2, beta=0.7, sigma2_0=1.0)
    # day 1: 0.1 + 0.2*1.0^2  + 0.7*1.0  = 1.0
    # day 2: 0.1 + 0.2*(-2)^2 + 0.7*1.0  = 1.6
    np.testing.assert_allclose(sigma2, [1.0, 1.0, 1.6])


def test_no_garch_effects_gives_constant_variance():
    eps = np.random.default_rng(0).normal(size=100)
    sigma2 = garch_variance(eps, omega=1.5, alpha=0.0, beta=0.0, sigma2_0=1.5)
    np.testing.assert_allclose(sigma2, 1.5)


def test_long_run_variance_and_half_life():
    assert TYPICAL.persistence == pytest.approx(0.98)
    assert TYPICAL.long_run_variance == pytest.approx(1.0)
    assert TYPICAL.half_life == pytest.approx(34.31, abs=0.01)


@pytest.mark.parametrize(
    "params",
    [
        GarchParams(0, omega=0.0, alpha=0.1, beta=0.8),   # omega must be > 0
        GarchParams(0, omega=0.1, alpha=-0.1, beta=0.8),  # alpha must be >= 0
        GarchParams(0, omega=0.1, alpha=0.2, beta=0.8),   # alpha + beta must be < 1
    ],
)
def test_invalid_parameters_are_rejected(params):
    assert not params.is_valid()


def test_typical_parameters_are_valid():
    assert TYPICAL.is_valid()


def test_log_likelihood_matches_scipy_normal():
    returns = np.random.default_rng(1).normal(0.05, 1.2, size=500)
    model = GARCH()
    sigma2 = model.conditional_variance(TYPICAL, returns)
    expected = stats.norm.logpdf(returns, loc=TYPICAL.mu, scale=np.sqrt(sigma2)).sum()
    assert model.log_likelihood(TYPICAL, returns) == pytest.approx(expected)


def test_garch_follows_the_common_interface():
    assert isinstance(GARCH(), VolatilityModel)