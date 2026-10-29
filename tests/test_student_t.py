"""Tests for GARCH(1,1) with Student-t innovations."""

import numpy as np
import pytest
from scipy import stats

from volrisk.models import GARCH, GarchParams, StudentTGARCH, StudentTParams, VolatilityModel

TYPICAL_T = StudentTParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88, nu=6.0)


def test_log_likelihood_matches_scipy_t():
    returns = np.random.default_rng(1).normal(0.05, 1.2, size=500)
    model = StudentTGARCH()
    sigma2 = model.conditional_variance(TYPICAL_T, returns)
    nu = TYPICAL_T.nu
    # scipy's t has variance nu/(nu-2); rescale so the variance is sigma2
    scale = np.sqrt(sigma2 * (nu - 2) / nu)
    expected = stats.t.logpdf(returns, df=nu, loc=TYPICAL_T.mu, scale=scale).sum()
    assert model.log_likelihood(TYPICAL_T, returns) == pytest.approx(expected)


def test_large_nu_approaches_normal_likelihood():
    returns = np.random.default_rng(2).normal(0.05, 1.2, size=500)
    normal = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)
    near_normal = StudentTParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88, nu=1e6)
    ll_normal = GARCH().log_likelihood(normal, returns)
    ll_t = StudentTGARCH().log_likelihood(near_normal, returns)
    assert ll_t == pytest.approx(ll_normal, rel=1e-4)


def test_same_variance_recursion_as_normal_garch():
    returns = np.random.default_rng(3).normal(size=200)
    normal = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)
    np.testing.assert_allclose(
        StudentTGARCH().conditional_variance(TYPICAL_T, returns),
        GARCH().conditional_variance(normal, returns),
    )


def test_nu_must_exceed_two():
    assert TYPICAL_T.is_valid()
    assert not StudentTParams(0.0, 0.02, 0.10, 0.88, nu=2.0).is_valid()


def test_shock_kurtosis_formula():
    assert TYPICAL_T.excess_kurtosis_of_shocks == pytest.approx(3.0)  # 6 / (6 - 4)
    assert StudentTParams(0.0, 0.02, 0.10, 0.88, nu=4.0).excess_kurtosis_of_shocks == float("inf")


def test_follows_the_common_interface():
    assert isinstance(StudentTGARCH(), VolatilityModel)