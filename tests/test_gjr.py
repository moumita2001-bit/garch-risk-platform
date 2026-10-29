"""Tests for GJR-GARCH(1,1) with Student-t innovations."""

import numpy as np
import pytest

from volrisk.models import GJRParams, GJRStudentTGARCH, StudentTGARCH, StudentTParams
from volrisk.models.simulate import simulate_garch

TRUE_GJR = GJRParams(mu=0.0, omega=0.02, alpha=0.03, beta=0.88, nu=6.0, gamma=0.12)


def test_negative_shock_raises_variance_more_than_positive():
    model = GJRStudentTGARCH()
    up = model.forecast_variance(TRUE_GJR, np.array([0.0, 0.0, 3.0]))
    down = model.forecast_variance(TRUE_GJR, np.array([0.0, 0.0, -3.0]))
    assert down - up == pytest.approx(TRUE_GJR.gamma * 9.0)  # gamma * 3^2


def test_zero_gamma_matches_student_t_garch():
    returns = np.random.default_rng(1).normal(size=500)
    t_params = StudentTParams(mu=0.0, omega=0.02, alpha=0.08, beta=0.9, nu=6.0)
    gjr_params = GJRParams(mu=0.0, omega=0.02, alpha=0.08, beta=0.9, nu=6.0, gamma=0.0)
    assert GJRStudentTGARCH().log_likelihood(gjr_params, returns) == pytest.approx(
        StudentTGARCH().log_likelihood(t_params, returns)
    )


def test_persistence_counts_gamma_at_half_weight():
    assert TRUE_GJR.persistence == pytest.approx(0.03 + 0.06 + 0.88)


def test_negative_gamma_is_invalid():
    bad = GJRParams(mu=0.0, omega=0.02, alpha=0.05, beta=0.88, nu=6.0, gamma=-0.01)
    assert not bad.is_valid()


@pytest.fixture(scope="module")
def gjr_fit():
    returns, _ = simulate_garch(TRUE_GJR, n=20_000, seed=11)
    return GJRStudentTGARCH(zero_mean=True).fit(returns), returns


def test_recovers_leverage_parameter(gjr_fit):
    fit, _ = gjr_fit
    assert fit.converged, fit.message
    assert fit.params.gamma == pytest.approx(TRUE_GJR.gamma, abs=0.04)
    assert fit.params.beta == pytest.approx(TRUE_GJR.beta, abs=0.03)
    assert fit.params.nu == pytest.approx(TRUE_GJR.nu, abs=1.5)


def test_gjr_beats_symmetric_model_on_leveraged_data(gjr_fit):
    fit, returns = gjr_fit
    symmetric = StudentTGARCH(zero_mean=True).fit(returns)
    assert fit.bic < symmetric.bic
