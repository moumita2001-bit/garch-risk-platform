"""Simulation and parameter-recovery tests for Student-t GARCH."""

import numpy as np
import pytest
from scipy import stats

from volrisk.models import GARCH, GarchParams, StudentTGARCH, StudentTParams
from volrisk.models.simulate import draw_shocks, simulate_garch

TRUE_T = StudentTParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88, nu=6.0)
TRUE_NORMAL = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)


@pytest.fixture(scope="module")
def t_data():
    returns, _ = simulate_garch(TRUE_T, n=20_000, seed=42)
    return returns


@pytest.fixture(scope="module")
def t_fit(t_data):
    return StudentTGARCH().fit(t_data)


def test_t_shocks_are_standardized():
    z = draw_shocks(TRUE_T, 1_000_000, np.random.default_rng(0))
    assert z.mean() == pytest.approx(0.0, abs=0.01)
    assert z.var() == pytest.approx(1.0, rel=0.02)


def test_t_shocks_give_fatter_tails_than_normal_shocks(t_data):
    normal_data, _ = simulate_garch(TRUE_NORMAL, n=20_000, seed=42)
    assert stats.kurtosis(t_data) > stats.kurtosis(normal_data) + 1.0


def test_recovers_true_parameters_including_nu(t_fit):
    assert t_fit.converged, t_fit.message
    est = t_fit.params
    assert est.alpha == pytest.approx(TRUE_T.alpha, abs=0.02)
    assert est.beta == pytest.approx(TRUE_T.beta, abs=0.03)
    assert est.nu == pytest.approx(TRUE_T.nu, abs=1.0)


def test_t_model_beats_normal_model_on_t_data(t_data, t_fit):
    normal_fit = GARCH().fit(t_data)
    assert t_fit.aic < normal_fit.aic
    assert t_fit.bic < normal_fit.bic


def test_t_model_on_normal_data_finds_large_nu():
    normal_data, _ = simulate_garch(TRUE_NORMAL, n=20_000, seed=42)
    est = StudentTGARCH().fit(normal_data).params
    assert est.nu > 30  # thin tails: the t model correctly collapses toward normal