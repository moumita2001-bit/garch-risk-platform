"""Parameter-recovery tests: simulate with known parameters, fit, compare."""

import pytest

from volrisk.models import GARCH, GarchParams
from volrisk.models.simulate import simulate_garch

TRUE = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)


@pytest.fixture(scope="module")
def recovered():
    returns, _ = simulate_garch(TRUE, n=20_000, seed=42)
    return GARCH().fit(returns), returns


def test_optimizer_converges(recovered):
    fit, _ = recovered
    assert fit.converged, fit.message


def test_recovers_true_parameters(recovered):
    est = recovered[0].params
    assert est.mu == pytest.approx(TRUE.mu, abs=0.02)
    assert est.omega == pytest.approx(TRUE.omega, abs=0.01)
    assert est.alpha == pytest.approx(TRUE.alpha, abs=0.02)
    assert est.beta == pytest.approx(TRUE.beta, abs=0.03)


def test_recovers_persistence_and_long_run_variance(recovered):
    est = recovered[0].params
    assert est.persistence == pytest.approx(TRUE.persistence, abs=0.01)
    assert est.long_run_variance == pytest.approx(TRUE.long_run_variance, rel=0.15)


def test_estimate_satisfies_constraints(recovered):
    assert recovered[0].params.is_valid()


def test_fit_beats_the_true_parameters_on_likelihood(recovered):
    # The MLE is, by definition, the best-fitting point for this sample,
    # so it must score at least as well as the true parameters.
    fit, returns = recovered
    assert fit.log_likelihood >= GARCH().log_likelihood(TRUE, returns)


def test_no_arch_effects_gives_alpha_near_zero():
    flat = GarchParams(mu=0.0, omega=1.0, alpha=0.0, beta=0.0)
    returns, _ = simulate_garch(flat, n=5_000, seed=42)
    est = GARCH().fit(returns).params
    assert est.alpha < 0.02
    assert est.long_run_variance == pytest.approx(1.0, rel=0.10)