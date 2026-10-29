"""Tests for one-day Value-at-Risk."""

import numpy as np
import pytest
from scipy import stats

from volrisk.models import GARCH, GarchParams, StudentTGARCH, StudentTParams
from volrisk.models.simulate import draw_shocks
from volrisk.risk import one_day_var, shock_quantile

NORMAL = GarchParams(mu=0.0, omega=0.02, alpha=0.10, beta=0.88)
T5 = StudentTParams(mu=0.0, omega=0.02, alpha=0.10, beta=0.88, nu=5.0)


def test_forecast_matches_hand_calculation():
    returns = np.array([1.0, -2.0, 0.5])
    params = GarchParams(mu=0.0, omega=0.1, alpha=0.2, beta=0.7)
    sigma2 = GARCH().conditional_variance(params, returns)
    expected = 0.1 + 0.2 * 0.5**2 + 0.7 * sigma2[-1]
    assert GARCH().forecast_variance(params, returns) == pytest.approx(expected)


def test_normal_99_quantile_is_textbook_value():
    assert shock_quantile(NORMAL, 0.01) == pytest.approx(-2.3263, abs=1e-4)


def test_t_quantile_matches_simulated_shocks():
    z = draw_shocks(T5, 2_000_000, np.random.default_rng(0))
    assert shock_quantile(T5, 0.01) == pytest.approx(np.quantile(z, 0.01), abs=0.02)


def test_t_var_is_larger_at_99_but_smaller_at_95():
    # Same variance, different shape: the t puts more mass far out in the tail
    # and less in the shoulders.
    assert shock_quantile(T5, 0.01) < shock_quantile(NORMAL, 0.01)
    assert shock_quantile(T5, 0.05) > shock_quantile(NORMAL, 0.05)


def test_var_scales_with_forecast_volatility():
    calm = np.full(200, 0.5)
    stressed = np.append(calm, -6.0)  # one crash day yesterday
    assert one_day_var(GARCH(), NORMAL, stressed) > one_day_var(GARCH(), NORMAL, calm)


def test_var_formula():
    returns = np.random.default_rng(1).normal(size=300)
    params = GarchParams(mu=0.05, omega=0.02, alpha=0.10, beta=0.88)
    sigma = np.sqrt(GARCH().forecast_variance(params, returns))
    expected = -(0.05 + sigma * stats.norm.ppf(0.01))
    assert one_day_var(GARCH(), params, returns, p=0.01) == pytest.approx(expected)


def test_works_with_student_t_model():
    returns = np.random.default_rng(2).normal(size=300)
    assert one_day_var(StudentTGARCH(), T5, returns) > 0


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        shock_quantile(NORMAL, 1.5)