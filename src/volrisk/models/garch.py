"""GARCH(1,1) written from scratch.

Model (returns in percent):
    r_t       = mu + eps_t,      eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma2_t  = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Parameters are estimated by maximum likelihood with scipy.
"""

from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import minimize

from volrisk.models.base import VolatilityModel

LOG_2PI = np.log(2 * np.pi)


@dataclass(frozen=True)
class GarchParams:
    """Parameters of a GARCH(1,1) model."""

    mu: float     # mean daily return
    omega: float  # baseline variance
    alpha: float  # reaction to yesterday's shock
    beta: float   # memory of yesterday's variance

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Level the variance reverts to: omega / (1 - persistence)."""
        return self.omega / (1 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to lose half its effect."""
        return np.log(0.5) / np.log(self.persistence)

    def is_valid(self) -> bool:
        """omega > 0, alpha >= 0, beta >= 0 and persistence < 1."""
        return (
            self.omega > 0
            and self.alpha >= 0
            and self.beta >= 0
            and self.persistence < 1
        )


@dataclass(frozen=True)
class GarchFit:
    """Result of fitting a GARCH-family model to a return series."""

    params: GarchParams
    log_likelihood: float
    n_obs: int
    converged: bool
    message: str

    @property
    def n_params(self) -> int:
        return len(fields(self.params))

    @property
    def aic(self) -> float:
        """Akaike information criterion (lower is better)."""
        return 2 * self.n_params - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion (lower is better)."""
        return self.n_params * np.log(self.n_obs) - 2 * self.log_likelihood


def garch_variance(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    sigma2_0: float,
    gamma: float = 0.0,
) -> np.ndarray:
    """Run the GARCH(1,1) recursion and return sigma^2_t for every day.

    gamma > 0 adds the GJR leverage term: negative shocks get weight
    alpha + gamma instead of alpha. gamma = 0 is plain GARCH(1,1).
    """
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_0
    for t in range(1, n):
        shock_weight = alpha + gamma * (eps[t - 1] < 0)
        sigma2[t] = omega + shock_weight * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


class GARCH(VolatilityModel):
    """GARCH(1,1) with normal innovations."""

    name = "GARCH(1,1)"

    def __init__(self, zero_mean: bool = False):
        """zero_mean=True fixes mu at 0 instead of estimating it, a common
        choice for 1-day VaR because the daily mean is tiny and noisy."""
        self.zero_mean = zero_mean

    def variance_path(self, params: GarchParams, eps: np.ndarray, sigma2_0: float) -> np.ndarray:
        """sigma^2_t for a series of shocks, starting the recursion at sigma2_0."""
        gamma = getattr(params, "gamma", 0.0)  # only GJR models have gamma
        return garch_variance(eps, params.omega, params.alpha, params.beta, sigma2_0, gamma)

    def conditional_variance(self, params: GarchParams, returns: np.ndarray) -> np.ndarray:
        eps = np.asarray(returns, dtype=float) - params.mu
        sigma2_0 = eps.var()  # start the recursion at the sample variance
        return self.variance_path(params, eps, sigma2_0)

    def log_likelihood(self, params: GarchParams, returns: np.ndarray) -> float:
        eps = np.asarray(returns, dtype=float) - params.mu
        sigma2 = self.conditional_variance(params, returns)
        return float(-0.5 * np.sum(LOG_2PI + np.log(sigma2) + eps**2 / sigma2))

    def forecast_variance(self, params: GarchParams, returns: np.ndarray) -> float:
        """One-step-ahead forecast sigma2_{T+1}, from today's shock and variance."""
        eps = np.asarray(returns, dtype=float) - params.mu
        # Append a placeholder for tomorrow: sigma2[T+1] depends only on day T.
        path = self.variance_path(params, np.append(eps, 0.0), eps.var())
        return float(path[-1])

    # --- Hooks that subclasses (e.g. Student-t, GJR) override ---------------

    def _make_params(self, x: np.ndarray) -> GarchParams:
        """Turn the optimizer's vector into a parameter object."""
        return GarchParams(*(float(v) for v in x))

    def _initial_guess(self, r: np.ndarray) -> np.ndarray:
        """Starting point: typical equity values, omega set so the
        long-run variance equals the sample variance."""
        alpha0, beta0 = 0.05, 0.90
        mu0 = 0.0 if self.zero_mean else r.mean()
        var0 = np.mean((r - mu0) ** 2)
        return np.array([mu0, var0 * (1 - alpha0 - beta0), alpha0, beta0])

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        mu_bound = (0.0, 0.0) if self.zero_mean else (None, None)
        return [mu_bound, (1e-8, None), (0.0, 1.0), (0.0, 1.0)]

    # --- Estimation ---------------------------------------------------------

    def fit(self, returns: np.ndarray) -> GarchFit:
        """Estimate the parameters by maximum likelihood.

        Returns should be in percent (e.g. 1.2 for a 1.2% day); the optimizer
        is much better behaved on that scale than on decimals like 0.012.
        """
        r = np.asarray(returns, dtype=float)

        def negative_log_likelihood(x: np.ndarray) -> float:
            params = self._make_params(x)
            if not params.is_valid():
                return 1e10  # steer the optimizer away from forbidden regions
            return -self.log_likelihood(params, r)

        # persistence must stay below 1 (with a small safety margin)
        stationarity = {
            "type": "ineq",
            "fun": lambda x: 1 - 1e-6 - self._make_params(x).persistence,
        }

        result = minimize(
            negative_log_likelihood,
            self._initial_guess(r),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=[stationarity],
            options={"maxiter": 500, "ftol": 1e-9},
        )

        return GarchFit(
            params=self._make_params(result.x),
            log_likelihood=float(-result.fun),
            n_obs=len(r),
            converged=bool(result.success),
            message=str(result.message),
        )
