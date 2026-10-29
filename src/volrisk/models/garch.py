"""GARCH(1,1) written from scratch.

Model (returns in percent):
    r_t       = mu + eps_t,      eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma2_t  = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Parameters are estimated by maximum likelihood with scipy.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from volrisk.models.base import VolatilityModel

LOG_2PI = np.log(2 * np.pi)
N_PARAMS = 4


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
        """Level the variance reverts to: omega / (1 - alpha - beta)."""
        return self.omega / (1 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to lose half its effect."""
        return np.log(0.5) / np.log(self.persistence)

    def is_valid(self) -> bool:
        """omega > 0, alpha >= 0, beta >= 0 and alpha + beta < 1."""
        return (
            self.omega > 0
            and self.alpha >= 0
            and self.beta >= 0
            and self.persistence < 1
        )


@dataclass(frozen=True)
class GarchFit:
    """Result of fitting GARCH(1,1) to a return series."""

    params: GarchParams
    log_likelihood: float
    n_obs: int
    converged: bool
    message: str

    @property
    def aic(self) -> float:
        """Akaike information criterion (lower is better)."""
        return 2 * N_PARAMS - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion (lower is better)."""
        return N_PARAMS * np.log(self.n_obs) - 2 * self.log_likelihood


def garch_variance(
    eps: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: float
) -> np.ndarray:
    """Run the GARCH(1,1) recursion and return sigma^2_t for every day."""
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_0
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


class GARCH(VolatilityModel):
    """GARCH(1,1) with normal innovations."""

    name = "GARCH(1,1)"

    def conditional_variance(self, params: GarchParams, returns: np.ndarray) -> np.ndarray:
        eps = np.asarray(returns, dtype=float) - params.mu
        sigma2_0 = eps.var()  # start the recursion at the sample variance
        return garch_variance(eps, params.omega, params.alpha, params.beta, sigma2_0)

    def log_likelihood(self, params: GarchParams, returns: np.ndarray) -> float:
        eps = np.asarray(returns, dtype=float) - params.mu
        sigma2 = self.conditional_variance(params, returns)
        return float(-0.5 * np.sum(LOG_2PI + np.log(sigma2) + eps**2 / sigma2))

    def fit(self, returns: np.ndarray) -> GarchFit:
        """Estimate (mu, omega, alpha, beta) by maximum likelihood.

        Returns should be in percent (e.g. 1.2 for a 1.2% day); the optimizer
        is much better behaved on that scale than on decimals like 0.012.
        """
        r = np.asarray(returns, dtype=float)

        # Starting guess: typical equity values, with omega chosen so the
        # long-run variance equals the sample variance.
        alpha0, beta0 = 0.05, 0.90
        x0 = np.array([r.mean(), r.var() * (1 - alpha0 - beta0), alpha0, beta0])

        def negative_log_likelihood(x: np.ndarray) -> float:
            params = GarchParams(*x)
            if not params.is_valid():
                return 1e10  # steer the optimizer away from forbidden regions
            return -self.log_likelihood(params, r)

        bounds = [(None, None), (1e-8, None), (0.0, 1.0), (0.0, 1.0)]
        # alpha + beta must stay below 1 (with a small safety margin)
        stationarity = {"type": "ineq", "fun": lambda x: 1 - 1e-6 - x[2] - x[3]}

        result = minimize(
            negative_log_likelihood,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[stationarity],
            options={"maxiter": 500, "ftol": 1e-9},
        )

        return GarchFit(
            params=GarchParams(*(float(v) for v in result.x)),
            log_likelihood=float(-result.fun),
            n_obs=len(r),
            converged=bool(result.success),
            message=str(result.message),
        )