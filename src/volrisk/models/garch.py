"""GARCH(1,1) written from scratch.

Model (returns in percent):
    r_t       = mu + eps_t,      eps_t = sigma_t * z_t,   z_t ~ N(0, 1)
    sigma2_t  = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

This file covers the variance recursion and the log-likelihood.
Fitting (maximum likelihood with scipy) is added in the next session.
"""

from dataclasses import dataclass

import numpy as np

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