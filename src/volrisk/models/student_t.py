"""GARCH(1,1) with Student-t innovations.

Same variance recursion as GARCH(1,1); only the shock distribution changes:
    eps_t = sigma_t * z_t,   z_t ~ standardized Student-t with nu degrees of freedom

"Standardized" means rescaled to variance 1, so sigma_t^2 is still the
conditional variance. Smaller nu = fatter tails; as nu grows the
distribution approaches the normal.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from volrisk.models.garch import GARCH, GarchParams

NU_MIN, NU_MAX = 2.05, 500.0  # nu must exceed 2 for the variance to exist


@dataclass(frozen=True)
class StudentTParams(GarchParams):
    """GARCH(1,1) parameters plus the degrees of freedom nu."""

    nu: float  # degrees of freedom (tail thickness)

    @property
    def excess_kurtosis_of_shocks(self) -> float:
        """Excess kurtosis of z_t: 6 / (nu - 4), infinite when nu <= 4."""
        return 6 / (self.nu - 4) if self.nu > 4 else float("inf")

    def is_valid(self) -> bool:
        return super().is_valid() and self.nu > 2


def student_t_log_likelihood(eps: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    """Sum of log densities of eps under a standardized Student-t with variance sigma2."""
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    terms = (
        const
        - 0.5 * np.log(sigma2)
        - 0.5 * (nu + 1) * np.log1p(eps**2 / ((nu - 2) * sigma2))
    )
    return float(np.sum(terms))


class StudentTGARCH(GARCH):
    """GARCH(1,1) with standardized Student-t innovations."""

    name = "GARCH(1,1)-t"

    def log_likelihood(self, params: StudentTParams, returns: np.ndarray) -> float:
        eps = np.asarray(returns, dtype=float) - params.mu
        sigma2 = self.conditional_variance(params, returns)
        return student_t_log_likelihood(eps, sigma2, params.nu)

    def _make_params(self, x: np.ndarray) -> StudentTParams:
        return StudentTParams(*(float(v) for v in x))

    def _initial_guess(self, r: np.ndarray) -> np.ndarray:
        return np.append(super()._initial_guess(r), 8.0)  # nu = 8: moderately fat tails

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        return super()._bounds() + [(NU_MIN, NU_MAX)]
    