"""Simulate returns from a GARCH(1,1) process with known parameters.

Used to test the estimator: simulate data where the true parameters are
known, fit the model, and check that it recovers them.

Passing GarchParams gives normal shocks; passing StudentTParams gives
standardized Student-t shocks with nu degrees of freedom.
"""

import numpy as np

from volrisk.models.garch import GarchParams
from volrisk.models.student_t import StudentTParams


def draw_shocks(params: GarchParams, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw z_t with mean 0 and variance 1 from the model's shock distribution."""
    if isinstance(params, StudentTParams):
        nu = params.nu
        # a raw t has variance nu / (nu - 2); rescale it to variance 1
        return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)
    return rng.standard_normal(size)


def simulate_garch(
    params: GarchParams, n: int, seed: int | None = None, burn_in: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns, sigma2) of length n from a GARCH(1,1) process.

    The first `burn_in` days are thrown away so the output does not depend
    on the starting value of the variance.
    """
    if not params.is_valid():
        raise ValueError(f"Parameters violate GARCH constraints: {params}")

    rng = np.random.default_rng(seed)
    total = n + burn_in
    z = draw_shocks(params, total, rng)

    sigma2 = np.empty(total)
    eps = np.empty(total)
    sigma2[0] = params.long_run_variance
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, total):
        sigma2[t] = params.omega + params.alpha * eps[t - 1] ** 2 + params.beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    returns = params.mu + eps
    return returns[burn_in:], sigma2[burn_in:]