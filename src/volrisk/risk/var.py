"""One-day Value-at-Risk from a GARCH volatility forecast.

    VaR_p = -(mu + sigma_{T+1} * q_p)

where q_p is the p-quantile (e.g. p = 0.01 for 99% VaR) of the model's
standardized shock distribution. VaR is reported as a positive loss in the
same units as the returns (percent).
"""

import numpy as np
from scipy import stats

from volrisk.models import GARCH, GarchParams, StudentTParams


def shock_quantile(params: GarchParams, p: float) -> float:
    """p-quantile of z_t (mean 0, variance 1) under the model's distribution."""
    if not 0 < p < 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if isinstance(params, StudentTParams):
        nu = params.nu
        return float(stats.t.ppf(p, df=nu) * np.sqrt((nu - 2) / nu))
    return float(stats.norm.ppf(p))


def one_day_var(
    model: GARCH, params: GarchParams, returns: np.ndarray, p: float = 0.01
) -> float:
    """Tomorrow's VaR at tail probability p (0.01 = 99% VaR), as a positive loss."""
    sigma = np.sqrt(model.forecast_variance(params, returns))
    return float(-(params.mu + sigma * shock_quantile(params, p)))