"""Rolling out-of-sample VaR backtest and the standard tests that judge it.

For each test day t, VaR is forecast using ONLY returns before day t:
  - the model is re-fitted every `refit_every` days on the last `window` days;
  - between re-fits the parameters are held fixed and the variance recursion
    is rolled forward one day at a time (as a production system would).

A breach (or "exception") is a day where the actual loss exceeded VaR.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from volrisk.models import GARCH
from volrisk.models.garch import garch_variance
from volrisk.risk.var import shock_quantile


def rolling_var(
    model: GARCH,
    returns: pd.Series,
    p: float = 0.01,
    window: int = 1000,
    refit_every: int = 21,
) -> pd.DataFrame:
    """Out-of-sample 1-day VaR for every day after the first `window` days.

    Returns a DataFrame indexed like `returns` with columns
    return, var (positive loss) and breach (True when -return > var).
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n <= window:
        raise ValueError(f"need more than {window} returns, got {n}")

    var = np.full(n, np.nan)
    for start in range(window, n, refit_every):
        stop = min(start + refit_every, n)
        train = r[start - window : start]
        params = model.fit(train).params

        # Roll the recursion through the training window and the next block.
        # sigma2[t] uses returns up to t-1 only, so it is a genuine forecast.
        # The starting variance comes from the training window alone.
        eps = r[start - window : stop] - params.mu
        sigma2 = garch_variance(eps, params.omega, params.alpha, params.beta, train.var())
        sigma = np.sqrt(sigma2[window:])
        var[start:stop] = -(params.mu + sigma * shock_quantile(params, p))

    out = pd.DataFrame({"return": r, "var": var}, index=returns.index).iloc[window:]
    out["breach"] = -out["return"] > out["var"]
    return out


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float

    @property
    def rejected(self) -> bool:
        """True if the model is rejected at the 5% level."""
        return self.p_value < 0.05


def kupiec_test(breaches: np.ndarray, p: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test.

    H0: the breach rate equals p. LR ~ chi-squared(1).
    """
    b = np.asarray(breaches, dtype=bool)
    n, x = len(b), int(b.sum())
    rate = x / n
    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1 - rate) + xlogy(x, rate)
    lr = float(-2 * (log_l0 - log_l1))
    return TestResult(lr, float(stats.chi2.sf(lr, df=1)))


def christoffersen_test(breaches: np.ndarray) -> TestResult:
    """Christoffersen (1998) independence test.

    H0: a breach today does not change the chance of a breach tomorrow
    (no clustering). LR ~ chi-squared(1).
    """
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr = float(-2 * (log_l0 - log_l1))
    return TestResult(lr, float(stats.chi2.sf(lr, df=1)))


def basel_zone(breaches_in_last_250_days: int) -> str:
    """Basel traffic light for 99% VaR over 250 trading days."""
    if breaches_in_last_250_days <= 4:
        return "green"
    if breaches_in_last_250_days <= 9:
        return "yellow"
    return "red"