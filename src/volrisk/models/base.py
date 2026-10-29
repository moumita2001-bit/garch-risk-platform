"""Common interface that every volatility model in VolRisk follows.

GARCH(1,1) is the first model; GJR-GARCH and Student-t variants come later.
Defining the interface once means the VaR engine and backtests can work with
any model without knowing which one it is.
"""

from abc import ABC, abstractmethod

import numpy as np


class VolatilityModel(ABC):
    """A model of time-varying (conditional) variance."""

    name: str = "base"

    @abstractmethod
    def conditional_variance(self, params, returns: np.ndarray) -> np.ndarray:
        """Return sigma^2_t for every day, given parameters and returns."""

    @abstractmethod
    def log_likelihood(self, params, returns: np.ndarray) -> float:
        """Return the total log-likelihood of the returns under the model."""

    @abstractmethod
    def fit(self, returns: np.ndarray):
        """Estimate the model's parameters from a return series."""