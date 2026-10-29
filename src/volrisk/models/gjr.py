"""GJR-GARCH(1,1) with Student-t innovations (Glosten, Jagannathan, Runkle 1993).

Adds the leverage effect: bad news raises volatility more than good news.
    sigma2_t = omega + (alpha + gamma * I[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma2_{t-1}

A negative shock gets weight alpha + gamma; a positive one gets alpha.
gamma = 0 gives back ordinary GARCH(1,1).
"""

from dataclasses import dataclass

import numpy as np

from volrisk.models.student_t import StudentTGARCH, StudentTParams


@dataclass(frozen=True)
class GJRParams(StudentTParams):
    """Student-t GARCH parameters plus the leverage coefficient gamma."""

    gamma: float  # extra reaction to negative shocks

    @property
    def persistence(self) -> float:
        # Negative shocks occur half the time (symmetric z), so gamma counts half.
        return self.alpha + self.gamma / 2 + self.beta

    def is_valid(self) -> bool:
        return super().is_valid() and self.gamma >= 0


class GJRStudentTGARCH(StudentTGARCH):
    """GJR-GARCH(1,1) with standardized Student-t innovations."""

    name = "GJR-GARCH(1,1)-t"

    def _make_params(self, x: np.ndarray) -> GJRParams:
        return GJRParams(*(float(v) for v in x))

    def _initial_guess(self, r: np.ndarray) -> np.ndarray:
        return np.append(super()._initial_guess(r), 0.05)  # gamma

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        return super()._bounds() + [(0.0, 1.0)]
