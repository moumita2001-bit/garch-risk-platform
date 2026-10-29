"""Risk measures built on the volatility models."""

from volrisk.risk.var import one_day_var, shock_quantile

__all__ = ["one_day_var", "shock_quantile"]
