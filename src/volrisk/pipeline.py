"""The daily VolRisk job: download -> clean -> fit -> VaR -> results file.

One function, `run`, does the whole job so a scheduler (cron today, AWS
Fargate in Week 8) only needs to call a single command:

    volrisk run

Every step is idempotent: downloads skip if already done today, and the
results file for a given date is simply overwritten if the job reruns.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from volrisk.models import GJRStudentTGARCH
from volrisk.risk import one_day_var

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TRADING_DAYS = 252


class PipelineError(RuntimeError):
    """Raised when the job cannot produce a trustworthy result."""


def _default_download_prices(ticker: str, start: str, force: bool) -> Path:
    from volrisk.data.download import download_ticker

    return download_ticker(ticker, start, force=force)


def _default_download_vix(force: bool) -> Path:
    from volrisk.data.fred import download_series

    return download_series("VIXCLS", force=force)


def _default_clean(ticker: str) -> Path:
    from volrisk.data.clean import clean_ticker

    return clean_ticker(ticker)


def latest_vix(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    vix = pd.read_csv(path, parse_dates=["date"]).dropna()
    last = vix.iloc[-1]
    return {"date": str(last["date"].date()), "value": float(last["vixcls"])}


def run(
    ticker: str = "SPY",
    start: str = "2008-01-01",
    force: bool = False,
    results_dir: Path = RESULTS_DIR,
    model=None,
    download_prices: Callable = _default_download_prices,
    download_vix: Callable = _default_download_vix,
    clean: Callable = _default_clean,
) -> Path:
    """Run the daily job and return the path of the results file it wrote."""
    model = model or GJRStudentTGARCH(zero_mean=True)

    logger.info("1/4 downloading %s and VIX", ticker)
    download_prices(ticker, start, force)
    try:
        vix_path = download_vix(force)
    except Exception as exc:  # VIX is context, not an input: keep going
        logger.warning("VIX download failed (%s); continuing without it", type(exc).__name__)
        vix_path = None

    logger.info("2/4 cleaning %s", ticker)
    processed = pd.read_csv(clean(ticker), parse_dates=["date"]).set_index("date")
    returns = (100 * processed["log_return"]).dropna()
    if len(returns) < 1000:
        raise PipelineError(f"only {len(returns)} returns; need at least 1,000 to fit")

    logger.info("3/4 fitting %s on %d days", model.name, len(returns))
    r = returns.to_numpy()
    fit = model.fit(r)
    if not fit.converged:
        raise PipelineError(f"model did not converge: {fit.message}")

    logger.info("4/4 computing VaR")
    sigma = float(np.sqrt(model.forecast_variance(fit.params, r)))
    as_of = returns.index[-1].date()
    result = {
        "ticker": ticker,
        "as_of": str(as_of),
        "forecast_for": "next trading day",
        "model": model.name,
        "n_obs": len(r),
        "params": {k: round(v, 6) for k, v in vars(fit.params).items()},
        "log_likelihood": round(fit.log_likelihood, 2),
        "forecast_vol_daily_pct": round(sigma, 4),
        "forecast_vol_annual_pct": round(sigma * np.sqrt(TRADING_DAYS), 2),
        "var_99_pct": round(one_day_var(model, fit.params, r, p=0.01), 4),
        "var_95_pct": round(one_day_var(model, fit.params, r, p=0.05), 4),
        "vix": latest_vix(vix_path),
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{ticker.lower()}_{as_of}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("saved %s: 99%% VaR %.2f%%", out.name, result["var_99_pct"])
    return out
