"""Week 1 exploration: do SPY returns show the patterns that justify GARCH?

Reads data/processed/spy.csv (made by volrisk.data.clean), checks the three
"stylized facts" of financial returns, saves charts to docs/figures/ and
writes the findings to docs/eda_summary.md.

  1. Volatility clustering: calm and turbulent periods group together
  2. Fat tails: extreme days happen far more often than a normal curve predicts
  3. Predictable volatility: big moves follow big moves (ARCH effects)

Usage:
    python notebooks/01_eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "spy.csv"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
SUMMARY_FILE = PROJECT_ROOT / "docs" / "eda_summary.md"

TRADING_DAYS = 252
# Label, date, and height (fraction of the top of the chart) so labels don't collide
CRISES = [
    ("2008 crisis", "2008-10-15", 0.92),
    ("COVID 2020", "2020-03-16", 0.92),
    ("Rates 2022", "2022-06-13", 0.72),
]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date")
    df["return_pct"] = 100 * df["log_return"]  # percent, easier to read
    return df


def summary_stats(r: pd.Series) -> dict:
    """Basic facts about daily returns (in percent)."""
    return {
        "days": len(r),
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "mean_daily_pct": r.mean(),
        "vol_daily_pct": r.std(),
        "vol_annual_pct": r.std() * np.sqrt(TRADING_DAYS),
        "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r),  # 0 for a normal distribution
        "worst_day_pct": r.min(),
        "worst_day": r.idxmin().date(),
        "best_day_pct": r.max(),
        "best_day": r.idxmax().date(),
    }


def tail_counts(r: pd.Series) -> dict:
    """How many 4-sigma days actually happened vs a normal curve's prediction."""
    z = (r - r.mean()) / r.std()
    actual = int((z.abs() > 4).sum())
    expected = len(r) * 2 * stats.norm.sf(4)
    return {"actual_4sd_days": actual, "normal_expected_4sd_days": expected}


def fig_returns(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["return_pct"], lw=0.5, color="#185FA5")
    top = df["return_pct"].max()
    for label, day, height in CRISES:
        ax.annotate(label, xy=(pd.Timestamp(day), top * height),
                    ha="center", fontsize=9, color="#712B13")
    ax.set_title("SPY daily returns: volatility comes in clusters")
    ax.set_ylabel("Daily return (%)")
    path = FIG_DIR / "01_returns_clustering.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_volatility(df: pd.DataFrame) -> Path:
    """Annualized volatility: 21-day rolling close-to-close vs Garman-Klass."""
    cc = df["log_return"].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
    gk = np.sqrt(df["rv_gk"].rolling(21).mean() * TRADING_DAYS) * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(cc.index, cc, lw=0.8, color="#888780", label="Close-to-close (21-day)")
    ax.plot(gk.index, gk, lw=0.8, color="#0F6E56", label="Garman-Klass (21-day)")
    ax.set_title("Annualized volatility over time: long calm stretches, sudden spikes")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend(frameon=False)
    path = FIG_DIR / "02_rolling_volatility.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_fat_tails(r: pd.Series) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.linspace(r.min(), r.max(), 500)
    ax1.hist(r, bins=150, density=True, color="#85B7EB", alpha=0.8, label="SPY returns")
    ax1.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color="#993C1D", lw=1.5,
             label="Normal curve, same mean and SD")
    ax1.set_xlim(-6, 6)
    ax1.set_title("Taller peak and fatter tails than a normal curve")
    ax1.set_xlabel("Daily return (%)")
    ax1.legend(frameon=False, fontsize=8)

    stats.probplot(r, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(markersize=2, color="#185FA5")
    ax2.get_lines()[1].set(color="#993C1D")
    ax2.set_title("Q-Q plot: points leave the line in both tails")
    ax2.set_xlabel("Normal distribution quantiles")
    ax2.set_ylabel("SPY daily return quantiles (%)")
    path = FIG_DIR / "03_fat_tails.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_acf(r: pd.Series) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    plot_acf(r, lags=30, ax=ax1, zero=False, title="Returns: almost no autocorrelation")
    plot_acf(r**2, lags=30, ax=ax2, zero=False,
             title="Squared returns: strong, slowly fading autocorrelation")
    for ax in (ax1, ax2):
        ax.set_xlabel("Lag (trading days)")
    ax1.set_ylabel("Autocorrelation")
    path = FIG_DIR / "04_acf_returns_vs_squared.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def arch_tests(r: pd.Series) -> dict:
    """Formal tests: is today's volatility predictable from the past?"""
    demeaned = r - r.mean()
    lm_stat, lm_p, _, _ = het_arch(demeaned, nlags=10)
    lb_ret = acorr_ljungbox(r, lags=[10]).iloc[0]
    lb_sq = acorr_ljungbox(r**2, lags=[10]).iloc[0]
    return {
        "arch_lm_stat": lm_stat, "arch_lm_p": lm_p,
        "lb_returns_stat": lb_ret["lb_stat"], "lb_returns_p": lb_ret["lb_pvalue"],
        "lb_squared_stat": lb_sq["lb_stat"], "lb_squared_p": lb_sq["lb_pvalue"],
    }


def fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def write_summary(s: dict, t: dict, a: dict, figs: list[Path]) -> None:
    rel = [f.relative_to(SUMMARY_FILE.parent).as_posix() for f in figs]
    verdict = (
        "Squared returns are strongly autocorrelated and the ARCH-LM test rejects "
        "constant volatility, so a GARCH model is justified."
        if a["arch_lm_p"] < 0.05 else
        "The ARCH-LM test does not reject constant volatility at 5%; revisit before fitting GARCH."
    )
    text = f"""# Week 1 exploration: SPY daily returns

Data: {s['days']:,} trading days from {s['start']} to {s['end']} (Tiingo, adjusted prices).
Generated by `notebooks/01_eda.py`.

## Summary statistics

| Measure | Value |
| --- | --- |
| Average daily return | {s['mean_daily_pct']:.3f}% |
| Daily volatility | {s['vol_daily_pct']:.2f}% |
| Annualized volatility | {s['vol_annual_pct']:.1f}% |
| Skewness | {s['skew']:.2f} |
| Excess kurtosis (0 for a normal curve) | {s['excess_kurtosis']:.1f} |
| Worst day | {s['worst_day_pct']:.2f}% on {s['worst_day']} |
| Best day | {s['best_day_pct']:.2f}% on {s['best_day']} |

## 1. Volatility clustering

![Daily returns]({rel[0]})

![Rolling volatility]({rel[1]})

Turbulent days bunch together around 2008, 2020 and 2022, separated by long calm stretches.

## 2. Fat tails

![Fat tails]({rel[2]})

Moves beyond 4 standard deviations happened on **{t['actual_4sd_days']} days**; a normal
curve predicts about **{t['normal_expected_4sd_days']:.1f}**. Excess kurtosis of
{s['excess_kurtosis']:.1f} supports Student-t errors in the models.

## 3. Volatility is predictable

![Autocorrelation]({rel[3]})

| Test (10 lags) | Statistic | p-value | Reading |
| --- | --- | --- | --- |
| Ljung-Box on returns | {a['lb_returns_stat']:.1f} | {fmt_p(a['lb_returns_p'])} | Significant but small: direction of returns is barely predictable |
| Ljung-Box on squared returns | {a['lb_squared_stat']:.1f} | {fmt_p(a['lb_squared_p'])} | Strong: size of moves is predictable |
| ARCH-LM | {a['arch_lm_stat']:.1f} | {fmt_p(a['arch_lm_p'])} | Volatility changes over time |

## Conclusion

{verdict}
"""
    SUMMARY_FILE.write_text(text)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    r = df["return_pct"]

    s, t, a = summary_stats(r), tail_counts(r), arch_tests(r)
    figs = [fig_returns(df), fig_volatility(df), fig_fat_tails(r), fig_acf(r)]
    write_summary(s, t, a, figs)

    print(f"Data: {s['days']:,} days, {s['start']} to {s['end']}")
    print(f"Annualized volatility: {s['vol_annual_pct']:.1f}%   "
          f"Excess kurtosis: {s['excess_kurtosis']:.1f}")
    print(f"4-sigma days: {t['actual_4sd_days']} actual vs "
          f"{t['normal_expected_4sd_days']:.1f} expected under a normal curve")
    print(f"ARCH-LM (10 lags): stat {a['arch_lm_stat']:.1f}, p {fmt_p(a['arch_lm_p'])}")
    print(f"Saved {len(figs)} charts to {FIG_DIR.relative_to(PROJECT_ROOT)}/ "
          f"and summary to {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()