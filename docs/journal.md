# VolRisk Build Journal

## Week 1 — Data pipeline and exploratory analysis (ended 2026-10-25)

**Goal:** Build a reliable daily price pipeline for SPY and confirm the
stylized facts that justify a GARCH model.

### What I built
- `volrisk.data.download`: Tiingo API client with retries and exponential
  backoff, a skip-if-already-downloaded-today guard, and a UTC audit log
  (`data/raw/download_log.csv`).
- `volrisk.data.clean`: validation, de-duplication, gap flags, log returns,
  and Garman-Klass realized variance. Output: 4,732 trading days
  (2008-01-03 to 2026-10-23).
- 13 unit tests, including a Monte Carlo simulation showing the
  Garman-Klass estimator is unbiased.
- `notebooks/01_eda.py`: a reproducible script that regenerates four charts
  and `docs/eda_summary.md` in one command.

### What the data says
| Stylized fact | Evidence |
|---|---|
| Volatility clustering | ARCH-LM(10) = 1,301, p < 0.0001 |
| Fat tails | Excess kurtosis 14.2; 37 four-sigma days vs 0.3 expected |
| Size of moves is far more predictable than direction | Ljung-Box(10): 4,023 on squared returns vs 84 on returns |

Annualized volatility is 19.7%. A constant-variance, normal-distribution
VaR would understate tail risk badly. That's the case for GARCH with
Student-t innovations.

### Decisions
- **Tiingo over Stooq.** Stooq blocks scripted downloads with a browser
  check, so I used it only for a one-time manual backfill.
- **Editable install (`pip install -e .`)** so tests and scripts import
  `volrisk` the same way production code will.
- **`python -m pytest`** to make sure the venv's pytest runs, not Anaconda's.
- **EDA as a `.py` script, not a notebook:** clean diffs in PRs, no hidden
  cell state, and one command reproduces every chart and number.

### Problems and fixes
- Home network stalls large transfers: pip downloads froze at ~32 KB and
  `git push` failed with "RPC failed; curl 56". Diagnosed by switching to
  a phone hotspot: everything succeeded there, so the cause is the home
  network, not the Mac or Python setup. Workaround: hotspot for big
  downloads and pushes; next step is an MTU test on the home router.
- Installed `ipykernel` over the hotspot so VS Code can run notebook cells.
- Corrected the EDA summary: the Ljung-Box test on returns is significant
  (p < 0.0001), so I changed "Weak" to "Significant but small". Fixed in
  the script so the wording survives regeneration.

### What I learned
- Small requests worked while large transfers failed, so testing on a
  second network was the quickest way to separate a local setup problem
  from a network problem.
- Being statistically significant isn't the same as being economically
  meaningful. With 4,700 observations, even tiny autocorrelation shows up
  as significant.

### Next week
Implement GARCH(1,1) from scratch: base model class, MLE with
`scipy.optimize`, a parameter-recovery test on simulated data, and
validation against the `arch` package.