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



## Week 2 — GARCH(1,1) from scratch

**Goal:** Implement GARCH(1,1) estimation myself and prove it is correct
before relying on it for VaR.

### What I built
- `models/base.py`: abstract `VolatilityModel` interface so later models
  (GJR-GARCH, Student-t) plug into the same VaR and backtest code.
- `models/garch.py`: variance recursion, Gaussian log-likelihood, and
  maximum-likelihood fit with `scipy.optimize` (SLSQP, with bounds and an
  alpha + beta < 1 stationarity constraint).
- `models/simulate.py`: GARCH simulator with burn-in and seeded randomness.
- 23 new tests (36 total): hand-calculated recursion, likelihood vs
  `scipy.stats`, statistical properties of simulated data, and
  parameter recovery on 20,000 simulated days.
- `notebooks/02_garch_validation.py`: side-by-side comparison with `arch`.

### Results on SPY (2008–2026)
| Parameter | Mine | arch |
|---|---|---|
| alpha | 0.1486 | 0.1486 |
| beta | 0.8289 | 0.8289 |
| alpha + beta | 0.9774 | 0.9774 |
| Long-run vol (annualized) | 18.8% | 18.8% |

Estimates match to 4 decimal places. Log-likelihoods differ by 0.3 because
the two implementations start the variance recursion differently (sample
variance vs exponentially weighted backcast).

### What I learned
- Validate an estimator twice: on simulated data where the truth is known
  (parameter recovery), and against a trusted reference on real data.
- Returns must be in percent: on decimal scale omega is ~0.000003 and the
  optimizer struggles.
- When alpha = 0, beta and omega are not separately identified; only the
  long-run variance is. Tests should check what the data can pin down.
- Simulated GARCH with normal shocks gives excess kurtosis ~2, while SPY
  shows 14.2. GARCH explains part of the fat tails; Student-t innovations
  are needed for the rest.

### Next
Student-t innovations, then turn conditional volatility into 1-day VaR.



## Week 3 — Student-t innovations

**Goal:** Test whether fat-tailed shocks explain SPY better than normal shocks.

### What I built
- Refactored `GARCH.fit()` into hooks (`_make_params`, `_initial_guess`,
  `_bounds`) so new models subclass instead of copying the optimizer.
  All 36 existing tests passed unchanged after the refactor.
- `StudentTGARCH`: standardized Student-t likelihood (via `gammaln`),
  nu bounded in [2.05, 500]. AIC/BIC now count parameters automatically.
- Simulator draws t shocks; 11 new tests (47 total), including recovery of
  nu = 6 from simulated data and AIC/BIC choosing the correct model.
- `notebooks/03_student_t_comparison.py` and `docs/student_t_comparison.md`.

### Results on SPY
| | Normal | Student-t |
|---|---|---|
| Log-likelihood | -6408.48 | -6268.92 |
| BIC | 12850.81 | 12580.15 |
| alpha + beta | 0.977 | 0.996 |
| nu | – | 5.46 |

- BIC favours Student-t by 271 points.
- Normal-GARCH residuals still have excess kurtosis 1.87, and 11 days beyond
  4 sigma against 0.3 expected under normality (14.8 under t).
- My estimates match `arch` (dist="t") to 3-4 decimal places.

### What I learned
- GARCH explains most of the fat tails, but not all; the shocks themselves
  are fat-tailed.
- Under t, crash days stop distorting the variance dynamics, and persistence
  rises to 0.996. That makes the long-run variance poorly identified
  (~34% annualized), so long-horizon forecasts need care.
- Refactor first and keep the tests green, then add the feature.

### Next
Turn the 1-day volatility forecast into Value-at-Risk.


## Week 4 — Value-at-Risk and backtesting

**Goal:** Turn the volatility forecast into 1-day 99% VaR and test it
out of sample on SPY.

### What I built
- One-step variance forecast and `one_day_var` (normal and Student-t quantiles).
- `rolling_var`: re-fit every 21 days on the previous 1,000 days; every
  forecast uses only earlier data. A test proves it: changing future
  returns leaves past forecasts unchanged (no look-ahead).
- Kupiec (breach rate) and Christoffersen (breach clustering) tests,
  Basel traffic-light zones.
- `zero_mean` option and GJR-GARCH-t, added through the existing fit hooks.
- 30 new tests (77 total).

### Results: 3,732 test days, 37 breaches expected
| Model | Breaches | Kupiec p | Christoffersen p |
|---|---|---|---|
| Normal GARCH | 87 | <0.001 | 0.208 |
| Student-t GARCH | 59 | 0.001 | 0.016 |
| Student-t, mu = 0 | 54 | 0.010 | 0.008 |
| GJR-t, mu = 0 | 48 | 0.092 | 0.155 |

### How I found the fix
- Splitting breaches by tail: 1.58% on losses vs 0.21% on gains, so total
  risk was about right but the distribution sat too far toward gains.
- Fixing mu = 0 removed 5 breaches; the rest was asymmetry.
- GJR on the full sample: gamma = 0.27, alpha = 0. Only down days raise
  SPY volatility (the leverage effect).

### What I learned
- A backtest that fails is useful: it shows where the model is wrong.
- Change one thing at a time and measure, rather than adding features
  until the numbers pass.
- 48 breaches still leans slightly high; the model passes, but only just.

### Next
FRED/VIX data and a one-command pipeline (Week 5).