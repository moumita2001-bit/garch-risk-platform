# References

Sources behind every model, test, formula, data set and tool used in VolRisk.
Each entry notes where it is used in the project.

## 1. Stylized facts of financial returns

**Cont, R. (2001).** Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
The standard summary of volatility clustering, fat tails and the leverage effect. Framework for the Week 1 EDA ([eda_summary.md](eda_summary.md)).

**Mandelbrot, B. (1963).** The variation of certain speculative prices. *Journal of Business*, 36(4), 394–419.
First documented that large price changes follow large changes, and that return tails are heavier than normal. Motivation for testing fat tails and clustering.

**Black, F. (1976).** Studies of stock price volatility changes. *Proceedings of the 1976 Meetings of the American Statistical Association, Business and Economic Statistics Section*, 177–181.
Origin of the leverage effect: falling prices raise volatility more than rising prices. Explains the negative skew (−0.29) in Week 1 and γ = 0.27 in Week 4.

## 2. Volatility models

**Engle, R. F. (1982).** Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
Introduced ARCH and the ARCH-LM test. The test (LM = 1,301) justified moving beyond constant volatility.

**Bollerslev, T. (1986).** Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
GARCH(1,1), implemented from scratch in Week 2 (`models/garch.py`), including the stationarity condition α + β < 1.

**Bollerslev, T. (1987).** A conditionally heteroskedastic time series model for speculative prices and rates of return. *Review of Economics and Statistics*, 69(3), 542–547.
GARCH with Student-t innovations, implemented in Week 3 (`models/student_t.py`).

**Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993).** On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
GJR-GARCH: negative shocks get extra weight γ. The production model since Week 4 (`models/gjr.py`).

**Nelson, D. B. (1991).** Conditional heteroskedasticity in asset returns: a new approach. *Econometrica*, 59(2), 347–370.
EGARCH, the main alternative asymmetric model. Considered and deferred as a future comparison.

**J.P. Morgan/Reuters (1996).** *RiskMetrics — Technical Document* (4th ed.). New York: J.P. Morgan.
The EWMA volatility model (λ = 0.94), a restricted GARCH with ω = 0 and α + β = 1. Discussed as a benchmark, not used as a model.

**Hansen, P. R., & Lunde, A. (2005).** A forecast comparison of volatility models: does anything beat a GARCH(1,1)? *Journal of Applied Econometrics*, 20(7), 873–889.
Evidence that GARCH(1,1) is hard to beat for daily forecasting. Supports choosing (1,1) over higher orders.

**Bollerslev, T., Chou, R. Y., & Kroner, K. F. (1992).** ARCH modeling in finance: a review of the theory and empirical evidence. *Journal of Econometrics*, 52(1–2), 5–59.
Survey of the ARCH family; background reading for model choice.

## 3. Estimation and model selection

**Akaike, H. (1974).** A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
AIC, computed in `GarchFit.aic`.

**Schwarz, G. (1978).** Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.
BIC, computed in `GarchFit.bic`. Student-t beat normal by 271 BIC points (Week 3).

**Kraft, D. (1988).** A software package for sequential quadratic programming. DFVLR-FB 88-28, Institut für Dynamik der Flugsysteme, Oberpfaffenhofen.
The SLSQP algorithm used by `scipy.optimize.minimize` for constrained maximum-likelihood estimation.

## 4. Diagnostic tests

**Ljung, G. M., & Box, G. E. P. (1978).** On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297–303.
Ljung-Box test on returns (84) and squared returns (4,023) in Week 1.

**Engle (1982)** above: ARCH-LM test.

## 5. Realized volatility estimators

**Garman, M. B., & Klass, M. J. (1980).** On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.
Garman-Klass range-based variance used in `data/clean.py`; unbiasedness verified by Monte Carlo test.

**Yang, D., & Zhang, Q. (2000).** Drift-independent volatility estimation based on high, low, open, and close prices. *Journal of Business*, 73(3), 477–491.
Extension that handles overnight gaps; noted as a future improvement.

## 6. Value-at-Risk and backtesting

**Jorion, P. (2006).** *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.
Reference text for VaR definitions and parametric VaR, VaR = −(μ + σ·q).

**Kupiec, P. H. (1995).** Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
Proportion-of-failures (unconditional coverage) test, implemented in `risk/backtest.py`.

**Christoffersen, P. F. (1998).** Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
Independence test for breach clustering, implemented in `risk/backtest.py`.

**Basel Committee on Banking Supervision (1996).** *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.* Bank for International Settlements.
The traffic-light zones (green 0–4, yellow 5–9, red 10+ breaches in 250 days), implemented in `basel_zone`.

**Basel Committee on Banking Supervision (2019).** *Minimum capital requirements for market risk* (FRTB). Bank for International Settlements.
Moves regulatory capital toward Expected Shortfall; cited as a planned extension.

**Basel Committee on Banking Supervision (2013).** *Principles for effective risk data aggregation and risk reporting* (BCBS 239). Bank for International Settlements.
Motivation for the UTC download audit logs and data lineage.

## 7. Implied volatility and the variance risk premium

**Cboe Global Markets.** *Cboe Volatility Index (VIX) White Paper.* Chicago Board Options Exchange.
How VIX is computed from S&P 500 option prices (Week 5).

**Carr, P., & Wu, L. (2009).** Variance risk premiums. *Review of Financial Studies*, 22(3), 1311–1341.
Documents implied variance exceeding realized variance; context for the +3.7-point gap found in Week 5.

**Bollerslev, T., Tauchen, G., & Zhou, H. (2009).** Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463–4492.
Further evidence and interpretation of the variance risk premium.

## 8. Textbooks

**Tsay, R. S. (2010).** *Analysis of Financial Time Series* (3rd ed.). Wiley.
Chapter 3 (conditional heteroscedastic models) used for GARCH theory, the likelihood and constraints.

**Hamilton, J. D. (1994).** *Time Series Analysis.* Princeton University Press.
Maximum-likelihood background.

## 9. Data sources

**Tiingo.** End-of-day stock prices API. https://www.tiingo.com — adjusted daily OHLC for SPY (primary source).

**Federal Reserve Bank of St. Louis.** FRED, series VIXCLS (CBOE Volatility Index: VIX). https://fred.stlouisfed.org/series/VIXCLS — daily VIX (Week 5).

**Stooq.** https://stooq.com — one-time manual historical SPY backfill.

## 10. Software

**Sheppard, K., et al.** *arch: ARCH models in Python.* https://github.com/bashtage/arch — reference implementation used to validate the from-scratch models (Weeks 2–3).

**Virtanen, P., et al. (2020).** SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. — optimization (`minimize`, SLSQP) and distributions.

**Harris, C. R., et al. (2020).** Array programming with NumPy. *Nature*, 585, 357–362.

**McKinney, W. (2010).** Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. — pandas.

**Seabold, S., & Perktold, J. (2010).** statsmodels: econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96. — Ljung-Box and ARCH-LM tests in the EDA.

## Formula index

| Formula | Where used | Source |
|---|---|---|
| r_t = ln(P_t / P_{t−1}) | `data/clean.py` | Tsay (2010), ch. 1 |
| σ²_GK = ½[ln(H/L)]² − (2 ln 2 − 1)[ln(C/O)]² | `data/clean.py` | Garman & Klass (1980) |
| σ²_t = ω + α ε²_{t−1} + β σ²_{t−1} | `models/garch.py` | Bollerslev (1986) |
| Long-run variance ω / (1 − α − β); half-life ln 0.5 / ln(α + β) | `GarchParams` | Tsay (2010), ch. 3 |
| Gaussian log-likelihood −½ Σ [ln 2π + ln σ²_t + ε²_t/σ²_t] | `GARCH.log_likelihood` | Bollerslev (1986); Hamilton (1994) |
| Standardized Student-t log-likelihood | `models/student_t.py` | Bollerslev (1987) |
| σ²_t = ω + (α + γ I[ε_{t−1} < 0]) ε²_{t−1} + β σ²_{t−1} | `models/gjr.py` | Glosten, Jagannathan & Runkle (1993) |
| AIC = 2k − 2ℓ; BIC = k ln n − 2ℓ | `GarchFit` | Akaike (1974); Schwarz (1978) |
| VaR_p = −(μ + σ_{T+1} q_p) | `risk/var.py` | Jorion (2006) |
| Kupiec LR_uc ~ χ²(1) | `kupiec_test` | Kupiec (1995) |
| Christoffersen LR_ind ~ χ²(1) | `christoffersen_test` | Christoffersen (1998) |
| Ljung-Box Q = n(n+2) Σ ρ̂²_k / (n−k) | `notebooks/01_eda.py` | Ljung & Box (1978) |
| ARCH-LM = n·R² ~ χ²(q) | `notebooks/01_eda.py` | Engle (1982) |