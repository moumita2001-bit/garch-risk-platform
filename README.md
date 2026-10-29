# garch-risk-platform
Cloud-deployed volatility and Value-at-Risk platform: GARCH models built from scratch in Python, served through a REST API with daily automated runs, backtesting and model monitoring on AWS.


   ## Architecture

   ![VolRisk platform architecture](docs/architecture.svg)

   ## Results so far

**Week 1: Stylized facts of SPY returns (2008–2026)**
- Excess kurtosis 14.2; 37 four-sigma days vs 0.3 expected under normality
- ARCH-LM = 1,301 (p < 0.0001): strong volatility clustering
- Details: [EDA summary](docs/eda_summary.md)

**Week 2: GARCH(1,1) built from scratch**
- Maximum-likelihood estimation with scipy, 36 tests including parameter recovery
- Matches the `arch` package to 4 decimal places (α = 0.149, β = 0.829)
- Details: [validation report](docs/garch_validation.md)