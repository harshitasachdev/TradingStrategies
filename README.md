# Trading Strategies — Harshita Sachdev

Quantitative trading strategy research built as part of the MSc Quantitative Finance programme at Singapore Management University. Three end-to-end projects covering statistical arbitrage, regime-aware portfolio construction, and technical momentum strategies — each implemented in Python with full backtesting and performance evaluation.

---

## Projects

### 1. Pairs Trading Based on Cointegration with ML Clustering
**File:** `Pairs_Trading_Strategy_Based_on_Cointegration_With_ML_Clustering.ipynb`

A market-neutral statistical arbitrage strategy applied to FX pairs and US equities, combining cointegration-based pair selection with ML clustering to identify tradeable pairs.

**Methodology**
- Screened 1,800+ US equities (filtered from 7,000 NASDAQ-listed stocks) using WRDS and Yahoo Finance data, retaining only tickers with 10 years of complete price, market cap, volume, and net profit margin data
- Applied pairwise OLS regression and Augmented Dickey-Fuller (ADF) tests to identify cointegrated pairs with stationary spread (p < 0.05)
- Validated mean-reversion properties using Hurst Exponent; used KMeans and DBSCAN clustering with PCA/t-SNE dimensionality reduction to group stocks by return behaviour before pair selection
- Constructed z-score spread signals with entry at ±1.5σ, stop-loss at ±2.5σ, and take-profit on spread convergence below ±0.5σ
- Applied hedge ratio from OLS regression to construct dollar-neutral long/short positions

**Results**
- Peak P&L of 368% over a two-year trading period
- Sharpe ratio: 1.16
- Maximum drawdown: 26%

---

### 2. Portfolio Construction using MVO with Alpha Overlay
**File:** `Project_25Jun_v2.ipynb`

A regime-aware, multi-asset portfolio that combines machine learning alpha forecasts with Mean-Variance Optimisation (MVO), benchmarked against a 60/40 SPY/TLT portfolio over 2013–2025.

**Methodology**
- Investment universe: 13 ETFs spanning equities, fixed income, commodities, real estate, and factor exposures (SPY, TLT, AGG, GLD, DBC, HYG, EFA, VNQ, TIP, BIL, MTUM, VLUE, USMV)
- Feature set: Fama-French five factors, macro indicators (CPI, unemployment, 10Y real rates), idiosyncratic ETF-level technical metrics, and predicted regime dummies
- Forecasted four macro regimes (Goldilocks, Heating Up, Slow Growth, Stagflation) using an SVM classifier trained on growth, inflation, and liquidity data — achieving 66% walk-forward accuracy
- Forecasted 1-month forward ETF returns using rolling Linear Regression, ElasticNet, and XGBoost models with a 5-year lookback window
- Translated predicted alphas into monthly portfolio weights using MVO (max Sharpe), with position size constraints (max 30% per asset) and long-only constraint
- Performed regime-specific performance attribution, decomposing returns by asset class, macro regime, and style factor

**Results**
- Outperformed 60/40 benchmark by ~50% in cumulative returns (2013–2025)
- Maximum drawdown: ~24%
- Attribution analysis showed outperformance concentrated in Goldilocks and Heating Up regimes; identified underperformance drivers in Stagflation regime

---

### 3. MACD-RSI Momentum Strategy — US Tech Equities
**File:** `MACD-RSI_Strategy.ipynb`

A dual-indicator momentum strategy combining MACD crossover signals with RSI trend confirmation, backtested across five large-cap US tech stocks over a 10-year horizon.

**Methodology**
- Universe: AAPL, MSFT, GOOGL, AMZN, NVDA — backtested 2015–2024 on $100,000 initial capital per stock
- Signal logic: long on bullish MACD crossover (12/26/9) with RSI < 50; short on bearish MACD crossover with RSI > 50 — RSI used as a trend direction filter, not an overbought/oversold indicator
- Custom RSI implementation using Wilder's smoothing method (14-day lookback); positions shifted by one day to eliminate lookahead bias; commission drag of $5 per trade included in capital simulation
- Evaluated using annualised Sharpe ratio, cumulative log returns, and peak-to-trough drawdown analysis with duration and magnitude

**Results**
- Best risk-adjusted performance on MSFT: Sharpe 0.85, cumulative return +4.7%
- Strategy performance varied significantly by volatility regime; trend-following signals underperformed on high-volatility names (NVDA, GOOGL), highlighting the need for regime-conditional RSI threshold calibration — identified as a direction for further research

---

## Tech Stack

- **Languages:** Python
- **Libraries:** pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn, plotly, yfinance, scipy
- **Techniques:** Cointegration, ADF testing, Hurst Exponent, KMeans/DBSCAN clustering, PCA, t-SNE, OLS regression, ElasticNet, XGBoost, SVM, Mean-Variance Optimisation, walk-forward validation
- **Data sources:** Yahoo Finance, WRDS, NASDAQ API, FRED (macro indicators)

---

## Contact

Harshita Sachdev
[LinkedIn](https://sg.linkedin.com/in/harshita-sachdev) · harshita.s.2024@mqf.smu.edu.sg
