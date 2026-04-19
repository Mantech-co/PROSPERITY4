# Options Trading: Data Analysis & Industrial Methods

---

## I. MARKET MICROSTRUCTURE DATA

### Order Book Analysis
- Full L2/L3 order book reconstruction — bid/ask queues, queue position, cancel/replace rates
- Order flow imbalance (OFI) — net signed order flow predicts short-term price moves
- Trade-through analysis — when orders cross the spread, directional signal
- Hidden order detection — inferring iceberg orders from fill patterns
- Latency arbitrage windows — stale quotes across venues

### Tick Data
- Trade-by-trade time-and-sales with exchange timestamps
- Aggressive vs passive fill classification (Lee-Ready, BVC algorithms)
- Intraday volume profiles — VWAP, TWAP curve modeling
- Microstructure noise filtering for realized volatility estimation

---

## II. VOLATILITY SURFACE ANALYSIS

This is the core of options trading.

### Surface Construction
- Fit implied vol surface from market quotes: strike x expiry -> IV
- Standard models: SVI (Stochastic Volatility Inspired), SABR, local vol (Dupire)
- Arbitrage-free constraints: calendar spread arb, butterfly arb, put-call parity
- Smoothing: cubic splines, kernel regression, neural net surface fitting

### Surface Dynamics
- Sticky strike vs sticky delta regimes — how surface moves when spot moves
- Skew sensitivity (dIV/dK) — put skew encodes tail risk demand
- Term structure of vol — contango/backwardation in VIX futures analog
- Realized vs implied vol spread — the fundamental edge in vol trading

### Greeks Surface
- Full delta, gamma, vega, theta, vanna, volga across the surface
- Vanna: dDelta/dVol — critical for skew hedging
- Volga: dVega/dVol — convexity of vol, drives risk reversals vs strangles pricing
- Dollar gamma/dollar vanna maps across surface

---

## III. STATISTICAL SIGNALS

### Volatility Forecasting
- GARCH family: GARCH(1,1), EGARCH (asymmetric), GJR-GARCH
- HAR-RV (Heterogeneous Autoregressive Realized Variance) — best empirical forecaster
- Realized volatility decomposition: continuous + jump components (BNS test)
- High-frequency realized variance estimators: two-scale, kernel-based (Barndorff-Nielsen)
- Options-implied forecasts vs realized — the risk premium

### Cross-Asset Signals
- Equity-vol correlation regimes — correlation between spot returns and IV changes
- Variance risk premium (VRP): E[RV] - IV^2 — systematically positive, harvested by vol sellers
- Cross-sectional IV rank among universe — relative value vol trades
- Correlation surface: implied correlation from index vol vs constituent vol (dispersion)
- Skew premium: 25-delta risk reversal vs realized skew

### Statistical Arbitrage on Surface
- Dispersion trading: long single-stock vol, short index vol (exploits correlation premium)
- Calendar spread richness/cheapness vs term structure model
- Horizontal spread (ratio spreads) relative value
- Fitting residuals from SVI surface — mispriced strikes

---

## IV. RISK FACTOR MODELS

### Factor Decomposition
- PCA on volatility surface — first 3 PCs explain ~95% of surface moves (level, slope, curvature)
- Factor loadings map exposures: parallel vol shift, term structure tilt, skew twist
- Covariance matrix estimation: Ledoit-Wolf shrinkage, factor covariance

### Greeks P&L Attribution
- Decompose daily P&L: delta P&L, gamma P&L (realized vol contribution), vega P&L (IV move), carry (theta)
- Gamma scalping: long gamma position delta-hedged continuously — profits from realized > implied vol
- Vega bucketing: isolate exposure to each expiry's IV independently

---

## V. EXECUTION & ALPHA SIGNAL STACK

### Market Making (MM)
- Avellaneda-Stoikov model for optimal bid-ask spread given inventory and vol
- Inventory risk management — skew quotes to shed position
- Adverse selection modeling — separate informed vs noise flow
- Quote stuffing detection, toxic flow filters
- Realized spread vs effective spread tracking

### Directional Options Trading
- Delta-1 with leverage: options as leveraged directional — analyze underlying with full quant stack
- Momentum signals on underlying translated to options positions
- Earnings volatility — IV crush post-event, straddle buying pre-event models
- Event vol surface: fit pre/post event IV separately, extract event vol component

### Volatility Arbitrage
- Pure vol positions: delta-hedge everything, isolate vega/gamma exposure
- Long gamma short vega: buy short-dated, sell long-dated (realized vs implied mismatch)
- Short gamma long vega: inverse — theta collection against tail exposure
- Skew arbitrage: over/underpriced OTM options vs model

---

## VI. PORTFOLIO CONSTRUCTION

### Greek Aggregation
- Net delta, gamma, vega, theta, vanna, volga across entire book
- Scenario analysis: stress the surface, stress spot simultaneously (vol-spot correlation)
- Tail Greeks: charm (delta decay), speed (gamma of gamma) for dynamic hedging

### Optimization
- Mean-variance in IV space — maximize Sharpe on vol positions
- Kelly criterion for position sizing given edge and variance of P&L
- Correlation-aware sizing — vol trades on correlated underlyings not independent
- VaR and CVaR on the Greeks vector, not just P&L history

### Hedging Strategy
- Delta hedge frequency optimization: transaction cost vs gamma P&L tradeoff
- Vega hedging with liquid instruments (ATM options, VIX products)
- Proxy hedging — when exact hedge too expensive, use correlated instrument

---

## VII. INDUSTRIAL TRADING STRATEGIES

### 1. Volatility Market Making
- Continuous two-sided quotes on options chains
- Real-time surface re-fitting as market moves
- Inventory-aware quoting — widen on concentrated risk
- Hedge delta continuously, let vega/gamma accumulate to natural limits

### 2. Statistical Vol Arb
- Identify mispriced vol nodes on surface vs fitted model
- Enter positions delta-neutral, vega-neutral across strikes
- Harvest mean-reversion of fitting residuals
- Firms: Citadel, SIG, Optiver, Jane Street

### 3. Dispersion / Correlation Trading
- Short index variance, long variance on constituents (or reverse)
- Pure correlation exposure — correlation risk premium harvesting
- Sizing by realized vs implied correlation spread

### 4. Gamma Scalping / Delta Hedging
- Buy cheap realized vol (buy options below fair IV)
- Continuously delta-hedge at optimal frequency
- P&L = 0.5 * Gamma * (sigma_realized^2 - sigma_implied^2) * S^2 * dt
- Net positive if realized vol systematically exceeds implied

### 5. Earnings / Event Volatility
- Model event vol component isolation
- Buy pre-event, sell post-event (IV crush)
- Or sell pre-event if IV overprices event move historically
- Requires accurate earnings move distribution model

### 6. Skew Trading
- Risk reversals: sell put skew when it's rich vs historical
- Butterfly spreads: isolate curvature richness
- Requires robust skew premium signal vs risk adjustment

### 7. Term Structure / Roll Yield
- VIX futures term structure — short front, long back in contango
- Options calendar spreads — harvest theta differential
- Roll yield harvesting with vol risk management

### 8. Cross-Asset Vol
- Equity vol vs rates vol correlations
- FX vol vs equity vol relationships
- Commodity vol term structure
- Statistical relationships break under stress — regime detection required

---

## VIII. REGIME & MACRO OVERLAY

- HMM (Hidden Markov Models) for vol regime detection: low/med/high vol regimes
- Risk-on/risk-off classification — changes hedge ratios, position sizing
- Macro factor sensitivity: rates, credit spreads, VIX term structure shape
- Liquidity regime monitoring — bid-ask spreads widen, reduce size

---

## IX. RISK MANAGEMENT DATA SYSTEMS

- Real-time Greeks aggregation across entire portfolio
- P&L explain: predicted vs actual — residuals flag model error
- Scenario grids: spot +-10%, vol +-30%, combined
- Liquidity-adjusted VaR — options illiquid, can't always hedge
- Concentration limits per expiry, per strike cluster, per underlying

---

Core edge: surface mispricings revert. Realized vol != implied vol over time. Correlation premium exists. All else is execution.
