# Data Analysis for Options Alpha Discovery

---

## I. ORDER BOOK RAW ANALYSIS

### Queue Dynamics
- **Queue position inference** — at what queue depth do fills happen? Reveals true liquidity
- **Cancel-to-trade ratio by price level** — high cancel rate at a level = spoofing / fake wall
- **Order arrival rate (Poisson intensity)** — does arrival rate spike before moves? Informed flow
- **Refresh rate of best bid/ask** — fast refresh = MM, slow = real resting order
- **Size distribution at top of book** — fat-tailed? Bimodal? Reveals participant composition

### Order Flow Imbalance (OFI)
- Raw: (buy volume - sell volume) at each tick
- Cumulative OFI vs price — divergence predicts reversal, convergence predicts continuation
- **OFI across multiple levels** — weight by distance from mid, captures depth pressure
- Cross-asset OFI correlation — does options flow lead underlying OFI?

### Trade Classification
- Lee-Ready: trades above mid = buyer-initiated, below = seller-initiated
- **VPIN (Volume-synchronized Probability of Informed Trading)** — rolling imbalance metric, spikes before volatility events
- Bulk volume classification — aggregate trade direction in short windows

---

## II. OPTIONS-SPECIFIC RAW DATA ANALYSIS

### Implied Volatility Extraction
- From raw option prices + underlying: back out IV per strike per expiry
- **IV residuals from a fitted surface** — the single most important signal. Mispriced options mean someone knows something or model is wrong
- Time series of IV residuals — mean-reverting? Momentum? Both at different frequencies

### Volatility Surface Shape Analysis
- **Skew slope** (dIV/dK normalized) — changes in skew *before* underlying moves
- **Term structure slope** — front vol rising relative to back = near-term fear
- **Butterfly (curvature)** — fat tails being priced in or out
- PCA on daily surface snapshots — decompose into level/slope/curvature factors, analyze factor time series independently

### Put-Call Parity Violations
- Raw: C - P vs S - K*e^(-rT) — any persistent deviation = funding cost signal or supply/demand imbalance
- Violations by strike — which strikes are being pushed off parity? Why?

### Open Interest + Volume Analysis
- OI concentration by strike — strikes with massive OI act as gravitational centers for price (pinning)
- Volume/OI ratio — high ratio = speculative, low = hedging. Different alpha implications
- **OI change overnight** — new positions being built. Directional signal
- Call/put OI ratio shifts across strikes — tracks hedging demand shift

---

## III. REALIZED VOLATILITY ANALYSIS

### Estimators (from raw OHLCV or tick)
- **Parkinson**: uses high-low range — more efficient than close-close
- **Garman-Klass**: OHLC — even more efficient
- **Yang-Zhang**: handles overnight gaps
- **Realized variance from ticks**: sum of squared log-returns at high frequency
- Compare estimators — when they diverge, microstructure noise is present

### Jump Detection
- **BNS test** (Barndorff-Nielsen-Shephard): separate continuous vol from jumps in realized variance
- Jump days vs non-jump days — do options misprice jump risk? Is IV elevated before jumps?
- Jump clustering — do jumps arrive in clusters? (Hawkes process)

### Volatility Autocorrelation
- ACF of |returns|, squared returns — long memory (ARFIMA, HAR structure)
- HAR-RV: RV today predicted by RV yesterday, last week, last month — each has independent forecasting power
- **Volatility of volatility** — how much does vol itself move? Drives volga exposure value

---

## IV. CROSS-SECTIONAL ANALYSIS (if multiple underlyings)

### Correlation Structure
- Rolling realized correlation matrix — how stable is it?
- **Implied vs realized correlation spread** — dispersion trade signal
- Correlation breakdown detection — when individual stock vols decouple from index vol

### Relative Value
- IV rank / IV percentile per underlying — cross-sectional richness/cheapness
- Skew rank cross-sectionally — which underlyings have abnormally steep put skew?
- Beta-adjusted vol comparison — after removing market beta, which residual vols are cheap?

---

## V. MICROSTRUCTURE NOISE IN OPTIONS

- Options tick data is sparse — bid-ask bounce dominates realized variance
- **Noise-to-signal ratio**: compare high-frequency RV to daily RV — large ratio = microstructure dominated
- Mid-price vs last-trade price — which is more informative?
- **Effective spread analysis**: (trade price - mid) x 2 = cost of immediacy per strike
- Wide effective spreads at OTM strikes = liquidity desert, arb signals there are unreliable

---

## VI. GAMMA EXPOSURE DERIVED SIGNALS

### Dealer Positioning Inference
- Aggregate dealer gamma by strike = sum of (open interest x gamma) per strike
- **Positive dealer gamma**: dealers long gamma -> they sell rallies, buy dips -> dampens vol
- **Negative dealer gamma**: dealers short gamma -> they buy rallies, sell dips -> amplifies vol
- Net GEX zero-crossing = volatility regime transition point

### Charm / Delta Decay
- As expiry approaches, delta of ITM options rises, OTM falls
- Dealers must rehedge — predictable, scheduled flow
- Largest charm exposure = most predictable intraday directional pressure near expiry

### Vanna Flow
- When spot moves, delta changes via vanna — dealers rehedge
- Direction of vanna flow is predictable from surface shape
- Large vanna exposure = vol moves cause systematic spot flows

---

## VII. TIME-SERIES SIGNALS ON RAW DATA

### Mean Reversion vs Momentum Detection
- **Hurst exponent** on IV time series — H < 0.5 mean-reverting, H > 0.5 trending
- Run this per strike, per expiry — different regions of surface have different dynamics
- **Half-life of mean reversion** (Ornstein-Uhlenbeck fit) — tells you holding period for arb

### Cointegration
- IV pairs across strikes — do two strikes share a long-run equilibrium?
- Spread between 25-delta put IV and 25-delta call IV — cointegrated with skew factor
- When spread deviates: reversion trade with known half-life

### Regime Detection on Raw Data
- **Markov Regime Switching** on vol series — identify low/high vol regimes from data alone
- Transition probabilities — how persistent is each regime?
- In competition: label each day by regime, train separate models per regime

---

## VIII. INFORMATION CONTENT ANALYSIS

### What Predicts What
- Does options IV lead realized vol? (Granger causality test)
- Does put/call volume ratio predict spot direction? (regression, lagged)
- Does skew change predict next-day spot move direction? (directional accuracy)
- Does term structure inversion predict volatility spikes? (event study)

### Event Study Methodology
- Align all large IV spikes, compute average spot path +-N days around event
- Align all skew inversions — what happens to spot after?
- **Cumulative abnormal return (CAR)** analysis: do options signals generate abnormal underlying returns?

---

## IX. WHAT ANALYSIS ACTUALLY FINDS ALPHA IN COMPETITIONS

Most competition alpha comes from:

1. **IV residuals from surface fit** — consistent mispricings in specific strikes
2. **OFI leading price** — options order flow precedes underlying move
3. **GEX sign changes** — predict vol regime shifts
4. **Realized vs implied spread z-score** — systematic vol selling or buying signal
5. **Skew changes leading spot** — put buyers know before price moves
6. **OI cliff analysis** — price pinning into expiry is mechanical and exploitable
7. **Vanna/charm flows near expiry** — scheduled, predictable, size-able

---

Raw data hierarchy of informativeness: **options flow > order book imbalance > realized vol estimators > price**. Work backwards from derivatives to spot.
