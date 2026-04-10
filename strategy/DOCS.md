# Strategy Framework Documentation

Reference for all helper classes, parameters, and coding guidelines in `main.py`.

---

## Architecture Overview

```
Trader (entry point)
├── __init__()          ← params, strategies, market makers created ONCE
└── run(state)          ← called every timestep
     ├── strat.reset(state)  ← refreshes per-tick state
     └── strat.run(state_dict)
          └── self.mm.execute(self, ...)  ← reuses existing MarketMaker
```

**Key principle:** Anything that doesn't change between ticks lives in `__init__`. The `run()` method only handles per-tick work.

---

## Classes

### `Logger`

Handles all output. Single instance shared across strategies.

| Method | Format | Purpose |
|--------|--------|---------|
| `log(**series)` | `LOGVIZ:{"key":value,...}` | Numeric series for the log visualizer |
| `log_order(side, price, qty, tag)` | `LOGORDER:SIDE:PRICE:QTY:TAG` | Per-order trade attribution |
| `debug(msg, tag, product)` | `LOGDBG:TAG:PRODUCT:MSG` | Text debug messages |
| `flush()` | — | Returns buffered output (when `auto_print=False`) |

**`log_order` tags:** Each order placed via `bid()`/`ask()` is automatically logged with its tag, so you can trace which decision produced which order.

---

### `MidPriceHelper`

Static utility class. All methods take an `OrderDepth` and return `Optional[float]`.

| Method | Formula | Use Case |
|--------|---------|----------|
| `get_vwap(od)` | `Σ(price × volume) / Σ(volume)` | Volume-weighted fair value across all levels |
| `get_bba_mid(od)` | `(best_bid + best_ask) / 2` | Simple midpoint of top-of-book |
| `get_pop(od)` | `(highest_vol_bid + highest_vol_ask) / 2` | Mid from most liquid price levels |

**When to use which:**
- `get_bba_mid` — fast, stable, good default for EMA input
- `get_vwap` — accounts for depth, better for trend signals (SMA/slope)
- `get_pop` — resistant to thin outlier levels, useful when book is skewed

---

### `PositionManager`

FIFO inventory tracker. Reconstructs average cost basis from trade history.

```python
pos_mgr = PositionManager("TOMATOES", state_dict)
pos_mgr.update(state.own_trades.get("TOMATOES", []))
avg = pos_mgr.get_avg_price(n_units=5.0)
pos_mgr.persist(state_dict)
```

| Method | Purpose |
|--------|---------|
| `update(trades)` | Process new trades into FIFO inventory layers |
| `get_avg_price(n_units)` | Weighted avg price of last N units (newest first) |
| `persist(state_dict)` | Save inventory + last trade time to state |

**FIFO netting:** When a trade in the opposite direction arrives, it closes the oldest layers first. If the trade reverses the position entirely, the remainder starts a new layer.

**State keys:** Uses `TOMATO_LAST_TRADE_TIME` (singular) for backwards compatibility with `old_main.py`.

---

### `BaseStrategy`

Abstract base class. All product strategies inherit from this.

**Class constant:**
- `PRODUCT = ""` — Override in subclass (e.g. `"EMERALDS"`)

**Lifecycle:**
1. `__init__(logger, params)` — Called once. Stores static config.
2. `reset(state)` — Called each tick. Refreshes position, capacity, order book.
3. `run(state_dict)` — Called each tick. Contains strategy logic. Returns `List[Order]`.

**Order placement:**

| Method | Action | Auto-logged as |
|--------|--------|----------------|
| `bid(price, qty, tag)` | Place buy order (clamped to `buy_capacity`) | `LOGORDER:BUY:...` |
| `ask(price, qty, tag)` | Place sell order (clamped to `sell_capacity`) | `LOGORDER:SELL:...` |

Both methods automatically prevent position limit breaches via real-time capacity tracking. You never need to manually check limits.

**`get_walls(min_vol=1)`** — Returns `(best_bid, best_ask)` where each level has at least `min_vol` volume. Used to determine where to place making orders.

---

### `MarketMaker`

Reusable helper that handles the full take/make cycle. Created once per strategy in `__init__`.

```python
self.mm = MarketMaker(aggression=0.0, fair_build_ratio=0.5, make=True)
```

#### Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `aggression` | `float` | `0.0` | Max loss per order for outlier removal |
| `fair_build_ratio` | `float` | `0.5` | Max `\|pos\|/limit` to allow fair-value building |
| `make` | `bool` | `True` | Whether to place maker (limit) orders |

#### Taking Logic (3 tiers)

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Netting at Fair Value                [TAKE_NET]     │
│ ─ Always active. Zero risk.                                 │
│ ─ Buy at fair if short, sell at fair if long                │
│ ─ Capped to position size (net toward 0)                    │
├─────────────────────────────────────────────────────────────┤
│ TIER 2a: Build at Fair Value                 [TAKE_FAIR]    │
│ ─ Active when |pos| < fair_build_ratio × limit              │
│ ─ Exhausts all volume at fair price                         │
│ ─ Zero risk (buying/selling at fair value)                  │
├─────────────────────────────────────────────────────────────┤
│ TIER 2b: Outlier Removal                     [TAKE/TAKE_AGG]│
│ ─ Better than fair → loss = 0 → always taken       [TAKE]  │
│ ─ Worse than fair → loss = vol × deviation  [TAKE_AGG]      │
│ ─ Only taken if loss ≤ aggression parameter                 │
└─────────────────────────────────────────────────────────────┘
```

**Loss formula:** `loss = volume × max(0, deviation_from_fair)`
- When price is better than fair: deviation ≤ 0 → loss = 0 → always passes
- When price is worse than fair: deviation > 0 → gated by `aggression`

**Example (aggression=5, fair=10000):**
```
Book: bid 9999 (vol=3), bid 9993 (vol=20)
Loss for 9999 bid: 3 × (10000 - 9999) = 3 ≤ 5 → TAKE_AGG (sell at 9999)
Making: overbid 9993 → place bid at 9994
Round trip: sold 9999, buy 9994 = 5 profit/unit
```

#### Making Logic

After taking, the maker places orders to capture spread:

1. **Overbid:** Find best bid below fair with vol > 1, place bid at `bp + 1`
2. **Undercut:** Find best ask above fair with vol > 1, place ask at `sp - 1`
3. Fill remaining capacity at these prices

---

## Execution Order (per tick)

```
1. Trader.run(state)
2.   Parse state_dict from traderData JSON
3.   For each product:
4.     strat.reset(state)          → refresh position, capacity, order_depth
5.     strat.run(state_dict)       → strategy-specific logic:
6.       a. Compute mid price target (EMA, fixed wall, etc.)
7.       b. mm.execute(...)         → taking + making
8.       c. Post-filters (trend, inventory) modify self.orders in-place
9.     result[product] = strat.orders
10.  Serialize state_dict → traderData JSON
```

---

## Adding a New Strategy

```python
class MyStrategy(BaseStrategy):
    PRODUCT = "MY_PRODUCT"
    
    def __init__(self, logger: Logger, params: Dict[str, Any]):
        super().__init__(logger, params)
        self.mm = MarketMaker(aggression=0, fair_build_ratio=0.5)
    
    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        # 1. Compute your mid price target
        mid = MidPriceHelper.get_bba_mid(self.order_depth)
        if mid is None:
            return []
        
        # 2. Run market maker
        bid_wall, ask_wall = self.get_walls()
        self.mm.execute(self, mid, bid_wall, ask_wall)
        
        # 3. Optional: post-filters on self.orders
        
        return self.orders
```

Then register in `Trader.__init__`:
```python
self.my_params = {"position_limit": 50}
self.my_strat = MyStrategy(self.logger, self.my_params)
```

And in `Trader.run()`:
```python
if "MY_PRODUCT" in state.order_depths:
    self.my_strat.reset(state)
    result["MY_PRODUCT"] = self.my_strat.run(state_dict)
```

---

## Coding Guidelines

### 1. Init vs Run

| Belongs in `__init__` | Belongs in `run` / `reset` |
|---|---|
| Logger, param dicts | Position, capacity |
| Strategy objects | Order book (`order_depth`) |
| MarketMaker instances | Orders list |
| Position limit | State dict reads/writes |
| Any config that is constant across ticks | Anything derived from current market state |

### 2. Order Placement

- **Always use `bid()` / `ask()`**, never `self.orders.append(Order(...))` directly. The helper methods enforce capacity limits and emit log lines.
- **Tag every order** with a descriptive tag (`TAKE`, `TAKE_NET`, `TAKE_FAIR`, `TAKE_AGG`, `MAKE`). This is critical for post-analysis.

### 3. State Persistence

- All cross-tick state goes through `state_dict` (serialized as `traderData` JSON).
- Use descriptive keys: `"TOMATOES_EMA"`, `"TOMATO_PRICES"`, `"TOMATO_SMAS"`.
- Always handle missing keys with `.get(key, default)`.

### 4. Post-Filters

Post-filters modify `self.orders` after `MarketMaker.execute()`. They are applied in sequence:

1. **Trend Filter** — Remove orders against the trend (slope-based).
2. **Inventory Filter** — Remove closing orders that would realize a loss vs avg cost.

Filters should log removals: `self.logger.log(orders_removed_avg=removed)`.

### 5. Naming Conventions

- **Strategy classes:** `{Product}Strategy` (e.g. `EmeraldStrategy`)
- **PRODUCT constant:** Uppercase product name as it appears in `state.order_depths`
- **Params:** `snake_case`, descriptive (e.g. `slope_threshold`, not `st`)
- **State keys:** `UPPER_SNAKE_CASE` (e.g. `TOMATOES_EMA`)

### 6. Performance

- Sort order book data once and reuse (`sorted_asks`, `sorted_bids`).
- Avoid creating objects in `run()` that could live in `__init__`.
- `PositionManager` is per-tick because it reads from `state_dict` which changes.
