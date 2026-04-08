# Prosperity4BT Core Logic Explanation

This document explains the core logic behind how backtesting, order matchmaking, and execution function within the `prosperity4bt` package. The implementation is primarily divided between `test_runner.py` and `tools/order_match_maker.py`.

## 1. Backtest Loop Logic
The backtesting loop is managed by the `TestRunner` class (`test_runner.py`). It simulates the sequential passage of time, feeding market data to the trading algorithm and processing its orders.

For each timestamp in the market data, the following steps occur execution:
1. **Initialize Trading State:** 
    - The `TradingState` is constructed for the current timestamp (`__initialize_trade_state`).
    - The market data (bids and asks) is translated into an `OrderDepth` for each product. Bids are mapped with positive volumes, while asks are given negative volumes.
    - Various `Observation` values (such as `ConversionObservation` for related products) are also populated if present in the data.
2. **Trader Invocation:** 
    - The user-provided algorithm is executed via `self.trader.run(state)`.
    - The algorithm’s standard output (print statements) is captured and appended to a sandbox lambda log for debugging purposes.
    - The algorithm returns its requested `orders`, `conversions`, and `trader_data` string (which is carried over to the next timestamp's state).
3. **Limit Enforcement:** 
    - The `__enforce_limits` method checks the generated orders against hardcoded position limits per product.
    - It calculates the maximum possible long and short exposure if all orders for a product were filled. 
    - **Important:** If an order combination for a product would exceed its maximum position `LIMITS`, *all* orders for that product are completely discarded for the current timestamp.
4. **Order Matching:** 
    - The accepted orders are passed to the `OrderMatchMaker`.
5. **Activity Logging:** 
    - The results, limits evaluation, and sandbox outputs are appended to the overall `BacktestResult`.

## 2. Match Making and Execution Logic
Once orders pass the limit checks, the `OrderMatchMaker` (`tools/order_match_maker.py`) attempts to execute them against the available market state. Order matching operates in two primary phases:

### Phase 1: Order Depth (Order Book) Matching
Algorithm orders are first matched against the existing `OrderDepth` limits available at the timestamp:
- **Buy Orders (Quantity > 0):** The matchmaker inspects the `sell_orders` in the order depth. It filters for available sell prices that are $\le$ the buy order's price. It sorts these available sell levels from lowest to highest. It then deducts volume from the algorithm's order and the respective order book level until the order is filled or available sell volume is exhausted.
- **Sell Orders (Quantity < 0):** The matchmaker inspects the `buy_orders` in the order depth. It filters for available buy prices that are $\ge$ the sell order's price. Buy prices are sorted highest to lowest. Matching occurs similarly, consuming volume until the order is filled or buy volume runs out.

### Phase 2: Market Trades Matching
If an algorithm order is not completely filled by the order book, the system optionally attempts to match the remainder against historical `MarketTrade` sequences (dependent on the `TradeMatchingMode`).
- **Buy Orders:** If historical market trades show sellers at a price $\le$ the algorithm's buy order price, the algorithm may "step in" and match against those historical seller quantities.
- **Sell Orders:** If historical market trades show buyers at a price $\ge$ the algorithm's sell order price, the algorithm matches against the historical buyer quantities.

### Order Execution & State Updates
Whenever any part of an order successfully matches (either from Phase 1 or Phase 2):
1. **Position Update:** The current position for the executed product is updated in `self.state.position`. Buy fills increase position, sell fills decrease it.
2. **Profit & Loss Update:** The overall PnL tracker for the backtest is updated in `self.back_data.profit_loss`. Buy orders subtract $(price \times volume)$, and sell orders add $(price \times volume)$ to the running balance.
3. **Trade Creation:** A new `Trade` record is generated denoting "SUBMISSION" as either the buyer or the seller. These trades are appended to the algorithm's `own_trades` dictionary to be accessible in the subsequent timestamp's state.

Unfilled portions of algorithm orders simply expire. They are not maintained as limit orders on an order book for the next timestamp.
