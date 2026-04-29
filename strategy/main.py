import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "position_limit": 10,
    "window_size": 40,        # Number of periods for rolling average/std
    "z_entry_threshold": 2.0, # Z-score required to enter a position
    "z_exit_threshold": 0.5,  # Z-score required to close a position
    "hedge_ratio": 1.48        # Asset A moves 1.5x as much as Asset B (Update this based on your calculation)
}

class Logger:
    """Logger with order-level trade attribution."""
    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    def _emit(self, line: str) -> None:
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)

    def log(self, **series: float) -> None:
        if not series:
            return
        clean: dict[str, float] = {}
        for k, v in series.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not clean:
            return
        self._emit(self.PREFIX + json.dumps(clean, separators=(',', ':')))

    def log_order(self, product: str, side: str, price: int, qty: int, tag: str) -> None:
        """Log an individual order placement for strategy-level trade attribution."""
        self._emit(f'LOGORDER:{product}:{side}:{price}:{qty}:{tag}')

    def debug(self, msg: str, tag: str = 'DBG', product: str = '') -> None:
        self._emit(f'LOGDBG:{tag}:{product}:{msg}')

    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out


class PairsStrategy:
    """Strategy for trading a correlated pair using bid-ask adjusted Z-score reversion."""
    
    def __init__(self, logger: Logger, params: Dict[str, Any], prod_a: str, prod_b: str):
        self.logger = logger
        self.params = params
        self.prod_a = prod_a
        self.prod_b = prod_b
        
        self.limit = params.get("position_limit", 20)
        self.window_size = params.get("window_size", 40)
        self.z_entry = params.get("z_entry_threshold", 2.0)
        self.z_exit = params.get("z_exit_threshold", 0.5)
        self.hedge_ratio = params.get("hedge_ratio", 1.0)

    def _size_pair(self, max_qty_a: int, max_qty_b: int) -> Tuple[int, int]:
        n = min(max_qty_a, self.limit)
        while n > 0 and round(n * self.hedge_ratio) > max_qty_b:
            n -= 1
        return n, round(n * self.hedge_ratio)

    def run(self, state: TradingState, state_dict: Dict[str, Any]) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {self.prod_a: [], self.prod_b: []}

        # 1. Ensure we have order book data for both assets
        if self.prod_a not in state.order_depths or self.prod_b not in state.order_depths:
            return orders

        depth_a = state.order_depths[self.prod_a]
        depth_b = state.order_depths[self.prod_b]

        # 2. Cannot trade if books are empty
        if not depth_a.buy_orders or not depth_a.sell_orders or not depth_b.buy_orders or not depth_b.sell_orders:
            return orders

        # 3. Extract Best Prices
        best_bid_a = max(depth_a.buy_orders.keys())
        best_ask_a = min(depth_a.sell_orders.keys())
        best_bid_b = max(depth_b.buy_orders.keys())
        best_ask_b = min(depth_b.sell_orders.keys())

        # 4. Calculate Mid Prices (For clean statistical tracking)
        mid_a = (best_bid_a + best_ask_a) / 2.0
        mid_b = (best_bid_b + best_ask_b) / 2.0

        # Track the mid-price spread for our rolling statistics
        mid_spread = mid_a - (self.hedge_ratio * mid_b)

        # 5. Persist spread history
        spread_history = state_dict.get("spread_history", [])
        spread_history.append(mid_spread)
        
        if len(spread_history) > self.window_size:
            spread_history.pop(0)
            
        state_dict["spread_history"] = spread_history

        # Wait until we have enough data to calculate reliable Z-score
        if len(spread_history) < self.window_size:
            return orders

        # 6. Calculate Rolling Statistics
        mean = sum(spread_history) / len(spread_history)
        variance = sum((x - mean) ** 2 for x in spread_history) / len(spread_history)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return orders

        # ─── BID-ASK ADJUSTED EXECUTION SPREADS ──────────────────────────────

        # What spread do we actually get if we SHORT the pair? (Sell A at bid, Buy B at ask)
        exec_short_spread = best_bid_a - (self.hedge_ratio * best_ask_b)
        z_score_short = (exec_short_spread - mean) / std_dev

        # What spread do we actually get if we LONG the pair? (Buy A at ask, Sell B at bid)
        exec_long_spread = best_ask_a - (self.hedge_ratio * best_bid_b)
        z_score_long = (exec_long_spread - mean) / std_dev

        # Mid Z-Score for clean exits
        mid_z_score = (mid_spread - mean) / std_dev

        # Log to visualize in IMC visualizer
        self.logger.log(
            mid_spread=mid_spread, 
            z_short=z_score_short, 
            z_long=z_score_long
        )

        # Retrieve current positions
        pos_a = int(state.position.get(self.prod_a, 0))
        pos_b = int(state.position.get(self.prod_b, 0))

        # ─── TRADING LOGIC ──────────────────────────────────────────────────

        # 1. ENTRY LOGIC: Short A, Long B (Checking the harsh 'short' execution spread)
        if z_score_short > self.z_entry:
            max_short_a = self.limit + pos_a 
            max_long_b = self.limit - pos_b
            
            trade_qty_a, trade_qty_b = self._size_pair(max_short_a, max_long_b)

            if trade_qty_a > 0:
                orders[self.prod_a].append(Order(self.prod_a, best_bid_a, -trade_qty_a))
                orders[self.prod_b].append(Order(self.prod_b, best_ask_b, trade_qty_b))
                
                self.logger.debug(f"Entered Short Pair | Adjusted Z: {z_score_short:.2f}")

        # 2. ENTRY LOGIC: Long A, Short B (Checking the harsh 'long' execution spread)
        elif z_score_long < -self.z_entry:
            max_long_a = self.limit - pos_a   
            max_short_b = self.limit + pos_b  
            
            trade_qty_a, trade_qty_b = self._size_pair(max_long_a, max_short_b)

            if trade_qty_a > 0:
                orders[self.prod_a].append(Order(self.prod_a, best_ask_a, trade_qty_a))
                orders[self.prod_b].append(Order(self.prod_b, best_bid_b, -trade_qty_b))
                
                self.logger.debug(f"Entered Long Pair | Adjusted Z: {z_score_long:.2f}")

        # 3. EXIT LOGIC (Reversion to mean)
        elif abs(mid_z_score) < self.z_exit:
            flattened = False
            if pos_a != 0:
                price_a = best_ask_a if pos_a < 0 else best_bid_a
                orders[self.prod_a].append(Order(self.prod_a, price_a, -pos_a))
                flattened = True
                
            if pos_b != 0:
                price_b = best_ask_b if pos_b < 0 else best_bid_b
                orders[self.prod_b].append(Order(self.prod_b, price_b, -pos_b))
                flattened = True

            if flattened:
                self.logger.debug(f"Pair Exit: Flattening positions | Mid Z: {mid_z_score:.2f}")

        return orders


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = TUNING_PARAMS
        
        # Instantiate our pairs strategy
        # NOTE: Replace 'ASSET_A' and 'ASSET_B' with actual product names
        self.pairs_strategy = PairsStrategy(
            logger=self.logger,
            params=self.params,
            prod_a="PEBBLES_XS",
            prod_b="UV_VISOR_AMBER"
        )
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_data = state.traderData if state.traderData else "{}"
        
        try:
            state_dict = json.loads(trader_data)
        except Exception:
            state_dict = {}

        # 1. Run the pairs strategy
        pair_orders = self.pairs_strategy.run(state, state_dict)
        
        # 2. Add valid orders to the result dictionary
        for prod, ord_list in pair_orders.items():
            if ord_list:
                result[prod] = ord_list

        # 3. Re-encode the updated persistent state dictionary
        new_trader_data = json.dumps(state_dict)

        # 4. Flush logger to print visualization outputs
        self.logger.flush()
        
        # Return orders, conversions (always 0 unless specified), and new state
        return result, 0, new_trader_data