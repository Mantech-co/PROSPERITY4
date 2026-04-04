import json
import math
from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict

class Trader:
    def __init__(self):
        # Prosperity 4 tutorial limits
        self.LIMIT = {
            "EMERALDS": 80,
            "TOMATOES": 80
        }
        
        # ---------------------------------------------------------
        # HYPERPARAMETERS (Optimizer Mode)
        # ---------------------------------------------------------
        # self.ma_window = int(os.getenv("OPT_MA_WINDOW", "3"))
        # self.thresh_a = float(os.getenv("OPT_THRESH_A", "0.1"))
        # self.thresh_b = float(os.getenv("OPT_THRESH_B", "0.175"))
        # self.thresh_c = float(os.getenv("OPT_THRESH_C", "10.0"))
        
        # ---------------------------------------------------------
        # HYPERPARAMETERS (Submission Mode - Uncomment for final submission)
        # ---------------------------------------------------------
        self.ma_window = 6
        self.thresh_a = 0.12
        self.thresh_b = 0.195
        self.thresh_c = 13.0

    def calculate_thresholds(self, position: int, limit: int) -> tuple[float, float]:
        """Calculates dynamic threshold percentages based on inventory."""
        a = self.thresh_a
        b = self.thresh_b
        c = self.thresh_c
        
        # BUY Threshold Logic
        if position <= 0:
            # We are short or flat: no penalty to buy
            buy_threshold = a
        else:
            # We are long: exponentially harder to get MORE long
            x = (position / limit) * 100
            buy_threshold = a + (b - a) * (1 - math.exp(-x / c))
            
        # SELL Threshold Logic
        if position >= 0:
            # We are long or flat: no penalty to sell
            sell_threshold = a
        else:
            # We are short: exponentially harder to get MORE short
            x = (abs(position) / limit) * 100
            sell_threshold = a + (b - a) * (1 - math.exp(-x / c))
            
        return buy_threshold, sell_threshold

    def tomato_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int,
        product_state: dict
    ) -> tuple[List[Order], dict]:
        orders: List[Order] = []
        limit = self.LIMIT[product]
        
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        
        if best_ask is None or best_bid is None:
            return orders, product_state
            
        # ---------------------------------------------------------
        # 1. Rolling Window Moving Average Logic
        # ---------------------------------------------------------
        if "BID_HISTORY" not in product_state:
            product_state["BID_HISTORY"] = []
        if "ASK_HISTORY" not in product_state:
            product_state["ASK_HISTORY"] = []
            
        bid_history = product_state["BID_HISTORY"]
        ask_history = product_state["ASK_HISTORY"]
        
        # Append current tick prices
        bid_history.append(best_bid)
        ask_history.append(best_ask)
        
        # Trim to optimized window size
        if len(bid_history) > self.ma_window:
            bid_history.pop(0)
        if len(ask_history) > self.ma_window:
            ask_history.pop(0)
            
        bid_mean = sum(bid_history) / len(bid_history)
        ask_mean = sum(ask_history) / len(ask_history)

        # ---------------------------------------------------------
        # 2. Dynamic Threshold Mathematics
        # ---------------------------------------------------------
        buy_threshold_pct, sell_threshold_pct = self.calculate_thresholds(position, limit)
        
        # Convert raw percentage (e.g., 0.1) into a decimal multiplier
        buy_multiplier = 1 - (buy_threshold_pct / 100)
        sell_multiplier = 1 + (sell_threshold_pct / 100)
        
        buy_vol = 0
        sell_vol = 0
        
        # ---------------------------------------------------------
        # 3. Aggressive Spread Crossing (The Alpha)
        # ---------------------------------------------------------
        if order_depth.sell_orders:
            ask_qty = abs(order_depth.sell_orders[best_ask])
            
            # Favorable deviation: ask is significantly lower than its recent mean
            if best_ask < (ask_mean * buy_multiplier):
                qty = min(ask_qty, limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty

        if order_depth.buy_orders:
            bid_qty = order_depth.buy_orders[best_bid]
            
            # Favorable deviation: bid is significantly higher than its recent mean
            if best_bid > (bid_mean * sell_multiplier):
                qty = min(bid_qty, limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty

        # ---------------------------------------------------------
        # 4. Passive Market Making (Safe Pennying)
        # ---------------------------------------------------------
        bid_capacity = limit - (position + buy_vol)
        ask_capacity = limit + (position - sell_vol)

        # Bug Fix: Ensure we never cross a 1-tick spread when resting orders
        quote_bid = min(best_bid + 1, best_ask - 1)
        quote_ask = max(best_ask - 1, best_bid + 1)

        if bid_capacity > 0:
            orders.append(Order(product, quote_bid, bid_capacity))
        if ask_capacity > 0:
            orders.append(Order(product, quote_ask, -ask_capacity))

        return orders, product_state

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        
        try:
            trader_state = json.loads(state.traderData)
        except Exception:
            trader_state = {}
            
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            
            # Isolate state dictionary per product
            if product not in trader_state:
                trader_state[product] = {}
                
            if product == "EMERALDS":
                pass
            elif product == "TOMATOES":
                orders, updated_state = self.tomato_orders(product, depth, pos, trader_state[product])
                result[product] = orders
                trader_state[product] = updated_state
                
        trader_data_out = json.dumps(trader_state)
        return result, 1, trader_data_out
