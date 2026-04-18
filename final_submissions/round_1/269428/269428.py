from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict
import json
import numpy as np 

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

class Trader:
    def __init__(self):
        self.logger = Logger()
        
        # Prosperity 4 tutorial limits
        self.LIMIT = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }
        
        # --- IPR Strategy Parameters ---
        self.slope = 0.00100253
        self.min_hold_limit = 75
        self.min_delta = 5.5
        self.empty_book_delta = 110  # Hyperparameter for empty book spread
        
        # --- ACO Strategy Parameters ---
        self.delta = 90 # Hyperparameter for empty book spread
        self.aco_aggressive_clear_limit = 40
        self.aco_soft_position_limit = 50
        self.aco_clear_width = 4
        
        # --- IPR State Variables (Defaults for Tick 0) ---
        self.ipr_start_mid_price = None
        self.ipr_start_timestamp = None
        self.ipr_microprice_history = [] 
        self.ipr_limit_reached = False
        self.last_best_bid = None
        self.last_best_ask = None

        # --- ACO State Variables (Defaults for Tick 0) ---
        self.prev_best_bid = None
        self.prev_best_ask = None

    def IPR_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int,
        timestamp: int
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.LIMIT[product]
        
        sell_orders = dict(order_depth.sell_orders)
        buy_orders = dict(order_depth.buy_orders)

        best_ask = min(sell_orders.keys()) if sell_orders else None
        best_bid = max(buy_orders.keys()) if buy_orders else None

        if self.ipr_start_mid_price is None:
            if best_ask is not None and best_bid is not None:
                self.ipr_start_mid_price = (best_bid + best_ask) / 2.0
                self.ipr_start_timestamp = timestamp
            else:
                return orders 

        time_elapsed = timestamp - self.ipr_start_timestamp
        fair_value = self.ipr_start_mid_price + (self.slope * time_elapsed)

        buy_vol = 0
        sell_vol = 0

        # --- INITIAL AGGRESSIVE BUY ---
        if not self.ipr_limit_reached:
            if best_ask is not None:
                ask_qty = -sell_orders[best_ask]
                qty = min(ask_qty, limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty
                    sell_orders[best_ask] += qty
                    if sell_orders[best_ask] >= 0:
                        del sell_orders[best_ask]
            
            if position + buy_vol >= limit:
                self.ipr_limit_reached = True

        # --- BUYBACK TO MIN HOLD LIMIT ---
        # If position dropped below min_hold_limit (e.g. outlier ask filled), buy back
        if sell_orders and position < self.min_hold_limit:
            needed = self.min_hold_limit - position
            for ask_price in sorted(sell_orders.keys()):
                if needed <= 0:
                    break
                available = -sell_orders[ask_price]
                qty = min(available, needed, limit - position - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    self.logger.log_order(product, "BUY", ask_price, qty, "BUYBACK")
                    buy_vol += qty
                    needed -= qty
                    sell_orders[ask_price] += qty
                    if sell_orders[ask_price] >= 0:
                        del sell_orders[ask_price]

        new_best_bid = max(buy_orders.keys()) if buy_orders else self.last_best_bid
        new_best_ask = min(sell_orders.keys()) if sell_orders else self.last_best_ask

        # Update historical bests for next tick
        if buy_orders:
            self.last_best_bid = max(buy_orders.keys())
        if sell_orders:
            self.last_best_ask = min(sell_orders.keys())

        # --- MARKET MAKING ---
        bid_capacity = limit - (position + buy_vol)
        ask_capacity = max(0, (position - sell_vol) - self.min_hold_limit)

        # Safely calculate microprice only when both sides are actively contested
        if buy_orders and sell_orders:
            bid_vol_live = buy_orders[max(buy_orders.keys())]
            ask_vol_live = abs(sell_orders[min(sell_orders.keys())])
            current_microprice = (max(buy_orders.keys()) * ask_vol_live + min(sell_orders.keys()) * bid_vol_live) / (bid_vol_live + ask_vol_live)
            self.ipr_microprice_history.append(current_microprice)
            if len(self.ipr_microprice_history) > 5:
                self.ipr_microprice_history.pop(0)

        # -- BID SIDE PRICING --
        if new_best_bid is not None:
            if buy_orders:
                quote_bid = new_best_bid + 1           # Competition exists: Undercut
                bid_tag = "REST_BID"
            else:
                quote_bid = new_best_bid - self.empty_book_delta  # No competition: Widen spread down
                bid_tag = "NO_COMP_BID"

            # Always bid regardless of phase
            if bid_capacity > 0:
                orders.append(Order(product, int(quote_bid), bid_capacity))
                self.logger.log_order(product, "BUY", int(quote_bid), bid_capacity, bid_tag)

        # -- ASK SIDE PRICING --
        if new_best_ask is not None:
            if sell_orders:
                quote_ask = new_best_ask - 1           # Competition exists: Overcut
                ask_tag = "REST_ASK"
                # Post-limit phase asks (only if min_delta above reference)
                if self.ipr_limit_reached and len(self.ipr_microprice_history) == 5:
                    past_avg_microprice = sum(self.ipr_microprice_history) / 5.0
                    if ask_capacity > 0 and (quote_ask - past_avg_microprice) >= self.min_delta:
                        orders.append(Order(product, int(quote_ask), -ask_capacity))
                        self.logger.log_order(product, "SELL", int(quote_ask), ask_capacity, ask_tag)
            else:
                quote_ask = new_best_ask + self.empty_book_delta  # No competition: Widen spread up
                ask_tag = "NO_COMP_ASK"
                # Empty ask book: outlier ask at max volume ignoring min_hold_limit
                if self.ipr_limit_reached:
                    empty_ask_capacity = limit + (position - sell_vol)
                    if empty_ask_capacity > 0:
                        orders.append(Order(product, int(quote_ask), -empty_ask_capacity))
                        self.logger.log_order(product, "SELL", int(quote_ask), empty_ask_capacity, ask_tag)

        return orders

    def ACO_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int,
        state: TradingState
    ) -> List[Order]:
        orders: List[Order] = []
        buy_vol = sell_vol = 0
        limit = self.LIMIT[product]
        
        # Calculate fair value as strictly 10000
        fair_value = 10000
        sell_orders = dict(order_depth.sell_orders)
        buy_orders = dict(order_depth.buy_orders)

        # 1. Take aggressive orders (Cross the spread if highly favorable)
        if sell_orders:
            current_best_ask = min(sell_orders.keys())
            ask_qty = -sell_orders[current_best_ask]
            if current_best_ask < fair_value:
                qty = min(ask_qty, limit - position)
                if qty > 0:
                    orders.append(Order(product, current_best_ask, qty))
                    self.logger.log_order(product, "BUY", int(current_best_ask), qty, "TAKE_ASK")

                    buy_vol += qty
                    sell_orders[current_best_ask] += qty
                    if sell_orders[current_best_ask] >= 0:
                        del sell_orders[current_best_ask]

        if buy_orders:
            current_best_bid = max(buy_orders.keys())
            bid_qty = buy_orders[current_best_bid]
            if current_best_bid > fair_value:
                qty = min(bid_qty, limit + position)
                if qty > 0:
                    orders.append(Order(product, current_best_bid, -qty))
                    self.logger.log_order(product, "SELL", int(current_best_bid), qty, "TAKE_BID")
                    sell_vol += qty
                    buy_orders[current_best_bid] -= qty
                    if buy_orders[current_best_bid] <= 0:
                        del buy_orders[current_best_bid]

        # --- 1.5. Aggressive Clearing (Panic Button at 70) ---
        position_after_take = position + buy_vol - sell_vol
        if abs(position_after_take) > self.aco_aggressive_clear_limit:
            fair_for_bid = fair_value + self.aco_clear_width
            fair_for_ask = fair_value - self.aco_clear_width

            if position_after_take > 0:  # Excess Long, need to sell
                for price in sorted(buy_orders.keys(), reverse=True):
                    if price >= fair_for_ask:
                        vol = buy_orders[price]
                        clear_qty = min(vol, position_after_take)
                        if clear_qty > 0:
                            orders.append(Order(product, price, -clear_qty))
                            self.logger.log_order(product, "SELL", int(price), clear_qty, "CLEAR_LONG")
                            sell_vol += clear_qty
                            position_after_take -= clear_qty
                            buy_orders[price] -= clear_qty
                            if buy_orders[price] <= 0:
                                del buy_orders[price]
                            if abs(position_after_take) <= self.aco_aggressive_clear_limit:
                                break

            elif position_after_take < 0:  # Excess Short, need to buy
                for price in sorted(sell_orders.keys()):
                    if price <= fair_for_bid:
                        vol = abs(sell_orders[price])
                        clear_qty = min(vol, abs(position_after_take))
                        if clear_qty > 0:
                            orders.append(Order(product, price, clear_qty))
                            self.logger.log_order(product, "BUY", int(price), clear_qty, "CLEAR_SHORT")
                            buy_vol += clear_qty
                            position_after_take += clear_qty
                            sell_orders[price] += clear_qty
                            if sell_orders[price] >= 0:
                                del sell_orders[price]
                            if abs(position_after_take) <= self.aco_aggressive_clear_limit:
                                break

        # Recalculate best bid and ask after taking aggressive/clearing orders
        new_best_bid = max(buy_orders.keys()) if buy_orders else (self.prev_best_bid or fair_value - 1)
        new_best_ask = min(sell_orders.keys()) if sell_orders else (self.prev_best_ask or fair_value + 1)
        
        # Update state for the next timestamp
        if buy_orders:
            self.prev_best_bid = max(buy_orders.keys())
        if sell_orders:
            self.prev_best_ask = min(sell_orders.keys())

        # 2. Market Make
        bid_capacity = limit - (position + buy_vol)
        ask_capacity = limit + (position - sell_vol)
        expected_pos = position + buy_vol - sell_vol

        # --- BID SIDE PRICING ---
        if buy_orders:
            quote_bid = new_best_bid + 1           # Competition exists: Undercut
            bid_tag = "REST_BID"
        else:
            quote_bid = 10000 - self.delta  # No competition: Widen spread down
            bid_tag = "NO_COMP_BID"
            
        if expected_pos < -self.aco_soft_position_limit:
            quote_bid += 1  # Apply passive skew up if heavily short
            
        if bid_capacity > 0 and quote_bid <= fair_value:
            orders.append(Order(product, int(quote_bid), bid_capacity))
            self.logger.log_order(product, "BUY", int(quote_bid), bid_capacity, bid_tag)

        # --- ASK SIDE PRICING ---
        if sell_orders:
            quote_ask = new_best_ask - 1           # Competition exists: Overcut
            ask_tag = "REST_ASK"
        else:
            quote_ask = new_best_ask + self.delta  # No competition: Widen spread up
            ask_tag = "NO_COMP_ASK"
            
        if expected_pos > self.aco_soft_position_limit:
            quote_ask -= 1  # Apply passive skew down if heavily long
            
        if ask_capacity > 0 and quote_ask >= fair_value:
            orders.append(Order(product, int(quote_ask), -ask_capacity))
            self.logger.log_order(product, "SELL", int(quote_ask), ask_capacity, ask_tag)

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # 1. Deserialize State from previous tick for BOTH strategies
        if state.traderData:
            try:
                saved_state = json.loads(state.traderData)
                # Load IPR state
                self.ipr_start_mid_price = saved_state.get("ipr_start_mid_price")
                self.ipr_start_timestamp = saved_state.get("ipr_start_timestamp")
                self.ipr_microprice_history = saved_state.get("ipr_microprice_history", [])
                self.ipr_limit_reached = saved_state.get("ipr_limit_reached", False)
                self.last_best_bid = saved_state.get("last_best_bid")
                self.last_best_ask = saved_state.get("last_best_ask")
                # Load ACO state
                self.prev_best_bid = saved_state.get("prev_best_bid")
                self.prev_best_ask = saved_state.get("prev_best_ask")
            except json.JSONDecodeError:
                # If JSON parsing fails, we gracefully continue with tick 0 defaults
                self.logger.debug("Failed to decode traderData", tag="ERROR")

        result: Dict[str, List[Order]] = {}
        self.logger.log(timestamp=state.timestamp, product=1)

        # Handle ASH_COATED_OSMIUM
        product_aco = "ASH_COATED_OSMIUM"
        pos_aco = state.position.get(product_aco, 0)
        if product_aco in state.order_depths:
            result[product_aco] = self.ACO_orders(product_aco, state.order_depths[product_aco], pos_aco, state)
        else:
            depth_ = OrderDepth()
            depth_.buy_orders, depth_.sell_orders = {}, {}
            result[product_aco] = self.ACO_orders(product_aco, depth_, pos_aco, state)

        # Handle INTARIAN_PEPPER_ROOT
        product_ipr = "INTARIAN_PEPPER_ROOT"
        pos_ipr = state.position.get(product_ipr, 0)
        if product_ipr in state.order_depths:
            result[product_ipr] = self.IPR_orders(product_ipr, state.order_depths[product_ipr], pos_ipr, state.timestamp)
        else:
            depth_ = OrderDepth()
            depth_.buy_orders, depth_.sell_orders = {}, {}
            result[product_ipr] = self.IPR_orders(product_ipr, depth_, pos_ipr, state.timestamp)
                
        # 2. Serialize State for the next tick for BOTH strategies
        current_state = {
            "ipr_start_mid_price": self.ipr_start_mid_price,
            "ipr_start_timestamp": self.ipr_start_timestamp,
            "ipr_microprice_history": self.ipr_microprice_history,
            "ipr_limit_reached": self.ipr_limit_reached,
            "last_best_bid": self.last_best_bid,
            "last_best_ask": self.last_best_ask,
            "prev_best_bid": self.prev_best_bid,
            "prev_best_ask": self.prev_best_ask
        }
        trader_data = json.dumps(current_state)
        
        # Flush the logger to output to the console
        self.logger.flush()
        
        return result, 1, trader_data