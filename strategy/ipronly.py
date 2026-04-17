from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict
import json

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
        self.LIMIT = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }
        
        # Strategy parameters
        self.slope = 1.000004085764449662451780476147e-03
        self.intercept = 11999.9749430865
        self.min_outlier_pos = 70
        
        # State variables
        self.ipr_limit_reached = False

    def IPR_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int,
        timestamp: int
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.LIMIT[product]
        fair_value = self.intercept + (self.slope * timestamp)
        
        # Sort sell orders ascending (lowest price first)
        sell_orders = sorted(order_depth.sell_orders.items())
        # Sort buy orders descending (highest price first)
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        current_pos = position

        # 1. INITIAL PHASE: Aggressive Buy until Limit 80
        if not self.ipr_limit_reached:
            for price, qty in sell_orders:
                take_qty = min(-qty, limit - current_pos)
                if take_qty > 0:
                    orders.append(Order(product, price, take_qty))
                    current_pos += take_qty
            
            if current_pos >= limit:
                self.ipr_limit_reached = True
            
            return orders

        # 2. OUTLIER PHASE: Take buy/sell orders relative to trend line
        
        # Take buy orders (to SELL) if price > fair_value
        # Condition: position must not decrease below 70
        if current_pos >= self.min_outlier_pos:
            for price, qty in buy_orders:
                if price > fair_value:
                    # Respect short limit (-80)
                    max_sell = current_pos - (-limit)
                    take_qty = min(qty, max_sell)
                    
                    if take_qty > 0:
                        orders.append(Order(product, price, -take_qty))
                        current_pos -= take_qty
                    
                    # Stop taking outliers if position decreases below 70
                    if current_pos < self.min_outlier_pos:
                        break

        # Take sell orders (to BUY) if price < fair_value
        # "only buy till that position is taken back, do not keep building position"
        for price, qty in sell_orders:
            if price < fair_value:
                take_qty = min(-qty, limit - current_pos)
                if take_qty > 0:
                    orders.append(Order(product, price, take_qty))
                    current_pos += take_qty
        
        return orders

    def ACO_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int
    ) -> List[Order]:
        orders: List[Order] = []
        # TODO: Implement Ash Coated Osmium logic
        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # 1. Deserialize State from previous tick
        if state.traderData:
            try:
                saved_state = json.loads(state.traderData)
                self.ipr_limit_reached = saved_state.get("ipr_limit_reached", False)
            except json.JSONDecodeError:
                self.logger.debug("Failed to decode traderData", tag="ERROR")

        result: Dict[str, List[Order]] = {}
        
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            
            if product == "ASH_COATED_OSMIUM":
                result[product] = self.ACO_orders(product, depth, pos)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.IPR_orders(product, depth, pos, state.timestamp)
                
        # 2. Serialize State for the next tick
        current_state = {
            "ipr_limit_reached": self.ipr_limit_reached
        }
        trader_data = json.dumps(current_state)
        
        self.logger.flush()
        
        return result, 1, trader_data