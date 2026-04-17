import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "position_limit": 20,
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

class BaseStrategy:
    """Base class for all product strategies."""
    PRODUCT = ""
    
    def __init__(self, logger: Logger, params: Dict[str, Any]):
        self.logger = logger
        self.params = params
        self.limit = params.get("position_limit", 20)
        self.orders: List[Order] = []
        self.state: Optional[TradingState] = None
        self.order_depth: Optional[OrderDepth] = None
        self.current_pos = 0
        self.buy_capacity = 0
        self.sell_capacity = 0

    def reset(self, state: TradingState):
        self.state = state
        self.order_depth = state.order_depths[self.PRODUCT]
        self.orders = []
        self.current_pos = int(state.position.get(self.PRODUCT, 0))
        self.buy_capacity = self.limit - self.current_pos
        self.sell_capacity = self.limit + self.current_pos

    def bid(self, price: int, quantity: int, tag: str = "MAKE"):
        if quantity <= 0 or self.buy_capacity <= 0:
            return
        exec_qty = min(quantity, self.buy_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), exec_qty))
        self.buy_capacity -= exec_qty
        self.logger.log_order(self.PRODUCT, "BUY", int(price), exec_qty, tag)

    def ask(self, price: int, quantity: int, tag: str = "MAKE"):
        if quantity <= 0 or self.sell_capacity <= 0:
            return
        exec_qty = min(quantity, self.sell_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), -exec_qty))
        self.sell_capacity -= exec_qty
        self.logger.log_order(self.PRODUCT, "SELL", int(price), exec_qty, tag)

    def _consolidate(self) -> List[Order]:
        """Merge orders at the same price level before submission."""
        merged: Dict[int, int] = {}
        for o in self.orders:
            merged[o.price] = merged.get(o.price, 0) + o.quantity
        return [Order(self.PRODUCT, price, qty) for price, qty in merged.items() if qty != 0]

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        return []

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = TUNING_PARAMS
        self.strategies: Dict[str, BaseStrategy] = {
            # "PRODUCT": BaseStrategy(self.logger, self.params)
        }
    
    def run(self, state: TradingState):
        # self.logger.log(timestamp = state.timestamp)
        result: Dict[str, List[Order]] = {}
        trader_data = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(trader_data)
        except Exception:
            state_dict = {}

        for product, strategy in self.strategies.items():
            if product in state.order_depths:
                strategy.reset(state)
                result[product] = strategy.run(state_dict)

        new_trader_data = json.dumps(state_dict)
        return result, 0, new_trader_data
