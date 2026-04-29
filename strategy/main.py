import json
from typing import List, Dict, Any, Optional
from datamodel import OrderDepth, TradingState, Order

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "position_limit": 10,
}

class Logger:
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
        self._emit(f'LOGORDER:{product}:{side}:{price}:{qty}:{tag}')

    def debug(self, msg: str, tag: str = 'DBG', product: str = '') -> None:
        self._emit(f'LOGDBG:{tag}:{product}:{msg}')

    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out


class BaseStrategy:
    PRODUCT = ""

    def __init__(self, logger: Logger, params: Dict[str, Any]):
        self.logger = logger
        self.params = params
        self.limit = params.get("position_limit", 10)
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
        merged: Dict[int, int] = {}
        for o in self.orders:
            merged[o.price] = merged.get(o.price, 0) + o.quantity
        return [Order(self.PRODUCT, price, qty) for price, qty in merged.items() if qty != 0]

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        return []


class FullLongStrategy(BaseStrategy):
    """Always drives position to +limit by hitting best ask."""

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        if not self.order_depth.sell_orders:
            return []
        best_ask = min(self.order_depth.sell_orders.keys())
        self.bid(best_ask, self.buy_capacity, tag="LONG")
        return self._consolidate()


class FullShortStrategy(BaseStrategy):
    """Always drives position to -limit by hitting best bid."""

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        if not self.order_depth.buy_orders:
            return []
        best_bid = max(self.order_depth.buy_orders.keys())
        self.ask(best_bid, self.sell_capacity, tag="SHORT")
        return self._consolidate()


def make_long(product: str, logger: Logger, params: Dict[str, Any]) -> FullLongStrategy:
    s = FullLongStrategy(logger, params)
    s.PRODUCT = product
    return s


def make_short(product: str, logger: Logger, params: Dict[str, Any]) -> FullShortStrategy:
    s = FullShortStrategy(logger, params)
    s.PRODUCT = product
    return s


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = TUNING_PARAMS
        self.strategies: Dict[str, BaseStrategy] = {
            "PEBBLES_XL":       make_long("PEBBLES_XL",       self.logger, self.params),
            "MICROCHIP_SQUARE": make_long("MICROCHIP_SQUARE", self.logger, self.params),
            "PEBBLES_XS":       make_short("PEBBLES_XS",       self.logger, self.params),
            "MICROCHIP_OVAL":   make_short("MICROCHIP_OVAL",   self.logger, self.params),
            "MICROCHIP_TRIANGLE": make_short("MICROCHIP_TRIANGLE", self.logger, self.params),
        }

    def run(self, state: TradingState):
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
