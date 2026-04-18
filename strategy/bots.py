import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

TUNING_PARAMS = {
    "position_limit": 20,
    "spread": 2,  # half-spread from mid
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
        merged: Dict[int, int] = {}
        for o in self.orders:
            merged[o.price] = merged.get(o.price, 0) + o.quantity
        return [Order(self.PRODUCT, price, qty) for price, qty in merged.items() if qty != 0]

    def mid_price(self) -> Optional[float]:
        """Inherited mid-price logic from main.py: inertial walls via filtered order book."""
        od = self.order_depth
        vol_threshold = self.params.get("vol_threshold", 20)
        price_threshold = self.params.get("price_threshold", 3.0)

        bid_orders = sorted(
            [(p, v) for p, v in od.buy_orders.items() if abs(v) <= vol_threshold],
            reverse=True,
        )
        ask_orders = sorted(
            [(p, v) for p, v in od.sell_orders.items() if abs(v) <= vol_threshold]
        )

        best_bid = bid_orders[0][0] if bid_orders else None
        best_ask = ask_orders[0][0] if ask_orders else None

        # fall back to raw best if filtered book is empty
        if best_bid is None and od.buy_orders:
            best_bid = max(od.buy_orders.keys())
        if best_ask is None and od.sell_orders:
            best_ask = min(od.sell_orders.keys())

        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        return []


class MarketOrderRemoverStrategy(BaseStrategy):
    """
    Absorbs all incoming market orders (orders priced through mid) and
    replaces them with passive limit bids/asks at mid ± spread.
    """
    PRODUCT = "PRODUCT_NAME"  # override per product

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        mid = self.mid_price()
        if mid is None:
            return []

        spread = self.params.get("spread", 2)
        self.logger.log(mid=mid)

        # absorb all buy orders
        for bp, bv in sorted(self.order_depth.buy_orders.items(), reverse=True):
            if self.sell_capacity <= 0:
                break
            self.ask(bp, min(abs(bv), self.sell_capacity), tag="ABSORB_BUY")

        # absorb all sell orders
        for sp, sv in sorted(self.order_depth.sell_orders.items()):
            if self.buy_capacity <= 0:
                break
            self.bid(sp, min(abs(sv), self.buy_capacity), tag="ABSORB_SELL")

        # place passive quotes at fixed spread from mid
        bid_price = math.floor(mid) - spread
        ask_price = math.ceil(mid) + spread

        if self.buy_capacity > 0:
            self.bid(bid_price, self.buy_capacity, tag="PASSIVE_BID")
        if self.sell_capacity > 0:
            self.ask(ask_price, self.sell_capacity, tag="PASSIVE_ASK")

        return self._consolidate()


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = {
            **TUNING_PARAMS,
            "vol_threshold": 20,
            "price_threshold": 3.0,
        }
        self.strategies: Dict[str, BaseStrategy] = {
            # "PRODUCT_NAME": MarketOrderRemoverStrategy(self.logger, self.params),
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

        return result, 0, json.dumps(state_dict)
