import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

TUNING_PARAMS = {
    "z_window": 20,
    "z_threshold": 2.0,
    "position_limit": 50,
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
        self.limit = params.get("position_limit", 50)
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

    def bid(self, price: int, quantity: int, tag: str = "BUY"):
        if quantity <= 0 or self.buy_capacity <= 0:
            return
        qty = min(quantity, self.buy_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), qty))
        self.buy_capacity -= qty
        self.logger.log_order(self.PRODUCT, "BUY", int(price), qty, tag)

    def ask(self, price: int, quantity: int, tag: str = "SELL"):
        if quantity <= 0 or self.sell_capacity <= 0:
            return
        qty = min(quantity, self.sell_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), -qty))
        self.sell_capacity -= qty
        self.logger.log_order(self.PRODUCT, "SELL", int(price), qty, tag)

    def _consolidate(self) -> List[Order]:
        merged: Dict[int, int] = {}
        for o in self.orders:
            merged[o.price] = merged.get(o.price, 0) + o.quantity
        return [Order(self.PRODUCT, price, qty) for price, qty in merged.items() if qty != 0]

    def mid_price(self) -> Optional[float]:
        bids = self.order_depth.buy_orders
        asks = self.order_depth.sell_orders
        if not bids or not asks:
            return None
        return (max(bids) + min(asks)) / 2.0


class HydrogelPackStrategy(BaseStrategy):
    PRODUCT = "HYDROGEL_PACK"

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        mid = self.mid_price()
        if mid is None:
            return []

        window = self.params.get("z_window", 20)
        threshold = self.params.get("z_threshold", 2.0)

        prices = state_dict.get("HP_PRICES", [])
        prices.append(mid)
        if len(prices) > window:
            prices.pop(0)
        state_dict["HP_PRICES"] = prices

        self.logger.log(mid=mid)

        if len(prices) < window:
            return []

        mean = sum(prices) / len(prices)
        variance = sum((x - mean) ** 2 for x in prices) / len(prices)
        std = math.sqrt(variance) if variance > 0 else 1e-6
        z = (mid - mean) / std

        self.logger.log(z_score=z)

        best_bid = max(self.order_depth.buy_orders)
        best_ask = min(self.order_depth.sell_orders)

        if z > threshold:
            # overpriced → sell at best bid (aggressive)
            self.ask(best_bid, self.sell_capacity, tag="Z_SELL")
        elif z < -threshold:
            # underpriced → buy at best ask (aggressive)
            self.bid(best_ask, self.buy_capacity, tag="Z_BUY")

        return self._consolidate()


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.hydrogel_params = {
            "z_window": TUNING_PARAMS["z_window"],
            "z_threshold": TUNING_PARAMS["z_threshold"],
            "position_limit": TUNING_PARAMS["position_limit"],
        }
        self.hydrogel_strat = HydrogelPackStrategy(self.logger, self.hydrogel_params)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        try:
            state_dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            state_dict = {}

        if "HYDROGEL_PACK" in state.order_depths:
            self.hydrogel_strat.reset(state)
            result["HYDROGEL_PACK"] = self.hydrogel_strat.run(state_dict)

        return result, 0, json.dumps(state_dict)
