import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from datamodel import OrderDepth, TradingState, Order, Trade

_PARAMS_FILE = Path(__file__).parent / "params.json"
if _PARAMS_FILE.exists():
    with open(_PARAMS_FILE) as _f:
        TUNING_PARAMS = json.load(_f)
else:
    TUNING_PARAMS = {}


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
        self.limit = params.get("position_limit", 100)
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

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        return []


WINDOW  = 10
ENTRY_Z = 0.85
EXIT_Z  = 0.84


def _zscore(buf: List[float], price: float):
    n = len(buf)
    mu = sum(buf) / n
    sigma = (sum((x - mu) ** 2 for x in buf) / n) ** 0.5
    if sigma < 1e-9:
        return None
    return (price - mu) / sigma


class VelvetfruitSignal:
    """Reads VELVETFRUIT_EXTRACT mid price, computes z-score, writes signal to state_dict."""
    PRODUCT = "VELVETFRUIT_EXTRACT"

    def __init__(self, logger: Logger):
        self.logger = logger

    def update(self, state: TradingState, state_dict: Dict[str, Any]) -> Optional[int]:
        od = state.order_depths.get(self.PRODUCT)
        if od is None:
            return state_dict.get("vf_signal", 0)

        best_bid = max(od.buy_orders.keys(), default=None)
        best_ask = min(od.sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None:
            return state_dict.get("vf_signal", 0)

        mid = (best_bid + best_ask) / 2.0

        buf: List[float] = state_dict.get("vf_buf", [])
        buf.append(mid)
        if len(buf) > WINDOW:
            buf = buf[-WINDOW:]
        state_dict["vf_buf"] = buf

        if len(buf) < WINDOW:
            return 0

        z = _zscore(buf[:-1], mid)
        if z is None:
            return state_dict.get("vf_signal", 0)

        signal: int = state_dict.get("vf_signal", 0)
        if signal == 0:
            if z < -ENTRY_Z:
                signal = 1
            elif z > ENTRY_Z:
                signal = -1
        elif signal == 1 and z >= -EXIT_Z:
            signal = 0
        elif signal == -1 and z <= EXIT_Z:
            signal = 0

        state_dict["vf_signal"] = signal
        state_dict["vf_z"] = z
        self.logger.log(vf_z=z, vf_mid=mid, vf_signal=signal)
        return signal


class Vev5200Strategy(BaseStrategy):
    PRODUCT = "VEV_5200"

    def _take_asks(self, qty: int, tag: str):
        for price in sorted(self.order_depth.sell_orders.keys()):
            if self.buy_capacity <= 0 or qty <= 0:
                break
            vol = min(abs(self.order_depth.sell_orders[price]), qty, self.buy_capacity)
            self.bid(price, vol, tag=tag)
            qty -= vol

    def _take_bids(self, qty: int, tag: str):
        for price in sorted(self.order_depth.buy_orders.keys(), reverse=True):
            if self.sell_capacity <= 0 or qty <= 0:
                break
            vol = min(abs(self.order_depth.buy_orders[price]), qty, self.sell_capacity)
            self.ask(price, vol, tag=tag)
            qty -= vol

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        signal: int = state_dict.get("vf_signal", 0)

        self.logger.log(vev_pos=self.current_pos, vev_signal=signal)

        if signal == 1:
            if self.buy_capacity > 0:
                self._take_asks(self.buy_capacity, tag="ENTER_LONG")
        elif signal == -1:
            if self.sell_capacity > 0:
                self._take_bids(self.sell_capacity, tag="ENTER_SHORT")
        else:
            if self.current_pos > 0:
                self._take_bids(self.current_pos, tag="EXIT_LONG")
            elif self.current_pos < 0:
                self._take_asks(abs(self.current_pos), tag="EXIT_SHORT")

        return self._consolidate()


class Trader:
    def __init__(self):
        self.logger  = Logger()
        self.signal  = VelvetfruitSignal(self.logger)
        self.vev_strat = Vev5200Strategy(self.logger, {
            "position_limit": 300,
        })

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        try:
            state_dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            state_dict = {}

        # 1. compute signal from VELVETFRUIT_EXTRACT
        self.signal.update(state, state_dict)

        # 2. execute on VEV_5200
        if "VEV_5200" in state.order_depths:
            self.vev_strat.reset(state)
            result["VEV_5200"] = self.vev_strat.run(state_dict)

        return result, 0, json.dumps(state_dict)
