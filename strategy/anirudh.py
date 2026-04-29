import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "GALAXY_SOUNDS_SOLAR_FLAMES": {
        "position_limit": 10,
        "base": 11000,
        "grid_limit": 10,
        "tiers": [(200, 0.30), (500, 0.80), (1000, 1.00)],
        "ema_period": 300,
        "obi_weight": 1.5,
        "skew_factor": 0.05,
    },
    "GALAXY_SOUNDS_DARK_MATTER": {
        "position_limit": 10,
        "base": 10250,
        "grid_limit": 10,
        "tiers": [(200, 0.30), (500, 0.80), (1000, 1.00)],
        "ema_period": 300,
        "obi_weight": 1.5,
        "skew_factor": 0.05,
    },
    "SNACKPACK_RASPBERRY": {
        "position_limit": 10,
        "base": 10100,
        "grid_limit": 10,
        "tiers": [(200, 0.80), (300, 1.00)],
        "ema_period": 300,
        "obi_weight": 1.5,
        "skew_factor": 0.05,
    },
    "SNACKPACK_VANILLA": {
        "position_limit": 10,
        "base": 10100,
        "grid_limit": 10,
        "tiers": [(200, 0.80), (300, 1.00)],
        "ema_period": 300,
        "obi_weight": 1.5,
        "skew_factor": 0.05,
    },
    "SNACKPACK_CHOCOLATE": {
        "position_limit": 10,
        "base": 9800,
        "grid_limit": 10,
        "tiers": [(200, 0.80), (300, 1.00)],
        "ema_period": 300,
        "obi_weight": 1.5,
        "skew_factor": 0.05,
    },
    "PEBBLES_XS":          {"position_limit": 10},
    "MICROCHIP_OVAL":      {"position_limit": 10},
    "UV_VISOR_AMBER":      {"position_limit": 10},
    "OXYGEN_SHAKE_GARLIC": {"position_limit": 10},
    "MICROCHIP_SQUARE":    {"position_limit": 10},
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

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        return []

class GridMMStrategy(BaseStrategy):
    PRODUCT = ""

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        depth = self.order_depth
        buys, sells = depth.buy_orders, depth.sell_orders
        if not buys or not sells:
            return []

        best_bid = max(buys.keys())
        best_ask = min(sells.keys())
        mid = (best_bid + best_ask) / 2

        alpha = 2 / (self.params.get("ema_period", 300) + 1)
        ema_key = f"ema_{self.PRODUCT}"
        ema = state_dict.get(ema_key)
        if ema is None:
            ema = mid
        else:
            ema = mid * alpha + ema * (1 - alpha)
        state_dict[ema_key] = ema

        bid_vol = sum(q for _, q in sorted(buys.items(), reverse=True)[:5])
        ask_vol = sum(abs(q) for _, q in sorted(sells.items())[:5])
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

        fair = ema + obi * self.params.get("obi_weight", 1.5)
        pos = self.current_pos
        pos_limit = self.limit
        grid_limit = self.params.get("grid_limit", pos_limit)

        sim_bid_pos = pos
        sim_ask_pos = pos

        if self.state.timestamp >= 900000:
            pos_limit = pos_limit // 2
            grid_limit = grid_limit // 2

            if pos > pos_limit:
                dump_qty = pos - pos_limit
                out_px = max(best_ask - 1, math.ceil(ema))
                self.orders.append(Order(self.PRODUCT, int(out_px), -dump_qty))
                sim_ask_pos -= dump_qty
            elif pos < -pos_limit:
                dump_qty = abs(pos + pos_limit)
                out_px = min(best_bid + 1, math.floor(ema))
                self.orders.append(Order(self.PRODUCT, int(out_px), dump_qty))
                sim_bid_pos += dump_qty

        mm_limit = pos_limit - grid_limit
        base_px = self.params["base"]

        for dist, frac in self.params["tiers"]:
            target_cumulative = int(grid_limit * frac)

            bid_allowance = target_cumulative - sim_bid_pos
            if bid_allowance > 0:
                self.orders.append(Order(self.PRODUCT, base_px - dist, bid_allowance))
                sim_bid_pos += bid_allowance

            ask_allowance = sim_ask_pos - (-target_cumulative)
            if ask_allowance > 0:
                self.orders.append(Order(self.PRODUCT, base_px + dist, -ask_allowance))
                sim_ask_pos -= ask_allowance

        grid_bid_volume = sim_bid_pos - pos
        grid_ask_volume = pos - sim_ask_pos

        bid_cap = pos_limit - pos
        ask_cap = pos_limit + pos

        mm_bid_qty = min(mm_limit, bid_cap - grid_bid_volume)
        mm_ask_qty = min(mm_limit, ask_cap - grid_ask_volume)

        skewed_fair = fair - pos * self.params.get("skew_factor", 0.05)

        if mm_bid_qty > 0:
            b_px = min(best_bid + 1, math.floor(skewed_fair) - 1)
            self.orders.append(Order(self.PRODUCT, int(b_px), mm_bid_qty))

        if mm_ask_qty > 0:
            a_px = max(best_ask - 1, math.ceil(skewed_fair) + 1)
            self.orders.append(Order(self.PRODUCT, int(a_px), -mm_ask_qty))

        return self._consolidate()

class ShortOnlyStrategy(BaseStrategy):
    PRODUCT = ""

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        depth = self.order_depth
        if not depth.buy_orders:
            return []
        best_bid = max(depth.buy_orders.keys())
        if self.current_pos > -self.limit:
            self.ask(best_bid, self.limit, "SHORT")
        return self._consolidate()

class LongOnlyStrategy(BaseStrategy):
    PRODUCT = ""

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        depth = self.order_depth
        if not depth.sell_orders:
            return []
        best_ask = min(depth.sell_orders.keys())
        if self.current_pos < self.limit:
            self.bid(best_ask, self.limit, "LONG")
        return self._consolidate()

class GalaxySoundsSolarFlames(GridMMStrategy):
    PRODUCT = "GALAXY_SOUNDS_SOLAR_FLAMES"

class GalaxySoundsDarkMatter(GridMMStrategy):
    PRODUCT = "GALAXY_SOUNDS_DARK_MATTER"

class SnackpackRaspberry(GridMMStrategy):
    PRODUCT = "SNACKPACK_RASPBERRY"

class SnackpackVanilla(GridMMStrategy):
    PRODUCT = "SNACKPACK_VANILLA"

class SnackpackChocolate(GridMMStrategy):
    PRODUCT = "SNACKPACK_CHOCOLATE"

class PebblesXS(ShortOnlyStrategy):
    PRODUCT = "PEBBLES_XS"

class MicrochipOval(ShortOnlyStrategy):
    PRODUCT = "MICROCHIP_OVAL"

class UVVisorAmber(ShortOnlyStrategy):
    PRODUCT = "UV_VISOR_AMBER"

class OxygenShakeGarlic(LongOnlyStrategy):
    PRODUCT = "OXYGEN_SHAKE_GARLIC"

class MicrochipSquare(LongOnlyStrategy):
    PRODUCT = "MICROCHIP_SQUARE"

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = TUNING_PARAMS
        self.strategies: Dict[str, BaseStrategy] = {
            "GALAXY_SOUNDS_SOLAR_FLAMES": GalaxySoundsSolarFlames(self.logger, TUNING_PARAMS["GALAXY_SOUNDS_SOLAR_FLAMES"]),
            "GALAXY_SOUNDS_DARK_MATTER":  GalaxySoundsDarkMatter(self.logger,  TUNING_PARAMS["GALAXY_SOUNDS_DARK_MATTER"]),
            "SNACKPACK_RASPBERRY":        SnackpackRaspberry(self.logger,       TUNING_PARAMS["SNACKPACK_RASPBERRY"]),
            "SNACKPACK_VANILLA":          SnackpackVanilla(self.logger,         TUNING_PARAMS["SNACKPACK_VANILLA"]),
            "SNACKPACK_CHOCOLATE":        SnackpackChocolate(self.logger,       TUNING_PARAMS["SNACKPACK_CHOCOLATE"]),
            "PEBBLES_XS":                 PebblesXS(self.logger,               TUNING_PARAMS["PEBBLES_XS"]),
            "MICROCHIP_OVAL":             MicrochipOval(self.logger,            TUNING_PARAMS["MICROCHIP_OVAL"]),
            "UV_VISOR_AMBER":             UVVisorAmber(self.logger,             TUNING_PARAMS["UV_VISOR_AMBER"]),
            "OXYGEN_SHAKE_GARLIC":        OxygenShakeGarlic(self.logger,        TUNING_PARAMS["OXYGEN_SHAKE_GARLIC"]),
            "MICROCHIP_SQUARE":           MicrochipSquare(self.logger,          TUNING_PARAMS["MICROCHIP_SQUARE"]),
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
