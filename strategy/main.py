import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "osmium_z_threshold": 2.0,
    "osmium_z_window": 200,
    "osmium_mm_limit": 80,  # 20% of 80
    "wall_inertia": 0.3,     # price EMA alpha
    "vol_alpha": 0.3,        # volume class-mean EMA alpha
    "outlier_frac": 0.2,     # reject obs if deviation > outlier_frac * thin_spread
    "position_limit": 80,
    "vol_threshold": 20,
    "price_threshold": 3.0
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
        self.limit = params.get("position_limit", 80)
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

    def update_inertial_walls(self, state_dict: Dict[str, Any], augment: bool = False) -> Tuple[Optional[float], Optional[float]]:
        P = self.PRODUCT
        vol_threshold = self.params.get("vol_threshold", 20)
        price_threshold = self.params.get("price_threshold", 10.0)

        # 1. Filter orders: volume <= vol_threshold
        bid_orders = sorted([(p, v) for p, v in self.order_depth.buy_orders.items() if abs(v) <= vol_threshold], reverse=True)
        ask_orders = sorted([(p, v) for p, v in self.order_depth.sell_orders.items() if abs(v) <= vol_threshold])

        prev_bid = state_dict.get(f"{P}_PBID")
        prev_ask = state_dict.get(f"{P}_PASK")

        # 2. Find best bid: highest price within deviation threshold from previous
        best_bid = None
        for p, v in bid_orders:
            if prev_bid is None or abs(p - prev_bid) <= price_threshold:
                best_bid = float(p)
                break

        # 3. Find best ask: lowest price within deviation threshold from previous
        best_ask = None
        for p, v in ask_orders:
            if prev_ask is None or abs(p - prev_ask) <= price_threshold:
                best_ask = float(p)
                break

        # 4. If left with no orders, augment value using previous value
        if best_bid is None:
            best_bid = prev_bid

        if best_ask is None:
            best_ask = prev_ask

        # Update state for next cycle
        if best_bid is not None:
            state_dict[f"{P}_PBID"] = best_bid

        if best_ask is not None:
            state_dict[f"{P}_PASK"] = best_ask

        # self.logger.log(best_bid=best_bid, best_ask=best_ask)
        return best_bid, best_ask

class OsmiumStrategy(BaseStrategy):
    PRODUCT = "ASH_COATED_OSMIUM"

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        bid_wall, ask_wall = self.update_inertial_walls(state_dict, augment=True)
        if bid_wall is None or ask_wall is None:
            return []

        mid = (bid_wall + ask_wall) / 2.0

        self.logger.log(mid=mid)

        # ── Z-score bookkeeping ─────────────────────────────────────────
        window = self.params.get("osmium_z_window", 20)
        prices = state_dict.get("OSMIUM_PRICES", [])
        prices.append(mid)
        if len(prices) > window: prices.pop(0)
        state_dict["OSMIUM_PRICES"] = prices

        z_score = None
        if len(prices) >= window:
            mean = sum(prices) / len(prices)
            variance = sum((x - mean) ** 2 for x in prices) / len(prices)
            std = math.sqrt(variance) if variance > 0 else 1e-6
            z_score = (mid - mean) / std
            self.logger.log(z_score=z_score)

        threshold = self.params.get("osmium_z_threshold", 2.0)
        mm_limit = self.params.get("osmium_mm_limit", 16)

        # ── 1. TAKING ──────────────────────────────────────────────────
        # Sell into bids sitting above mid (free edge)
        for bp, bv in sorted(self.order_depth.buy_orders.items(), reverse=True):
            if bp <= mid or self.sell_capacity <= 0: break
            self.ask(bp, min(abs(bv), self.sell_capacity), tag="TAKE_BID")

        # Buy from asks sitting below mid
        for sp, sv in sorted(self.order_depth.sell_orders.items()):
            if sp >= mid or self.buy_capacity <= 0: break
            self.bid(sp, min(abs(sv), self.buy_capacity), tag="TAKE_ASK")

        # ── 2. Z-SCORE signal (passive orders at mid) ──────────────────
        z_signal = None   # 'BUY' or 'SELL'
        z_price  = round(mid)
        if z_score is not None:
            if z_score > threshold:
                z_signal = 'SELL'   # overpriced → short
            elif z_score < -threshold:
                z_signal = 'BUY'    # underpriced → long

        # ── 3. MARKET MAKING ───────────────────────────────────────────
        mm_bid_p = round(bid_wall) + 1
        mm_ask_p = round(ask_wall) - 1

        # Z-score takes precedence: suppress MM side that would cross it
        spread_ok  = mm_bid_p < mm_ask_p
        do_mm_bid  = spread_ok and not (z_signal == 'SELL' and mm_bid_p >= z_price)
        do_mm_ask  = spread_ok and not (z_signal == 'BUY'  and mm_ask_p <= z_price)

        # Apply z-score first so it gets capacity priority
        if z_signal == 'SELL' and self.sell_capacity > 0:
            self.ask(z_price, self.sell_capacity, tag="Z_ASK")
        elif z_signal == 'BUY' and self.buy_capacity > 0:
            self.bid(z_price, self.buy_capacity, tag="Z_BID")

        # Apply MM with remaining capacity
        if do_mm_bid and self.buy_capacity > 0:
            self.bid(mm_bid_p, min(mm_limit, self.buy_capacity), tag="MM_BID")
        if do_mm_ask and self.sell_capacity > 0:
            self.ask(mm_ask_p, min(mm_limit, self.sell_capacity), tag="MM_ASK")

        return self._consolidate()

class PepperRootStrategy(BaseStrategy):
    PRODUCT = "INTARIAN_PEPPER_ROOT"

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        bid_wall, ask_wall = self.update_inertial_walls(state_dict)
        if bid_wall is None or ask_wall is None:
            return []
        
        # Simple Market Maker for Pepper Root
        mid = (bid_wall + ask_wall) / 2.0
        self.bid(round(bid_wall), self.limit, tag="MM_BID")
        self.ask(round(ask_wall), self.limit, tag="MM_ASK")
        return self._consolidate()

class IntarianRootStrategy(BaseStrategy):
    PRODUCT = "INTARIAN_ROOT"

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        # Buy and hold: take cheapest asks until at position limit
        for price, vol in sorted(self.order_depth.sell_orders.items()):
            if self.buy_capacity <= 0:
                break
            self.bid(price, min(abs(vol), self.buy_capacity), tag="BUY_HOLD")
        return self._consolidate()

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.osmium_params = {
            "osmium_z_threshold": TUNING_PARAMS["osmium_z_threshold"],
            "osmium_z_window": TUNING_PARAMS["osmium_z_window"],
            "osmium_mm_limit": TUNING_PARAMS["osmium_mm_limit"],
            "wall_inertia": TUNING_PARAMS["wall_inertia"],
            "vol_alpha": TUNING_PARAMS["vol_alpha"],
            "outlier_frac": TUNING_PARAMS["outlier_frac"],
            "vol_threshold": TUNING_PARAMS["vol_threshold"],
            "price_threshold": TUNING_PARAMS["price_threshold"],
            "position_limit": 80
        }
        self.pepper_params = {
            "wall_inertia": TUNING_PARAMS["wall_inertia"],
            "vol_alpha": TUNING_PARAMS["vol_alpha"],
            "outlier_frac": TUNING_PARAMS["outlier_frac"],
            "vol_threshold": TUNING_PARAMS["vol_threshold"],
            "price_threshold": TUNING_PARAMS["price_threshold"],
            "position_limit": 80
        }
        self.intarian_root_params = {
            "position_limit": 80
        }
        self.osmium_strat = OsmiumStrategy(self.logger, self.osmium_params)
        self.pepper_strat = PepperRootStrategy(self.logger, self.pepper_params)
        self.intarian_root_strat = IntarianRootStrategy(self.logger, self.intarian_root_params)
    
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        traderData = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(traderData)
        except Exception:
            state_dict = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            self.osmium_strat.reset(state)
            result["ASH_COATED_OSMIUM"] = self.osmium_strat.run(state_dict)

        # if "INTARIAN_PEPPER_ROOT" in state.order_depths:
        #     self.pepper_strat.reset(state)
        #     result["INTARIAN_PEPPER_ROOT"] = self.pepper_strat.run(state_dict)

        if "INTARIAN_ROOT" in state.order_depths:
            self.intarian_root_strat.reset(state)
            result["INTARIAN_ROOT"] = self.intarian_root_strat.run(state_dict)

        traderData = json.dumps(state_dict)
        return result, 0, traderData
