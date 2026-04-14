import json
import math
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

TUNING_PARAMS = {
    "osmium_z_threshold": 2.0,
    "osmium_z_window": 20,
    "osmium_mm_limit": 16,  # 20% of 80
    "wall_inertia": 0.5,
    "position_limit": 80
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

    def get_lr_prediction(self, history: List[float]) -> Optional[float]:
        if len(history) < 3:
            return None
        # y_t = (4*y3 + y2 - 2*y1) / 3
        return (4.0 * history[-1] + history[-2] - 2.0 * history[-3]) / 3.0

    def update_inertial_walls(self, state_dict: Dict[str, Any], augment: bool = False) -> Tuple[Optional[float], Optional[float]]:
        prev_bid = state_dict.get(f"{self.PRODUCT}_INERTIAL_BID")
        prev_ask = state_dict.get(f"{self.PRODUCT}_INERTIAL_ASK")
        alpha = self.params.get("wall_inertia", 0.5)

        buy_orders = self.order_depth.buy_orders
        sell_orders = self.order_depth.sell_orders

        # Bid Wall
        if not buy_orders:
            if augment:
                history = state_dict.get(f"{self.PRODUCT}_BID_HIST", [])
                inertial_bid = self.get_lr_prediction(history) or prev_bid
            else:
                inertial_bid = prev_bid
        else:
            best_bid = max(buy_orders.keys())
            if prev_bid is None:
                inertial_bid = best_bid
            else:
                tentative_bid = best_bid * alpha + (1 - alpha) * prev_bid
                filtered_bids = {p: v for p, v in buy_orders.items() if p < tentative_bid + 1}
                inertial_bid = max(filtered_bids.keys()) if filtered_bids else tentative_bid

        # Ask Wall
        if not sell_orders:
            if augment:
                history = state_dict.get(f"{self.PRODUCT}_ASK_HIST", [])
                inertial_ask = self.get_lr_prediction(history) or prev_ask
            else:
                inertial_ask = prev_ask
        else:
            best_ask = min(sell_orders.keys())
            if prev_ask is None:
                inertial_ask = best_ask
            else:
                tentative_ask = best_ask * alpha + (1 - alpha) * prev_ask
                filtered_asks = {p: v for p, v in sell_orders.items() if p > tentative_ask - 1}
                inertial_ask = min(filtered_asks.keys()) if filtered_asks else tentative_ask

        # Update History for LR
        if inertial_bid is not None:
            bid_hist = state_dict.get(f"{self.PRODUCT}_BID_HIST", [])
            bid_hist.append(inertial_bid)
            if len(bid_hist) > 3: bid_hist.pop(0)
            state_dict[f"{self.PRODUCT}_BID_HIST"] = bid_hist
            state_dict[f"{self.PRODUCT}_INERTIAL_BID"] = inertial_bid

        if inertial_ask is not None:
            ask_hist = state_dict.get(f"{self.PRODUCT}_ASK_HIST", [])
            ask_hist.append(inertial_ask)
            if len(ask_hist) > 3: ask_hist.pop(0)
            state_dict[f"{self.PRODUCT}_ASK_HIST"] = ask_hist
            state_dict[f"{self.PRODUCT}_INERTIAL_ASK"] = inertial_ask

        return inertial_bid, inertial_ask

class OsmiumStrategy(BaseStrategy):
    PRODUCT = "ASH_COATED_OSMIUM"

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        bid_wall, ask_wall = self.update_inertial_walls(state_dict, augment=True)
        if bid_wall is None or ask_wall is None:
            return []

        mid_price = (bid_wall + ask_wall) / 2.0
        
        # Z-Score Calculation
        window = self.params.get("osmium_z_window", 20)
        prices = state_dict.get("OSMIUM_PRICES", [])
        prices.append(mid_price)
        if len(prices) > window:
            prices.pop(0)
        state_dict["OSMIUM_PRICES"] = prices

        if len(prices) >= window:
            mean = sum(prices) / len(prices)
            variance = sum((x - mean) ** 2 for x in prices) / len(prices)
            std = math.sqrt(variance) if variance > 0 else 1e-6
            z_score = (mid_price - mean) / std
            
            threshold = self.params.get("osmium_z_threshold", 2.0)
            
            # Directional component
            if z_score > threshold:
                # Overpriced -> Short
                target_pos = -self.limit
                if self.current_pos > target_pos:
                    # Take bids
                    sorted_bids = sorted(self.order_depth.buy_orders.items(), reverse=True)
                    for bp, bv in sorted_bids:
                        if self.current_pos <= target_pos: break
                        vol = min(abs(bv), self.current_pos - target_pos)
                        self.ask(bp, vol, tag="Z_TAKE_BID")
            elif z_score < -threshold:
                # Underpriced -> Long
                target_pos = self.limit
                if self.current_pos < target_pos:
                    # Take asks
                    sorted_asks = sorted(self.order_depth.sell_orders.items())
                    for sp, sv in sorted_asks:
                        if self.current_pos >= target_pos: break
                        vol = min(abs(sv), target_pos - self.current_pos)
                        self.bid(sp, vol, tag="Z_TAKE_ASK")

        # Market Making component (20% of limit)
        mm_limit = self.params.get("osmium_mm_limit", 16)
        # Place orders around mid-price
        # Overbid bid_wall if possible, underbid ask_wall
        mm_bid = round(bid_wall)
        mm_ask = round(ask_wall)
        
        # We cap MM orders to mm_limit
        self.bid(mm_bid, mm_limit, tag="MM_BID")
        self.ask(mm_ask, mm_limit, tag="MM_ASK")

        return self.orders

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
        return self.orders

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.osmium_params = {
            "osmium_z_threshold": TUNING_PARAMS["osmium_z_threshold"],
            "osmium_z_window": TUNING_PARAMS["osmium_z_window"],
            "osmium_mm_limit": TUNING_PARAMS["osmium_mm_limit"],
            "wall_inertia": TUNING_PARAMS["wall_inertia"],
            "position_limit": 80
        }
        self.pepper_params = {
            "wall_inertia": TUNING_PARAMS["wall_inertia"],
            "position_limit": 80
        }
        self.osmium_strat = OsmiumStrategy(self.logger, self.osmium_params)
        self.pepper_strat = PepperRootStrategy(self.logger, self.pepper_params)
    
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

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            self.pepper_strat.reset(state)
            result["INTARIAN_PEPPER_ROOT"] = self.pepper_strat.run(state_dict)

        traderData = json.dumps(state_dict)
        return result, 0, traderData
