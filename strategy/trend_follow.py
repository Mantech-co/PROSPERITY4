from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict
import json


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


class Trader:
    # ============================================================
    # Strategy parameters
    # ============================================================
    TARGET_PRODUCTS = [
        "MICROCHIP_RECTANGLE",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ]
    POSITION_LIMIT = 10

    # EMA spans 3000 most-recent mid-price observations.
    # alpha = 2 / (N+1) is the standard pandas-style EMA decay.
    EMA_SPAN = 3000

    # Each Prosperity tick advances the clock by 100 timestamps,
    # so one tick == one sample for the slope calculation.
    SAMPLE_PERIOD = 100

    # Slope thresholds (units = price / timestamp)
    SLOPE_LONG_THRESHOLD = 0.20    # slope > +0.20 -> long full size
    SLOPE_SHORT_THRESHOLD = -0.20  # slope < -0.20 -> short full size
    SLOPE_FLAT_UPPER = 0.10        # |slope| <= 0.10 -> flatten to 0
    SLOPE_FLAT_LOWER = -0.10
    # In the dead bands (-0.20, -0.10) and (0.10, 0.20) the position is held.

    # Don't trade until the EMA has seen enough samples to be meaningful.
    WARMUP_TICKS = 100

    def __init__(self):
        self.logger = Logger()
        self.alpha = 2.0 / (self.EMA_SPAN + 1)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _mid_price(order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _market_take_buy(product: str, order_depth: OrderDepth, qty: int) -> List[Order]:
        """Lift offers (cheapest first) until qty units are bought or book is empty."""
        orders: List[Order] = []
        if qty <= 0:
            return orders
        remaining = qty
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if remaining <= 0:
                break
            ask_vol = -order_depth.sell_orders[ask_price]  # asks are stored as negatives
            if ask_vol <= 0:
                continue
            take = min(remaining, ask_vol)
            orders.append(Order(product, ask_price, take))
            remaining -= take
        return orders

    @staticmethod
    def _market_take_sell(product: str, order_depth: OrderDepth, qty: int) -> List[Order]:
        """Hit bids (highest first) until qty units are sold or book is empty."""
        orders: List[Order] = []
        if qty <= 0:
            return orders
        remaining = qty
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if remaining <= 0:
                break
            bid_vol = order_depth.buy_orders[bid_price]
            if bid_vol <= 0:
                continue
            take = min(remaining, bid_vol)
            orders.append(Order(product, bid_price, -take))
            remaining -= take
        return orders

    # ============================================================
    # Per-product strategy
    # ============================================================
    def _trade_ema_slope(self, product: str, state: TradingState, mem: dict) -> List[Order]:
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders

        order_depth = state.order_depths[product]
        mid = self._mid_price(order_depth)
        if mid is None:
            return orders

        position = int(state.position.get(product, 0))

        # ---- update EMA ----
        prev_ema = mem.get("ema")
        if prev_ema is None:
            # First observation: seed EMA, no slope yet, do nothing.
            mem["ema"] = mid
            mem["count"] = 1
            self.logger.log(**{
                f"{product}_mid": mid,
                f"{product}_ema": mid,
                f"{product}_slope": 0.0,
                f"{product}_pos": position,
            })
            return orders

        new_ema = self.alpha * mid + (1.0 - self.alpha) * prev_ema
        # Slope = (ema(t) - ema(t-100)) / 100. Previous EMA value IS ema(t-100)
        # because every Prosperity tick advances the clock by exactly 100.
        slope = (new_ema - prev_ema) / float(self.SAMPLE_PERIOD)

        mem["ema"] = new_ema
        mem["count"] = int(mem.get("count", 0)) + 1

        self.logger.log(**{
            f"{product}_mid": mid,
            f"{product}_ema": new_ema,
            f"{product}_slope": slope,
            f"{product}_pos": position,
        })

        # ---- warmup gate ----
        if mem["count"] < self.WARMUP_TICKS:
            return orders

        # ---- trading logic (market taking) ----
        if slope > self.SLOPE_LONG_THRESHOLD:
            # Go long to +POSITION_LIMIT
            need = self.POSITION_LIMIT - position
            if need > 0:
                orders.extend(self._market_take_buy(product, order_depth, need))
                self.logger.debug(f"slope={slope:.4f} -> LONG need={need}", "ENTER", product)

        elif slope < self.SLOPE_SHORT_THRESHOLD:
            # Go short to -POSITION_LIMIT
            need = position - (-self.POSITION_LIMIT)  # how many to sell
            if need > 0:
                orders.extend(self._market_take_sell(product, order_depth, need))
                self.logger.debug(f"slope={slope:.4f} -> SHORT need={need}", "ENTER", product)

        elif self.SLOPE_FLAT_LOWER <= slope <= self.SLOPE_FLAT_UPPER:
            # Flatten to 0
            if position > 0:
                orders.extend(self._market_take_sell(product, order_depth, position))
                self.logger.debug(f"slope={slope:.4f} -> FLAT sell {position}", "EXIT", product)
            elif position < 0:
                orders.extend(self._market_take_buy(product, order_depth, -position))
                self.logger.debug(f"slope={slope:.4f} -> FLAT buy {-position}", "EXIT", product)
        # else: slope in dead band (-0.20, -0.10) or (0.10, 0.20) -> hold position

        return orders

    # ============================================================
    # Prosperity 4 entry point
    # ============================================================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Restore per-product memory from traderData (JSON)
        try:
            memory = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(memory, dict):
                memory = {}
        except (json.JSONDecodeError, TypeError):
            memory = {}

        for product in self.TARGET_PRODUCTS:
            if product not in memory or not isinstance(memory[product], dict):
                memory[product] = {}
            orders = self._trade_ema_slope(product, state, memory[product])
            if orders:
                result[product] = orders

        trader_data = json.dumps(memory)
        self.logger.flush()
        return result, 1, trader_data