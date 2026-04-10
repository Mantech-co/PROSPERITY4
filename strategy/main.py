import json
from typing import List, Dict, Any, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order, Trade

# ── Strategy Parameters ─────────────────────────────────────────────────────

# --- Global Tuning ---
TUNING_PARAMS = {
    "alpha": 0.5,
    "n_sma": 20,
    "m_slope": 30,
    "slope_threshold": 0.2
}

# --- Emeralds Strategy Defaults ---
EMERALD_LIMIT    = 80

# --- Tomatoes Strategy Defaults ---
TOMATO_LIMIT           = 80
TOMATO_ALPHA           = 0.5
TOMATO_N_SMA           = 20
TOMATO_M_SLOPE         = 30
TOMATO_SLOPE_THRESHOLD = 0.2
TOMATO_AVG_UNITS       = 5.0

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

    def log_order(self, side: str, price: int, qty: int, tag: str) -> None:
        """Log an individual order placement for strategy-level trade attribution."""
        self._emit(f'LOGORDER:{side}:{price}:{qty}:{tag}')

    def debug(self, msg: str, tag: str = 'DBG', product: str = '') -> None:
        self._emit(f'LOGDBG:{tag}:{product}:{msg}')

    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out

class MidPriceHelper:
    """Utility class for various mid-price calculations."""
    
    @staticmethod
    def get_vwap(order_depth: OrderDepth) -> Optional[float]:
        total_vol = 0
        total_val = 0
        for bp, bv in order_depth.buy_orders.items():
            total_val += bp * bv
            total_vol += bv
        for sp, sv in order_depth.sell_orders.items():
            total_val += sp * abs(sv)
            total_vol += abs(sv)
        return total_val / total_vol if total_vol > 0 else None

    @staticmethod
    def get_bba_mid(order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def get_pop(order_depth: OrderDepth) -> Optional[float]:
        """Returns mid price based on most popular (highest volume) bid and ask."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        pop_bid = max(order_depth.buy_orders.items(), key=lambda x: x[1])[0]
        pop_ask = max(order_depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]
        return (pop_bid + pop_ask) / 2.0

class PositionManager:
    """Manages FIFO inventory and state persistence."""
    
    def __init__(self, product: str, state_dict: Dict[str, Any]):
        self.product = product
        self.inventory = state_dict.get("INVENTORY", {}).get(product, [])
        key_suffix = "TOMATO" if product == "TOMATOES" else product
        self.last_trade_time = state_dict.get(f"{key_suffix}_LAST_TRADE_TIME", -1)

    def update(self, trades: List[Trade]):
        for trade in trades:
            if trade.timestamp <= self.last_trade_time:
                continue
            
            trade_qty = trade.quantity if trade.buyer == "SUBMISSION" else -trade.quantity
            
            if not self.inventory or (self.inventory[0][0] * trade_qty >= 0):
                self.inventory.append([trade_qty, trade.price])
            else:
                temp_qty = trade_qty
                while temp_qty != 0 and self.inventory:
                    if abs(temp_qty) >= abs(self.inventory[0][0]):
                        temp_qty += self.inventory[0][0]
                        self.inventory.pop(0)
                    else:
                        self.inventory[0][0] += temp_qty
                        temp_qty = 0
                if temp_qty != 0:
                    self.inventory.append([temp_qty, trade.price])
            
            self.last_trade_time = max(self.last_trade_time, trade.timestamp)

    def get_avg_price(self, n_units: float = 5.0) -> Optional[float]:
        accumulated_units = 0.0
        weighted_sum = 0.0
        for layer_qty, layer_price in reversed(self.inventory):
            needed = n_units - accumulated_units
            contribution = min(abs(layer_qty), needed)
            weighted_sum += contribution * layer_price
            accumulated_units += contribution
            if accumulated_units >= n_units:
                break
        return weighted_sum / accumulated_units if accumulated_units > 0 else None

    def persist(self, state_dict: Dict[str, Any]):
        if "INVENTORY" not in state_dict:
            state_dict["INVENTORY"] = {}
        state_dict["INVENTORY"][self.product] = self.inventory
        key_suffix = "TOMATO" if self.product == "TOMATOES" else self.product
        state_dict[f"{key_suffix}_LAST_TRADE_TIME"] = self.last_trade_time

class BaseStrategy:
    """Base class for all product strategies with real-time capacity tracking."""
    PRODUCT = ""
    
    def __init__(self, logger: Logger, params: Dict[str, Any]):
        """Called once at Trader init. Stores static config."""
        self.logger = logger
        self.params = params
        self.limit = params.get("position_limit", 20)
        self.orders: List[Order] = []
        # Per-tick state (set in reset)
        self.state: Optional[TradingState] = None
        self.order_depth: Optional[OrderDepth] = None
        self.current_pos = 0
        self.buy_capacity = 0
        self.sell_capacity = 0

    def reset(self, state: TradingState):
        """Called each tick to refresh per-tick state."""
        self.state = state
        self.order_depth = state.order_depths[self.PRODUCT]
        self.orders = []
        self.current_pos = int(state.position.get(self.PRODUCT, 0))
        self.buy_capacity = self.limit - self.current_pos
        self.sell_capacity = self.limit + self.current_pos

    def bid(self, price: int, quantity: int, tag: str = "MAKE"):
        """Place a bid, clamping to available buy capacity."""
        if quantity <= 0 or self.buy_capacity <= 0:
            return
        exec_qty = min(quantity, self.buy_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), exec_qty))
        self.buy_capacity -= exec_qty
        self.sell_capacity += exec_qty
        self.logger.log_order("BUY", int(price), exec_qty, tag)

    def ask(self, price: int, quantity: int, tag: str = "MAKE"):
        """Place an ask, clamping to available sell capacity."""
        if quantity <= 0 or self.sell_capacity <= 0:
            return
        exec_qty = min(quantity, self.sell_capacity)
        self.orders.append(Order(self.PRODUCT, int(price), -exec_qty))
        self.sell_capacity -= exec_qty
        self.buy_capacity += exec_qty
        self.logger.log_order("SELL", int(price), exec_qty, tag)

    def get_walls(self, min_vol: int = 1) -> Tuple[Optional[int], Optional[int]]:
        """Find best bid/ask walls with at least min_vol."""
        bid_wall, ask_wall = None, None
        for px, vol in sorted(self.order_depth.buy_orders.items(), reverse=True):
            if abs(vol) >= min_vol:
                bid_wall = px
                break
        for px, vol in sorted(self.order_depth.sell_orders.items()):
            if abs(vol) >= min_vol:
                ask_wall = px
                break
        return bid_wall, ask_wall

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        raise NotImplementedError

class MarketMaker:
    """Helper class to handle market making: taking (netting, fair-build, outlier removal) and making."""
    
    def __init__(self, aggression: float = 0.0, fair_build_ratio: float = 0.5, make: bool = True):
        """Called once at strategy init.
        
        Args:
            aggression: Max loss per order for outlier removal.
                        0   = only take profitable orders (better than fair).
                        > 0 = also take lossy outliers up to this loss budget.
                        Loss formula: volume * max(0, deviation_from_fair).
            fair_build_ratio: Max position as fraction of limit to allow fair-value building.
                              e.g. 0.5 = build at fair only if |pos| < 50% of limit.
            make: Whether to place maker orders (overbid/undercut).
        """
        self.aggression = aggression
        self.fair_build_ratio = fair_build_ratio
        self.make = make

    def execute(self, strat: BaseStrategy, mid_target: float, bid_wall: Optional[int], ask_wall: Optional[int]):
        """Called each tick. Operates on the strategy's current per-tick state."""
        if bid_wall is None or ask_wall is None:
            return
        
        # Sort once, reuse for both taking and making
        sorted_asks = sorted(strat.order_depth.sell_orders.items())
        sorted_bids = sorted(strat.order_depth.buy_orders.items(), reverse=True)
        low_pos = abs(strat.current_pos) < self.fair_build_ratio * strat.limit

        # ── TAKING: Buy side (taking asks) ──
        for sp, sv in sorted_asks:
            vol = abs(sv)
            if sp == mid_target:
                # 1. Netting at fair (always, no risk)
                if strat.current_pos < 0:
                    strat.bid(sp, min(vol, abs(strat.current_pos)), tag="TAKE_NET")
                # 2a. Build at fair (if position < 50% limit)
                if low_pos:
                    strat.bid(sp, vol, tag="TAKE_FAIR")
            else:
                # 2b. Outlier removal (gated by aggression)
                # loss < 0 when sp < mid (profitable), loss > 0 when sp > mid (lossy)
                loss = vol * max(0, sp - mid_target)
                if loss <= self.aggression:
                    strat.bid(sp, vol, tag="TAKE" if sp < mid_target else "TAKE_AGG")

        # ── TAKING: Sell side (taking bids) ──
        for bp, bv in sorted_bids:
            vol = abs(bv)
            if bp == mid_target:
                # 1. Netting at fair (always, no risk)
                if strat.current_pos > 0:
                    strat.ask(bp, min(vol, abs(strat.current_pos)), tag="TAKE_NET")
                # 2a. Build at fair (if position < 50% limit)
                if low_pos:
                    strat.ask(bp, vol, tag="TAKE_FAIR")
            else:
                # 2b. Outlier removal (gated by aggression)
                loss = vol * max(0, mid_target - bp)
                if loss <= self.aggression:
                    strat.ask(bp, vol, tag="TAKE" if bp > mid_target else "TAKE_AGG")

        # ── MAKING ──
        if not self.make:
            return

        # bid wall and ask wall need to be informed of outliers
        # if an outlier is detected, the bid wall and ask wall should be adjusted
        # to not place orders at the outlier price
        

        bid_price = bid_wall + (1 if bid_wall < mid_target-1 else -1)
        ask_price = ask_wall - (1 if ask_wall > mid_target+1 else -1)
        
        for bp, bv in sorted_bids:
            if bv > 1 and bp + 1 < mid_target:
                bid_price = max(bid_price, bp + 1)
                break
            elif bp < mid_target:
                bid_price = max(bid_price, bp)
                break
                
        for sp, sv in sorted_asks:
            if abs(sv) > 1 and sp - 1 > mid_target:
                ask_price = min(ask_price, sp - 1)
                break
            elif sp > mid_target:
                ask_price = min(ask_price, sp)
                break

        if bid_price < mid_target:
            strat.bid(bid_price, strat.buy_capacity)
                
        if ask_price > mid_target:
            strat.ask(ask_price, strat.sell_capacity)

class EmeraldStrategy(BaseStrategy):
    """Strategy for EMERALDS: Pure Market Making around 10000."""
    PRODUCT = "EMERALDS"
    
    def __init__(self, logger: Logger, params: Dict[str, Any]):
        super().__init__(logger, params)
        self.mm = MarketMaker()
    
    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        bid_wall, ask_wall = self.get_walls()
        self.mm.execute(self, 10000, bid_wall, ask_wall)
        return self.orders

class TomatoStrategy(BaseStrategy):
    """Strategy for TOMATOES: Trend following + Market Making with Inventory Filtering."""
    PRODUCT = "TOMATOES"
    
    def __init__(self, logger: Logger, params: Dict[str, Any]):
        super().__init__(logger, params)
        self.mm = MarketMaker()
    
    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        # 1. Mid Price and EMA
        vwap = MidPriceHelper.get_vwap(self.order_depth)
        market_mid = MidPriceHelper.get_bba_mid(self.order_depth)
        if market_mid is None or vwap is None:
            return []
            
        alpha = self.params["alpha"]
        tomatoes_ema = state_dict.get("TOMATOES_EMA", market_mid)
        tomatoes_ema = alpha * market_mid + (1 - alpha) * tomatoes_ema
        state_dict["TOMATOES_EMA"] = tomatoes_ema
        mid_price_target = round(tomatoes_ema)

        # 2. SMA and Slope
        n_sma = self.params["n_sma"]
        m_slope = self.params["m_slope"]
        slope_threshold = self.params["slope_threshold"]
        
        tomato_prices = state_dict.get("TOMATO_PRICES", [])
        tomato_prices.append(vwap)
        if len(tomato_prices) > n_sma:
            tomato_prices.pop(0)
        state_dict["TOMATO_PRICES"] = tomato_prices
        
        current_sma = sum(tomato_prices) / len(tomato_prices)
        tomato_smas = state_dict.get("TOMATO_SMAS", [])
        tomato_smas.append(current_sma)
        if len(tomato_smas) > m_slope:
            tomato_smas.pop(0)
        state_dict["TOMATO_SMAS"] = tomato_smas
        
        slope = 0
        if len(tomato_smas) == m_slope:
            slope = (tomato_smas[-1] - tomato_smas[0]) / (m_slope - 1)
            
        # self.logger.log(mid_price=mid_price_target, vwap=vwap, sma=current_sma, slope=4940 + 10*slope)

        # 3. Position and Inventory Manager
        pos_mgr = PositionManager(self.PRODUCT, state_dict)
        pos_mgr.update(self.state.own_trades.get(self.PRODUCT, []))
        avg_price = pos_mgr.get_avg_price(self.params.get("inventory_avg_units", 5.0))
        pos_mgr.persist(state_dict)

        # 4. Order Generation
        bid_wall, ask_wall = self.get_walls()
        self.mm.execute(self, mid_price_target, bid_wall, ask_wall)

        # 5. Trend Filter
        if abs(slope) > slope_threshold:
            if slope > 0: # Bullish
                self.orders = [o for o in self.orders if o.quantity > 0]
            elif slope < 0: # Bearish
                self.orders = [o for o in self.orders if o.quantity < 0]

        # 6. Inventory Favourable Filter
        if avg_price is not None and len(pos_mgr.inventory) > 0:
            pre_filter = len(self.orders)
            is_long = pos_mgr.inventory[0][0] > 0
            filtered = []
            for o in self.orders:
                if is_long and o.quantity < 0: # Selling out of a long
                    if o.price > avg_price:
                        filtered.append(o)
                elif not is_long and o.quantity > 0: # Buying back a short
                    if o.price < avg_price:
                        filtered.append(o)
                else:
                    filtered.append(o)
            self.orders = filtered
            removed = pre_filter - len(self.orders)
            if removed > 0:
                self.logger.log(orders_removed_avg=removed, avg_price=avg_price)

        return self.orders

class Trader:
    def __init__(self):
        self.logger = Logger()

        # Build param dicts once
        self.emerald_params = {
            "position_limit": TUNING_PARAMS.get("position_limit", EMERALD_LIMIT)
        }
        self.tomato_params = {
            "alpha": TUNING_PARAMS.get("alpha", TOMATO_ALPHA),
            "n_sma": TUNING_PARAMS.get("n_sma", TOMATO_N_SMA),
            "m_slope": TUNING_PARAMS.get("m_slope", TOMATO_M_SLOPE),
            "slope_threshold": TUNING_PARAMS.get("slope_threshold", TOMATO_SLOPE_THRESHOLD),
            "position_limit": TUNING_PARAMS.get("position_limit", TOMATO_LIMIT),
            "inventory_avg_units": TUNING_PARAMS.get("inventory_avg_units", TOMATO_AVG_UNITS)
        }

        # Instantiate strategies once
        self.emerald_strat = EmeraldStrategy(self.logger, self.emerald_params)
        self.tomato_strat = TomatoStrategy(self.logger, self.tomato_params)
    
    def run(self, state: TradingState):
        """Main dispatcher — only per-tick work happens here."""
        result: Dict[str, List[Order]] = {}
        
        # 1. Parse Persistence State
        traderData = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(traderData)
        except Exception:
            state_dict = {}

        # 2. Reset and run strategies
        if "EMERALDS" in state.order_depths:
            self.emerald_strat.reset(state)
            result["EMERALDS"] = self.emerald_strat.run(state_dict)

        if "TOMATOES" in state.order_depths:
            self.tomato_strat.reset(state)
            result["TOMATOES"] = self.tomato_strat.run(state_dict)

        # 3. Serialize Persistence State
        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData
