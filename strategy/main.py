
import json
from typing import List
from datamodel import OrderDepth, UserId, TradingState, Order

# ── Tuning Injection ────────────────────────────────────────────────────────
# A single shared JSON file for all tuned hyperparameters.
# _TUNING_PARAMS_FILENAME = "params.json"

# def _load_tuning_params() -> dict:
#     """Load hyperparameters from params.json in the same directory, or fail silently."""
#     params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _TUNING_PARAMS_FILENAME)
#     try:
#         with open(params_path, "r") as f:
#             return json.load(f)
#     except Exception:
#         return {}

# TUNING_PARAMS = _load_tuning_params()
TUNING_PARAMS = {}

# ── Product-Specific Parameter Initialization ───────────────────────────────

_TOMATO_DEFAULTS = {
    "alpha": 0.5,
    "n_sma": 20,
    "m_slope": 30,
    "slope_threshold": 0.2,
    "position_limit": 80
}

# Merge shared tuning overrides into commodity defaults
TOMATO_PARAMS = {**_TOMATO_DEFAULTS, **TUNING_PARAMS}

_EMERALD_DEFAULTS = {
    "position_limit": 80,
}

EMERALD_PARAMS = {**_EMERALD_DEFAULTS, **TUNING_PARAMS}

class Logger:

    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, **series: float) -> None:
        if not series:
            return
        # Convert everything to float; skip non-numeric silently
        clean: dict[str, float] = {}
        for k, v in series.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not clean:
            return

        line = self.PREFIX + json.dumps(clean, separators=(',', ':'))
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)

    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out

    def reset(self) -> None:
        self._buffer.clear()

    def debug(self, msg: str, tag: str = 'DBG', product: str = '') -> None:
        """Log a text debug message visible in the Logs tab of the visualizer.
        
        Args:
            msg:     Free-form debug text.
            tag:     Severity tag – 'INFO', 'WARN', 'ERR', or 'DBG' (default).
            product: Optional product context (e.g. 'TOMATOES') for position/PnL lookup.
        
        Usage:
            self.logger.debug("spread too wide, skipping", tag="WARN", product="TOMATOES")
        """
        line = f'LOGDBG:{tag}:{product}:{msg}'
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)


class Trader:

    def __init__(self):
        self.logger = Logger()
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        result = {}
        
        traderData = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(traderData)
        except Exception:
            state_dict = {}
        
        tomatoes_ema = state_dict.get("TOMATOES_EMA", None)
        # ── FIFO Inventory & Rolling Average (5 Units) ─────────────────────────
        # Maintain inventory layers (FIFO) and calculate rolling avg of last 5 units.
        inventory = state_dict.get("INVENTORY", {}).get("TOMATOES", [])
        last_processed_time = state_dict.get("TOMATO_LAST_TRADE_TIME", -1)
        
        tomatoes_trades = state.own_trades.get("TOMATOES", [])
        for trade in tomatoes_trades:
            if trade.timestamp <= last_processed_time:
                continue
            
            # Buyer is "SUBMISSION" means we bought (positive qty)
            trade_qty = trade.quantity if trade.buyer == "SUBMISSION" else -trade.quantity
            
            # If inventory is empty or trade is in same direction, add a new layer
            if not inventory or (inventory[0][0] * trade_qty >= 0):
                inventory.append([trade_qty, trade.price])
            else:
                # Opposite direction: FIFO netting
                temp_qty = trade_qty
                while temp_qty != 0 and inventory:
                    if abs(temp_qty) >= abs(inventory[0][0]):
                        temp_qty += inventory[0][0] # e.g., -10 (sell) + 5 (oldest buy) = -5 remaining sell
                        inventory.pop(0)
                    else:
                        inventory[0][0] += temp_qty
                        temp_qty = 0
                
                # If trade reversed the position, add the remainder as a new layer
                if temp_qty != 0:
                    inventory.append([temp_qty, trade.price])
            
            last_processed_time = max(last_processed_time, trade.timestamp)

        # Calculate weighted average of the LAST 5 units (newest to oldest)
        target_units = 5.0
        accumulated_units = 0.0
        weighted_sum = 0.0
        
        for layer_qty, layer_price in reversed(inventory):
            needed = target_units - accumulated_units
            contribution = min(abs(layer_qty), needed)
            weighted_sum += contribution * layer_price
            accumulated_units += contribution
            if accumulated_units >= target_units:
                break
        
        tomatoes_avg_price = weighted_sum / accumulated_units if accumulated_units > 0 else None
        
        # Persist inventory and last processed time
        if "INVENTORY" not in state_dict:
            state_dict["INVENTORY"] = {}
        state_dict["INVENTORY"]["TOMATOES"] = inventory
        state_dict["TOMATO_LAST_TRADE_TIME"] = last_processed_time


        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "EMERALDS":
                current_position = int(state.position.get(product, 0))
                position_limit = EMERALD_PARAMS["position_limit"]
                max_bid_pos = position_limit - current_position
                max_ask_pos = position_limit + current_position

                bid_orders = order_depth.buy_orders.items()
                ask_orders = order_depth.sell_orders.items()

                if not bid_orders or not ask_orders:
                    continue

                bid_wall = max(bid_orders, key=lambda x: x[0])[0]
                ask_wall = min(ask_orders, key=lambda x: x[0])[0]

                ##########################################################
                ####### 1. TAKING
                ##########################################################
                for sp, sv in ask_orders:
                    if sp <= 10000 - 1:
                        orders.append(Order(product, sp, -sv))
                        max_bid_pos -= abs(sv)
                    elif sp == 10000 and current_position < 0:
                            volume = min(-sv,  abs(current_position))
                            orders.append(Order(product, sp, volume))
                            max_bid_pos -= abs(volume)

                for bp, bv in bid_orders:
                    if bp >= 10000 + 1:
                        orders.append(Order(product, bp, -bv))
                        max_ask_pos -= abs(bv)
                    elif bp == 10000 and current_position > 0:
                            volume = min(bv,  current_position)
                            orders.append(Order(product, bp, -volume))
                            max_ask_pos -= abs(volume)

                ###########################################################
                ####### 2. MAKING
                ###########################################################
                bid_price = int(bid_wall + 1) # base case
                ask_price = int(ask_wall - 1) # base case

                # OVERBIDDING: overbid best bid that is still under the mid wall
                for bp, bv in bid_orders:
                    overbidding_price = bp + 1
                    if bv > 1 and overbidding_price < 10000:
                        bid_price = max(bid_price, overbidding_price)
                        break
                    elif bp < 10000:
                        bid_price = max(bid_price, bp)
                        break

                # UNDERBIDDING: underbid best ask that is still over the mid wall
                for sp, sv in ask_orders:
                    underbidding_price = sp - 1
                    if sv > 1 and underbidding_price > 10000:
                        ask_price = min(ask_price, underbidding_price)
                        break
                    elif sp > 10000:
                        ask_price = min(ask_price, sp)
                        break

                # POST ORDERS
                if bid_price < 10000:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > 10000:
                    orders.append(Order(product, ask_price, -max_ask_pos))
            
            elif product == "TOMATOES":
                current_position = int(state.position.get(product, 0))
                position_limit = TOMATO_PARAMS["position_limit"]
                alpha = TOMATO_PARAMS["alpha"]
                n_sma = TOMATO_PARAMS["n_sma"]
                m_slope = TOMATO_PARAMS["m_slope"]
                slope_threshold = TOMATO_PARAMS["slope_threshold"]

                max_bid_pos = position_limit - current_position
                max_ask_pos = position_limit + current_position

                bid_orders = list(order_depth.buy_orders.items())
                ask_orders = list(order_depth.sell_orders.items())

                if not bid_orders or not ask_orders:
                    continue

                total_vol = 0
                total_val = 0
                for bp, bv in bid_orders:
                    total_val += bp * bv
                    total_vol += bv
                for sp, sv in ask_orders:
                    total_val += sp * abs(sv)
                    total_vol += abs(sv)

                if total_vol > 0:
                    vwap = total_val / total_vol
                else:
                    continue

                bid_wall = max(bid_orders, key=lambda x: x[0])[0]
                ask_wall = min(ask_orders, key=lambda x: x[0])[0]
                market_mid = (bid_wall + ask_wall) / 2.0

                if tomatoes_ema is None:
                    tomatoes_ema = market_mid
                else:
                    tomatoes_ema = alpha * market_mid + (1 - alpha) * tomatoes_ema

                mid_price = round(tomatoes_ema)

                # SMA and Slope calculation
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
                    
                self.logger.log(mid_price=mid_price, vwap=vwap, sma=current_sma, slope=4940 + 10*slope)

                ##########################################################
                ####### 1. TAKING
                ##########################################################
                for sp, sv in ask_orders:
                    if sp <= mid_price - 1:
                        orders.append(Order(product, sp, -sv))
                        max_bid_pos -= abs(sv)
                    elif sp == mid_price and current_position < 0:
                        volume = min(-sv,  abs(current_position))
                        orders.append(Order(product, sp, volume))
                        max_bid_pos -= abs(volume)

                for bp, bv in bid_orders:
                    if bp >= mid_price + 1:
                        orders.append(Order(product, bp, -bv))
                        max_ask_pos -= abs(bv)
                    elif bp == mid_price and current_position > 0:
                        volume = min(bv,  current_position)
                        orders.append(Order(product, bp, -volume))
                        max_ask_pos -= abs(volume)

                ###########################################################
                ####### 2. MAKING
                ###########################################################
                bid_price = int(bid_wall + 1) # base case
                ask_price = int(ask_wall - 1) # base case

                # OVERBIDDING: overbid best bid that is still under the mid wall
                for bp, bv in bid_orders:
                    overbidding_price = bp + 1
                    if bv > 1 and overbidding_price < mid_price:
                        bid_price = max(bid_price, overbidding_price)
                        break
                    elif bp < mid_price:
                        bid_price = max(bid_price, bp)
                        break

                # UNDERBIDDING: underbid best ask that is still over the mid wall
                for sp, sv in ask_orders:
                    underbidding_price = sp - 1
                    if sv > 1 and underbidding_price > mid_price:
                        ask_price = min(ask_price, underbidding_price)
                        break
                    elif sp > mid_price:
                        ask_price = min(ask_price, sp)
                        break

                # POST ORDERS
                if bid_price < mid_price:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > mid_price:
                    orders.append(Order(product, ask_price, -max_ask_pos))

                # Directional Filter based on Slope
                if abs(slope) > slope_threshold:
                    if slope > 0:
                        # Bullish trend: Keep only Buy orders
                        orders = [o for o in orders if o.quantity > 0]
                        # self.logger.debug("Bullish trend - filtering for Buys only", product=product)
                    elif slope < 0:
                        # Bearish trend: Keep only Sell orders
                        orders = [o for o in orders if o.quantity < 0]
                        # self.logger.debug("Bearish trend - filtering for Sells only", product=product)

                # ── Last-Mile Favourable Filtering ─────────────────────────────
                # Only sell if price > average cost (long); buy if price < avg (short)
                if tomatoes_avg_price is not None:
                    pre_filter_count = len(orders)
                    # For long positions (inventory[0][0] > 0), only sell if price > avg
                    # For short positions (inventory[0][0] < 0), only buy if price < avg
                    is_long = inventory[0][0] > 0
                    
                    filtered_orders = []
                    for o in orders:
                        if is_long and o.quantity < 0: # We are long, this is a sell order
                            if o.price > tomatoes_avg_price:
                                filtered_orders.append(o)
                        elif not is_long and o.quantity > 0: # We are short, this is a buy order
                            if o.price < tomatoes_avg_price:
                                filtered_orders.append(o)
                        else:
                            # Not a closing/netting order in this direction, or same-direction addition
                            filtered_orders.append(o)
                    
                    orders = filtered_orders
                    removed = pre_filter_count - len(orders)
                    if removed > 0:
                        self.logger.log(orders_removed_avg=removed, avg_price=tomatoes_avg_price)


            if orders:
                result[product] = orders
    
        if tomatoes_ema is not None:
            state_dict["TOMATOES_EMA"] = tomatoes_ema
        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData

