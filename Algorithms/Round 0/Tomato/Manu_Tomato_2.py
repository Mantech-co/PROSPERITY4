import json
import math
from typing import List
from datamodel import OrderDepth, UserId, TradingState, Order

class Logger:

    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, timestamp: int, **series: float) -> None:
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

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "EMERALDS":
                current_position = int(state.position.get(product, 0))
                position_limit = 80
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
                if bid_price < 10000 and max_bid_pos > 0:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > 10000 and max_ask_pos > 0:
                    orders.append(Order(product, ask_price, -max_ask_pos))
            
            elif product == "TOMATOES":
                current_position = int(state.position.get(product, 0))
                position_limit = 80 
                
                max_bid_pos = position_limit - current_position
                max_ask_pos = position_limit + current_position

                bid_orders = list(order_depth.buy_orders.items())
                ask_orders = list(order_depth.sell_orders.items())

                if not bid_orders or not ask_orders:
                    continue

                ##########################################################
                ####### 1. ROBUST WALL IDENTIFICATION
                ##########################################################
                # Filter for deep liquidity (assuming exchange wall is > 30 lots)
                valid_bids = [(p, v) for p, v in bid_orders if v >= 30]
                valid_asks = [(p, v) for p, v in ask_orders if abs(v) >= 30]

                # Retrieve saved Wall Mid (Stateful Memory)
                wall_mid = state_dict.get("TOMATOES_WALL_MID", None)

                if valid_bids and valid_asks:
                    bid_wall_price, bid_wall_vol = max(valid_bids, key=lambda x: x[0])
                    ask_wall_price, ask_wall_vol = min(valid_asks, key=lambda x: x[0])
                    ask_wall_vol = abs(ask_wall_vol)

                    raw_mid = (bid_wall_price + ask_wall_price) / 2.0

                    ##########################################################
                    ####### 2. MICRO-PRICE ROUNDING (Imbalance)
                    ##########################################################
                    if raw_mid % 1 == 0.5:
                        if bid_wall_vol > ask_wall_vol:
                            wall_mid = math.ceil(raw_mid)  # Buy pressure, round up
                        elif ask_wall_vol > bid_wall_vol:
                            wall_mid = math.floor(raw_mid) # Sell pressure, round down
                        else:
                            wall_mid = round(raw_mid)
                    else:
                        wall_mid = round(raw_mid)
                    
                    # Update state
                    state_dict["TOMATOES_WALL_MID"] = wall_mid

                if wall_mid is None:
                    continue # Skip tick if we have no historical or current fair value

                ##########################################################
                ####### 3. DYNAMIC INVENTORY SKEW
                ##########################################################
                inventory_fraction = current_position / position_limit
                max_skew = 2 # Max ticks we are willing to shift our fair value estimate
                
                # If long, skew is positive (lowers fair value -> lowers bids/asks)
                # If short, skew is negative (raises fair value -> raises bids/asks)
                skew = int(round(inventory_fraction * max_skew))
                
                adjusted_fair_value = wall_mid - skew
                
                self.logger.log(state.timestamp, TOMATO_ADJ_MID=adjusted_fair_value, TOMATO_RAW_MID=wall_mid)

                ##########################################################
                ####### 4. TAKING (Aggressive)
                ##########################################################
                for sp, sv in ask_orders:
                    if sp < adjusted_fair_value:
                        volume = min(abs(sv), max_bid_pos)
                        if volume > 0:
                            orders.append(Order(product, sp, volume))
                            max_bid_pos -= volume
                    elif sp == adjusted_fair_value and current_position < 0:
                        # Only take at fair value if it helps reduce short position
                        volume = min(abs(sv), abs(current_position))
                        if volume > 0:
                            orders.append(Order(product, sp, volume))
                            max_bid_pos -= volume

                for bp, bv in bid_orders:
                    if bp > adjusted_fair_value:
                        volume = min(bv, max_ask_pos)
                        if volume > 0:
                            orders.append(Order(product, bp, -volume))
                            max_ask_pos -= volume
                    elif bp == adjusted_fair_value and current_position > 0:
                        # Only take at fair value if it helps reduce long position
                        volume = min(bv, current_position)
                        if volume > 0:
                            orders.append(Order(product, bp, -volume))
                            max_ask_pos -= volume

                ###########################################################
                ####### 5. MAKING (Passive)
                ###########################################################
                best_bid = max(bid_orders, key=lambda x: x[0])[0]
                best_ask = min(ask_orders, key=lambda x: x[0])[0]

                # Attempt to penny the best quotes, but strictly bound them 
                # by our inventory-skewed adjusted fair value.
                my_bid = min(best_bid + 1, adjusted_fair_value - 1)
                my_ask = max(best_ask - 1, adjusted_fair_value + 1)
                
                if max_bid_pos > 0:
                    orders.append(Order(product, my_bid, max_bid_pos))
                if max_ask_pos > 0:
                    orders.append(Order(product, my_ask, -max_ask_pos))

            if orders:
                result[product] = orders
    
        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData