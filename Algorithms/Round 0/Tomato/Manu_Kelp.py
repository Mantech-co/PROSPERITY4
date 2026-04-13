import json
from typing import List, Tuple
from datamodel import OrderDepth, UserId, TradingState, Order
import math

class Product:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

# Hyperparameters
PARAMS = {
    Product.TOMATOES: {
        "ema_alpha": 0.4, # Tunes the EMA decay rate 
        "take_percentage": 0.15, # 15% of total limit (80) allocated to taking 
        "take_width": 1,
        "default_edge": 1,
        "disregard_edge": 1,
        "join_edge": 0,
    }
}

class Logger:
    PREFIX = 'LOGVIZ:'
    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []
    def log(self, timestamp: int, **series: float) -> None:
        if not series: return
        clean: dict[str, float] = {}
        for k, v in series.items():
            try: clean[k] = float(v)
            except (TypeError, ValueError): pass
        if not clean: return
        line = self.PREFIX + json.dumps(clean, separators=(',', ':'))
        if self._auto_print: print(line)
        else: self._buffer.append(line)
    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out
    def reset(self) -> None:
        self._buffer.clear()

class Trader:
    def __init__(self, params=None):
        self.logger = Logger()
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.EMERALDS: 80, Product.TOMATOES: 80}

    def take_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int, buy_order_volume: int, sell_order_volume: int, take_limit: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[product]
        
        # Take from Asks (Buy orders)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            
            if best_ask <= int(round(fair_value - take_width)):
                global_buy_room = position_limit - (position + buy_order_volume)
                buy_room = min(global_buy_room, take_limit - buy_order_volume)
                quantity = min(best_ask_amount, buy_room)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Take from Bids (Sell orders)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            if best_bid >= int(round(fair_value + take_width)):
                global_sell_room = position_limit + (position - sell_order_volume)
                sell_room = min(global_sell_room, take_limit - sell_order_volume)
                quantity = min(best_bid_amount, sell_room)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
                        
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float, default_edge: float, make_limit: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[product]
        
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = int(round(fair_value + default_edge))
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = int(round(fair_value - default_edge))
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # Make Buy Order
        global_buy_room = position_limit - (position + buy_order_volume)
        buy_quantity = min(global_buy_room, make_limit)
        if buy_quantity > 0:
            orders.append(Order(product, int(round(bid)), buy_quantity))
            buy_order_volume += buy_quantity

        # Make Sell Order
        global_sell_room = position_limit + (position - sell_order_volume)
        sell_quantity = min(global_sell_room, make_limit)
        if sell_quantity > 0:
            orders.append(Order(product, int(round(ask)), -sell_quantity))
            sell_order_volume += sell_quantity
            
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        result = {}
        
        traderData = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(traderData)
        except Exception:
            state_dict = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == Product.EMERALDS:
                current_position = int(state.position.get(product, 0))
                # 80 limit mapping directly to logic
                position_limit = 80
                max_bid_pos = position_limit - current_position
                max_ask_pos = position_limit + current_position

                bid_orders = order_depth.buy_orders.items()
                ask_orders = order_depth.sell_orders.items()

                if not bid_orders or not ask_orders:
                    continue

                bid_wall = max(bid_orders, key=lambda x: x[0])[0]
                ask_wall = min(ask_orders, key=lambda x: x[0])[0]

                # 1. TAKING
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

                # 2. MAKING
                bid_price = int(bid_wall + 1)
                ask_price = int(ask_wall - 1)

                for bp, bv in bid_orders:
                    overbidding_price = bp + 1
                    if bv > 1 and overbidding_price < 10000:
                        bid_price = max(bid_price, overbidding_price)
                        break
                    elif bp < 10000:
                        bid_price = max(bid_price, bp)
                        break

                for sp, sv in ask_orders:
                    underbidding_price = sp - 1
                    if sv > 1 and underbidding_price > 10000:
                        ask_price = min(ask_price, underbidding_price)
                        break
                    elif sp > 10000:
                        ask_price = min(ask_price, sp)
                        break

                if bid_price < 10000:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > 10000:
                    orders.append(Order(product, ask_price, -max_ask_pos))

                if orders:
                    result[product] = orders

            elif product == Product.TOMATOES:
                if product in self.params:
                    position = int(state.position.get(product, 0))
                    
                    bid_orders = list(order_depth.buy_orders.items())
                    ask_orders = list(order_depth.sell_orders.items())

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

                    # Random Walk EMA Estimate
                    alpha = self.params[product]["ema_alpha"]
                    tomatoes_ema = state_dict.get("TOMATOES_EMA", vwap)
                    tomatoes_ema = alpha * vwap + (1 - alpha) * tomatoes_ema
                    state_dict["TOMATOES_EMA"] = tomatoes_ema
                    
                    fair_value = tomatoes_ema

                    # Static Allocation Formula
                    take_pct = self.params[product]["take_percentage"]
                    take_limit = int(self.LIMIT[product] * take_pct)
                    make_limit = self.LIMIT[product] - take_limit
                    
                    buy_vol = 0
                    sell_vol = 0
                    
                    take_orders, buy_vol, sell_vol = self.take_orders(
                        product, order_depth, fair_value,
                        self.params[product]["take_width"], position,
                        buy_vol, sell_vol, take_limit
                    )
                    
                    make_orders, buy_vol, sell_vol = self.make_orders(
                        product, order_depth, fair_value, position,
                        buy_vol, sell_vol,
                        self.params[product]["disregard_edge"],
                        self.params[product]["join_edge"],
                        self.params[product]["default_edge"],
                        make_limit
                    )
                    
                    orders = take_orders + make_orders
                    if orders:
                        result[product] = orders

        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData
