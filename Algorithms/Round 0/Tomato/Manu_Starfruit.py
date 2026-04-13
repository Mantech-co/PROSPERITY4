import json
from typing import List, Tuple
from datamodel import OrderDepth, UserId, TradingState, Order
import math

class Product:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

PARAMS = {
    Product.TOMATOES: {
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    }
}

class Logger:
    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    def log(self, timestamp: int, **series: float) -> None:
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
    def __init__(self, params=None):
        self.logger = Logger()
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.EMERALDS: 80, Product.TOMATOES: 80}

    def take_best_orders(
        self, product: str, fair_value: int, take_width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0
    ) -> Tuple[int, int]:
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(round(bid)), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(round(ask)), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int
    ) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = int(round(fair_value - width))
        fair_for_ask = int(round(fair_value + width))

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def tomatoes_fair_value(self, order_depth: OrderDepth, state_dict: dict) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.TOMATOES]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.TOMATOES]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if state_dict.get("tomatoes_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = state_dict["tomatoes_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if state_dict.get("tomatoes_last_price", None) is not None:
                last_price = state_dict["tomatoes_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.TOMATOES]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            state_dict["tomatoes_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int, prevent_adverse: bool = False, adverse_volume: int = 0
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int, position: int, buy_order_volume: int, sell_order_volume: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int, disregard_edge: float, join_edge: float, default_edge: float, manage_position: bool = False, soft_position_limit: int = 0
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
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

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        result = {}
        
        traderData = state.traderData if state.traderData else "{}"
        try:
            state_dict = json.loads(traderData)
        except Exception:
            state_dict = {}

        tomatoes_avg = state_dict.get("TOMATOES_AVG", None)

        tomatotrades = state.own_trades.get("TOMATOES", [])
        if tomatotrades:
            for i in tomatotrades:
                quantity = i.quantity*(1 if i.buyer == "SUBMISSION" else -1)
                self.logger.log(state.timestamp, TOMATO_TRADE=quantity)
                if tomatoes_avg is None:
                    tomatoes_avg = (i.price, quantity)
                else:
                    total_quantity = tomatoes_avg[1] + quantity
                    if total_quantity != 0:
                        tomatoes_avg = ((tomatoes_avg[0] * tomatoes_avg[1] + i.price * quantity) / total_quantity, total_quantity)
                    else:
                        tomatoes_avg = None
            state_dict["TOMATOES_AVG"] = tomatoes_avg

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == Product.EMERALDS:
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
                if bid_price < 10000:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > 10000:
                    orders.append(Order(product, ask_price, -max_ask_pos))

                if orders:
                    result[product] = orders

            elif product == Product.TOMATOES:
                if product in self.params:
                    position = int(state.position.get(product, 0))
                    fair_value = self.tomatoes_fair_value(order_depth, state_dict)
                    
                    if fair_value is not None:
                        take_orders, buy_vol, sell_vol = self.take_orders(
                            product, order_depth, fair_value,
                            self.params[product]["take_width"], position,
                            self.params[product]["prevent_adverse"],
                            self.params[product]["adverse_volume"]
                        )
                        clear_orders, buy_vol, sell_vol = self.clear_orders(
                            product, order_depth, fair_value,
                            self.params[product]["clear_width"], position,
                            buy_vol, sell_vol
                        )
                        make_orders, _, _ = self.make_orders(
                            product, order_depth, fair_value, position,
                            buy_vol, sell_vol,
                            self.params[product]["disregard_edge"],
                            self.params[product]["join_edge"],
                            self.params[product]["default_edge"]
                        )
                        
                        orders = take_orders + clear_orders + make_orders
                        if orders:
                            result[product] = orders

        if tomatoes_avg is not None:
            self.logger.log(state.timestamp, TOMATOES_POS=tomatoes_avg[1], TOMATOES_AVG=tomatoes_avg[0])
            state_dict["TOMATOES_AVG"] = tomatoes_avg

        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData
