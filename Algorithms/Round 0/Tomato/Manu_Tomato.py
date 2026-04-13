import json
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

        tomatoes_ema = state_dict.get("TOMATOES_EMA", None)
        tomatoes_avg = state_dict.get("TOMATOES_AVG", None)

        tomatotrades = state.own_trades.get("TOMATOES", [])
        if tomatotrades:
            for i in tomatotrades:
                # Process each tomato trade
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
                if bid_price < 10000:
                    orders.append(Order(product, bid_price, max_bid_pos))
                if ask_price > 10000:
                    orders.append(Order(product, ask_price, -max_ask_pos))
            
            elif product == "TOMATOES":
                current_position = int(state.position.get(product, 0))
                position_limit = 80 # default assumption matching emeralds
                alpha = 0.5
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

                if tomatoes_ema is None:
                    tomatoes_ema = vwap
                else:
                    tomatoes_ema = alpha * vwap + (1 - alpha) * tomatoes_ema

                mid_price = round(tomatoes_ema)

                bid_wall = max(bid_orders, key=lambda x: x[0])[0]
                ask_wall = min(ask_orders, key=lambda x: x[0])[0]

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
            if tomatoes_avg is not None:
                for i in orders:
                    if tomatoes_avg[1] > 0 and i.price < tomatoes_avg[0] and i.quantity < 0:
                            orders.remove(i)
                    elif tomatoes_avg[1] < 0 and i.price > tomatoes_avg[0] and i.quantity > 0:
                            orders.remove(i)

            if orders:
                result[product] = orders
    
        if tomatoes_ema is not None:
            state_dict["TOMATOES_EMA"] = tomatoes_ema
        if tomatoes_avg is not None:
            self.logger.log(state.timestamp, TOMATOES_POS=tomatoes_avg[1], TOMATOES_AVG=tomatoes_avg[0])
            state_dict["TOMATOES_AVG"] = tomatoes_avg
        traderData = json.dumps(state_dict)
        conversions = 0
        return result, conversions, traderData