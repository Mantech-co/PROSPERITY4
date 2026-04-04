from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict

class Trader:
    def __init__(self):
        # Prosperity 4 tutorial limits
        self.LIMIT = {
            "EMERALDS": 80,
            "TOMATOES": 80
        }

    def tomato_orders(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int
    ) -> List[Order]:
        orders: List[Order] = []
        buy_vol = sell_vol = 0
        limit = self.LIMIT[product]
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        fair_value = (best_ask + best_bid)/2
        # 1. Take aggressive orders (Cross the spread if highly favorable)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_qty = -order_depth.sell_orders[best_ask]
            # If someone is foolishly selling below 10,000, snap it up
            if best_ask < fair_value:
                qty = min(ask_qty, limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_qty = order_depth.buy_orders[best_bid]
            # If someone is foolishly buying above 10,000, dump it to them
            if best_bid > fair_value:
                qty = min(bid_qty, limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty

        # 2. Market Make (Provide passive liquidity at 1 tick away from fair)
        # Calculate remaining capacity after our aggressive orders above
        bid_capacity = limit - (position + buy_vol)
        ask_capacity = limit + (position - sell_vol)

        # Rest bids at 9999, asks at 10001
        quote_bid = best_bid +1
        quote_ask = best_ask -1

        if bid_capacity > 0:
            orders.append(Order(product, quote_bid, bid_capacity))
        if ask_capacity > 0:
            orders.append(Order(product, quote_ask, -ask_capacity))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        

        
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            
            if product == "EMERALDS":
                pass
            elif product == "TOMATOES":
                result[product] = self.tomato_orders(product, depth, pos)
                
        trader_data = "TOMATO_MM_V1"
        return result, 1, trader_data
