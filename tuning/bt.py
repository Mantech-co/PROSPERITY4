"""
Minimal backtester for hyperparameter tuning.
Reads data/round5/prices*.csv, runs strategy/anirudh.py Trader, outputs PnL.
"""
import sys
import os
import csv
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prosperity4bt"))

from datamodel import OrderDepth, TradingState, Order, Observation, Listing

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "round5")
DAYS = [2, 3, 4]


def load_prices(day: int) -> dict:
    """Returns {timestamp: {product: {buy_orders, sell_orders, mid}}}"""
    path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
    data = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            product = row["product"]
            buy_orders = {}
            sell_orders = {}
            for i in range(1, 4):
                bp, bv = row.get(f"bid_price_{i}", ""), row.get(f"bid_volume_{i}", "")
                if bp and bv:
                    buy_orders[int(bp)] = int(bv)
                ap, av = row.get(f"ask_price_{i}", ""), row.get(f"ask_volume_{i}", "")
                if ap and av:
                    sell_orders[int(ap)] = -int(av)  # negative as per convention
            mid = float(row["mid_price"]) if row["mid_price"] else None
            data[ts][product] = {"buy_orders": buy_orders, "sell_orders": sell_orders, "mid": mid}
    return data


def match_orders(orders: list, order_depths: dict, positions: dict, cash: dict):
    """Fill orders against orderbook depth, update positions and cash in-place."""
    for order in orders:
        sym = order.symbol
        qty = order.quantity
        depth = order_depths[sym]

        if qty > 0:  # buy: match against asks <= order.price
            for price in sorted(p for p in depth.sell_orders if p <= order.price):
                avail = abs(depth.sell_orders[price])
                fill = min(qty, avail)
                positions[sym] = positions.get(sym, 0) + fill
                cash[sym] = cash.get(sym, 0.0) - price * fill
                depth.sell_orders[price] += fill  # less negative
                if depth.sell_orders[price] == 0:
                    del depth.sell_orders[price]
                qty -= fill
                if qty == 0:
                    break

        elif qty < 0:  # sell: match against bids >= order.price
            sell_qty = abs(qty)
            for price in sorted((p for p in depth.buy_orders if p >= order.price), reverse=True):
                avail = depth.buy_orders[price]
                fill = min(sell_qty, avail)
                positions[sym] = positions.get(sym, 0) - fill
                cash[sym] = cash.get(sym, 0.0) + price * fill
                depth.buy_orders[price] -= fill
                if depth.buy_orders[price] == 0:
                    del depth.buy_orders[price]
                sell_qty -= fill
                if sell_qty == 0:
                    break


def run_day(trader, day: int, positions: dict, cash: dict) -> float:
    price_data = load_prices(day)
    timestamps = sorted(price_data.keys())
    products = set(p for ts in price_data.values() for p in ts)

    trader_data = ""
    last_mid = {}

    for ts in timestamps:
        order_depths = {}
        ts_data = price_data[ts]
        for product in products:
            if product not in ts_data:
                continue
            od = OrderDepth()
            od.buy_orders = dict(ts_data[product]["buy_orders"])
            od.sell_orders = dict(ts_data[product]["sell_orders"])
            order_depths[product] = od
            if ts_data[product]["mid"] is not None:
                last_mid[product] = ts_data[product]["mid"]

        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings={p: Listing(p, p, 1) for p in order_depths},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=dict(positions),
            observations=Observation({}, {}),
        )

        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            orders_dict, _, trader_data = trader.run(state)

        for product, orders in orders_dict.items():
            if product not in order_depths:
                continue
            match_orders(orders, order_depths, positions, cash)

    # mark-to-market at end of day
    pnl = sum(cash.values())
    for sym, pos in positions.items():
        if sym in last_mid:
            pnl += pos * last_mid[sym]
    return pnl


def run(params=None):
    from strategy.anirudh import Trader, TUNING_PARAMS

    if params is not None:
        for product, p in params.items():
            if product in TUNING_PARAMS:
                TUNING_PARAMS[product].update(p)

    trader = Trader()
    positions = {}
    cash = {}
    total_pnl = 0.0

    for day in DAYS:
        cash_before = sum(cash.values())
        run_day(trader, day, positions, cash)
        day_pnl = sum(cash.values()) - cash_before
        print(f"day {day}: pnl={day_pnl:+.0f}  cumulative={sum(cash.values()):.0f}  pos={dict(positions)}")

    print(f"\ntotal cash: {sum(cash.values()):.0f}")
    print("\nper-product cash:")
    for sym, val in sorted(cash.items(), key=lambda x: -x[1]):
        print(f"  {sym:<35} {val:+,.0f}")
    print("final positions:", {k: v for k, v in positions.items() if v != 0})
    return sum(cash.values())


if __name__ == "__main__":
    run()
