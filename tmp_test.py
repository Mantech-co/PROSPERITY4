"""
Quick validation tests for:
1. __enforce_limits: orders clipped (not dropped) when exceeding position limits
2. _build_position_plot: position resets to 0 at day boundaries
"""

# ── Test 1: enforce_limits clipping ───────────────────────────────────────────
# Reproduce just the logic without importing the full module stack
print("=" * 60)
print("TEST 1: __enforce_limits clipping logic")
print("=" * 60)

LIMITS = {'TOMATOES': 80}

class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

def enforce_limits(position, product_orders, limit):
    """Replicate the new __enforce_limits logic."""
    log = []

    # Clip buy orders (most aggressive = highest price first)
    buy_orders = sorted([o for o in product_orders if o.quantity > 0], key=lambda o: o.price, reverse=True)
    buy_capacity = max(0, limit - position)
    for order in buy_orders:
        if order.quantity > buy_capacity:
            if buy_capacity == 0:
                product_orders.remove(order)
                log.append(f"Buy @ {order.price} x{order.quantity} DROPPED")
            else:
                log.append(f"Buy @ {order.price} CLIPPED {order.quantity} -> {buy_capacity}")
                order.quantity = buy_capacity
                buy_capacity = 0
        else:
            buy_capacity -= order.quantity

    # Clip sell orders (most aggressive = lowest price first)
    sell_orders = sorted([o for o in product_orders if o.quantity < 0], key=lambda o: o.price)
    sell_capacity = max(0, limit + position)
    for order in sell_orders:
        sell_qty = abs(order.quantity)
        if sell_qty > sell_capacity:
            if sell_capacity == 0:
                product_orders.remove(order)
                log.append(f"Sell @ {order.price} x{sell_qty} DROPPED")
            else:
                log.append(f"Sell @ {order.price} CLIPPED {sell_qty} -> {sell_capacity}")
                order.quantity = -sell_capacity
                sell_capacity = 0
        else:
            sell_capacity -= sell_qty

    return product_orders, log


# Case 1: position=75, buy 10 -> clipped to 5
orders = [Order('T', 100, 10)]
result, log = enforce_limits(75, orders, 80)
print(f"pos=75, buy 10  -> qty={result[0].quantity}  log={log}")
assert result[0].quantity == 5, f"Expected 5, got {result[0].quantity}"
print("  PASS ✓\n")

# Case 2: position=75, buy 3 -> unchanged
orders = [Order('T', 100, 3)]
result, log = enforce_limits(75, orders, 80)
print(f"pos=75, buy 3   -> qty={result[0].quantity}  log={log}")
assert result[0].quantity == 3
print("  PASS ✓\n")

# Case 3: position=80, buy 1 -> DROPPED
orders = [Order('T', 100, 1)]
result, log = enforce_limits(80, orders, 80)
print(f"pos=80, buy 1   -> remaining={result}  log={log}")
assert result == []
print("  PASS ✓\n")

# Case 4: position=75, buy 10 + sell 3 -> buy clipped to 5, sell untouched
orders = [Order('T', 100, 10), Order('T', 90, -3)]
result, log = enforce_limits(75, orders, 80)
buys = [o for o in result if o.quantity > 0]
sells = [o for o in result if o.quantity < 0]
print(f"pos=75, buy 10 + sell 3 -> buy={buys[0].quantity if buys else 'DROP'}, sell={sells[0].quantity if sells else 'DROP'}  log={log}")
assert buys[0].quantity == 5
assert sells[0].quantity == -3
print("  PASS ✓\n")

# Case 5: position=-78, sell 5 -> clipped to 2
orders = [Order('T', 90, -5)]
result, log = enforce_limits(-78, orders, 80)
print(f"pos=-78, sell 5 -> qty={result[0].quantity}  log={log}")
assert result[0].quantity == -2, f"Expected -2, got {result[0].quantity}"
print("  PASS ✓\n")

# Case 6: position=-80, sell 1 -> DROPPED
orders = [Order('T', 90, -1)]
result, log = enforce_limits(-80, orders, 80)
print(f"pos=-80, sell 1 -> remaining={result}  log={log}")
assert result == []
print("  PASS ✓\n")

# ── Test 2: position plot day-boundary reset ───────────────────────────────────
print("=" * 60)
print("TEST 2: _build_position_plot day-boundary reset logic")
print("=" * 60)

def simulate_position_plot(trades):
    """Replicate the _build_position_plot accumulation logic."""
    pos_by_sym = {}
    for tr in trades:
        ts = tr['timestamp']
        plot_ts = ts
        # Backtester logs: no 'day' field -> derive day_seg from ts // 1_000_000
        if 'day' in tr:
            day_seg = tr['day']
        else:
            day_seg = plot_ts // 1_000_000

        is_buyer = tr.get('buyer') == 'SUBMISSION'
        is_sell  = tr.get('seller') == 'SUBMISSION'
        delta = 0
        if is_buyer: delta += tr['qty']
        if is_sell:  delta -= tr['qty']
        if delta != 0:
            pos_by_sym.setdefault(tr['sym'], []).append((day_seg, plot_ts, delta))

    result = {}
    for sym, events in pos_by_sym.items():
        events = sorted(events, key=lambda x: x[1])
        cum = 0
        last_day_seg = events[0][0]
        vals = []
        for day_seg, plot_ts, delta in events:
            if day_seg != last_day_seg:
                cum = 0  # reset at day boundary
                last_day_seg = day_seg
            cum += delta
            vals.append((plot_ts, cum))
        result[sym] = vals
    return result

# Multi-day: day 0 builds to 80, day 1 resets and builds to 80 again
# OLD bug: would show 160. New fix: shows 80 at day 1.
trades = [
    {'sym': 'T', 'timestamp': 500, 'qty': 80, 'buyer': 'SUBMISSION', 'seller': ''},     # day 0
    {'sym': 'T', 'timestamp': 1_000_100, 'qty': 80, 'buyer': 'SUBMISSION', 'seller': ''}, # day 1
]
result = simulate_position_plot(trades)
vals = result['T']
print(f"Multi-day trades: {vals}")
assert vals[0] == (500, 80), f"Day 0 end: {vals[0]}"
assert vals[1] == (1_000_100, 80), f"Day 1 end: {vals[1]}, expected (1000100, 80) not 160"
print("  Position resets to 0 at day boundary  PASS ✓\n")

# Single-day: no reset needed
trades2 = [
    {'sym': 'T', 'timestamp': 100, 'qty': 40, 'buyer': 'SUBMISSION', 'seller': ''},
    {'sym': 'T', 'timestamp': 200, 'qty': 40, 'buyer': 'SUBMISSION', 'seller': ''},
]
result2 = simulate_position_plot(trades2)
vals2 = result2['T']
print(f"Single-day trades: {vals2}")
assert vals2[-1][1] == 80, f"Expected 80, got {vals2[-1][1]}"
print("  Single-day accumulation unchanged  PASS ✓\n")

# Self-trade: buyer=SUBMISSION AND seller=SUBMISSION -> delta=0
trades3 = [
    {'sym': 'T', 'timestamp': 100, 'qty': 80, 'buyer': 'SUBMISSION', 'seller': 'SUBMISSION'},
]
result3 = simulate_position_plot(trades3)
print(f"Self-trade (buyer=seller=SUBMISSION): result={result3}")
assert 'T' not in result3, "Self-trade delta=0, should produce no event"
print("  Self-trade correctly produces no position delta  PASS ✓\n")

print("All tests passed! ✓")
