import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

LOG_FILE = "backtests/2026-04-25_11-58-18.log"

with open(LOG_FILE) as f:
    data = json.load(f)

# parse: LOGORDER:PRODUCT:SIDE:PRICE:QTY:TAG
orders = []
for entry in data["logs"]:
    ts = entry["timestamp"]
    for line in entry["lambdaLog"].split("\n"):
        if line.startswith("LOGORDER:"):
            parts = line.split(":")
            if len(parts) >= 6:
                _, product, side, price, qty, tag = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                orders.append({
                    "timestamp": ts,
                    "product": product,
                    "side": side,
                    "price": int(price),
                    "qty": int(qty),
                    "tag": tag,
                })

products = sorted(set(o["product"] for o in orders))
n = len(products)
fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), squeeze=False)

for i, product in enumerate(products):
    ax = axes[i][0]
    prod_orders = [o for o in orders if o["product"] == product]

    buy_ts = [o["timestamp"] for o in prod_orders if o["side"] == "BUY"]
    buy_px = [o["price"] for o in prod_orders if o["side"] == "BUY"]
    buy_qty = [o["qty"] for o in prod_orders if o["side"] == "BUY"]

    sell_ts = [o["timestamp"] for o in prod_orders if o["side"] == "SELL"]
    sell_px = [o["price"] for o in prod_orders if o["side"] == "SELL"]
    sell_qty = [o["qty"] for o in prod_orders if o["side"] == "SELL"]

    ax.scatter(buy_ts, buy_px, s=[q * 0.5 for q in buy_qty], c="green", alpha=0.6, marker="^", label="BUY", zorder=3)
    ax.scatter(sell_ts, sell_px, s=[q * 0.5 for q in sell_qty], c="red", alpha=0.6, marker="v", label="SELL", zorder=3)

    ax.set_title(product, fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Orders Scatter — backtests/2026-04-25_11-58-18.log", fontsize=15, y=1.01)
plt.tight_layout()
out = "analysis/order_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
