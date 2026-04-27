import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "round_4")
PRODUCT = "HYDROGEL_PACK"
DAY = 1
POV = "Mark 14"

prices_df = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{DAY}.csv", sep=";")
trades_df = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{DAY}.csv", sep=";")

prices = prices_df[prices_df["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
trades = trades_df[trades_df["symbol"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
trades = pd.merge_asof(trades, prices[["timestamp", "mid_price"]], on="timestamp", direction="nearest")

# tag order book rows where 3 levels exist on a side → infer Mark 22 at best
prices["tag_bid"] = prices["bid_price_3"].notna()
prices["tag_ask"] = prices["ask_price_3"].notna()

# Mark 38 trade timestamps
mark38_ts = set(
    trades[(trades["buyer"] == "Mark 38") | (trades["seller"] == "Mark 38")]["timestamp"]
)

# tagged bid/ask rows not coinciding with a Mark 38 trade (within ±500 ts)
def not_near_mark38(ts_series):
    return ts_series.apply(
        lambda t: not any(abs(t - m) <= 500 for m in mark38_ts)
    )

tagged_bids = prices[prices["tag_bid"] & not_near_mark38(prices["timestamp"])]
tagged_asks = prices[prices["tag_ask"] & not_near_mark38(prices["timestamp"])]

all_traders = sorted(set(trades["buyer"].tolist()) | set(trades["seller"].tolist()))
_palette = [
    "#e6194b", "#7fff00", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]
trader_color = {t: _palette[i % len(_palette)] for i, t in enumerate(all_traders)}

# synthetic trades: tagged bid orders → buys, tagged ask orders → sells
syn_buys = tagged_bids[["timestamp", "bid_price_1", "bid_volume_1", "mid_price"]].rename(
    columns={"bid_price_1": "price", "bid_volume_1": "quantity"}
).assign(side=1)
syn_sells = tagged_asks[["timestamp", "ask_price_1", "ask_volume_1", "mid_price"]].rename(
    columns={"ask_price_1": "price", "ask_volume_1": "quantity"}
).assign(side=-1)

syn = pd.concat([syn_buys, syn_sells]).sort_values("timestamp").reset_index(drop=True)
syn["pos_delta"] = syn["side"] * syn["quantity"]
syn["cash_delta"] = -syn["side"] * syn["price"] * syn["quantity"]
syn["position"] = syn["pos_delta"].cumsum()
syn["realized_cash"] = syn["cash_delta"].cumsum()
syn["pnl"] = syn["realized_cash"] + syn["position"] * syn["mid_price"]

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

ax1.plot(prices["timestamp"], prices["mid_price"], color="steelblue", lw=1.2, zorder=2)
ax1.plot(prices["timestamp"], prices["bid_price_1"], color="#00b300", lw=0.7, alpha=0.4, zorder=1)
ax1.plot(prices["timestamp"], prices["ask_price_1"], color="#cc0000", lw=0.7, alpha=0.4, zorder=1)

# all trades — small, faded
for trader, grp in trades.groupby("buyer"):
    ax1.scatter(grp["timestamp"], grp["price"],
                color=trader_color[trader], marker="^", s=20, alpha=0.3, zorder=4)
for trader, grp in trades.groupby("seller"):
    ax1.scatter(grp["timestamp"], grp["price"],
                color=trader_color[trader], marker="v", s=20, alpha=0.3, zorder=4)

# tagged Mark 22 order book entries — diamonds, black outline
ax1.scatter(tagged_bids["timestamp"], tagged_bids["bid_price_1"],
            color="cyan", marker="D", s=30, zorder=7,
            edgecolors="black", linewidths=0.8, label="M22 tag bid")
ax1.scatter(tagged_asks["timestamp"], tagged_asks["ask_price_1"],
            color="magenta", marker="D", s=30, zorder=7,
            edgecolors="black", linewidths=0.8, label="M22 tag ask")

legend_handles = [
    mlines.Line2D([], [], color="steelblue", lw=1.2, label="mid"),
    mlines.Line2D([], [], color="#00b300", lw=0.7, label="bid1"),
    mlines.Line2D([], [], color="#cc0000", lw=0.7, label="ask1"),
    mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=6, label="buy"),
    mlines.Line2D([], [], color="gray", marker="v", linestyle="None", markersize=6, label="sell"),
    mlines.Line2D([], [], color="cyan", marker="D", linestyle="None", markersize=6,
                  markeredgecolor="black", label="M22 tag bid (3-level, no M38)"),
    mlines.Line2D([], [], color="magenta", marker="D", linestyle="None", markersize=6,
                  markeredgecolor="black", label="M22 tag ask (3-level, no M38)"),
] + [
    mlines.Line2D([], [], color=trader_color[tr], marker="s", linestyle="None", markersize=6, label=tr)
    for tr in all_traders
]
ax1.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=3)
ax1.set_ylabel("Price")
ax1.set_title(f"{PRODUCT}  |  Day {DAY}  |  POV: {POV}", fontsize=10)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.step(syn["timestamp"], syn["position"], where="post", color="darkorange", lw=1.2)
ax2.axhline(0, color="black", lw=0.5, linestyle="--")
ax2.set_ylabel("Position\n(M22 synthetic)", fontsize=8)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3.step(syn["timestamp"], syn["pnl"], where="post", color="purple", lw=1.2)
ax3.axhline(0, color="black", lw=0.5, linestyle="--")
ax3.set_ylabel("PnL\n(M22 synthetic)", fontsize=8)
ax3.set_xlabel("Timestamp")
ax3.grid(True, alpha=0.3)

plt.show()
