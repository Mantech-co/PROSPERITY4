import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "round_4")
PRODUCT = "VELVETFRUIT_EXTRACT"

_palette = [
    "#e6194b", "#7fff00", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]

ALL_TRADERS = ["Mark 01", "Mark 14", "Mark 22", "Mark 49", "Mark 55", "Mark 67"]
KEY_MAP = {str(i + 1): t for i, t in enumerate(ALL_TRADERS)}  # "1"->Mark 01 ... "6"->Mark 67
TRADER_COLOR = {t: _palette[i % len(_palette)] for i, t in enumerate(ALL_TRADERS)}

state = {"day": 1, "show": {t: True for t in ALL_TRADERS}}

fig, ax = plt.subplots(figsize=(16, 6))


def load(day):
    prices_df = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{day}.csv", sep=";")
    trades_df = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{day}.csv", sep=";")
    prices = prices_df[prices_df["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    trades = trades_df[trades_df["symbol"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    trades = pd.merge_asof(trades, prices[["timestamp", "mid_price"]], on="timestamp", direction="nearest")
    return prices, trades


def draw():
    day = state["day"]
    active = [t for t, on in state["show"].items() if on]
    prices, trades = load(day)

    ax.cla()
    ax.plot(prices["timestamp"], prices["mid_price"], color="steelblue", lw=1.2, zorder=2, label="mid")
    ax.plot(prices["timestamp"], prices["bid_price_1"], color="#00b300", lw=0.7, alpha=0.4, zorder=1, label="bid1")
    ax.plot(prices["timestamp"], prices["ask_price_1"], color="#cc0000", lw=0.7, alpha=0.4, zorder=1, label="ask1")

    for trader in active:
        color = TRADER_COLOR[trader]
        buys = trades[trades["buyer"] == trader]
        sells = trades[trades["seller"] == trader]
        ax.scatter(buys["timestamp"], buys["price"],
                   color=color, marker="^", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.scatter(sells["timestamp"], sells["price"],
                   color=color, marker="v", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)

    for _, row in trades[trades["buyer"].isin(active) | trades["seller"].isin(active)].iterrows():
        ax.text(row["timestamp"], row["price"], str(int(row["quantity"])),
                fontsize=6, ha="left", va="bottom", color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    key_hints = "  |  " + "  ".join(f"{k}={t}" for k, t in KEY_MAP.items())
    legend_handles = [
        mlines.Line2D([], [], color="steelblue", lw=1.2, label="mid"),
        mlines.Line2D([], [], color="#00b300", lw=0.7, label="bid1"),
        mlines.Line2D([], [], color="#cc0000", lw=0.7, label="ask1"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=6, label="buy"),
        mlines.Line2D([], [], color="gray", marker="v", linestyle="None", markersize=6, label="sell"),
    ] + [
        mlines.Line2D([], [], color=TRADER_COLOR[t], marker="s", linestyle="None", markersize=7,
                      label=f"{KEY_MAP_INV[t]}={t}" + ("" if state["show"][t] else " [OFF]"),
                      alpha=1.0 if state["show"][t] else 0.3)
        for t in ALL_TRADERS
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=3)
    ax.set_ylabel("Price")
    ax.set_title(f"{PRODUCT}  |  Day {day}  |  ←→ day{key_hints}", fontsize=9)
    ax.set_xlabel("Timestamp")
    ax.grid(True, alpha=0.3)
    fig.canvas.draw_idle()


KEY_MAP_INV = {t: k for k, t in KEY_MAP.items()}


def on_key(event):
    if event.key == "right":
        state["day"] = min(state["day"] + 1, 3)
    elif event.key == "left":
        state["day"] = max(state["day"] - 1, 1)
    elif event.key in KEY_MAP:
        t = KEY_MAP[event.key]
        state["show"][t] = not state["show"][t]
    else:
        return
    draw()


fig.canvas.mpl_connect("key_press_event", on_key)
draw()
plt.tight_layout()
plt.show()
