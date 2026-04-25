import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))
WINDOW = 1000


def load_velvetfruit() -> pd.DataFrame:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == "VELVETFRUIT_EXTRACT"])
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["day", "timestamp"]).reset_index(drop=True)
    max_ts = data["timestamp"].max() + 100
    data["global_ts"] = data["day"] * max_ts + data["timestamp"]
    return data


def main():
    hydrogel = load_velvetfruit()
    mid = hydrogel["mid_price"]
    ts = hydrogel["global_ts"].values
    price = mid.values
    best_bid = hydrogel["bid_price_1"].values
    best_ask = hydrogel["ask_price_1"].values
    n = len(ts)

    state = {"start": 0, "ema_span": 10, "vol_win": 20, "window": WINDOW}

    def compute():
        ema = mid.ewm(span=state["ema_span"], adjust=False).mean()
        vol = ema.rolling(state["vol_win"]).std().values
        ema = ema.values
        return ema, vol

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [7, 3]})
    line1, = ax1.plot([], [], label="mid price")
    line_ema, = ax1.plot([], [], color="red", linewidth=1)
    line_bid, = ax1.plot([], [], color="green", linewidth=0.8, linestyle="--", label="best bid")
    line_ask, = ax1.plot([], [], color="purple", linewidth=0.8, linestyle="--", label="best ask")
    ax1.set_ylabel("mid price")
    ax1.legend()

    line2, = ax2.plot([], [], color="orange")
    ax2.set_xlabel("timestamp")
    ax2.set_ylabel("std")

    def draw():
        ema, vol = compute()
        start, end = state["start"], min(state["start"] + state["window"], n)
        sl = slice(start, end)
        line1.set_data(ts[sl], price[sl])
        line_ema.set_data(ts[sl], ema[sl])
        line_bid.set_data(ts[sl], best_bid[sl])
        line_ask.set_data(ts[sl], best_ask[sl])
        line2.set_data(ts[sl], vol[sl])
        all_prices = np.concatenate([price[sl], best_bid[sl], best_ask[sl]])
        for ax, data in [(ax1, all_prices), (ax2, vol[sl])]:
            ax.set_xlim(ts[start], ts[end - 1])
            valid = data[~np.isnan(data)]
            if len(valid):
                pad = (valid.max() - valid.min()) * 0.05 or 1
                ax.set_ylim(valid.min() - pad, valid.max() + pad)
        line_ema.set_label(f"EMA({state['ema_span']})")
        ax1.set_title(f"VELVETFRUIT_EXTRACT mid price")
        ax2.set_title(f"rolling volatility (window={state['vol_win']})")
        ax1.legend(loc="upper left")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["start"] = min(state["start"] + WINDOW, n - WINDOW)
        elif event.key == "left":
            state["start"] = max(state["start"] - WINDOW, 0)
        elif event.key == "d":
            state["ema_span"] = min(state["ema_span"] + 1, 200)
        elif event.key == "a":
            state["ema_span"] = max(state["ema_span"] - 1, 2)
        elif event.key == "l":
            state["vol_win"] = min(state["vol_win"] + 1, 200)
        elif event.key == "j":
            state["vol_win"] = max(state["vol_win"] - 1, 2)
        elif event.key == "2":
            state["window"] = min(state["window"] + 100, n)
        elif event.key == "1":
            state["window"] = max(state["window"] - 100, 100)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
