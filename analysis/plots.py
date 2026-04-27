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
    data = load_velvetfruit()
    mid = data["mid_price"]
    ts = data["global_ts"].values
    price = mid.values
    n = len(ts)

    state = {"start": 0, "window": WINDOW, "fast": 10, "slow": 50}

    def compute():
        fast_ema = mid.ewm(span=state["fast"], adjust=False).mean().values
        slow_ema = mid.ewm(span=state["slow"], adjust=False).mean().values
        cross = np.sign(fast_ema - slow_ema)
        cross_up = np.where((cross[1:] > 0) & (cross[:-1] <= 0))[0] + 1
        cross_dn = np.where((cross[1:] < 0) & (cross[:-1] >= 0))[0] + 1
        return fast_ema, slow_ema, cross_up, cross_dn

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [7, 3]})

    line_price, = ax1.plot([], [], color="white", linewidth=0.7, label="mid price")
    line_fast,  = ax1.plot([], [], color="#39ff6e", linewidth=1.2)
    line_slow,  = ax1.plot([], [], color="#ff3d5a", linewidth=1.2)
    scat_up = ax1.scatter([], [], marker="^", color="#39ff6e", s=60, zorder=6, label="crossover up")
    scat_dn = ax1.scatter([], [], marker="v", color="#ff3d5a", s=60, zorder=6, label="crossover dn")
    ax1.set_ylabel("mid price")
    ax1.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a2e")

    line_diff, = ax2.plot([], [], color="#ff9f43", linewidth=0.9)
    ax2.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("timestamp")
    ax2.set_ylabel("fast − slow EMA")
    ax2.set_facecolor("#111111")

    def draw():
        fast_ema, slow_ema, cross_up, cross_dn = compute()
        s, e = state["start"], min(state["start"] + state["window"], n)
        sl = slice(s, e)

        line_price.set_data(ts[sl], price[sl])
        line_fast.set_data(ts[sl], fast_ema[sl])
        line_slow.set_data(ts[sl], slow_ema[sl])

        diff = fast_ema - slow_ema
        line_diff.set_data(ts[sl], diff[sl])

        up_in = cross_up[(cross_up >= s) & (cross_up < e)]
        dn_in = cross_dn[(cross_dn >= s) & (cross_dn < e)]
        scat_up.set_offsets(np.c_[ts[up_in], price[up_in]] if len(up_in) else np.empty((0, 2)))
        scat_dn.set_offsets(np.c_[ts[dn_in], price[dn_in]] if len(dn_in) else np.empty((0, 2)))

        for ax, ydata in [(ax1, np.concatenate([price[sl], fast_ema[sl], slow_ema[sl]])),
                          (ax2, diff[sl])]:
            ax.set_xlim(ts[s], ts[e - 1])
            valid = ydata[~np.isnan(ydata)]
            if len(valid):
                pad = (valid.max() - valid.min()) * 0.05 or 1
                ax.set_ylim(valid.min() - pad, valid.max() + pad)

        line_fast.set_label(f"EMA fast ({state['fast']})")
        line_slow.set_label(f"EMA slow ({state['slow']})")
        ax1.set_title(
            f"VELVETFRUIT_EXTRACT — EMA crossover  "
            f"[fast={state['fast']} | slow={state['slow']}]  "
            f"crossovers in view: ↑{len(up_in)} ↓{len(dn_in)}",
            color="white"
        )
        ax1.legend(loc="upper left", fontsize=8, facecolor="#222", labelcolor="white")
        fig.canvas.draw_idle()

    def on_key(event):
        if   event.key == "right": state["start"] = min(state["start"] + WINDOW, n - WINDOW)
        elif event.key == "left":  state["start"] = max(state["start"] - WINDOW, 0)
        elif event.key == "2":     state["window"] = min(state["window"] + 100, n)
        elif event.key == "1":     state["window"] = max(state["window"] - 100, 100)
        # fast EMA: a/d
        elif event.key == "d":     state["fast"] = min(state["fast"] + 1, state["slow"] - 1)
        elif event.key == "a":     state["fast"] = max(state["fast"] - 1, 2)
        # slow EMA: j/l
        elif event.key == "l":     state["slow"] = min(state["slow"] + 1, 500)
        elif event.key == "j":     state["slow"] = max(state["slow"] - 1, state["fast"] + 1)
        else: return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
