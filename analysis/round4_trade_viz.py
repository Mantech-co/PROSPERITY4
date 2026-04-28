import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "round_4")
PRODUCT = "VEV_5300"
POV = "Mark 38"

_palette = [
    "#e6194b", "#7fff00", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

ALL_VOLS = [1, 2, 3, 4, 5, 6]
state = {"day": 1, "show": {v: True for v in ALL_VOLS}}

fig1, ax1 = plt.subplots(figsize=(16, 6))
fig2, ax_vc = plt.subplots(figsize=(7, 4))
fig3, ax_ia = plt.subplots(figsize=(8, 5))
fig4, ax_streak = plt.subplots(figsize=(8, 5))
fig5, ax_cluster = plt.subplots(figsize=(8, 5))


def load(day):
    prices_df = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{day}.csv", sep=";")
    trades_df = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{day}.csv", sep=";")
    prices = prices_df[prices_df["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    trades = trades_df[trades_df["symbol"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    trades = pd.merge_asof(trades, prices[["timestamp", "mid_price"]], on="timestamp", direction="nearest")
    return prices, trades


def draw():
    day = state["day"]
    active_vols = [v for v, on in state["show"].items() if on]
    prices, trades = load(day)

    all_traders = sorted(set(trades["buyer"].tolist()) | set(trades["seller"].tolist()))
    trader_color = {t: _palette[i % len(_palette)] for i, t in enumerate(all_traders)}

    m38_m14 = trades.copy()
    filtered = m38_m14[m38_m14["quantity"].isin(active_vols)].copy().reset_index(drop=True)

    vol_label = ",".join(str(v) for v in sorted(active_vols)) if active_vols else "none"
    toggle_hint = "  |  2/3 toggle vol"

    # main price chart
    ax1.cla()
    ax1.plot(prices["timestamp"], prices["mid_price"], color="steelblue", lw=1.2, zorder=2)
    ax1.plot(prices["timestamp"], prices["bid_price_1"], color="#00b300", lw=0.7, alpha=0.4, zorder=1)
    ax1.plot(prices["timestamp"], prices["ask_price_1"], color="#cc0000", lw=0.7, alpha=0.4, zorder=1)
    for trader, grp in filtered.groupby("buyer"):
        ax1.scatter(grp["timestamp"], grp["price"],
                    color=trader_color[trader], marker="^", s=60, zorder=5,
                    edgecolors="black", linewidths=0.8)
    for trader, grp in filtered.groupby("seller"):
        ax1.scatter(grp["timestamp"], grp["price"],
                    color=trader_color[trader], marker="v", s=60, zorder=5,
                    edgecolors="black", linewidths=0.8)
    for _, row in filtered.iterrows():
        ax1.text(row["timestamp"], row["price"], str(int(row["quantity"])),
                 fontsize=6, ha="left", va="bottom", color="black",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    legend_handles = [
        mlines.Line2D([], [], color="steelblue", lw=1.2, label="mid"),
        mlines.Line2D([], [], color="#00b300", lw=0.7, label="bid1"),
        mlines.Line2D([], [], color="#cc0000", lw=0.7, label="ask1"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=6, label="buy"),
        mlines.Line2D([], [], color="gray", marker="v", linestyle="None", markersize=6, label="sell"),
    ] + [
        mlines.Line2D([], [], color=trader_color[tr], marker="s", linestyle="None", markersize=6, label=tr)
        for tr in all_traders
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=3)
    ax1.set_ylabel("Price")
    ax1.set_title(f"{PRODUCT}  |  Day {day}  |  vol=[{vol_label}]  |  ←→ day{toggle_hint}", fontsize=10)
    ax1.set_xlabel("Timestamp")
    ax1.grid(True, alpha=0.3)
    fig1.canvas.draw_idle()

    # vol count — all M38-M14 trades
    ax_vc.cla()
    vol_counts = m38_m14["quantity"].value_counts().sort_index()
    colors = ["#4363d8" if v in active_vols else "#aaaaaa" for v in vol_counts.index]
    if not vol_counts.empty:
        ax_vc.bar(vol_counts.index.astype(str), vol_counts.values,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax_vc.set_xlabel("Volume")
    ax_vc.set_ylabel("# Trades")
    ax_vc.set_title(f"Volume vs Trade Count (all)  |  {PRODUCT} Day {day}")
    ax_vc.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.canvas.draw_idle()

    # IAT vs volume
    ax_ia.cla()
    if len(filtered) > 1:
        m38s = filtered.sort_values("timestamp").reset_index(drop=True)
        iat = m38s["timestamp"].diff().dropna()
        tail = m38s.iloc[1:].reset_index(drop=True)
        tail["iat"] = iat.values
        buys  = tail[tail["buyer"]  == POV]
        sells = tail[tail["seller"] == POV]
        ax_ia.scatter(buys["iat"],  buys["quantity"],  color="#4363d8", marker="^",
                      edgecolors="black", linewidths=0.5, s=60, alpha=0.9, label=f"{POV} buy")
        ax_ia.scatter(sells["iat"], sells["quantity"], color="#e6194b", marker="v",
                      edgecolors="black", linewidths=0.5, s=60, alpha=0.9, label=f"{POV} sell")
        ax_ia.legend(fontsize=8)
    ax_ia.set_xlabel("IAT (ts)")
    ax_ia.set_ylabel("Volume")
    ax_ia.set_title(f"IAT vs Volume — vol=[{vol_label}]  |  {PRODUCT} Day {day}")
    ax_ia.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.canvas.draw_idle()

    # streak lengths
    ax_streak.cla()
    m38s_all = m38_m14.sort_values("timestamp").reset_index(drop=True)
    m38s_all["side"] = m38s_all.apply(lambda r: "buy" if r["buyer"] == POV else "sell", axis=1)
    buy_streaks, sell_streaks = [], []
    cur_side, cur_len = None, 0
    for side in m38s_all["side"]:
        if side == cur_side:
            cur_len += 1
        else:
            if cur_side == "buy":  buy_streaks.append(cur_len)
            elif cur_side == "sell": sell_streaks.append(cur_len)
            cur_side, cur_len = side, 1
    if cur_side == "buy":  buy_streaks.append(cur_len)
    elif cur_side == "sell": sell_streaks.append(cur_len)

    all_lens = sorted(set(buy_streaks + sell_streaks))
    buy_counts  = {l: buy_streaks.count(l)  for l in all_lens}
    sell_counts = {l: sell_streaks.count(l) for l in all_lens}
    x = range(len(all_lens))
    w = 0.35
    ax_streak.bar([i - w/2 for i in x], [buy_counts.get(l, 0)  for l in all_lens],
                  width=w, color="#4363d8", edgecolor="black", linewidth=0.5, label="buy streak")
    ax_streak.bar([i + w/2 for i in x], [sell_counts.get(l, 0) for l in all_lens],
                  width=w, color="#e6194b", edgecolor="black", linewidth=0.5, label="sell streak")
    ax_streak.set_xticks(list(x))
    ax_streak.set_xticklabels([str(l) for l in all_lens])
    ax_streak.set_xlabel("Streak Length")
    ax_streak.set_ylabel("Count")
    ax_streak.set_title(f"Contiguous Buy/Sell Streak Length — {POV}  |  {PRODUCT} Day {day}")
    ax_streak.legend(fontsize=8)
    ax_streak.grid(True, alpha=0.3, axis="y")
    fig4.tight_layout()
    fig4.canvas.draw_idle()

    # IAT by side-transition type
    ax_cluster.cla()
    m38s_all2 = m38_m14.sort_values("timestamp").reset_index(drop=True)
    m38s_all2["side"] = m38s_all2.apply(lambda r: "B" if r["buyer"] == POV else "S", axis=1)
    m38s_all2["iat"] = m38s_all2["timestamp"].diff()
    m38s_all2["prev_side"] = m38s_all2["side"].shift(1)
    m38s_all2 = m38s_all2.dropna(subset=["iat", "prev_side"])
    m38s_all2["transition"] = m38s_all2["prev_side"] + "→" + m38s_all2["side"]

    transition_order = ["B→B", "B→S", "S→B", "S→S"]
    transition_colors = {"B→B": "#4363d8", "B→S": "#f58231", "S→B": "#42d4f4", "S→S": "#e6194b"}
    data = [m38s_all2[m38s_all2["transition"] == t]["iat"].dropna().values for t in transition_order]
    bp = ax_cluster.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", lw=2))
    for patch, t in zip(bp["boxes"], transition_order):
        patch.set_facecolor(transition_colors[t])
        patch.set_alpha(0.75)
    for i, (t, d) in enumerate(zip(transition_order, data), 1):
        ax_cluster.scatter([i] * len(d), d, alpha=0.4, s=15, color=transition_colors[t], zorder=5)
    ax_cluster.set_xticks(range(1, len(transition_order) + 1))
    ax_cluster.set_xticklabels(transition_order)
    ax_cluster.set_ylabel("IAT (ts)")
    ax_cluster.set_title(f"IAT by Side Transition — M38  |  {PRODUCT} Day {day}\n"
                         f"same-side IAT < cross-side → clustering", fontsize=9)
    ax_cluster.grid(True, alpha=0.3, axis="y")
    fig5.tight_layout()
    fig5.canvas.draw_idle()


def on_key(event):
    if event.key == "right":
        state["day"] = min(state["day"] + 1, 3)
    elif event.key == "left":
        state["day"] = max(state["day"] - 1, 1)
    elif event.key in [str(v) for v in ALL_VOLS]:
        v = int(event.key)
        state["show"][v] = not state["show"][v]
    else:
        return
    draw()


fig1.canvas.mpl_connect("key_press_event", on_key)
draw()
plt.show()
