import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "round_4")
PRODUCT = "HYDROGEL_PACK"
DAYS = [1, 2, 3]

# grid search ranges
WINDOWS = np.arange(5, 1001, 20)       # EMA span
Z_THRESHOLDS = np.arange(0.1, 6.1, 0.2)


def load_day(day):
    prices_df = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{day}.csv", sep=";")
    trades_df = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{day}.csv", sep=";")

    prices = prices_df[prices_df["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    trades = trades_df[
        (trades_df["symbol"] == PRODUCT) &
        (trades_df["buyer"].isin(["Mark 38", "Mark 14"])) &
        (trades_df["seller"].isin(["Mark 38", "Mark 14"]))
    ].copy().sort_values("timestamp").reset_index(drop=True)

    trades = pd.merge_asof(trades, prices[["timestamp", "mid_price"]], on="timestamp", direction="nearest")
    end_mid = prices["mid_price"].iloc[-1]
    return prices, trades, end_mid


def simulate(prices, trades, end_mid, window, z_thresh):
    mid = prices["mid_price"].values
    ts = prices["timestamp"].values

    ema = pd.Series(mid).ewm(span=window, adjust=False).mean().values
    rol_std = pd.Series(mid).rolling(window, min_periods=2).std().values

    ema_map = dict(zip(ts, ema))
    std_map = dict(zip(ts, rol_std))

    trade_pnls = []
    for _, row in trades.iterrows():
        t = row["timestamp"]
        price = row["price"]
        qty = row["quantity"]

        e = ema_map.get(t)
        s = std_map.get(t)
        if e is None or s is None or s == 0 or np.isnan(s):
            continue

        z = (price - e) / s
        if abs(z) < z_thresh:
            continue

        if z < -z_thresh:
            trade_pnls.append((end_mid - price) * qty)
        else:
            trade_pnls.append((price - end_mid) * qty)

    return trade_pnls


# load all days once
day_data = [load_day(d) for d in DAYS]

# grid search
results = np.zeros((len(WINDOWS), len(Z_THRESHOLDS)))

for i, w in enumerate(WINDOWS):
    for j, z in enumerate(Z_THRESHOLDS):
        all_pnls = []
        for prices, trades, end_mid in day_data:
            all_pnls.extend(simulate(prices, trades, end_mid, w, z))
        if len(all_pnls) == 0:
            results[i, j] = 0.0
        else:
            arr = np.array(all_pnls)
            cutoff = np.percentile(arr, 99)
            results[i, j] = arr[arr >= cutoff].sum()

# plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# heatmap
im = axes[0].imshow(
    results, aspect="auto", origin="lower",
    extent=[Z_THRESHOLDS[0], Z_THRESHOLDS[-1], WINDOWS[0], WINDOWS[-1]],
    cmap="RdYlGn"
)
plt.colorbar(im, ax=axes[0], label="Total PnL")
axes[0].set_xlabel("Z-Score Threshold")
axes[0].set_ylabel("EMA Window")
axes[0].set_title(f"Grid Search — {PRODUCT}  |  Days {DAYS}")

best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
axes[0].plot(Z_THRESHOLDS[best_j], WINDOWS[best_i], "k*", markersize=12, label=f"best: w={WINDOWS[best_i]}, z={Z_THRESHOLDS[best_j]:.1f}")
axes[0].legend(fontsize=8)

# best window slice
axes[1].plot(Z_THRESHOLDS, results[best_i, :], color="steelblue", lw=1.5, label=f"window={WINDOWS[best_i]}")
axes[1].axvline(Z_THRESHOLDS[best_j], color="red", lw=0.8, linestyle="--")
axes[1].set_xlabel("Z-Score Threshold")
axes[1].set_ylabel("Total PnL")
axes[1].set_title(f"PnL vs Z-Threshold @ best window")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.suptitle(f"Best: window={WINDOWS[best_i]}, z_thresh={Z_THRESHOLDS[best_j]:.1f}, PnL={results[best_i, best_j]:.1f}", fontsize=11)
plt.tight_layout()
plt.show()

best_w = WINDOWS[best_i]
best_z = Z_THRESHOLDS[best_j]
print(f"Best window={best_w}, z_thresh={best_z:.2f}, total_pnl={results[best_i, best_j]:.2f}")

# plot best params across all days
fig2, axes2 = plt.subplots(len(DAYS), 2, figsize=(16, 5 * len(DAYS)), squeeze=False)

for row, (day, (prices, trades, end_mid)) in enumerate(zip(DAYS, day_data)):
    mid = prices["mid_price"].values
    ts = prices["timestamp"].values

    ema = pd.Series(mid).ewm(span=best_w, adjust=False).mean().values
    rol_std = pd.Series(mid).rolling(best_w, min_periods=2).std().values
    ema_map = dict(zip(ts, ema))
    std_map = dict(zip(ts, rol_std))

    cum_pnl, pnl_ts = [], []
    running = 0.0
    buy_ts, buy_px, sell_ts, sell_px = [], [], [], []

    for _, r in trades.iterrows():
        t, price, qty = r["timestamp"], r["price"], r["quantity"]
        e = ema_map.get(t)
        s = std_map.get(t)
        if e is None or s is None or s == 0 or np.isnan(s):
            continue
        z = (price - e) / s
        if abs(z) < best_z:
            continue
        trade_pnl = (end_mid - price) * qty if z < -best_z else (price - end_mid) * qty
        running += trade_pnl
        (buy_ts if z < -best_z else sell_ts).append(t)
        (buy_px if z < -best_z else sell_px).append(price)
        cum_pnl.append(running); pnl_ts.append(t)

    ax_p = axes2[row, 0]
    ax_p.plot(ts, mid, color="steelblue", lw=1.0, label="mid")
    ax_p.plot(ts, ema, color="orange", lw=1.0, linestyle="--", label=f"EMA({best_w})")
    ax_p.fill_between(ts, ema - best_z * rol_std, ema + best_z * rol_std,
                      alpha=0.15, color="orange", label=f"±{best_z:.1f}σ")
    ax_p.scatter(buy_ts,  buy_px,  marker="^", color="green", s=50, zorder=5, label="buy signal")
    ax_p.scatter(sell_ts, sell_px, marker="v", color="red",   s=50, zorder=5, label="sell signal")
    ax_p.set_title(f"Day {day}  |  signals")
    ax_p.set_ylabel("Price"); ax_p.grid(True, alpha=0.3); ax_p.legend(fontsize=7)

    ax_c = axes2[row, 1]
    if pnl_ts:
        ax_c.step(pnl_ts, cum_pnl, where="post", color="purple", lw=1.5)
        ax_c.axhline(0, color="black", lw=0.5, linestyle="--")
    ax_c.set_title(f"Day {day}  |  cumulative PnL  (total={running:.1f})")
    ax_c.set_ylabel("PnL"); ax_c.set_xlabel("Timestamp"); ax_c.grid(True, alpha=0.3)

plt.suptitle(f"{PRODUCT}  |  best params: window={best_w}, z={best_z:.1f}", fontsize=11)
plt.tight_layout()
plt.show()
