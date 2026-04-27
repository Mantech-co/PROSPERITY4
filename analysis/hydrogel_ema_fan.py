import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))
N_EMAS = 50
EMA_MIN = 5
EMA_MAX = 600


def load() -> np.ndarray:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == "HYDROGEL_PACK"])
    data = pd.concat(dfs, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
    return data["mid_price"].values.astype(np.float64)


def compute_ema(prices: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def count_long_crossovers(ema_matrix: np.ndarray) -> np.ndarray:
    """
    ema_matrix: (K, n) — rows are individual EMAs ordered by increasing span.
    For every pair (i, j) where i < j (faster < slower span), a long crossover
    at tick t means ema[i,t-1] < ema[j,t-1]  and  ema[i,t] >= ema[j,t].
    Returns array of length n with crossover counts per tick.
    """
    K, n = ema_matrix.shape
    # sign matrix: +1 where faster > slower, -1 otherwise  shape (pairs, n)
    # build via broadcasting upper-triangle pairs
    above = np.zeros((K, K, n), dtype=np.int8)
    for i in range(K):
        for j in range(i + 1, K):
            above[i, j] = (ema_matrix[i] >= ema_matrix[j]).astype(np.int8)

    # long crossover: was below (above==0) at t-1, now above (above==1) at t
    prev = above[:, :, :-1]   # (K, K, n-1)
    curr = above[:, :, 1:]    # (K, K, n-1)
    crossovers = ((curr == 1) & (prev == 0))  # (K, K, n-1)

    counts = np.zeros(n, dtype=np.int32)
    counts[1:] = crossovers.sum(axis=(0, 1))
    return counts


def _lis(seq: np.ndarray) -> int:
    tails = []
    for x in seq:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def alignment_scores(ema_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    At each tick, EMAs are ordered by increasing span.
    long_score  = LIS length  (how many are stacked fast→slow ascending)
    short_score = LDS length  (how many are stacked fast→slow descending)
    """
    K, n = ema_matrix.shape
    long_score  = np.empty(n, dtype=np.int32)
    short_score = np.empty(n, dtype=np.int32)
    for t in range(n):
        vals = ema_matrix[:, t]
        long_score[t]  = _lis(vals)
        short_score[t] = _lis(-vals)
    return long_score, short_score


def main():
    prices = load()
    n = len(prices)
    spans = np.unique(np.linspace(EMA_MIN, EMA_MAX, N_EMAS).astype(int))
    K = len(spans)
    print(f"{K} EMAs, spans {spans[0]}..{spans[-1]}, {n} ticks")

    ema_matrix = np.stack([compute_ema(prices, s) for s in spans])  # (K, n)

    print(f"Computing alignment scores (LIS/LDS) across {n} ticks...")
    long_score, short_score = alignment_scores(ema_matrix)

    BURN = 500
    prices      = prices[BURN:]
    ema_matrix  = ema_matrix[:, BURN:]
    long_score  = long_score[BURN:]
    short_score = short_score[BURN:]
    n = len(prices)
    print(f"Max long alignment: {long_score.max()}  Max short alignment: {short_score.max()}")

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 1, K))

    fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(prices, color="white", linewidth=0.8, alpha=0.9, label="mid price", zorder=K + 1)
    for color, row in zip(colors, ema_matrix):
        ax1.plot(row, color=color, linewidth=0.5, alpha=0.55)

    ax1.set_facecolor("#0d0d0d")
    ax1.set_title(f"HYDROGEL_PACK — {K} EMAs (spans {spans[0]}–{spans[-1]})", color="white", fontsize=12)
    ax1.set_ylabel("price", color="white")
    ax1.tick_params(colors="white", labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.legend(fontsize=8, facecolor="#222", labelcolor="white")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor("#0d0d0d")
    t = np.arange(n)
    ax2.fill_between(t,  long_score,  color="#00e5ff", alpha=0.5, linewidth=0, label="long LIS")
    ax2.fill_between(t, -short_score, color="#ff4d6d", alpha=0.5, linewidth=0, label="short LDS")
    ax2.plot(t,  long_score,  color="#00e5ff", linewidth=0.6, alpha=0.9)
    ax2.plot(t, -short_score, color="#ff4d6d", linewidth=0.6, alpha=0.9)
    ax2.axhline(0, color="#555", linewidth=0.5)
    ax2.set_ylabel("alignment\n(LIS/LDS)", color="white", fontsize=9)
    ax2.set_xlabel("tick", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="upper left")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=spans[0], vmax=spans[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.01)
    cbar.set_label("EMA span", fontsize=10, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    out = Path(__file__).parent / "hydrogel_ema_fan.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
