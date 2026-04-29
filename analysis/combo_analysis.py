"""
Combo analysis: within each product group, plot sum and difference of
mid prices for all C(5,2) pairs and C(5,3) triples.
"""
import itertools
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "round5"
OUT_DIR  = Path(__file__).parent.parent / "analysis_output" / "combo_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG   = "#1a1a2e"
AXBG = "#111111"
COLORS = ["#39ff6e", "#ff3d5a", "#ff9f43", "#54a0ff", "#5f27cd",
          "#ee5a24", "#00d2d3", "#ffd32a", "#c56cf0", "#aaa69d"]

# ── load data ─────────────────────────────────────────────────────────────────
dfs = []
for f in sorted(DATA_DIR.glob("prices_round_5_day_*.csv")):
    dfs.append(pd.read_csv(f, sep=";"))
raw = pd.concat(dfs, ignore_index=True)

# global timestamp (day * offset + timestamp)
max_ts = raw["timestamp"].max() + 100
raw["gts"] = raw["day"] * max_ts + raw["timestamp"]

# pivot to wide: index=gts, columns=product
wide = raw.pivot_table(index="gts", columns="product", values="mid_price")
wide.sort_index(inplace=True)
ts = wide.index.values

# ── group products ─────────────────────────────────────────────────────────────
groups: dict[str, list[str]] = defaultdict(list)
for p in wide.columns:
    groups[p.split("_")[0]].append(p)

# short name helper
def short(name: str) -> str:
    parts = name.split("_")
    # drop group prefix (first token or first two for GALAXY_SOUNDS / UV_VISOR etc.)
    # just take last part(s) to keep labels readable
    return "_".join(parts[1:]) if len(parts) > 1 else name


# ── plot helpers ───────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(AXBG)
    ax.tick_params(colors="grey", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("grey")
    ax.yaxis.label.set_color("grey")


def plot_series(ax, x, y, color, label, lw=0.7):
    ax.plot(x, y, color=color, linewidth=lw, label=label)
    style_ax(ax)


def add_stats(ax, y):
    """Overlay mean ± 1σ bands."""
    m, s = np.nanmean(y), np.nanstd(y)
    ax.axhline(m,      color="white",  linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(m + s,  color="#aaa",   linewidth=0.4, linestyle=":",  alpha=0.5)
    ax.axhline(m - s,  color="#aaa",   linewidth=0.4, linestyle=":",  alpha=0.5)
    ax.set_ylim(m - 4*s, m + 4*s)


# ══════════════════════════════════════════════════════════════════════════════
# PAIRS
# ══════════════════════════════════════════════════════════════════════════════
for group_key, members in sorted(groups.items()):
    pairs = list(itertools.combinations(members, 2))  # C(5,2)=10
    n = len(pairs)                                     # 10

    # Layout: n rows × 2 cols  (col0=sum, col1=diff)
    fig, axes = plt.subplots(n, 2, figsize=(16, n * 2.2),
                             facecolor=BG, constrained_layout=True)
    fig.suptitle(f"{group_key}  —  pairs (C(5,2)=10)  |  col0: sum   col1: diff",
                 color="white", fontsize=11)

    for row, (a, b) in enumerate(pairs):
        sa, sb = wide[a], wide[b]
        s_sum  = sa + sb
        s_diff = sa - sb

        la, lb = short(a), short(b)
        c0, c1 = COLORS[row % len(COLORS)], COLORS[(row + 5) % len(COLORS)]

        ax_sum  = axes[row, 0]
        ax_diff = axes[row, 1]

        plot_series(ax_sum,  ts, s_sum.values,  c0, f"{la}+{lb}")
        add_stats(ax_sum, s_sum.values)
        ax_sum.set_title(f"{la} + {lb}  (sum)", fontsize=7)

        plot_series(ax_diff, ts, s_diff.values, c1, f"{la}−{lb}")
        add_stats(ax_diff, s_diff.values)
        ax_diff.set_title(f"{la} − {lb}  (diff)", fontsize=7)

    out = OUT_DIR / f"pairs_{group_key.lower()}.png"
    fig.savefig(out, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# TRIPLES
# ══════════════════════════════════════════════════════════════════════════════
for group_key, members in sorted(groups.items()):
    triples = list(itertools.combinations(members, 3))  # C(5,3)=10
    n = len(triples)

    # Layout: n rows × 4 cols  (col0=sum, col1=A+B−C, col2=A+C−B, col3=B+C−A)
    fig, axes = plt.subplots(n, 4, figsize=(22, n * 2.2),
                             facecolor=BG, constrained_layout=True)
    fig.suptitle(
        f"{group_key}  —  triples (C(5,3)=10)  |  col0:sum  col1:A+B−C  col2:A+C−B  col3:B+C−A",
        color="white", fontsize=10)

    for row, (a, b, c) in enumerate(triples):
        sa, sb, sc = wide[a], wide[b], wide[c]
        la, lb, lc = short(a), short(b), short(c)

        combos = [
            (sa + sb + sc,   f"{la}+{lb}+{lc}",   "sum"),
            (sa + sb - sc,   f"{la}+{lb}−{lc}",   "A+B−C"),
            (sa + sc - sb,   f"{la}+{lc}−{lb}",   "A+C−B"),
            (sb + sc - sa,   f"{lb}+{lc}−{la}",   "B+C−A"),
        ]

        for col, (series, label, kind) in enumerate(combos):
            ax = axes[row, col]
            color = COLORS[(row * 4 + col) % len(COLORS)]
            plot_series(ax, ts, series.values, color, label)
            add_stats(ax, series.values)
            ax.set_title(f"{kind}: {label}", fontsize=6)

    out = OUT_DIR / f"triples_{group_key.lower()}.png"
    fig.savefig(out, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")

print("done")
