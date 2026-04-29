"""
Manual Gamble Analysis R3→R4
Q: How many people gambled more on manual and got rank boost from it?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

CB_UID  = "0b4fea6a-6afe-4f2c-b414-7e9a8a804555"
CB_NAME = "CarbonBlack"
COLS = ("pos", "pos_change", "uid", "name", "country", "score")

BLUE = "#4878CF"; ORG = "#E8762B"; GRN = "#59A14F"; RED = "#E15759"; PUR = "#B07AA1"

def load(path):
    df = pd.read_csv(path, header=None, names=COLS)
    return df.drop_duplicates(subset="uid", keep="first")

r3_overall = load(os.path.join(BASE, "../round 3/leaderboard_data_round3.csv"))
r3_algo    = load(os.path.join(BASE, "../round 3/leaderboard_data_round3_algo.csv"))
r3_manual  = load(os.path.join(BASE, "../round 3/leaderboard_data_round3_manual.csv"))
r4_overall = load(os.path.join(BASE, "leaderboard_data_round4_overall.csv"))
r4_algo    = load(os.path.join(BASE, "leaderboard_data_round4_algo.csv"))
r4_manual  = load(os.path.join(BASE, "leaderboard_data_round4_manual.csv"))

df = r4_overall[["uid","name","country","pos","score"]].rename(
    columns={"pos":"r4_pos","score":"r4_score"})
df = df.merge(r3_overall[["uid","pos","score"]].rename(
    columns={"pos":"r3_pos","score":"r3_score"}), on="uid", how="left")
df = df.merge(r4_manual[["uid","pos","score"]].rename(
    columns={"pos":"r4_manual_pos","score":"r4_manual"}), on="uid", how="left")
df = df.merge(r4_algo[["uid","pos","score"]].rename(
    columns={"pos":"r4_algo_pos","score":"r4_algo"}), on="uid", how="left")
df = df.merge(r3_manual[["uid","pos","score"]].rename(
    columns={"pos":"r3_manual_pos","score":"r3_manual"}), on="uid", how="left")
df = df.merge(r3_algo[["uid","pos","score"]].rename(
    columns={"pos":"r3_algo_pos","score":"r3_algo"}), on="uid", how="left")

# positive rank_delta = improved (lower rank number = better)
df["rank_delta"]       = df["r3_pos"].astype(float) - df["r4_pos"].astype(float)
df["manual_delta"]     = df["r3_manual_pos"].astype(float) - df["r4_manual_pos"].astype(float)
df["algo_delta"]       = df["r3_algo_pos"].astype(float) - df["r4_algo_pos"].astype(float)
df["manual_share_r4"]  = df["r4_manual"] / (df["r4_manual"] + df["r4_algo"])

m_valid = df["r4_manual"].dropna()
p50  = m_valid.median()
p75  = np.percentile(m_valid, 75)
p90  = np.percentile(m_valid, 90)
p95  = np.percentile(m_valid, 95)

# Classify gamblers
df["gambler_p75"] = df["r4_manual"] >= p75   # top 25% manual
df["gambler_p90"] = df["r4_manual"] >= p90   # top 10% manual
df["gambler_p95"] = df["r4_manual"] >= p95   # top 5%  manual

# "Manual-driven improver": improved overall rank AND had outlier manual
df["improved"]       = df["rank_delta"] > 0
df["manual_outlier"] = df["gambler_p75"]      # our primary threshold

cb = df[df["uid"] == CB_UID].iloc[0] if CB_UID in df["uid"].values else None

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")

# ── helpers ──────────────────────────────────────────────────────────────────
def add_cb(ax, x_val, y_val, label=CB_NAME, color=RED, size=90, zorder=6):
    if cb is not None:
        ax.scatter(x_val, y_val, color=color, s=size, zorder=zorder,
                   marker="*", edgecolors="black", linewidths=0.5, label=label)

# ════════════════════════════════════════════════════════════════════════════
# Plot A — Manual score distribution & gamble spectrum
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("R4 Manual Score — Who Gambled?", fontsize=15, fontweight="bold")

# A1: Full histogram with percentile bands
ax = axes[0]
ms = df["r4_manual"].dropna()
ax.hist(ms, bins=120, color=BLUE, alpha=0.6, edgecolor="none", label="All teams")
ax.axvline(p50, color="gray",  linestyle="--", lw=1.5, label=f"Median {p50:.0f}")
ax.axvline(p75, color=ORG,    linestyle="--", lw=1.5, label=f"p75  {p75:.0f}")
ax.axvline(p90, color=PUR,    linestyle="--", lw=1.5, label=f"p90  {p90:.0f}")
ax.axvline(p95, color=RED,    linestyle="--", lw=1.5, label=f"p95  {p95:.0f}")
if cb is not None:
    ax.axvline(cb["r4_manual"], color="black", lw=2.5,
               label=f"{CB_NAME}  {cb['r4_manual']:.0f}")
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Count")
ax.set_title("Manual Score Distribution")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

# A2: CDF
ax = axes[1]
ms_s = np.sort(ms)
cdf  = np.arange(1, len(ms_s)+1) / len(ms_s)
ax.plot(ms_s, cdf, color=BLUE, lw=2)
for pct, val, col in [(75, p75, ORG), (90, p90, PUR), (95, p95, RED)]:
    ax.axvline(val, color=col, lw=1.5, linestyle="--")
    ax.axhline(pct/100, color=col, lw=0.8, linestyle=":")
    ax.annotate(f"p{pct}={val:.0f}", xy=(val, pct/100), xytext=(5,4),
                textcoords="offset points", fontsize=8, color=col)
if cb is not None:
    cb_pct = (ms < cb["r4_manual"]).mean() * 100
    ax.axvline(cb["r4_manual"], color="black", lw=2.5,
               label=f"{CB_NAME}: top {100-cb_pct:.1f}% (p{cb_pct:.0f})")
    ax.legend(fontsize=9)
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("CDF")
ax.set_title("CDF — Manual Score")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax.spines[["top","right"]].set_visible(False)

# A3: Gambler count at each threshold
ax = axes[2]
thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
counts = [(df["r4_manual"] >= np.percentile(m_valid, t)).sum() for t in thresholds]
improved_counts = [
    ((df["r4_manual"] >= np.percentile(m_valid, t)) & (df["improved"])).sum()
    for t in thresholds
]
x = np.arange(len(thresholds))
ax.bar(x, counts, color=BLUE, alpha=0.6, label="Total gamblers")
ax.bar(x, improved_counts, color=GRN, alpha=0.85, label="Gamblers who improved rank")
ax.set_xticks(x)
ax.set_xticklabels([f"Top {100-t}%\n(p{t}+)" for t in thresholds], fontsize=8)
ax.set_xlabel("Manual Score Threshold"); ax.set_ylabel("# Teams")
ax.set_title("Gamblers vs Rank-Improvers by Threshold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "gamble_A_distribution.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot B — Manual score vs Rank Delta scatter (the key plot)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("R4 Manual Score vs Overall Rank Change (R3→R4)", fontsize=15, fontweight="bold")

valid = df[["r4_manual","rank_delta"]].dropna()

# B1: Full scatter
ax = axes[0]
not_gamble = valid[valid["r4_manual"] < p75]
gamble     = valid[valid["r4_manual"] >= p75]
ax.scatter(not_gamble["r4_manual"], not_gamble["rank_delta"],
           alpha=0.25, s=6, color=BLUE, label=f"Below p75 (n={len(not_gamble)})")
ax.scatter(gamble["r4_manual"], gamble["rank_delta"],
           alpha=0.45, s=10, color=ORG, label=f"Gamblers p75+ (n={len(gamble)})")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(p75, color=ORG, lw=1.2, linestyle="--", alpha=0.7)
ax.axvline(p90, color=PUR, lw=1.2, linestyle="--", alpha=0.7, label=f"p90={p90:.0f}")
if cb is not None:
    add_cb(ax, cb["r4_manual"], cb["rank_delta"])
    ax.legend(fontsize=9)
corr = valid["r4_manual"].corr(valid["rank_delta"])
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Rank Delta (positive=improved)")
ax.set_title(f"Manual Score vs Rank Delta (r={corr:.3f})")
ax.spines[["top","right"]].set_visible(False)

# B2: Zoomed — top 500 only
ax = axes[1]
top500 = df[df["r4_pos"] <= 500][["r4_manual","rank_delta","name","uid"]].dropna(subset=["r4_manual","rank_delta"])
ng_t = top500[top500["r4_manual"] < p75]
g_t  = top500[top500["r4_manual"] >= p75]
ax.scatter(ng_t["r4_manual"], ng_t["rank_delta"], alpha=0.4, s=15, color=BLUE, label=f"<p75 (n={len(ng_t)})")
ax.scatter(g_t["r4_manual"],  g_t["rank_delta"],  alpha=0.6, s=20, color=ORG,  label=f"p75+ (n={len(g_t)})")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(p75, color=ORG, lw=1.2, linestyle="--", alpha=0.7, label=f"p75={p75:.0f}")
# annotate top manual gamblers in top500
top_man_t = g_t.nlargest(10, "r4_manual")
for _, row in top_man_t.iterrows():
    ax.annotate(str(row["name"])[:14], (row["r4_manual"], row["rank_delta"]),
                fontsize=6, xytext=(4,2), textcoords="offset points", color="darkorange")
if cb is not None and cb["r4_pos"] <= 500:
    add_cb(ax, cb["r4_manual"], cb["rank_delta"])
ax.legend(fontsize=9)
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Rank Delta (positive=improved)")
ax.set_title("Manual vs Rank Delta — Top 500 Teams")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "gamble_B_manual_vs_rankdelta.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot C — Quadrant analysis: gamble × rank outcome
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Gamble Outcome Quadrant Analysis", fontsize=15, fontweight="bold")

for ax, label, data in [
    (axes[0], "All Teams", df.dropna(subset=["r4_manual","rank_delta"])),
    (axes[1], "Top 500 (R4)", df[df["r4_pos"]<=500].dropna(subset=["r4_manual","rank_delta"])),
]:
    hi_manual = data["r4_manual"] >= p75
    improved  = data["rank_delta"] > 0

    # Q1: high manual + improved  (lucky gamblers)
    # Q2: low manual  + improved  (algo-driven improvers)
    # Q3: high manual + dropped   (unlucky gamblers)
    # Q4: low manual  + dropped
    q1 = (hi_manual &  improved).sum()
    q2 = (~hi_manual & improved).sum()
    q3 = (hi_manual & ~improved).sum()
    q4 = (~hi_manual & ~improved).sum()
    total = len(data)

    ax.scatter(data[hi_manual  & improved]["r4_manual"],
               data[hi_manual  & improved]["rank_delta"],
               alpha=0.4, s=8, color=GRN,
               label=f"High manual + Improved (Q1): {q1} ({q1/total*100:.1f}%)")
    ax.scatter(data[~hi_manual & improved]["r4_manual"],
               data[~hi_manual & improved]["rank_delta"],
               alpha=0.3, s=6, color=BLUE,
               label=f"Low manual + Improved (Q2): {q2} ({q2/total*100:.1f}%)")
    ax.scatter(data[hi_manual  & ~improved]["r4_manual"],
               data[hi_manual  & ~improved]["rank_delta"],
               alpha=0.4, s=8, color=ORG,
               label=f"High manual + Dropped (Q3): {q3} ({q3/total*100:.1f}%)")
    ax.scatter(data[~hi_manual & ~improved]["r4_manual"],
               data[~hi_manual & ~improved]["rank_delta"],
               alpha=0.2, s=5, color=RED,
               label=f"Low manual + Dropped (Q4): {q4} ({q4/total*100:.1f}%)")

    ax.axhline(0, color="black", lw=1)
    ax.axvline(p75, color="black", lw=1, linestyle="--")
    ax.text(p75+500, data["rank_delta"].max()*0.9, "p75 threshold", fontsize=8, color="gray")
    ax.text(data["r4_manual"].min(), data["rank_delta"].max()*0.9,
            "Q2\nAlgo improvers", fontsize=9, color=BLUE, ha="left")
    ax.text(data["r4_manual"].max()*0.7, data["rank_delta"].max()*0.9,
            "Q1\nLucky gamblers", fontsize=9, color=GRN, ha="left")
    ax.text(data["r4_manual"].max()*0.7, data["rank_delta"].min()*0.9,
            "Q3\nGamblers who\nlost rank", fontsize=9, color=ORG, ha="left")

    if cb is not None:
        add_cb(ax, cb["r4_manual"], cb["rank_delta"])

    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Rank Delta")
    ax.set_title(f"Quadrant: {label}")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "gamble_C_quadrant.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot D — Top manual gamblers who improved rank most
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Top Gamblers: Rank Boosted by Manual in R4", fontsize=15, fontweight="bold")

# D1: top 30 by manual score who improved rank (top 500)
ax = axes[0]
manual_improvers = df[(df["r4_manual"] >= p75) & (df["improved"])].copy()
manual_improvers = manual_improvers.dropna(subset=["r4_manual","r4_algo","rank_delta"])
top30_gambler = manual_improvers.nlargest(30, "r4_manual")[
    ["name","r4_manual","r4_algo","rank_delta"]].sort_values("r4_manual")

colors_bar = [RED if str(r["name"]) == CB_NAME else GRN for _, r in top30_gambler.iterrows()]
ax.barh(top30_gambler["name"].astype(str), top30_gambler["r4_manual"],
        color=colors_bar, alpha=0.85, label="Manual Score")
ax2 = ax.twiny()
ax2.barh(top30_gambler["name"].astype(str), top30_gambler["rank_delta"],
         color="gray", alpha=0.3, label="Rank Delta")
ax.set_xlabel("R4 Manual Score", color=GRN)
ax2.set_xlabel("Rank Improvement →", color="gray")
ax.set_title("Top 30 Gamblers (p75+) who Improved Rank\n(sorted by manual score)")
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top","right"]].set_visible(False)

# D2: manual score vs rank delta — with manual score as bubble size
ax = axes[1]
plot_df = df.dropna(subset=["r4_manual","rank_delta","r4_algo"]).copy()
plot_df["manual_pct"] = plot_df["r4_manual"].rank(pct=True)

# size = manual score magnitude (normalized)
s_norm = (plot_df["r4_manual"] - plot_df["r4_manual"].min()) / (
          plot_df["r4_manual"].max() - plot_df["r4_manual"].min())
sizes = 5 + s_norm * 60

not_gamble_m = plot_df[plot_df["r4_manual"] < p75]
gamble_m     = plot_df[plot_df["r4_manual"] >= p75]

ax.scatter(not_gamble_m["r4_algo"], not_gamble_m["rank_delta"],
           s=5, alpha=0.2, color=BLUE, label="Conservative (<p75 manual)")
ax.scatter(gamble_m["r4_algo"], gamble_m["rank_delta"],
           s=gamble_m["r4_manual"]/gamble_m["r4_manual"].max()*60 + 8,
           alpha=0.5, color=ORG,
           label=f"Gamblers (p75+ manual, n={len(gamble_m)})\n[bubble size ∝ manual score]")
ax.axhline(0, color="black", lw=0.8)
if cb is not None:
    add_cb(ax, cb["r4_algo"], cb["rank_delta"])
    ax.legend(fontsize=9)
ax.set_xlabel("R4 Algo Score"); ax.set_ylabel("Rank Delta")
ax.set_title("Algo Score vs Rank Delta\n(gamblers shown larger by manual score)")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "gamble_D_top_gamblers.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot E — Summary stats: gamble payoff analysis
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Gamble Payoff Summary R4 — Manual Score Driven Rank Shifts", fontsize=14, fontweight="bold")

# E1: Box plot — rank delta by manual score quartile
ax = axes[0,0]
df["manual_quartile"] = pd.qcut(df["r4_manual"].dropna(), 4,
                                 labels=["Q1\n(Conservative)","Q2","Q3","Q4\n(Gamblers)"])
box_data = [df[df["manual_quartile"]==q]["rank_delta"].dropna().values
            for q in ["Q1\n(Conservative)","Q2","Q3","Q4\n(Gamblers)"]]
bp = ax.boxplot(box_data, labels=["Q1\n(Conservative)","Q2","Q3","Q4\n(Gamblers)"],
                patch_artist=True, medianprops=dict(color="black", lw=2), showfliers=False)
colors_bp = [BLUE, "#87CEEB", ORG, RED]
for patch, col in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(col); patch.set_alpha(0.7)
if cb is not None:
    q_cb = df[df["uid"]==CB_UID]["manual_quartile"].values
    if len(q_cb):
        q_idx = ["Q1\n(Conservative)","Q2","Q3","Q4\n(Gamblers)"].index(str(q_cb[0])) + 1
        ax.scatter([q_idx], [cb["rank_delta"]], color=RED, s=100, zorder=5, marker="*",
                   edgecolors="black", lw=0.5, label=CB_NAME)
        ax.legend(fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Manual Score Quartile"); ax.set_ylabel("Rank Delta")
ax.set_title("Rank Delta by Manual Score Quartile")
ax.spines[["top","right"]].set_visible(False)

# E2: % improved by manual quartile
ax = axes[0,1]
q_stats = df.groupby("manual_quartile", observed=True).agg(
    pct_improved=("improved", "mean"),
    count=("improved", "count"),
    median_delta=("rank_delta", "median")
).reset_index()
bar_colors_q = [BLUE, "#87CEEB", ORG, RED]
bars = ax.bar(q_stats["manual_quartile"].astype(str), q_stats["pct_improved"]*100,
              color=bar_colors_q, alpha=0.85)
for bar, row in zip(bars, q_stats.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
            f'{row.pct_improved*100:.1f}%\n(n={row.count})',
            ha='center', fontsize=8)
ax.set_xlabel("Manual Score Quartile"); ax.set_ylabel("% Teams Improved Rank")
ax.set_title("% Rank Improvers by Manual Quartile")
ax.spines[["top","right"]].set_visible(False)

# E3: manual score contribution to overall rank improvement (scatter)
ax = axes[1,0]
plot_d = df.dropna(subset=["rank_delta","manual_delta","r4_manual"])
hi = plot_d["r4_manual"] >= p75
ax.scatter(plot_d[~hi]["manual_delta"], plot_d[~hi]["rank_delta"],
           alpha=0.2, s=5, color=BLUE, label="Conservative")
ax.scatter(plot_d[hi]["manual_delta"], plot_d[hi]["rank_delta"],
           alpha=0.4, s=10, color=ORG, label="Gamblers (p75+)")
ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
if cb is not None:
    add_cb(ax, cb["manual_delta"], cb["rank_delta"])
    ax.legend(fontsize=9)
corr2 = plot_d["manual_delta"].corr(plot_d["rank_delta"])
ax.set_xlabel("Manual Rank Delta (positive=improved manual rank)")
ax.set_ylabel("Overall Rank Delta")
ax.set_title(f"Manual Rank Improvement → Overall Rank (r={corr2:.3f})")
ax.spines[["top","right"]].set_visible(False)

# E4: How many in each band had outlier manual?
ax = axes[1,1]
bands = [(1,100),(101,250),(251,500),(501,1000),(1001,2000),(2001,5000)]
band_labels_e = ["1-100","101-250","251-500","501-1k","1k-2k","2k-5k"]
gambler_counts_e, total_counts_e, pct_gamblers_e = [], [], []
for lo, hi_b in bands:
    band_df = df[(df["r4_pos"] >= lo) & (df["r4_pos"] <= hi_b)]
    g = (band_df["r4_manual"] >= p75).sum()
    t = len(band_df)
    gambler_counts_e.append(g)
    total_counts_e.append(t)
    pct_gamblers_e.append(g/t*100 if t > 0 else 0)

x = np.arange(len(band_labels_e))
ax.bar(x, pct_gamblers_e, color=ORG, alpha=0.85)
ax.axhline(25, color="gray", lw=1.5, linestyle="--", label="Expected 25% (if random)")
for i, (pct, g, t) in enumerate(zip(pct_gamblers_e, gambler_counts_e, total_counts_e)):
    ax.text(i, pct+0.5, f"{pct:.0f}%\n({g}/{t})", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(band_labels_e)
ax.set_xlabel("R4 Rank Band"); ax.set_ylabel("% with Outlier Manual (p75+)")
ax.set_title("% Gamblers in Each R4 Rank Band")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "gamble_E_summary.png")

# ════════════════════════════════════════════════════════════════════════════
# Print key numbers
# ════════════════════════════════════════════════════════════════════════════
print("\n====== GAMBLE ANALYSIS SUMMARY ======")
has_both = df.dropna(subset=["r4_manual","rank_delta"])
n = len(has_both)
print(f"Total teams (with manual+rank data): {n}")
print(f"\nManual score percentiles:")
for pct in [25,50,75,90,95]:
    print(f"  p{pct}: {np.percentile(m_valid, pct):.0f}")

print(f"\nGamblers (p75+ manual = {p75:.0f}+):")
gamblers = has_both[has_both["r4_manual"] >= p75]
print(f"  Total gamblers: {len(gamblers)} ({len(gamblers)/n*100:.1f}%)")
print(f"  Gamblers who improved rank: {gamblers['improved'].sum()} ({gamblers['improved'].mean()*100:.1f}%)")
non_gamblers = has_both[has_both["r4_manual"] < p75]
print(f"\nNon-gamblers (below p75):")
print(f"  Total: {len(non_gamblers)} ({len(non_gamblers)/n*100:.1f}%)")
print(f"  Non-gamblers who improved rank: {non_gamblers['improved'].sum()} ({non_gamblers['improved'].mean()*100:.1f}%)")

print(f"\nTop-500 breakdown:")
t500 = df[df["r4_pos"] <= 500].dropna(subset=["r4_manual"])
g500 = (t500["r4_manual"] >= p75).sum()
print(f"  Gamblers in top 500: {g500}/{len(t500)} ({g500/len(t500)*100:.1f}%)")

if cb is not None:
    cb_pct_manual = (m_valid < cb["r4_manual"]).mean() * 100
    print(f"\nCarbonBlack:")
    print(f"  R4 manual score: {cb['r4_manual']:.0f} (p{cb_pct_manual:.0f})")
    print(f"  R4 algo score:   {cb['r4_algo']:.0f}")
    print(f"  R3→R4 rank:      {int(cb['r3_pos'])} → {int(cb['r4_pos'])} (Δ{cb['rank_delta']:+.0f})")
    print(f"  Manual rank:     {int(cb['r4_manual_pos'])}")
    print(f"  Algo rank:       {int(cb['r4_algo_pos'])}")

print("\nAll plots saved to", OUT)
