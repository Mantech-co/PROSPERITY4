import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

CB_UID  = "0b4fea6a-6afe-4f2c-b414-7e9a8a804555"
CB_NAME = "CarbonBlack"

def load_lb(path, cols=("pos","pos_change","uid","name","country","score")):
    df = pd.read_csv(path, header=None, names=cols)
    df = df.drop_duplicates(subset="uid", keep="first")
    return df

r3_overall = load_lb(os.path.join(BASE, "../round 3/leaderboard_data_round3.csv"))
r3_algo    = load_lb(os.path.join(BASE, "../round 3/leaderboard_data_round3_algo.csv"))
r3_manual  = load_lb(os.path.join(BASE, "../round 3/leaderboard_data_round3_manual.csv"))

r4_overall = load_lb(os.path.join(BASE, "leaderboard_data_round4_overall.csv"))
r4_algo    = load_lb(os.path.join(BASE, "leaderboard_data_round4_algo.csv"))
r4_manual  = load_lb(os.path.join(BASE, "leaderboard_data_round4_manual.csv"))

# Build merged df: inner join on r4 participants
df = r4_overall[["uid","name","country","pos","score"]].rename(
    columns={"pos":"r4_overall_pos","score":"r4_overall_score"})

df = df.merge(
    r3_overall[["uid","pos","score"]].rename(columns={"pos":"r3_overall_pos","score":"r3_overall_score"}),
    on="uid", how="left")
df = df.merge(
    r4_algo[["uid","pos","score"]].rename(columns={"pos":"r4_algo_pos","score":"r4_algo_score"}),
    on="uid", how="left")
df = df.merge(
    r4_manual[["uid","pos","score"]].rename(columns={"pos":"r4_manual_pos","score":"r4_manual_score"}),
    on="uid", how="left")
df = df.merge(
    r3_algo[["uid","pos","score"]].rename(columns={"pos":"r3_algo_pos","score":"r3_algo_score"}),
    on="uid", how="left")
df = df.merge(
    r3_manual[["uid","pos","score"]].rename(columns={"pos":"r3_manual_pos","score":"r3_manual_score"}),
    on="uid", how="left")

df["r4_ind_overall"] = df["r4_overall_score"]
df["r4_ind_algo"]    = df["r4_algo_score"]
df["r4_ind_manual"]  = df["r4_manual_score"]

# positive = improved rank
df["rank_delta"] = df["r3_overall_pos"].astype(float) - df["r4_overall_pos"].astype(float)
df["improved"]   = df["rank_delta"] > 0
df["dropped"]    = df["rank_delta"] < 0

cb = df[df["uid"] == CB_UID].iloc[0] if CB_UID in df["uid"].values else None

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")

BLUE = "#4878CF"
ORG  = "#E8762B"
GRN  = "#59A14F"
RED  = "#E15759"

# ════════════════════════════════════════════════════════════════════════════
# Plot 1 — Rank Delta R3→R4
# ════════════════════════════════════════════════════════════════════════════
rd = df["rank_delta"].dropna()
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Rank Delta Analysis (R3 → R4)", fontsize=16, fontweight="bold")

ax = axes[0, 0]
bins = np.arange(rd.min() - 50, rd.max() + 50, 50)
ax.hist(rd[rd < 0], bins=bins, color=RED, alpha=0.7, label="Dropped")
ax.hist(rd[rd > 0], bins=bins, color=GRN, alpha=0.7, label="Improved")
ax.hist(rd[rd == 0], bins=bins, color="gray", alpha=0.7, label="No change")
ax.axvline(0, color="black", linewidth=1)
ax.axvline(rd.mean(), color=ORG, linestyle="--", linewidth=1.5, label=f"Mean {rd.mean():.0f}")
if cb is not None:
    ax.axvline(cb["rank_delta"], color=RED, linewidth=2, label=f"{CB_NAME} {cb['rank_delta']:.0f}")
ax.set_xlabel("Rank Delta (positive = improved)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Rank Changes")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

ax = axes[0, 1]
rd_sorted = np.sort(rd)
cdf = np.arange(1, len(rd_sorted)+1) / len(rd_sorted)
neg_mask = rd_sorted < 0
pos_mask = rd_sorted >= 0
ax.fill_between(rd_sorted[neg_mask], cdf[neg_mask], alpha=0.2, color=RED)
ax.fill_between(rd_sorted[pos_mask], cdf[pos_mask], 1, alpha=0.2, color=GRN)
ax.plot(rd_sorted, cdf, color=BLUE, linewidth=2)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
pct_dropped  = (rd < 0).mean() * 100
pct_improved = (rd > 0).mean() * 100
ax.annotate(f"{pct_dropped:.1f}% dropped",  xy=(rd_sorted[0], 0.05),  fontsize=10, color=RED)
ax.annotate(f"{pct_improved:.1f}% improved", xy=(rd_sorted[-1], 0.95), fontsize=10, color=GRN, ha="right")
if cb is not None:
    ax.axvline(cb["rank_delta"], color=RED, linewidth=2, linestyle=":")
ax.set_xlabel("Rank Delta")
ax.set_ylabel("CDF")
ax.set_title("CDF of Rank Delta")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 0]
colors = [GRN if v > 0 else RED if v < 0 else "gray" for v in df["rank_delta"]]
ax.scatter(df["r3_overall_pos"], df["rank_delta"], c=colors, alpha=0.3, s=5)
if cb is not None:
    ax.scatter(cb["r3_overall_pos"], cb["rank_delta"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("R3 Starting Position")
ax.set_ylabel("Rank Delta")
ax.set_title("R3 Starting Position vs Rank Change")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 1]
df_valid = df.dropna(subset=["rank_delta","name"])
top_imp = df_valid.nlargest(20, "rank_delta")[["name","rank_delta"]]
top_drp = df_valid.nsmallest(20, "rank_delta")[["name","rank_delta"]]
combined = pd.concat([top_drp, top_imp])
combined["name"] = combined["name"].astype(str)
bar_colors = [RED if v < 0 else GRN for v in combined["rank_delta"]]
ax.barh(combined["name"], combined["rank_delta"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Rank Delta")
ax.set_title("Top 20 Most Improved & Dropped")
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot1_rank_delta.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 2 — Score Distributions
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("R4 Score Distributions", fontsize=16, fontweight="bold")

def hist_clipped(ax, data, color, title, xlabel, cb_val=None, clip_pct=(1, 99)):
    lo, hi = np.percentile(data.dropna(), clip_pct)
    clipped = data[(data >= lo) & (data <= hi)].dropna()
    ax.hist(clipped, bins=80, color=color, alpha=0.85, edgecolor="none")
    n_clip = len(data.dropna()) - len(clipped)
    ax.set_title(f"{title}" + (f"\n({n_clip} outliers clipped)" if n_clip else ""))
    if cb_val is not None:
        ax.axvline(cb_val, color=RED, linewidth=2, label=CB_NAME)
        ax.legend(fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.spines[["top","right"]].set_visible(False)

for ax, col, title, color in [
    (axes[0,0], "r4_overall_score", "R4 Overall Score",  BLUE),
    (axes[0,1], "r4_ind_manual",    "R4 Manual Score",   ORG),
    (axes[0,2], "r4_ind_algo",      "R4 Algo Score",     GRN),
]:
    cb_val = cb[col] if cb is not None else None
    hist_clipped(ax, df[col], color, title, col, cb_val)

ax = axes[1,0]
cb_val = cb["r4_ind_overall"] if cb is not None else None
hist_clipped(ax, df["r4_ind_overall"], BLUE, "R4 Round-Only Overall", "R4 Individual Overall Score", cb_val)

ax = axes[1,1]
bp_data = [df["r3_overall_score"].dropna().values, df["r4_ind_overall"].dropna().values]
bp = ax.boxplot(bp_data, labels=["R3 Overall", "R4 Individual Overall"], patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], [BLUE, GRN]):
    patch.set_facecolor(color); patch.set_alpha(0.6)
if cb is not None:
    ax.scatter([1, 2], [cb["r3_overall_score"], cb["r4_ind_overall"]], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=9)
ax.set_ylabel("Score")
ax.set_title("R3 vs R4 Individual Overall")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,2]
m = df["r4_ind_manual"].dropna()
a = df.loc[m.index, "r4_ind_algo"]
ax.scatter(m, a, alpha=0.3, s=8, color="purple")
if cb is not None:
    ax.scatter(cb["r4_ind_manual"], cb["r4_ind_algo"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=9)
corr = m.corr(a)
ax.set_xlabel("R4 Individual Manual")
ax.set_ylabel("R4 Individual Algo")
ax.set_title(f"R4 Manual vs Algo (r={corr:.3f})")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot2_score_distributions.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 3 — R3 vs R4 Rank
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("R3 Rank vs R4 Rank", fontsize=16, fontweight="bold")

r3p = df["r3_overall_pos"].astype(float)
r4p = df["r4_overall_pos"].astype(float)
delta = df["rank_delta"]

ax = axes[0]
c = [GRN if v > 0 else RED if v < 0 else "gray" for v in delta]
ax.scatter(r3p, r4p, c=c, alpha=0.3, s=5)
lim = max(r3p.max(), r4p.max())
ax.plot([0,lim],[0,lim], "k--", linewidth=0.8, label="No change")
if cb is not None:
    ax.scatter(cb["r3_overall_pos"], cb["r4_overall_pos"], color=RED, s=80, zorder=5, label=CB_NAME)
ax.legend(fontsize=8)
ax.set_xlabel("R3 Rank"); ax.set_ylabel("R4 Rank")
ax.set_title("R3 vs R4 Rank")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
norm = mcolors.TwoSlopeNorm(vmin=delta.min(), vcenter=0, vmax=delta.max())
sc = ax.scatter(r3p, r4p, c=delta, cmap="RdYlGn", norm=norm, alpha=0.4, s=5)
ax.plot([0,lim],[0,lim], "k--", linewidth=0.8)
if cb is not None:
    ax.scatter(cb["r3_overall_pos"], cb["r4_overall_pos"], color="red", s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="Rank Delta")
ax.set_xlabel("R3 Rank"); ax.set_ylabel("R4 Rank")
ax.set_title("R3 vs R4 Rank (colored by delta)")
ax.spines[["top","right"]].set_visible(False)

ax = axes[2]
top500 = df[r4p <= 500]
r3p_t = top500["r3_overall_pos"].astype(float)
r4p_t = top500["r4_overall_pos"].astype(float)
d_t   = top500["rank_delta"]
sc2 = ax.scatter(r3p_t, r4p_t, c=d_t, cmap="RdYlGn",
                 norm=mcolors.TwoSlopeNorm(vmin=d_t.min(), vcenter=0, vmax=d_t.max()),
                 alpha=0.6, s=15)
ax.plot([0,500],[0,500], "k--", linewidth=0.8)
if cb is not None and cb["r4_overall_pos"] <= 500:
    ax.scatter(cb["r3_overall_pos"], cb["r4_overall_pos"], color="red", s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
plt.colorbar(sc2, ax=ax, label="Rank Delta")
ax.set_xlabel("R3 Rank"); ax.set_ylabel("R4 Rank")
ax.set_title("R3 vs R4 Rank — Top 500")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot3_r3_vs_r4_rank.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 4 — Score Improvement
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Score Improvement R3 → R4", fontsize=16, fontweight="bold")

def clip_series(s, pct=(1, 99)):
    lo, hi = np.percentile(s.dropna(), pct)
    return s.clip(lo, hi)

scatter_pairs = [
    (axes[0,0], "r3_overall_score", "r4_ind_overall", "R3 Overall Score", "R4 Individual Overall", BLUE),
    (axes[0,1], "r3_overall_pos",   "r4_ind_overall", "R3 Rank",          "R4 Individual Overall", ORG),
    (axes[0,2], "r3_manual_score",  "r4_ind_manual",  "R3 Manual Score",  "R4 Individual Manual",  ORG),
    (axes[1,0], "r3_algo_score",    "r4_ind_algo",    "R3 Algo Score",    "R4 Individual Algo",    GRN),
]
for ax, xcol, ycol, xl, yl, color in scatter_pairs:
    x = df[xcol].astype(float)
    y = clip_series(df[ycol].astype(float))
    valid = x.notna() & y.notna()
    ax.scatter(x[valid], y[valid], alpha=0.2, s=5, color=color)
    if cb is not None:
        ax.scatter(cb[xcol], cb[ycol], color=RED, s=80, zorder=5, label=CB_NAME)
        ax.legend(fontsize=8)
    r = x[valid].corr(df[ycol].astype(float)[valid])
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{xl} vs {yl} (r={r:.3f})")
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
df["r4_decile"] = pd.qcut(df["r4_overall_pos"].astype(float), 10, labels=[f"D{i}" for i in range(1,11)])
decile_avg = df.groupby("r4_decile")[["r4_ind_manual","r4_ind_algo"]].mean()
x = np.arange(len(decile_avg))
w = 0.4
ax.bar(x - w/2, decile_avg["r4_ind_manual"], w, color=ORG, label="Manual", alpha=0.85)
ax.bar(x + w/2, decile_avg["r4_ind_algo"],   w, color=GRN, label="Algo",   alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(decile_avg.index, fontsize=8)
ax.set_xlabel("R4 Rank Decile (D1=top)"); ax.set_ylabel("Avg Score")
ax.set_title("Avg Manual vs Algo Split by Decile")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

ax = axes[1,2]
total = decile_avg["r4_ind_manual"] + decile_avg["r4_ind_algo"]
ax.plot(decile_avg.index, decile_avg["r4_ind_algo"]   / total * 100, color=GRN, marker="o", label="% Algo")
ax.plot(decile_avg.index, decile_avg["r4_ind_manual"] / total * 100, color=ORG, marker="o", label="% Manual")
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("R4 Rank Decile"); ax.set_ylabel("%")
ax.set_title("Algo vs Manual % by R4 Rank Decile")
ax.legend(fontsize=9); ax.tick_params(axis="x", labelsize=8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot4_score_improvement.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 5 — Country Analysis
# ════════════════════════════════════════════════════════════════════════════
cg = df.groupby("country")
country_stats = cg.agg(
    team_count=("uid","count"),
    median_r4_overall=("r4_overall_score","median"),
    median_rank_delta=("rank_delta","median"),
    median_r4_ind=("r4_ind_overall","median"),
).reset_index()

min_teams = 10
cs_filt = country_stats[country_stats["team_count"] >= min_teams]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Country-Level Analysis (min {min_teams} teams)", fontsize=16, fontweight="bold")

ax = axes[0,0]
cs_sorted = cs_filt.sort_values("median_r4_overall", ascending=True)
bar_colors = [RED if c == "IN" else BLUE for c in cs_sorted["country"]]
ax.barh(cs_sorted["country"], cs_sorted["median_r4_overall"], color=bar_colors, alpha=0.85)
ax.set_xlabel("Median R4 Overall Score")
ax.set_title("Median R4 Score by Country")
ax.spines[["top","right"]].set_visible(False)

ax = axes[0,1]
top_imp = cs_filt.nlargest(20, "median_rank_delta")
bar_colors2 = [RED if v < 0 else GRN for v in top_imp["median_rank_delta"]]
ax.barh(top_imp["country"], top_imp["median_rank_delta"], color=bar_colors2, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Median Rank Delta")
ax.set_title("Median Rank Improvement by Country (top 20)")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,0]
scatter_c = [RED if c == "IN" else BLUE for c in cs_filt["country"]]
ax.scatter(cs_filt["team_count"], cs_filt["median_r4_overall"], c=scatter_c, s=60, alpha=0.8)
for _, row in cs_filt.iterrows():
    ax.annotate(row["country"], (row["team_count"], row["median_r4_overall"]),
                fontsize=7, xytext=(3,3), textcoords="offset points")
ax.set_xlabel("Team Count"); ax.set_ylabel("Median R4 Score")
ax.set_title("Team Count vs Median R4 Score per Country")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
top15 = cs_filt.nlargest(15, "median_r4_ind")
bar_colors3 = [RED if c == "IN" else GRN for c in top15["country"]]
ax.bar(top15["country"], top15["median_r4_ind"], color=bar_colors3, alpha=0.85)
ax.set_xlabel("Country"); ax.set_ylabel("Median R4 Individual Overall")
ax.set_title("Best R4 Round Performance by Country (top 15)")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot5_country.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 6 — Manual/Algo Rank Analysis
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Manual Score Clustering & Rank Movements (R4)", fontsize=16, fontweight="bold")

ax = axes[0,0]
manual_scores = df["r4_ind_manual"].dropna()
top_vals = manual_scores.value_counts().nlargest(25).sort_index()
ax.bar(range(len(top_vals)), top_vals.values, color=ORG, alpha=0.85)
ax.set_xticks(range(len(top_vals)))
ax.set_xticklabels([f"{v:.0f}" for v in top_vals.index], rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Manual Score Value"); ax.set_ylabel("Count")
ax.set_title("Most Common R4 Manual Score Values (bid clusters)")
ax.spines[["top","right"]].set_visible(False)

ax = axes[0,1]
m_rank = df["r4_manual_pos"].astype(float)
a_rank = df["r4_algo_pos"].astype(float)
ax.scatter(m_rank, a_rank, alpha=0.3, s=5, color="purple")
lim = max(m_rank.max(), a_rank.max())
ax.plot([0,lim],[0,lim], "k--", linewidth=0.8, label="Equal rank")
if cb is not None:
    ax.scatter(cb["r4_manual_pos"], cb["r4_algo_pos"], color=RED, s=80, zorder=5, label=CB_NAME)
ax.legend(fontsize=8)
ax.set_xlabel("R4 Manual Rank"); ax.set_ylabel("R4 Algo Rank")
ax.set_title("Manual Rank vs Algo Rank (R4)")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,0]
r3a = df["r3_algo_pos"].astype(float)
r4a = df["r4_algo_pos"].astype(float)
d_algo = r3a - r4a
valid = r3a.notna() & r4a.notna()
norm_a = mcolors.TwoSlopeNorm(vmin=d_algo[valid].min(), vcenter=0, vmax=d_algo[valid].max())
sc = ax.scatter(r3a[valid], r4a[valid], c=d_algo[valid], cmap="RdYlGn", norm=norm_a, alpha=0.4, s=5)
ax.plot([0,r3a.max()],[0,r3a.max()], "k--", linewidth=0.8)
if cb is not None:
    ax.scatter(cb["r3_algo_pos"], cb["r4_algo_pos"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="Algo Rank Delta")
ax.set_xlabel("R3 Algo Rank"); ax.set_ylabel("R4 Algo Rank")
ax.set_title("Algo Rank: R3 vs R4")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
r3m = df["r3_manual_pos"].astype(float)
r4m = df["r4_manual_pos"].astype(float)
d_man = r3m - r4m
valid_m = r3m.notna() & r4m.notna()
norm_m = mcolors.TwoSlopeNorm(vmin=d_man[valid_m].min(), vcenter=0, vmax=d_man[valid_m].max())
sc2 = ax.scatter(r3m[valid_m], r4m[valid_m], c=d_man[valid_m], cmap="RdYlGn", norm=norm_m, alpha=0.4, s=5)
ax.plot([0,r3m.max()],[0,r3m.max()], "k--", linewidth=0.8)
if cb is not None:
    ax.scatter(cb["r3_manual_pos"], cb["r4_manual_pos"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
plt.colorbar(sc2, ax=ax, label="Manual Rank Delta")
ax.set_xlabel("R3 Manual Rank"); ax.set_ylabel("R4 Manual Rank")
ax.set_title("Manual Rank: R3 vs R4")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot6_manual_algo_ranks.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 7 — Top Performers
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Top Performers Deep Dive (R4)", fontsize=16, fontweight="bold")

ax = axes[0,0]
top30 = df.nsmallest(30, "r4_overall_pos")[["name","r4_ind_manual","r4_ind_algo"]].set_index("name")
top30 = top30.sort_values("r4_ind_algo", ascending=True)
ax.barh(top30.index, top30["r4_ind_manual"], color=ORG, alpha=0.85, label="Manual")
ax.barh(top30.index, top30["r4_ind_algo"], left=top30["r4_ind_manual"], color=GRN, alpha=0.85, label="Algo")
ax.set_xlabel("R4 Individual Score")
ax.set_title("Top 30 Teams: R4 Score Breakdown")
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

ax = axes[0,1]
top30_algo = df.nsmallest(30, "r4_algo_pos")[["name","r4_ind_algo","r4_algo_pos"]].sort_values("r4_ind_algo")
bar_c = [RED if n == CB_NAME else BLUE for n in top30_algo["name"]]
ax.barh(top30_algo["name"], top30_algo["r4_ind_algo"], color=bar_c, alpha=0.85)
ax.set_xlabel("R4 Individual Algo Score")
ax.set_title("Top 30 by R4 Individual Algo")
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,0]
top30_man = df.nsmallest(30, "r4_manual_pos")[["name","r4_ind_manual","r4_manual_pos"]].sort_values("r4_ind_manual")
bar_c2 = [RED if n == CB_NAME else ORG for n in top30_man["name"]]
ax.barh(top30_man["name"], top30_man["r4_ind_manual"], color=bar_c2, alpha=0.85)
ax.set_xlabel("R4 Individual Manual Score")
ax.set_title("Top 30 by R4 Individual Manual")
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
top200 = df.nsmallest(200, "r4_overall_pos").sort_values("r4_overall_pos")
ax.fill_between(range(len(top200)), top200["r4_overall_score"].values, alpha=0.4, color=BLUE, label="R4 Score")
ax.fill_between(range(len(top200)), top200["r3_overall_score"].fillna(0).values, alpha=0.4, color=ORG, label="R3 Score")
ax.fill_between(range(len(top200)),
                top200["r3_overall_score"].fillna(0).values,
                top200["r4_overall_score"].values,
                alpha=0.4, color=GRN, label="R4 Gain")
if cb is not None and cb["r4_overall_pos"] <= 200:
    idx = (top200["uid"] == CB_UID).values.nonzero()[0]
    if len(idx):
        ax.scatter(idx[0], cb["r4_overall_score"], color=RED, s=80, zorder=5, label=CB_NAME)
ax.set_xlabel("R4 Rank"); ax.set_ylabel("Score")
ax.set_title("Top 200: R4 vs R3 Score")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot7_top_performers.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 8 — Rank Band Migration R3→R4
# ════════════════════════════════════════════════════════════════════════════
bands = [(1,100,"Top 100"),(101,500,"101-500"),(501,1000,"501-1k"),
         (1001,2000,"1k-2k"),(2001,3000,"2k-3k"),(3001,5000,"3k-5k"),(5001,99999,"5k+")]

def get_band(pos):
    for lo, hi, label in bands:
        if lo <= pos <= hi:
            return label
    return "5k+"

df["r3_band"] = df["r3_overall_pos"].astype(float).apply(lambda x: get_band(x) if not np.isnan(x) else None)
df["r4_band"] = df["r4_overall_pos"].astype(float).apply(lambda x: get_band(x) if not np.isnan(x) else None)

band_labels = [b[2] for b in bands]
matrix = pd.DataFrame(0, index=band_labels, columns=band_labels)
for _, row in df.iterrows():
    if row["r3_band"] and row["r4_band"]:
        matrix.loc[row["r3_band"], row["r4_band"]] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Rank Band Migration (R3 → R4)", fontsize=16, fontweight="bold")

ax = axes[0]
im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(band_labels))); ax.set_xticklabels(band_labels, rotation=30, ha="right")
ax.set_yticks(range(len(band_labels))); ax.set_yticklabels(band_labels)
ax.set_xlabel("R4 Band"); ax.set_ylabel("R3 Band")
ax.set_title("Rank Band Migration Heatmap")
for i in range(len(band_labels)):
    for j in range(len(band_labels)):
        v = matrix.values[i,j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > matrix.values.max()*0.5 else "black")
plt.colorbar(im, ax=ax, label="# Teams")

ax = axes[1]
pcts = []
for band in band_labels:
    row_data = matrix.loc[band]
    total = row_data.sum()
    if total == 0:
        pcts.append((0, 0, 0)); continue
    band_idx = band_labels.index(band)
    improved = row_data.iloc[:band_idx].sum()
    stayed   = row_data.iloc[band_idx]
    dropped  = row_data.iloc[band_idx+1:].sum()
    pcts.append((improved/total*100, stayed/total*100, dropped/total*100))

x = np.arange(len(band_labels))
impr = [p[0] for p in pcts]
stay = [p[1] for p in pcts]
drop = [p[2] for p in pcts]
ax.bar(x, drop,  color=RED,  label="Dropped band",  alpha=0.85)
ax.bar(x, stay,  bottom=drop, color=BLUE, label="Stayed",       alpha=0.85)
ax.bar(x, impr,  bottom=[d+s for d,s in zip(drop,stay)], color=GRN, label="Improved band", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(band_labels, rotation=30, ha="right")
ax.set_xlabel("R3 Band"); ax.set_ylabel("%")
ax.set_title("% Improved / Stayed / Dropped by R3 Band")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot8_rank_migration.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 9 — Manual Deep Dive
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("R4 Manual Trading Deep Dive", fontsize=16, fontweight="bold")

ax = axes[0,0]
ms = df["r4_ind_manual"].dropna().sort_values()
ax.hist(ms, bins=150, color=ORG, alpha=0.85, edgecolor="none")
ax.axvline(ms.median(), color=BLUE, linestyle="--", linewidth=1.5, label=f"Median {ms.median():.0f}")
if cb is not None:
    ax.axvline(cb["r4_ind_manual"], color=RED, linewidth=2, label=f"{CB_NAME} {cb['r4_ind_manual']:.0f}")
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Count")
ax.set_title("R4 Manual Score Full Distribution")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

ax = axes[0,1]
ms_sorted = np.sort(ms)
cdf_m = np.arange(1, len(ms_sorted)+1) / len(ms_sorted)
ax.step(ms_sorted, cdf_m, where="post", color=ORG, linewidth=2)
ax.fill_between(ms_sorted, cdf_m, step="post", alpha=0.15, color=ORG)
if cb is not None:
    pct_below = (ms < cb["r4_ind_manual"]).mean() * 100
    ax.axvline(cb["r4_ind_manual"], color=RED, linewidth=2, label=f"{CB_NAME} (top {100-pct_below:.1f}%)")
    ax.legend(fontsize=9)
for pct in [0.25, 0.5, 0.75, 0.9]:
    v = np.percentile(ms_sorted, pct*100)
    ax.annotate(f"p{int(pct*100)}={v:.0f}", xy=(v, pct), xytext=(4,4),
                textcoords="offset points", fontsize=8, color="dimgray")
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("CDF")
ax.set_title("CDF of R4 Manual Scores")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,0]
ax.scatter(df["r4_manual_pos"].astype(float), df["r4_ind_manual"].astype(float),
           alpha=0.3, s=5, color=ORG)
if cb is not None:
    ax.scatter(cb["r4_manual_pos"], cb["r4_ind_manual"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
ax.set_xlabel("R4 Manual Rank"); ax.set_ylabel("R4 Manual Score")
ax.set_title("Manual Rank vs Manual Score (R4)")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
ax.scatter(df["r4_algo_pos"].astype(float), df["r4_ind_algo"].astype(float),
           alpha=0.3, s=5, color=GRN)
if cb is not None:
    ax.scatter(cb["r4_algo_pos"], cb["r4_ind_algo"], color=RED, s=80, zorder=5, label=CB_NAME)
    ax.legend(fontsize=8)
ax.set_xlabel("R4 Algo Rank"); ax.set_ylabel("R4 Algo Score")
ax.set_title("Algo Rank vs Algo Score (R4)")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot9_manual_deep_dive.png")

# ════════════════════════════════════════════════════════════════════════════
# Plot 10 — India Focus
# ════════════════════════════════════════════════════════════════════════════
india = df[df["country"] == "IN"].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("India Focus — R4", fontsize=16, fontweight="bold")

ax = axes[0,0]
ax.hist(df["r4_overall_pos"].astype(float), bins=80, color=BLUE, alpha=0.5, label="All teams", density=True)
ax.hist(india["r4_overall_pos"].astype(float), bins=40, color=ORG, alpha=0.8, label="India", density=True)
if cb is not None:
    ax.axvline(cb["r4_overall_pos"], color=RED, linewidth=2, label=f"{CB_NAME} #{int(cb['r4_overall_pos'])}")
ax.set_xlabel("R4 Overall Rank"); ax.set_ylabel("Density")
ax.set_title("R4 Overall Rank Distribution: India vs All")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

ax = axes[0,1]
top_india = india.nsmallest(25, "r4_overall_pos")[["name","r4_overall_pos","r4_ind_overall"]].sort_values("r4_ind_overall")
bar_c = [RED if n == CB_NAME else ORG for n in top_india["name"]]
ax.barh(top_india["name"], top_india["r4_ind_overall"], color=bar_c, alpha=0.85)
ax.set_xlabel("R4 Individual Overall Score")
ax.set_title("Top 25 India Teams by R4 Individual Score")
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,0]
ax.scatter(india["r3_overall_pos"].astype(float), india["r4_overall_pos"].astype(float),
           alpha=0.6, s=20, color=ORG, label="India")
lim = max(india["r3_overall_pos"].dropna().max(), india["r4_overall_pos"].max())
ax.plot([0,lim],[0,lim], "k--", linewidth=0.8)
if cb is not None:
    ax.scatter(cb["r3_overall_pos"], cb["r4_overall_pos"], color=RED, s=100, zorder=5, label=CB_NAME)
ax.legend(fontsize=9)
ax.set_xlabel("R3 Overall Rank"); ax.set_ylabel("R4 Overall Rank")
ax.set_title("India: R3 vs R4 Rank")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1,1]
ax.hist(india["rank_delta"].dropna(), bins=40, color=ORG, alpha=0.85, edgecolor="none")
ax.axvline(0, color="black", linewidth=1)
ax.axvline(india["rank_delta"].mean(), color=BLUE, linestyle="--", linewidth=1.5,
           label=f"India mean {india['rank_delta'].mean():.0f}")
ax.axvline(df["rank_delta"].mean(), color="gray", linestyle="--", linewidth=1.5,
           label=f"Global mean {df['rank_delta'].mean():.0f}")
if cb is not None:
    ax.axvline(cb["rank_delta"], color=RED, linewidth=2, label=f"{CB_NAME} {cb['rank_delta']:.0f}")
ax.set_xlabel("Rank Delta"); ax.set_ylabel("Count")
ax.set_title("India Rank Delta Distribution (R3→R4)")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "plot10_india_focus.png")

print("\nAll plots saved to", OUT)
if cb is not None:
    print(f"\n{CB_NAME} summary:")
    print(f"  R3 rank: {int(cb['r3_overall_pos'])} → R4 rank: {int(cb['r4_overall_pos'])} (Δ {cb['rank_delta']:+.0f})")
    print(f"  R4 overall score: {cb['r4_overall_score']:.2f}")
    print(f"  R4 individual: overall={cb['r4_ind_overall']:.2f}, algo={cb['r4_ind_algo']:.2f}, manual={cb['r4_ind_manual']:.2f}")
    print(f"  R4 algo rank: {int(cb['r4_algo_pos'])}, manual rank: {int(cb['r4_manual_pos'])}")
