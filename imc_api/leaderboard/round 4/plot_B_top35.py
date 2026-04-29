import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "plots")

CB_UID  = "0b4fea6a-6afe-4f2c-b414-7e9a8a804555"
CB_NAME = "CarbonBlack"
COLS = ("pos", "pos_change", "uid", "name", "country", "score")
BLUE = "#4878CF"; ORG = "#E8762B"; RED = "#E15759"

def load(path):
    df = pd.read_csv(path, header=None, names=COLS)
    return df.drop_duplicates(subset="uid", keep="first")

r3_overall = load(os.path.join(BASE, "../round 3/leaderboard_data_round3.csv"))
r3_manual  = load(os.path.join(BASE, "../round 3/leaderboard_data_round3_manual.csv"))
r3_algo    = load(os.path.join(BASE, "../round 3/leaderboard_data_round3_algo.csv"))
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

df["rank_delta"] = df["r3_pos"].astype(float) - df["r4_pos"].astype(float)

m_valid = df["r4_manual"].dropna()
p75 = np.percentile(m_valid, 75)
p90 = np.percentile(m_valid, 90)

cb = df[df["uid"] == CB_UID].iloc[0] if CB_UID in df["uid"].values else None

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("R4 Manual Score vs Overall Rank Change (R3→R4)", fontsize=15, fontweight="bold")

valid = df[["r4_manual","rank_delta"]].dropna()

# B1: Full scatter (unchanged)
ax = axes[0]
not_gamble = valid[valid["r4_manual"] < p75]
gamble     = valid[valid["r4_manual"] >= p75]
ax.scatter(not_gamble["r4_manual"], not_gamble["rank_delta"],
           alpha=0.25, s=6, color=BLUE, label=f"Below p75 (n={len(not_gamble)})")
ax.scatter(gamble["r4_manual"], gamble["rank_delta"],
           alpha=0.45, s=10, color=ORG, label=f"Gamblers p75+ (n={len(gamble)})")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(p75, color=ORG, lw=1.2, linestyle="--", alpha=0.7)
ax.axvline(p90, color="#B07AA1", lw=1.2, linestyle="--", alpha=0.7, label=f"p90={p90:.0f}")
if cb is not None:
    ax.scatter(cb["r4_manual"], cb["rank_delta"], color=RED, s=90, zorder=6,
               marker="*", edgecolors="black", linewidths=0.5, label=CB_NAME)
    ax.legend(fontsize=9)
corr = valid["r4_manual"].corr(valid["rank_delta"])
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Rank Delta (positive=improved)")
ax.set_title(f"Manual Score vs Rank Delta (r={corr:.3f})")
ax.spines[["top","right"]].set_visible(False)

# B2: Top 35
ax = axes[1]
top35 = df[df["r4_pos"] <= 35][["r4_manual","rank_delta","name","uid","r4_pos"]].dropna(subset=["r4_manual","rank_delta"])
ng_t = top35[top35["r4_manual"] < p75]
g_t  = top35[top35["r4_manual"] >= p75]
ax.scatter(ng_t["r4_manual"], ng_t["rank_delta"], alpha=0.6, s=40, color=BLUE, label=f"<p75 (n={len(ng_t)})")
ax.scatter(g_t["r4_manual"],  g_t["rank_delta"],  alpha=0.8, s=50, color=ORG,  label=f"p75+ (n={len(g_t)})")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(p75, color=ORG, lw=1.2, linestyle="--", alpha=0.7, label=f"p75={p75:.0f}")

# annotate all teams
for _, row in top35.iterrows():
    is_cb = row["uid"] == CB_UID
    color = RED if is_cb else ("darkorange" if row["r4_manual"] >= p75 else "steelblue")
    ax.annotate(f"{int(row['r4_pos'])}. {str(row['name'])[:14]}",
                (row["r4_manual"], row["rank_delta"]),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points", color=color)

if cb is not None and cb["r4_pos"] <= 35:
    ax.scatter(cb["r4_manual"], cb["rank_delta"], color=RED, s=100, zorder=6,
               marker="*", edgecolors="black", linewidths=0.5, label=CB_NAME)

ax.legend(fontsize=9)
ax.set_xlabel("R4 Manual Score"); ax.set_ylabel("Rank Delta (positive=improved)")
ax.set_title("Manual vs Rank Delta — Top 35 Teams")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT, "gamble_B_top35.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")
