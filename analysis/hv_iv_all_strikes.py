import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

iv_dir = Path("data/iv")
files = sorted(iv_dir.glob("iv_underlying_mid_*.csv"))
steps_per_year = 10_000 * 252

fig, ax = plt.subplots(figsize=(16, 7))
colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

for f, color in zip(files, colors):
    strike = f.stem.split("_")[-1]
    df = pd.read_csv(f).sort_values("global_ts").reset_index(drop=True)
    df["mid_iv"] = (df["ask_iv"] + df["bid_iv"]) / 2
    df["log_ret"] = np.log(df["S_mid"]).diff()
    df["hv"] = df["log_ret"].rolling(13300).std() * np.sqrt(steps_per_year)

    ax.plot(df["global_ts"], df["mid_iv"], lw=0.7, color=color, label=f"IV K={strike}")
    # ax.plot(df["global_ts"], df["hv"], lw=0.9, color=color, ls="--", alpha=0.6)

# dummy lines for legend
ax.plot([], [], "k-", lw=1.5, label="mid IV (solid)")
# ax.plot([], [], "k--", lw=1.5, alpha=0.6, label="HV n=13300 (dashed)")

ax.set_xlabel("global timestamp")
ax.set_ylabel("volatility (annualised)")
ax.set_title("HV (window=13300) vs mid-IV — all strikes")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/hv_iv_all_strikes.png", dpi=150)
plt.close()
print("saved analysis/hv_iv_all_strikes.png")
