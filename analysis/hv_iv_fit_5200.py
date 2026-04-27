import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/iv/iv_underlying_mid_5200.csv")
df = df.sort_values("global_ts").reset_index(drop=True)

df["mid_iv"] = (df["ask_iv"] + df["bid_iv"]) / 2
df["log_ret"] = np.log(df["S_mid"]).diff()

# expanding HV from t=0: std of all log returns up to each point, annualised
steps_per_year = 10_000 * 252
df["hv"] = df["log_ret"].expanding().std() * np.sqrt(steps_per_year)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df["global_ts"], df["mid_iv"], lw=0.6, label="mid IV", alpha=0.85)
axes[0].plot(df["global_ts"], df["hv"], lw=0.8, label="HV (expanding from t=0)", alpha=0.85)
axes[0].set_xlabel("global timestamp")
axes[0].set_ylabel("volatility")
axes[0].set_title("Expanding HV vs mid-IV — strike 5200")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

mask = df["hv"].notna() & df["mid_iv"].notna()
axes[1].scatter(df.loc[mask, "hv"], df.loc[mask, "mid_iv"], s=1, alpha=0.3, color="purple")
lims = [min(df.loc[mask, "hv"].min(), df.loc[mask, "mid_iv"].min()),
        max(df.loc[mask, "hv"].max(), df.loc[mask, "mid_iv"].max())]
axes[1].plot(lims, lims, "r--", lw=1, label="y=x")
axes[1].set_xlabel("HV (expanding)")
axes[1].set_ylabel("mid IV")
axes[1].set_title("HV vs IV scatter")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/hv_iv_fit_5200.png", dpi=150)
plt.close()

final_hv = df["hv"].iloc[-1]
print(f"final expanding HV = {final_hv:.4f}")
print("saved analysis/hv_iv_fit_5200.png")
