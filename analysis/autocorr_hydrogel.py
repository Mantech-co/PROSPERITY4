import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))


def load_hydrogel() -> pd.DataFrame:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == "HYDROGEL_PACK"])
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["day", "timestamp"]).reset_index(drop=True)
    max_ts = data["timestamp"].max() + 100
    data["global_ts"] = data["day"] * max_ts + data["timestamp"]
    return data


def main():
    hydrogel = load_hydrogel()
    mid = np.diff(hydrogel["mid_price"].values[:1000])

    # ACF manually
    max_lag = 50
    n = len(mid)
    mean = mid.mean()
    var = ((mid - mean) ** 2).mean()
    acf = np.array([
        ((mid[:n - k] - mean) * (mid[k:] - mean)).mean() / var
        for k in range(max_lag + 1)
    ])
    conf = 1.96 / np.sqrt(n)

    # Durbin-Watson
    dw = durbin_watson(mid)

    # Ljung-Box
    lb = acorr_ljungbox(mid, lags=[10, 20, 50], return_df=True)

    print(f"N = {n}")
    print(f"Durbin-Watson: {dw:.4f}  (2=no autocorr, <2=pos, >2=neg)")
    print("\nLjung-Box test:")
    print(lb.to_string())
    print(f"\n95% confidence band: ±{conf:.4f}")
    print(f"ACF lags 1-5: {acf[1:6]}")

    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("HYDROGEL_PACK mid-price difference — 6-plot diagnostic", fontsize=13)

    # 1. Run sequence
    ax = axes[0, 0]
    ax.plot(mid, linewidth=0.6, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Run sequence")
    ax.set_xlabel("Index")
    ax.set_ylabel("Δ mid-price")

    # 2. Lag-1 scatter
    ax = axes[0, 1]
    ax.scatter(mid[:-1], mid[1:], s=4, alpha=0.4, color="steelblue")
    ax.set_title("Lag-1 plot")
    ax.set_xlabel("y(t)")
    ax.set_ylabel("y(t+1)")

    # 3. ACF
    plot_acf(mid, lags=50, ax=axes[0, 2], color="steelblue", title="ACF")

    # 4. Histogram
    ax = axes[1, 0]
    ax.hist(mid, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_title("Histogram")
    ax.set_xlabel("Δ mid-price")
    ax.set_ylabel("Count")

    # 5. Q-Q plot
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(mid)
    ax.scatter(osm, osr, s=6, alpha=0.5, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1)
    ax.set_title("Normal Q-Q plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")

    # 6. PACF
    plot_pacf(mid, lags=50, ax=axes[1, 2], color="steelblue", title="PACF")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "hydrogel_6plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
