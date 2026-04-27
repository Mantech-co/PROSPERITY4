import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

IV_DIR = Path(__file__).parent.parent / "data" / "iv"
TS_SPAN = 1_000_000


def load_iv_data():
    files = sorted(IV_DIR.glob("iv_underlying_mid_*.csv"))
    if not files:
        raise FileNotFoundError(f"No IV CSVs found in {IV_DIR}. Run compute_iv.py first.")
    dfs = [pd.read_csv(f) for f in tqdm(files, desc="loading IV CSVs")]
    return pd.concat(dfs, ignore_index=True)


def main():
    ivdf = load_iv_data()

    # use ask_iv, drop NaNs
    ivdf = ivdf[["global_ts", "moneyness", "ask_iv"]].dropna(subset=["ask_iv"])
    ivdf = ivdf.rename(columns={"ask_iv": "iv"})
    ivdf = ivdf.sort_values("global_ts").reset_index(drop=True)

    # bin timestamps, fit parabola per bin
    n_bins = 100
    ivdf["ts_bin"] = pd.cut(ivdf["global_ts"], bins=n_bins, labels=False)

    bin_centers = []
    coeffs = []  # (a, b, c) for each bin

    for b, grp in tqdm(ivdf.groupby("ts_bin"), desc="fitting parabolas", total=n_bins):
        if len(grp) < 5:
            continue
        m = grp["moneyness"].values
        iv = grp["iv"].values
        try:
            a, b_coef, c = np.polyfit(m, iv, 2)
            ts_center = grp["global_ts"].mean()
            bin_centers.append(ts_center)
            coeffs.append((a, b_coef, c))
        except Exception:
            continue

    bin_centers = np.array(bin_centers)
    coeffs = np.array(coeffs)  # shape (N, 3)

    # build surface grid
    m_grid = np.linspace(ivdf["moneyness"].quantile(0.02), ivdf["moneyness"].quantile(0.98), 80)
    T_grid = bin_centers

    # Z[i, j] = a[i]*m[j]^2 + b[i]*m[j] + c[i]
    Z = (
        coeffs[:, 0:1] * m_grid[None, :]**2
        + coeffs[:, 1:2] * m_grid[None, :]
        + coeffs[:, 2:3]
    )

    TT, MM = np.meshgrid(T_grid, m_grid, indexing="ij")

    sample = ivdf.sample(frac=0.03, random_state=42)

    views = [
        ("default",   30, -60),
        ("front",     10,  -90),
        ("side",      10,    0),
        ("top",       90,  -90),
        ("diagonal",  25,  -45),
        ("rear",      20,  120),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), subplot_kw={"projection": "3d"})
    axes_flat = axes.flatten()

    for ax, (label, elev, azim) in zip(axes_flat, views):
        surf = ax.plot_surface(TT, MM, Z, cmap="plasma", alpha=0.85, linewidth=0, antialiased=True)
        ax.scatter(
            sample["global_ts"], sample["moneyness"], sample["iv"],
            s=2, alpha=0.3, color="cyan", zorder=5
        )
        ax.set_xlabel("Timestamp", fontsize=8)
        ax.set_ylabel("ln(K/S)", fontsize=8)
        ax.set_zlabel("Ask IV", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)

    fig.suptitle("VEV Ask IV — Parabolic Vol Surface (multiple views)", fontsize=13)
    fig.colorbar(surf, ax=axes_flat, shrink=0.4, label="Ask IV", location="right")
    plt.tight_layout()
    out = Path(__file__).parent / "vol_surface_parabola.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {out}")

    # parabola coefficients over time
    fig2, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["a (curvature)", "b (skew)", "c (ATM level)"]
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(bin_centers, coeffs[:, i], color="#ff3d5a", linewidth=1.5)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.2)
        for d in [1, 2]:
            ax.axvline(d * TS_SPAN, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("Global Timestamp")
    fig2.suptitle("Parabola Fit Coefficients over Time", fontsize=12)
    plt.tight_layout()
    out2 = Path(__file__).parent / "vol_surface_coeffs.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
