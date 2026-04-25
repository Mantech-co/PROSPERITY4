import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PRICE_FILES = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"))


def load() -> np.ndarray:
    dfs = []
    for f in PRICE_FILES:
        df = pd.read_csv(f, sep=";")
        dfs.append(df[df["product"] == "VELVETFRUIT_EXTRACT"])
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return data["mid_price"].values.astype(np.float32)


def simulate_vectorized(prices, window, entries, exits):
    """
    Vectorized sim over all (entry, exit) pairs simultaneously.
    Since window is fixed, z-scores are identical for all combos.
    entries: (E,), exits: (X,) -> returns pnl grid (E, X)
    """
    n = len(prices)
    E, X = len(entries), len(exits)
    M = E * X

    en = np.repeat(entries, X).astype(np.float32)   # (M,)
    ex = np.tile(exits, E).astype(np.float32)        # (M,)
    valid = ex < en

    pnl = np.zeros(M, dtype=np.float32)
    position = np.zeros(M, dtype=np.int8)    # +1 long, -1 short, 0 flat
    entry_px = np.zeros(M, dtype=np.float32)

    for i in range(window, n):
        wp = prices[i - window:i]
        mu = wp.mean()
        sigma = wp.std()
        if sigma < 1e-9:
            continue
        z = np.float32((prices[i] - mu) / sigma)
        p = prices[i]

        flat  = position == 0
        long  = position == 1
        short = position == -1

        # entry signals
        go_long  = flat & valid & (z < -en)
        go_short = flat & valid & (z >  en)

        # exit signals
        exit_long  = long  & (z >= -ex)
        exit_short = short & (z <=  ex)

        # apply exits first
        pnl += exit_long  * (p - entry_px)
        pnl += exit_short * (entry_px - p)
        position[exit_long | exit_short] = 0

        # apply entries
        position[go_long]  = 1
        position[go_short] = -1
        entry_px[go_long | go_short] = p

    # close open positions at end
    p = prices[-1]
    pnl += (position == 1)  * (p - entry_px)
    pnl += (position == -1) * (entry_px - p)

    return pnl.reshape(E, X)


def poly2d_features(en, ex, deg=5):
    feats = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            feats.append((en ** i) * (ex ** j))
    return np.column_stack(feats)


def main():
    prices = load()
    WINDOW = 10

    entries = np.linspace(0.05, 5.0, 100)
    exits   = np.linspace(0.00, 4.9, 100)

    print(f"Grid: {len(entries)}x{len(exits)} = {len(entries)*len(exits)} combos")

    import time
    t0 = time.time()
    pnl_grid = simulate_vectorized(prices, WINDOW, entries, exits)
    print(f"Simulation done in {time.time()-t0:.2f}s")

    # mask invalid (exit >= entry)
    ENG, EXG = np.meshgrid(entries, exits, indexing="ij")
    mask = EXG < ENG
    pnl_grid[~mask] = np.nan

    # best
    flat_idx = np.nanargmax(pnl_grid)
    ei, xi = np.unravel_index(flat_idx, pnl_grid.shape)
    best_en, best_ex, best_pnl = entries[ei], exits[xi], pnl_grid[ei, xi]
    print(f"Best: entry={best_en:.3f}, exit={best_ex:.3f}, PnL={best_pnl:.2f}")

    # --- polynomial surface fit on valid points ---
    valid_mask = mask.ravel()
    en_arr = ENG.ravel()[valid_mask]
    ex_arr = EXG.ravel()[valid_mask]
    pnl_arr = pnl_grid.ravel()[valid_mask]

    DEG = 5
    X = poly2d_features(en_arr, ex_arr, deg=DEG)
    coeffs, _, _, _ = np.linalg.lstsq(X, pnl_arr, rcond=None)
    pnl_pred = X @ coeffs
    r2 = 1 - np.sum((pnl_arr - pnl_pred)**2) / np.sum((pnl_arr - pnl_arr.mean())**2)
    print(f"Poly deg={DEG} R² = {r2:.4f}")

    # fitted surface on fine grid
    en_fine = np.linspace(entries.min(), entries.max(), 400)
    ex_fine = np.linspace(exits.min(), exits.max(), 400)
    ENF, EXF = np.meshgrid(en_fine, ex_fine, indexing="ij")
    vmask = EXF < ENF
    Xf = poly2d_features(ENF[vmask], EXF[vmask], deg=DEG)
    Z = np.full(ENF.shape, np.nan)
    Z[vmask] = Xf @ coeffs

    # clamp for display
    vmin, vmax = np.nanpercentile(pnl_grid[mask], 2), np.nanpercentile(pnl_grid[mask], 98)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"VELVETFRUIT_EXTRACT  window=10  |  {len(entries)}×{len(exits)} grid  |  "
        f"poly deg={DEG} R²={r2:.3f}  |  best PnL={best_pnl:.2f} (entry={best_en:.2f}, exit={best_ex:.2f})",
        fontsize=10
    )

    ax1 = fig.add_subplot(131)
    im = ax1.pcolormesh(ENG, EXG, np.clip(pnl_grid, vmin, vmax), cmap="RdYlGn", shading="auto")
    plt.colorbar(im, ax=ax1, label="PnL")
    ax1.scatter(best_en, best_ex, color="blue", s=80, marker="*", zorder=5, label=f"best")
    ax1.set_xlabel("entry z"); ax1.set_ylabel("exit z")
    ax1.set_title("Actual PnL"); ax1.legend()

    ax2 = fig.add_subplot(132)
    cf = ax2.contourf(ENF, EXF, np.clip(Z, vmin, vmax), levels=40, cmap="RdYlGn")
    plt.colorbar(cf, ax=ax2, label="PnL (fitted)")
    ax2.scatter(best_en, best_ex, color="blue", s=80, marker="*", zorder=5)
    ax2.set_xlabel("entry z"); ax2.set_ylabel("exit z")
    ax2.set_title("Polynomial surface fit")

    ax3 = fig.add_subplot(133, projection="3d")
    stride = 10
    ax3.plot_surface(ENF[::stride, ::stride], EXF[::stride, ::stride], Z[::stride, ::stride],
                     cmap="RdYlGn", alpha=0.85, linewidth=0)
    ax3.set_xlabel("entry z"); ax3.set_ylabel("exit z"); ax3.set_zlabel("PnL")
    ax3.set_title("3D surface")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "velvetfruit_surface2.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
