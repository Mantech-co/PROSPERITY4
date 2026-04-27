import numpy as np
import math
import matplotlib.pyplot as plt


def _canonical_prices(S, S0, u, n_steps):
    """
    Map simulated prices to canonical levels using searchsorted on precomputed levels.
    Levels: S0*u^net for net>=0, S0*d^|net| for net<0 — pure powers only, no u*d.
    Returns (unique_canonical_prices, group_index_per_path).
    """
    d = 1.0 / u
    j = np.arange(n_steps + 1)
    net = 2 * j - n_steps                                        # -n_steps..n_steps step 2
    levels = np.where(net >= 0, S0 * u ** net, S0 * d ** (-net)) # ascending

    log_lev = np.log(levels)
    log_S   = np.log(S)

    idx = np.searchsorted(log_lev, log_S)
    lo  = np.clip(idx - 1, 0, n_steps)
    hi  = np.clip(idx,     0, n_steps)
    idx = np.where(np.abs(log_S - log_lev[hi]) < np.abs(log_S - log_lev[lo]), hi, lo)

    unique_idx, inverse = np.unique(idx, return_inverse=True)
    return levels[unique_idx], inverse


def mc_chooser(
    S0: float,
    K: float,
    N: int,
    sigma: float,
    s: int,
    choose_day: int = 10,
    n_paths: int = 200_000,
    seed: int = None,
    n_bins: int = 100,
    csv_path: str = None,
    plot_paths: bool = False,
    n_plot: int = 500,
) -> tuple[float, float]:
    """
    Chooser option (co_50): at end of choose_day, becomes call if S > K, put if S < K.
    Expires at end of N days. Returns (price, stderr).
    """
    rng = np.random.default_rng(seed)
    total_steps = N * s
    choose_step = choose_day * s
    dt = 1.0 / s
    u = np.exp(sigma * math.sqrt(dt))
    print(u)
    d = 1.0 / u
    p = (1.0 - d) / (u - d)

    up = rng.random((n_paths, total_steps)) < p
    moves = np.where(up, u, d)

    S = np.empty((n_paths, total_steps + 1))
    S[:, 0] = S0
    for t in range(total_steps):
        S[:, t + 1] = S[:, t] * moves[:, t]

    S_choose = S[:, choose_step]
    S_final  = S[:, -1]

    is_call = S_choose > K
    payoffs = np.where(is_call,
                       np.maximum(S_final - K, 0.0),
                       np.maximum(K - S_final, 0.0))

    price  = payoffs.mean()
    stderr = payoffs.std() / math.sqrt(n_paths)

    if csv_path:
        canonical, inverse = _canonical_prices(S_choose, S0, u, choose_step)
        grp_sum   = np.bincount(inverse, weights=payoffs, minlength=len(canonical))
        grp_count = np.bincount(inverse, minlength=len(canonical))
        grp_mean  = grp_sum / grp_count

        rows = np.column_stack([canonical, grp_mean, grp_count])
        np.savetxt(csv_path, rows, delimiter=",", header=f"spot_at_day{choose_day},expected_payoff,n_paths", comments="")

    if plot_paths:
        days = np.linspace(0, N, total_steps + 1)
        idx  = rng.choice(n_paths, size=min(n_plot, n_paths), replace=False)

        fig, ax = plt.subplots(figsize=(11, 6))
        for i in idx:
            color = "steelblue" if is_call[i] else "tomato"
            ax.plot(days, S[i], color=color, linewidth=0.4, alpha=0.4)

        ax.axvline(choose_day, color="black",  linestyle="--", linewidth=1.2, label=f"choice day {choose_day}")
        ax.axhline(K,          color="gold",   linestyle="--", linewidth=1.2, label=f"K={K}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Spot price")
        ax.set_title(f"Chooser option paths (blue=call, red=put, n={min(n_plot, n_paths)})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return price, stderr


if __name__ == "__main__":
    S0    = 50
    K     = 50
    N     = 15
    sigma = 2.51 / math.sqrt(252)
    s     = 4

    price, se = mc_chooser(S0, K, N, sigma, s, choose_day=10, n_paths=800_0000, seed=42, csv_path="chooser_payout.csv")
    print(f"chooser (choose@day10, expire@day15)  price={price:.4f}  ±{1.96*se:.4f} (95% CI)")
