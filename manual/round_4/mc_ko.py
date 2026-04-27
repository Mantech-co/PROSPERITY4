import numpy as np
import math


def _canonical_prices(S, S0, u, n_steps):
    d = 1.0 / u
    j = np.arange(n_steps + 1)
    net = 2 * j - n_steps
    levels = np.where(net >= 0, S0 * u ** net, S0 * d ** (-net))

    log_lev = np.log(levels)
    log_S   = np.log(S)

    idx = np.searchsorted(log_lev, log_S)
    lo  = np.clip(idx - 1, 0, n_steps)
    hi  = np.clip(idx,     0, n_steps)
    idx = np.where(np.abs(log_S - log_lev[hi]) < np.abs(log_S - log_lev[lo]), hi, lo)

    unique_idx, inverse = np.unique(idx, return_inverse=True)
    return levels[unique_idx], inverse


def mc_knockout_put(
    S0: float,
    K: float,
    B: float,
    N: int,
    sigma: float,
    s: int,
    n_paths: int = 200_000,
    barrier_type: str = "down",
    seed: int = None,
    n_bins: int = 100,
    csv_path: str = None,
) -> tuple[float, float]:
    """
    Monte Carlo pricer for a knock-out put option (r=0, risk-neutral).
    Returns (price, stderr).

    down-and-out put: pays max(K-S_T, 0) if S never drops below B
    up-and-out   put: pays max(K-S_T, 0) if S never rises above B
    """
    rng = np.random.default_rng(seed)
    total_steps = N * s
    dt = 1.0 / s
    u = np.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)

    # at each step: multiply by u with prob p, by d with prob 1-p
    up = rng.random((n_paths, total_steps)) < p
    moves = np.where(up, u, d)

    S = np.empty((n_paths, total_steps + 1))
    S[:, 0] = S0
    for t in range(total_steps):
        S[:, t + 1] = S[:, t] * moves[:, t]

    # Knock-out check across all steps including S0
    if barrier_type == "down":
        alive = np.all(S > B, axis=1)
    elif barrier_type == "up":
        alive = np.all(S < B, axis=1)
    else:
        raise ValueError("barrier_type must be 'down' or 'up'")

    payoffs = np.where(alive, np.maximum(K - S[:, -1], 0.0), 0.0)

    price  = payoffs.mean()
    stderr = payoffs.std() / math.sqrt(n_paths)

    if csv_path:
        S_final = S[:, -1]
        canonical, inverse = _canonical_prices(S_final, S0, u, total_steps)
        grp_sum   = np.bincount(inverse, weights=payoffs, minlength=len(canonical))
        grp_count = np.bincount(inverse, minlength=len(canonical))
        grp_mean  = grp_sum / grp_count

        rows = np.column_stack([canonical, grp_mean, grp_count])
        np.savetxt(csv_path, rows, delimiter=",", header="spot_at_expiry,expected_payoff,n_paths", comments="")

    return price, stderr


if __name__ == "__main__":
    S0    = 50
    K     = 45
    N     = 15
    sigma = 2.51 / math.sqrt(252)
    s     = 4

    for B in [35]:
        price, se = mc_knockout_put(S0, K, B, N, sigma, s, n_paths=500_000_0, seed=42, csv_path=f"ko_payout_B{B}.csv")
        print(f"down-and-out put  B={B}  price={price:.4f}  ±{1.96*se:.4f} (95% CI)")

    price, se = mc_knockout_put(S0, K, 0, N, sigma, s, n_paths=500_000_0, seed=42)
    print(f"vanilla put              price={price:.4f}  ±{1.96*se:.4f} (95% CI)")
