import numpy as np
import math


def regroup(csv_path: str, S0: float, sigma: float, s: int, n_steps: int, out_path: str = None, tol: float = 0.001):
    data    = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    spots   = data[:, 0]
    payoffs = data[:, 1]
    n_paths = data[:, 2]

    u = np.exp(sigma * math.sqrt(1.0 / s))
    d = 1.0 / u

    # net moves = 2j - n_steps, so step is always 2 → only even powers reachable
    step = 2
    levels = [S0 * d**k for k in range(0, n_steps + 1, step)] + \
             [S0 * u**k for k in range(step, n_steps + 1, step)]
    levels = sorted(levels)

    groups = {}  # canonical_level -> [weighted_payoff_sum, path_count]
    unmatched = []

    for spot, payoff, n in zip(spots, payoffs, n_paths):
        matched = None
        for level in levels:
            if abs(spot - level) / level <= tol:
                matched = level
                break
        if matched is not None:
            if matched not in groups:
                groups[matched] = [0.0, 0.0]
            groups[matched][0] += payoff * n
            groups[matched][1] += n
        else:
            unmatched.append(spot)

    if unmatched:
        print(f"Warning: {len(unmatched)} rows unmatched: {unmatched[:5]}")

    rows = []
    for level in sorted(groups):
        ws, nc = groups[level]
        rows.append([level, ws / nc, nc])

    out = out_path or csv_path
    header_line = open(csv_path).readline().strip()
    col0 = header_line.split(",")[0]
    np.savetxt(out, rows, delimiter=",",
               header=f"{col0},expected_payoff,n_paths", comments="")
    print(f"{len(rows)} canonical groups written to {out}")


if __name__ == "__main__":
    sigma = 2.51 / math.sqrt(252)
    s     = 4

    regroup("chooser_payout.csv", S0=50, sigma=sigma, s=s, n_steps=40, out_path="chooser_payout_grouped.csv")
    regroup("ko_payout_B35.csv",  S0=50, sigma=sigma, s=s, n_steps=60, out_path="ko_payout_B35_grouped.csv")
