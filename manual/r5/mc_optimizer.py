"""
Monte Carlo portfolio optimizer.
Sample product returns from per-product distributions, run optimizer, aggregate results.

Edit PRODUCTS to define your distributions.

  Symmetric  (normal):
    {"mean": 0.20, "delta_95": 0.10}
    → 95% of outcomes within [mean-delta_95, mean+delta_95]

  Asymmetric (skew-normal):
    {"mean": 0.15, "up_95": 0.18, "down_95": 0.08}
    → 95% confident return won't exceed mean+up_95
    → 95% confident return won't fall below mean-down_95
    skewness direction is inferred automatically
"""

import numpy as np
from scipy.stats import norm, skewnorm
from scipy.optimize import fsolve
from collections import defaultdict
from portfolio_optimizer import optimize_portfolio

_Z95 = 1.959964  # norm.ppf(0.975)

# --- Define your product distributions here ---
PRODUCTS = {
        "A": {"mean": 55.00, "up_95": 2, "down_95": 15},
    "B": {"mean": 2.5 , "delta_95": 1.25},
    "C": {"mean": 25.00, "up_95": 5, "down_95": 10},
    "D": {"mean": 7.5,   "up_95": 2, "down_95": 2},
    "E": {"mean": 35.00,"up_95": 7, "down_95": 5},
    "G": {"mean": 30.00, "up_95": 5, "down_95": 5},
    "H": {"mean": 45, "up_95": 5, "down_95": 10},
    "I": {"mean": 7.00, "up_95": 5, "down_95": 3},
    "J": {"mean": 7.5, "up_95": 2, "down_95": 2},
    
}

N_SAMPLES = 100_000
SEED = 42


def _fit_skewnorm(mean: float, p5: float, p95: float) -> tuple:
    scale0 = (p95 - p5) / (2 * _Z95)
    a0 = ((p95 - mean) - (mean - p5)) / scale0

    def loc_from_mean(a, scale):
        # skewnorm analytical mean: loc + scale * delta * sqrt(2/pi)
        delta = a / np.sqrt(1 + a ** 2)
        return mean - scale * delta * np.sqrt(2 / np.pi)

    def equations(params):
        a, scale = params
        scale = abs(scale)
        loc = loc_from_mean(a, scale)
        d = skewnorm(a, loc, scale)
        return [d.ppf(0.05) - p5, d.ppf(0.95) - p95]

    for a_init in [a0, a0 * 2, a0 / 2, 0.1 * np.sign(a0)]:
        sol, _, ier, _ = fsolve(equations, [a_init, scale0], full_output=True)
        if ier == 1:
            a, scale = sol
            break
    else:
        a, scale = sol  # best effort

    scale = abs(scale)
    loc = loc_from_mean(a, scale)
    return {"a": float(a), "loc": float(loc), "scale": float(scale)}


def _build_dist_params(cfg: dict) -> tuple:
    mean = cfg["mean"]
    if "delta_95" in cfg:
        return ("norm", {"loc": mean, "scale": cfg["delta_95"] / _Z95})
    p5  = mean - cfg["down_95"]
    p95 = mean + cfg["up_95"]
    return ("skewnorm", _fit_skewnorm(mean, p5, p95))


_COMPILED = {name: _build_dist_params(cfg) for name, cfg in PRODUCTS.items()}
_DIST = {"norm": norm, "skewnorm": skewnorm}


def sample_returns(rng: np.random.Generator) -> dict[str, float]:
    out = {}
    for name, (dist_type, params) in _COMPILED.items():
        out[name] = float(_DIST[dist_type].rvs(**params, random_state=rng))
    return out


def run_mc(n: int = N_SAMPLES) -> dict:
    rng = np.random.default_rng(SEED)
    alloc_samples = defaultdict(list)
    return_samples = defaultdict(list)
    pnl_samples = []

    for _ in range(n):
        returns = sample_returns(rng)
        allocs, _, pnl = optimize_portfolio(returns)
        for p, a in allocs.items():
            alloc_samples[p].append(a)
        for p, r in returns.items():
            return_samples[p].append(r)
        pnl_samples.append(pnl)

    return {
        "allocs": {p: np.array(v) for p, v in alloc_samples.items()},
        "returns": {p: np.array(v) for p, v in return_samples.items()},
        "pnl": np.array(pnl_samples),
    }


def print_summary(results: dict) -> None:
    allocs = results["allocs"]
    pnl = results["pnl"]

    print(f"\n{'Product':<10} {'Mean Alloc':>12} {'Std':>10} {'P(>0)':>8}")
    print("-" * 44)
    for p in sorted(allocs):
        a = allocs[p]
        print(f"{p:<10} {a.mean()*100:>11.2f}% {a.std()*100:>9.2f}% {(a > 1e-9).mean()*100:>7.1f}%")

    print(f"\nPnL  mean={pnl.mean()*100:.2f}%  std={pnl.std()*100:.2f}%"
          f"  p5={np.percentile(pnl,5)*100:.2f}%  p95={np.percentile(pnl,95)*100:.2f}%")


def export_csv(results: dict, path: str = "mc_simulations.csv") -> None:
    import csv
    products = sorted(results["allocs"].keys())
    n = len(results["pnl"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            [f"return_{p}" for p in products]
            + [f"alloc_{p}" for p in products]
            + ["pnl"]
        )
        writer.writerow(header)
        for i in range(n):
            row = (
                [results["returns"][p][i] for p in products]
                + [results["allocs"][p][i] for p in products]
                + [results["pnl"][i]]
            )
            writer.writerow(row)
    print(f"Saved {n} simulations → {path}")


if __name__ == "__main__":
    results = run_mc()
    print_summary(results)
    export_csv(results)
