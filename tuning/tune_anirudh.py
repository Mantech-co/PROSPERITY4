"""
Tune anirudh.py tier parameters (grid distances + cumulative fractions).

For 3-tier products (GALAXY_*): sweeps d1, f1, d2, f2, d3 → tiers = [(d1,f1),(d2,f2),(d3,1.0)]
For 2-tier products (SNACKPACK_*): sweeps d1, f1, d2        → tiers = [(d1,f1),(d2,1.0)]

Usage:
    python tuning/tune_anirudh.py GALAXY_SOUNDS_SOLAR_FLAMES
    python tuning/tune_anirudh.py SNACKPACK_RASPBERRY --degree 3
    python tuning/tune_anirudh.py --list
"""

import sys
import csv
import itertools
import argparse
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prosperity4bt"))

import warnings
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import strategy.anirudh as anirudh_module
from tuning.bt_gpu import run_sweep

RESULTS_DIR = PROJECT_ROOT / "tuning" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameter grids ───────────────────────────────────────────────────────────

PARAM_GRIDS: dict[str, dict] = {
    "GALAXY_SOUNDS_SOLAR_FLAMES": {
        "_tiers": 3,
        "d1": [100, 150, 200, 300],
        "f1": [0.20, 0.30, 0.40],
        "d2": [400, 500, 700],
        "f2": [0.60, 0.70, 0.80],
        "d3": [800, 1000, 1500],
    },
    "GALAXY_SOUNDS_DARK_MATTER": {
        "_tiers": 3,
        "d1": [100, 150, 200, 300],
        "f1": [0.20, 0.30, 0.40],
        "d2": [400, 500, 700],
        "f2": [0.60, 0.70, 0.80],
        "d3": [800, 1000, 1500],
    },
    "SNACKPACK_RASPBERRY": {
        "_tiers": 2,
        "d1": [100, 150, 200, 250],
        "f1": [0.40, 0.60, 0.70, 0.80],
        "d2": [250, 300, 400, 500],
    },
    "SNACKPACK_VANILLA": {
        "_tiers": 2,
        "d1": [100, 150, 200, 250],
        "f1": [0.40, 0.60, 0.70, 0.80],
        "d2": [250, 300, 400, 500],
    },
    "SNACKPACK_CHOCOLATE": {
        "_tiers": 2,
        "d1": [100, 150, 200, 250],
        "f1": [0.40, 0.60, 0.70, 0.80],
        "d2": [250, 300, 400, 500],
    },
}


# ── Tier helpers ──────────────────────────────────────────────────────────────

def _make_tiers(n: int, combo: dict) -> list[tuple]:
    if n == 3:
        return [(combo["d1"], combo["f1"]),
                (combo["d2"], combo["f2"]),
                (combo["d3"], 1.0)]
    return [(combo["d1"], combo["f1"]),
            (combo["d2"], 1.0)]


def _tiers_to_combo(tiers: list[tuple], n: int) -> dict:
    if n == 2:
        return {"d1": tiers[0][0], "f1": tiers[0][1], "d2": tiers[1][0]}
    return {"d1": tiers[0][0], "f1": tiers[0][1],
            "d2": tiers[1][0], "f2": tiers[1][1], "d3": tiers[2][0]}


def _is_valid(n: int, combo: dict) -> bool:
    if n == 3:
        return (combo["d1"] < combo["d2"] < combo["d3"]
                and combo["f1"] < combo["f2"])
    return combo["d1"] < combo["d2"]


def _combo_to_vec(n: int, combo: dict) -> list[float]:
    if n == 3:
        return [combo["d1"], combo["f1"], combo["d2"], combo["f2"], combo["d3"]]
    return [combo["d1"], combo["f1"], combo["d2"]]


# ── CSV ───────────────────────────────────────────────────────────────────────

def _save_csv(product: str, combos: list, keys: list, n: int, cash: np.ndarray):
    path = RESULTS_DIR / f"{product}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["product_pnl", "tiers"])
        for combo, pnl in zip(combos, cash):
            w.writerow([combo[k] for k in keys] + [f"{pnl:.0f}", str(_make_tiers(n, combo))])
    print(f"Results saved → {path}")


# ── Polynomial fit & optimise ─────────────────────────────────────────────────

def _bounds_from_grid(grid: dict, keys: list) -> list[tuple]:
    return [(min(grid[k]), max(grid[k])) for k in keys]


def poly_optimise(product: str, n: int, keys: list, grid: dict,
                  X: np.ndarray, y: np.ndarray, degree: int,
                  fixed_params: dict) -> list[tuple]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    model = Ridge(alpha=1.0).fit(Xp, y)

    bounds = _bounds_from_grid(grid, keys)

    def neg_pred(x):
        return -model.predict(poly.transform(x.reshape(1, -1)))[0]

    def constraint_valid(x):
        combo = dict(zip(keys, x))
        return 1.0 if _is_valid(n, combo) else -1.0

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      result = differential_evolution(
        neg_pred, bounds,
        constraints=NonlinearConstraint(constraint_valid, 0, np.inf),
          seed=42, maxiter=2000, tol=1e-8, polish=True,
      )

    x_opt = result.x
    pred_pnl = -result.fun
    print(f"\nPoly (degree={degree}) predicted optimum: {pred_pnl:+,.0f}")
    print(f"  raw params: {dict(zip(keys, x_opt))}")

    rounded = [round(v) if k.startswith("d") else round(v, 3) for k, v in zip(keys, x_opt)]
    tiers = _make_tiers(n, dict(zip(keys, rounded)))
    return tiers


# ── Tuner ─────────────────────────────────────────────────────────────────────

def tune(product: str, poly_degree: int = 2) -> list[tuple]:
    if product not in PARAM_GRIDS:
        print(f"No grid for '{product}'. Run --list.")
        sys.exit(1)

    grid = PARAM_GRIDS[product]
    n    = grid["_tiers"]
    keys = [k for k in grid if not k.startswith("_")]
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*[grid[k] for k in keys])
              if _is_valid(n, dict(zip(keys, vals)))]

    fixed_params = anirudh_module.TUNING_PARAMS[product]

    print(f"\n{'='*60}")
    print(f"Tuning: {product}  ({len(combos)} combos, GPU sweep)")
    print(f"{'='*60}")

    # ── GPU sweep: all combos at once ────────────────────────────────────
    cash_arr = run_sweep(product, combos, n, fixed_params)

    _save_csv(product, combos, keys, n, cash_arr)

    # Build & sort results
    results = sorted(zip(cash_arr, combos), key=lambda x: -x[0])

    print(f"\nTop 10 by {product} PnL (grid):")
    for pnl, combo in results[:10]:
        tiers = _make_tiers(n, combo)
        print(f"  {pnl:+10,.0f}   {tiers}")

    # ── Polynomial fit & optimise ────────────────────────────────────────
    X = np.array([_combo_to_vec(n, c) for c in combos], dtype=float)
    y = cash_arr.astype(float)
    best_poly_tiers = poly_optimise(product, n, keys, grid, X, y, poly_degree, fixed_params)

    # Verify poly suggestion with GPU
    poly_combo = _tiers_to_combo(best_poly_tiers, n)
    poly_cash = run_sweep(product, [poly_combo], n, fixed_params)[0]
    print(f"  actual PnL: {poly_cash:+,.0f}   {best_poly_tiers}")

    # Append poly result to CSV
    csv_path = RESULTS_DIR / f"{product}.csv"
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([poly_combo.get(k, "") for k in keys] + [f"{poly_cash:.0f}", f"POLY:{best_poly_tiers}"])

    best_grid_pnl, best_grid_combo = results[0]
    best_grid_tiers = _make_tiers(n, best_grid_combo)

    if poly_cash > best_grid_pnl:
        print(f"\nPoly beats grid (+{poly_cash - best_grid_pnl:,.0f}) — best: {best_poly_tiers}")
        return best_poly_tiers
    else:
        print(f"\nGrid still best — best: {best_grid_tiers}")
        return best_grid_tiers


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("product", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    if args.list or not args.product:
        print("Tunable products:")
        for p in PARAM_GRIDS:
            print(f"  {p}")
        sys.exit(0)

    tune(args.product, poly_degree=args.degree)


if __name__ == "__main__":
    main()
