"""
Tune anirudh.py tier parameters (grid distances + cumulative fractions).

For 3-tier products (GALAXY_*): sweeps d1, f1, d2, f2, d3 → tiers = [(d1,f1),(d2,f2),(d3,1.0)]
For 2-tier products (SNACKPACK_*): sweeps d1, f1, d2        → tiers = [(d1,f1),(d2,1.0)]

Usage:
    python tuning/tune_anirudh.py GALAXY_SOUNDS_SOLAR_FLAMES
    python tuning/tune_anirudh.py SNACKPACK_RASPBERRY
    python tuning/tune_anirudh.py --list
"""

import sys
import itertools
import argparse
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prosperity4bt"))

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import strategy.anirudh as anirudh_module
from tuning.bt import load_prices, match_orders, DAYS
from datamodel import OrderDepth, TradingState, Order, Observation, Listing
import io, contextlib

# ── Parameter grids ───────────────────────────────────────────────────────────
# _tiers: number of grid tiers (2 or 3)
# d1 < d2 < d3 enforced; f1 < f2 enforced; last tier fraction is always 1.0

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


def _is_valid(n: int, combo: dict) -> bool:
    if n == 3:
        return (combo["d1"] < combo["d2"] < combo["d3"]
                and combo["f1"] < combo["f2"])
    return combo["d1"] < combo["d2"]


# ── Core runner ───────────────────────────────────────────────────────────────

def run_combo(product: str, tiers: list[tuple]) -> tuple[float, float]:
    anirudh_module.TUNING_PARAMS[product]["tiers"] = tiers
    trader = anirudh_module.Trader()

    positions: dict = {}
    cash: dict = {}

    for day in DAYS:
        price_data = load_prices(day)
        timestamps = sorted(price_data.keys())
        products = set(p for ts in price_data.values() for p in ts)
        trader_data = ""

        for ts in timestamps:
            order_depths = {}
            ts_data = price_data[ts]
            for prod in products:
                if prod not in ts_data:
                    continue
                od = OrderDepth()
                od.buy_orders = dict(ts_data[prod]["buy_orders"])
                od.sell_orders = dict(ts_data[prod]["sell_orders"])
                order_depths[prod] = od

            state = TradingState(
                traderData=trader_data,
                timestamp=ts,
                listings={p: Listing(p, p, 1) for p in order_depths},
                order_depths=order_depths,
                own_trades={},
                market_trades={},
                position=dict(positions),
                observations=Observation({}, {}),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                orders_dict, _, trader_data = trader.run(state)
            for prod, orders in orders_dict.items():
                if prod in order_depths:
                    match_orders(orders, order_depths, positions, cash)

    product_pnl = cash.get(product, 0.0)
    total_pnl = sum(cash.values())
    return product_pnl, total_pnl


# ── Polynomial fit & optimise ─────────────────────────────────────────────────

def _combo_to_vec(n_tiers: int, combo: dict) -> list[float]:
    if n_tiers == 3:
        return [combo["d1"], combo["f1"], combo["d2"], combo["f2"], combo["d3"]]
    return [combo["d1"], combo["f1"], combo["d2"]]


def _vec_to_tiers(n_tiers: int, keys: list[str], x: np.ndarray) -> list[tuple]:
    combo = dict(zip(keys, x))
    return _make_tiers(n_tiers, combo)


def _bounds_from_grid(grid: dict) -> list[tuple]:
    keys = [k for k in grid if not k.startswith("_")]
    return [(min(grid[k]), max(grid[k])) for k in keys]


def poly_optimise(product: str, n_tiers: int, keys: list[str], grid: dict,
                  X: np.ndarray, y: np.ndarray, degree: int) -> list[tuple]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    model = Ridge(alpha=1.0).fit(Xp, y)

    bounds = _bounds_from_grid(grid)

    def neg_pred(x):
        xp = poly.transform(x.reshape(1, -1))
        return -model.predict(xp)[0]

    def constraint_valid(x):
        combo = dict(zip(keys, x))
        return 1.0 if _is_valid(n_tiers, combo) else -1.0

    result = differential_evolution(
        neg_pred, bounds,
        constraints=NonlinearConstraint(constraint_valid, 0, np.inf),
        seed=42, maxiter=2000, tol=1e-8, polish=True,
    )

    x_opt = result.x
    pred_pnl = -result.fun
    print(f"\nPoly (degree={degree}) predicted optimum: {pred_pnl:+,.0f}")
    print(f"  raw x: {dict(zip(keys, x_opt))}")

    # round distances to int, keep fractions as-is
    rounded = []
    for k, v in zip(keys, x_opt):
        rounded.append(round(v) if k.startswith("d") else round(v, 3))
    tiers = _vec_to_tiers(n_tiers, keys, np.array(rounded))
    return tiers


# ── Tuner ─────────────────────────────────────────────────────────────────────

def tune(product: str, poly_degree: int = 2) -> list[tuple]:
    if product not in PARAM_GRIDS:
        print(f"No grid for '{product}'. Run --list.")
        sys.exit(1)

    grid  = PARAM_GRIDS[product]
    n     = grid["_tiers"]
    keys  = [k for k in grid if not k.startswith("_")]
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*[grid[k] for k in keys])
              if _is_valid(n, dict(zip(keys, vals)))]

    print(f"\n{'='*60}")
    print(f"Tuning: {product}  ({len(combos)} valid combos × {len(DAYS)} days)")
    print(f"{'='*60}\n")

    original_tiers = deepcopy(anirudh_module.TUNING_PARAMS[product]["tiers"])
    results = []
    X_rows, y_vals = [], []

    for i, combo in enumerate(combos):
        tiers = _make_tiers(n, combo)
        prod_pnl, total_pnl = run_combo(product, tiers)
        results.append((prod_pnl, total_pnl, tiers))
        X_rows.append(_combo_to_vec(n, combo))
        y_vals.append(prod_pnl)
        tag = " | ".join(f"({d},{f})" for d, f in tiers)
        print(f"[{i+1:3d}/{len(combos)}] {tag}  →  {prod_pnl:+,.0f}")

    anirudh_module.TUNING_PARAMS[product]["tiers"] = original_tiers

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*60}")
    print(f"Top 10 by {product} PnL (grid):")
    print(f"{'='*60}")
    for prod_pnl, total_pnl, tiers in results[:10]:
        print(f"  {prod_pnl:+10,.0f}  (total {total_pnl:+,.0f})   {tiers}")

    # polynomial fit
    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    best_poly = poly_optimise(product, n, keys, grid, X, y, poly_degree)

    # verify poly suggestion by actually running it
    prod_pnl_poly, total_pnl_poly = run_combo(product, best_poly)
    anirudh_module.TUNING_PARAMS[product]["tiers"] = original_tiers
    print(f"  actual PnL: {prod_pnl_poly:+,.0f}  (total {total_pnl_poly:+,.0f})   {best_poly}")

    # pick whichever is better
    best_grid = results[0][2]
    if prod_pnl_poly > results[0][0]:
        print(f"\nPoly beats grid — using poly tiers: {best_poly}")
        return best_poly
    else:
        print(f"\nGrid still best — using grid tiers: {best_grid}")
        return best_grid


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
