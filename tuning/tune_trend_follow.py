"""
Tune trend_follow.py — single shared parameter set across all 50 products.
Optimises total PnL summed over all products.

Fill in PARAM_GRID below, then run:
    python tuning/tune_trend_follow.py
    python tuning/tune_trend_follow.py --degree 3
"""

import sys, csv, itertools, argparse, warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, NonlinearConstraint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prosperity4bt"))

from strategy.trend_follow import PRODUCTS
from tuning.bt_gpu import preload_product, DEVICE

RESULTS_DIR = PROJECT_ROOT / "tuning" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KEYS = ["ema_period", "confirmation_coefficient", "entry_threshold",
        "exit_threshold", "take_profit"]

# ── Define your bounds here ───────────────────────────────────────────────────

PARAM_GRID: dict = {
    # "ema_period":               [...],
    # "confirmation_coefficient": [...],
    # "entry_threshold":          [...],
    # "exit_threshold":           [...],
    # "take_profit":              [...],
}

# ── Validity ──────────────────────────────────────────────────────────────────

def _is_valid(combo: dict) -> bool:
    return combo["exit_threshold"] < combo["entry_threshold"]


# ── GPU sweep for one product ─────────────────────────────────────────────────

def _sweep_product(price_data: np.ndarray, combos: list, p_lim: float) -> np.ndarray:
    N = len(combos)
    zeros = torch.zeros(N, dtype=torch.float64, device=DEVICE)

    def _t(key):
        return torch.tensor([c[key] for c in combos], dtype=torch.float64, device=DEVICE)

    alpha     = 2.0 / (_t("ema_period") + 1.0)
    eff_thr   = _t("confirmation_coefficient") / _t("ema_period")
    entry_thr = _t("entry_threshold")
    exit_thr  = _t("exit_threshold")
    tp        = _t("take_profit")
    lim       = torch.full((N,), p_lim, dtype=torch.float64, device=DEVICE)

    ema       = torch.full((N,), -1.0, dtype=torch.float64, device=DEVICE)
    prev_ema  = torch.full((N,), -1.0, dtype=torch.float64, device=DEVICE)
    prev_ts   = torch.full((N,), -1.0, dtype=torch.float64, device=DEVICE)
    conf      = zeros.clone()
    in_pos    = torch.zeros(N, dtype=torch.bool,    device=DEVICE)
    pos_dir   = zeros.clone()
    entry_mid = zeros.clone()
    pos       = zeros.clone()
    cash      = zeros.clone()

    for t in range(price_data.shape[0]):
        row = price_data[t]
        BB1 = float(row[0]); BA1 = float(row[6]); ts = float(row[12])
        if BA1 <= 0 or BB1 <= 0:
            continue

        mid   = (BB1 + BA1) / 2.0
        mid_t = torch.full((N,), mid, dtype=torch.float64, device=DEVICE)

        # EMA
        uninit = ema < 0
        ema = torch.where(uninit, mid_t, mid_t * alpha + ema * (1.0 - alpha))

        # Slope
        dt_vec   = torch.full((N,), ts, dtype=torch.float64, device=DEVICE) - prev_ts
        valid_dt = (prev_ema >= 0) & (dt_vec > 0)
        dt_safe  = torch.where(dt_vec > 0, dt_vec, torch.full((N,), 100.0, dtype=torch.float64, device=DEVICE))
        slope    = torch.where(valid_dt, (ema - prev_ema) / dt_safe, zeros)

        prev_ema = ema.clone()
        prev_ts  = torch.full((N,), ts, dtype=torch.float64, device=DEVICE)

        # Sync stale in_pos
        stale    = in_pos & (pos == 0)
        in_pos   = in_pos & ~stale
        pos_dir  = torch.where(stale, zeros, pos_dir)
        entry_mid = torch.where(stale, zeros, entry_mid)
        conf     = torch.where(stale, zeros, conf)

        # Exit
        pct_gain     = torch.where(in_pos & (entry_mid > 0),
                           (mid_t - entry_mid) / entry_mid * pos_dir, zeros)
        should_close = in_pos & ((pct_gain >= tp) | (slope.abs() < exit_thr))

        close_long  = should_close & (pos_dir > 0) & (pos > 0)
        cash = torch.where(close_long,  cash + BB1 * pos,       cash)
        pos  = torch.where(close_long,  zeros,                   pos)

        close_short = should_close & (pos_dir < 0) & (pos < 0)
        cash = torch.where(close_short, cash - BA1 * pos.abs(), cash)
        pos  = torch.where(close_short, zeros,                   pos)

        in_pos    = in_pos & ~should_close
        pos_dir   = torch.where(should_close, zeros, pos_dir)
        entry_mid = torch.where(should_close, zeros, entry_mid)
        conf      = torch.where(should_close, zeros, conf)

        # Confirmation
        above_thr = ~in_pos & (slope.abs() >= entry_thr)
        conf = torch.where(above_thr, conf + slope * dt_safe, zeros)

        # Entry
        go_long  = ~in_pos & (conf >  eff_thr)
        go_short = ~in_pos & (conf < -eff_thr)

        cash      = torch.where(go_long,  cash - BA1 * lim, cash)
        cash      = torch.where(go_short, cash + BB1 * lim, cash)
        pos       = torch.where(go_long,  lim,  torch.where(go_short, -lim, pos))
        in_pos    = in_pos | go_long | go_short
        pos_dir   = torch.where(go_long,  torch.ones_like(pos_dir),
                    torch.where(go_short, -torch.ones_like(pos_dir), pos_dir))
        entry_mid = torch.where(go_long | go_short, mid_t, entry_mid)
        conf      = torch.where(go_long | go_short, zeros, conf)

    return cash.cpu().numpy()


# ── Sweep all products, return (N, n_products) cash matrix ───────────────────

def run_sweep_all(combos: list, p_lim: float = 10.0) -> np.ndarray:
    """Returns shape (N_combos, N_products). Loads each product once."""
    results = []
    for i, product in enumerate(PRODUCTS):
        print(f"  [{i+1:2d}/{len(PRODUCTS)}] {product}", flush=True)
        try:
            price_data = preload_product(product)
            cash = _sweep_product(price_data, combos, p_lim)
        except Exception as e:
            print(f"    skipped: {e}")
            cash = np.zeros(len(combos))
        results.append(cash)
    return np.stack(results, axis=1)   # (N_combos, N_products)


# ── CSV ───────────────────────────────────────────────────────────────────────

def _save_csv(combos: list, cash_matrix: np.ndarray):
    path = RESULTS_DIR / "tf_all_products.csv"
    total = cash_matrix.sum(axis=1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KEYS + ["total_pnl"] + PRODUCTS)
        for combo, row, t in zip(combos, cash_matrix, total):
            w.writerow([combo[k] for k in KEYS] + [f"{t:.0f}"] +
                       [f"{v:.0f}" for v in row])
    print(f"Results saved → {path}")


# ── Poly fit & optimise ───────────────────────────────────────────────────────

def poly_optimise(combos: list, total_cash: np.ndarray, degree: int) -> dict:
    X = np.array([[c[k] for k in KEYS] for c in combos], dtype=float)
    y = total_cash.astype(float)

    poly  = PolynomialFeatures(degree=degree, include_bias=True)
    model = Ridge(alpha=1.0).fit(poly.fit_transform(X), y)

    bounds = [(min(PARAM_GRID[k]), max(PARAM_GRID[k])) for k in KEYS]

    def neg_pred(x):
        return -model.predict(poly.transform(x.reshape(1, -1)))[0]

    def constraint(x):
        return 1.0 if _is_valid(dict(zip(KEYS, x))) else -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = differential_evolution(
            neg_pred, bounds,
            constraints=NonlinearConstraint(constraint, 0, np.inf),
            seed=42, maxiter=2000, tol=1e-8, polish=True,
        )

    x_opt    = result.x
    pred_pnl = -result.fun
    rounded  = {k: (round(v) if k == "ema_period" else round(v, 6))
                for k, v in zip(KEYS, x_opt)}
    print(f"\nPoly (degree={degree}) predicted total PnL: {pred_pnl:+,.0f}")
    print(f"  params: {rounded}")
    return rounded


# ── Tuner ─────────────────────────────────────────────────────────────────────

def tune(poly_degree: int = 2):
    if not all(k in PARAM_GRID for k in KEYS):
        missing = [k for k in KEYS if k not in PARAM_GRID]
        print(f"Fill in PARAM_GRID — missing keys: {missing}")
        sys.exit(1)

    combos = [dict(zip(KEYS, vals))
              for vals in itertools.product(*[PARAM_GRID[k] for k in KEYS])
              if _is_valid(dict(zip(KEYS, vals)))]

    print(f"\n{'='*60}")
    print(f"Sweeping {len(combos)} combos × {len(PRODUCTS)} products (GPU)")
    print(f"{'='*60}\n")

    cash_matrix = run_sweep_all(combos)
    total_cash  = cash_matrix.sum(axis=1)

    _save_csv(combos, cash_matrix)

    ranked = sorted(zip(total_cash, combos), key=lambda x: -x[0])
    print(f"\nTop 10 combos by total PnL:")
    for pnl, combo in ranked[:10]:
        print(f"  {pnl:+10,.0f}   {combo}")

    # Poly optimise
    best_poly = poly_optimise(combos, total_cash, poly_degree)
    poly_matrix = run_sweep_all([best_poly])
    poly_total  = poly_matrix.sum()
    print(f"  actual total PnL: {poly_total:+,.0f}")

    # Append poly row
    csv_path = RESULTS_DIR / "tf_all_products.csv"
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([best_poly.get(k, "") for k in KEYS] +
                   [f"{poly_total:.0f}  # POLY"] +
                   [f"{v:.0f}" for v in poly_matrix[0]])

    best_grid_pnl, best_grid_combo = ranked[0]
    best = best_poly if poly_total > best_grid_pnl else best_grid_combo
    tag  = "poly" if poly_total > best_grid_pnl else "grid"

    print(f"\nBest params ({tag}):")
    for k, v in best.items():
        print(f"  {k:<30} = {v}")

    return best


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()
    tune(poly_degree=args.degree)


if __name__ == "__main__":
    main()
