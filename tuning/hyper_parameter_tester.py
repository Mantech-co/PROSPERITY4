"""
Hyperparameter Grid-Search Tester for TOMATOES strategy.

Iterates through all combinations of parameter ranges, writes each
combination to tomato_params.json, runs a backtest, and saves the log
with a descriptive filename.

Supports two modes:
  --mode parallel   (default) uses multiprocessing; falls back to sequential on failure
  --mode sequential runs one backtest at a time

Usage:
    python tuning/hyper_parameter_tester.py --mode sequential
"""

import sys
import os
import json
import itertools
import argparse
import time
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Shared tuning file in strategy/
TUNING_PARAMS_FILE = PROJECT_ROOT / "strategy" / "params.json"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prosperity4bt"))

BACKTEST_OUTPUT_DIR = PROJECT_ROOT / "backtests" / "tuning_runs"

# ── Parameter Grid ────────────────────────────────────────────────────────────
# Edit these ranges to control what gets tested.
# Each key must match a key in tomato_params.json.
PARAM_GRID = {
    "m_slope":         list(range(20, 40, 1)),
    "slope_threshold": [i/100 for i in range(0, 20, 1)],
}

# Fixed params (not swept, but always written)
FIXED_PARAMS = {
    "alpha": 0.5,
    "n_sma": 20,
    "position_limit": 80,
}

# Backtest config
ROUND_DAY = ["0"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_combinations(grid: dict) -> list[dict]:
    """Cartesian product of parameter grid → list of param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo_vals in itertools.product(*values):
        combo = dict(zip(keys, combo_vals))
        combos.append(combo)
    return combos


def _combo_to_filename(combo: dict) -> str:
    """Create a descriptive filename from a parameter combination.
    Uses '--' to separate key=value pairs (since param names contain '_').
    """
    parts = []
    for k, v in sorted(combo.items()):
        # Shorten key names for readability
        short = k.replace("slope_threshold", "st").replace("threshold", "th")
        if isinstance(v, float):
            parts.append(f"{short}={v:.3f}")
        else:
            parts.append(f"{short}={v}")
    return "--".join(parts) + ".log"


def _write_params(combo: dict):
    """Write current parameter combination to the shared tuning file."""
    full_params = {**FIXED_PARAMS, **combo}
    with open(TUNING_PARAMS_FILE, "w") as f:
        json.dump(full_params, f, indent=4)


def _run_single_backtest(combo: dict, product: str, output_file: Path, index: int, total: int):
    """Run a single backtest for one parameter combination."""
    _write_params(combo)
    
    # Flush prints to ensure visibility even when redirected
    def qprint(*args):
        print(*args)
        sys.stdout.flush()

    # Clear modules correctly
    import importlib
    to_delete = [m for m in sys.modules if "strategy" in m or "prosperity4bt" in m]
    for m in to_delete:
        del sys.modules[m]

    try:
        from strategy.main import Trader
        from prosperity4bt import run_backtest
        
        combo_str = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        qprint(f"\n[{index+1}/{total}] Starting backtest ({product}): {combo_str}")
        qprint(f"         Output log: {output_file.name}")

        run_backtest(
            trader_class=Trader,
            round_day=ROUND_DAY,
            output_file=str(output_file),
            print_output=False,
            show_progress=False,
        )
        qprint(f"         [OK] Backtest complete.")
        return True, combo, str(output_file)
    except Exception as e:
        qprint(f"  [FAIL] FAILED: {e}")
        return False, combo, str(e)


def _run_single_for_pool(args):
    """Wrapper for multiprocessing pool — takes a single tuple argument."""
    combo, product, output_file, index, total = args
    return _run_single_backtest(combo, product, Path(output_file), index, total)


# ── Execution Modes ──────────────────────────────────────────────────────────

def run_sequential(combos: list[dict], product: str):
    """Run all backtests sequentially with a progress bar."""
    total = len(combos)
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    start = time.time()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Sequential backtests", unit="run")
    except ImportError:
        pbar = None

    for i, combo in enumerate(combos):
        fname = f"{product}--" + _combo_to_filename(combo)
        output_file = BACKTEST_OUTPUT_DIR / fname
        result = _run_single_backtest(combo, product, output_file, i, total)
        results.append(result)
        if pbar:
            pbar.update(1)
        else:
            pct = (i + 1) / total * 100
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  Progress: {i+1}/{total} ({pct:.0f}%) | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    if pbar:
        pbar.close()

    elapsed = time.time() - start
    success = sum(1 for r in results if r[0])
    print(f"\n{'='*60}")
    print(f"Completed {total} backtests in {elapsed:.1f}s")
    print(f"  [OK] Succeeded: {success}")
    print(f"  [FAIL] Failed:    {total - success}")
    print(f"  Logs in: {BACKTEST_OUTPUT_DIR}")
    return results


def run_parallel(combos: list[dict], product: str, max_workers: int = None):
    """Run backtests in parallel using multiprocessing."""
    import multiprocessing as mp

    total = len(combos)
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # cap at 4 to avoid thrashing

    print(f"Starting parallel execution with {max_workers} workers ({product})...")

    # Prepare args for pool
    pool_args = []
    for i, combo in enumerate(combos):
        fname = f"{product}--" + _combo_to_filename(combo)
        output_file = BACKTEST_OUTPUT_DIR / fname
        pool_args.append((combo, product, str(output_file), i, total))

    start = time.time()

    try:
        from tqdm import tqdm
        with mp.Pool(max_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_single_for_pool, pool_args),
                total=total,
                desc="Parallel backtests",
                unit="run",
            ))
    except ImportError:
        # No tqdm — manual progress
        with mp.Pool(max_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(_run_single_for_pool, pool_args)):
                results.append(result)
                pct = (i + 1) / total * 100
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  Progress: {i+1}/{total} ({pct:.0f}%) | "
                      f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    elapsed = time.time() - start
    success = sum(1 for r in results if r[0])
    print(f"\n{'='*60}")
    print(f"Completed {total} backtests in {elapsed:.1f}s")
    print(f"  [OK] Succeeded: {success}")
    print(f"  [FAIL] Failed:    {total - success}")
    print(f"  Logs in: {BACKTEST_OUTPUT_DIR}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tomato Hyperparameter Grid Search")
    parser.add_argument(
        "--mode", choices=["parallel", "sequential"], default="parallel",
        help="Execution mode (default: parallel, falls back to sequential)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: min(cpu_count, 4))",
    )
    parser.add_argument(
        "--product", type=str, default="TOMATOES",
        help="Product name for log tagging (default: TOMATOES)",
    )
    args = parser.parse_args()
    product = args.product

    combos = _build_combinations(PARAM_GRID)
    total = len(combos)
    print(f"{'='*60}")
    print(f"Hyperparameter Grid Search: {product}")
    print(f"{'='*60}")
    print(f"Parameters being swept:")
    for k, v in PARAM_GRID.items():
        print(f"  {k}: {v}")
    print(f"Fixed: {FIXED_PARAMS}")
    print(f"Total combinations: {total}")
    print(f"Mode: {args.mode}")
    print(f"Round/Day: {ROUND_DAY}")
    print(f"{'='*60}")

    if args.mode == "parallel":
        try:
            run_parallel(combos, product, max_workers=args.workers)
        except Exception as e:
            print(f"\n[WARN] Parallel execution failed: {e}")
            print("Falling back to sequential mode...\n")
            run_sequential(combos, product)
    else:
        run_sequential(combos, product)


if __name__ == "__main__":
    main()
