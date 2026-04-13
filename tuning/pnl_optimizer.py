"""
PnL Optimizer for hyperparameter tuning.

Scans the results_[product].csv file and performs an Nth degree polynomial fit
on each parameter to identify the optimal configuration according to the fit.

Features:
- Adjustable polynomial degree (N).
- Comparison between raw best and fit-based best results.
- Robust handling of single-value parameters.
- Progress visualization with tqdm.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_TEMPLATE = PROJECT_ROOT / "tuning" / "results_{product}.csv"


def fit_and_optimize(df, param_name, pnl_col, degree=2):
    """
    Fit an Nth degree polynomial to (param, PnL) and find the optimal point.
    """
    # Group by parameter and calculate mean PnL
    avg_pnl = df.groupby(param_name)[pnl_col].mean().reset_index()
    x = avg_pnl[param_name].values
    y = avg_pnl[pnl_col].values
    
    if len(x) < 2:
        return None, f"{param_name}: only 1 unique value ({x[0]}), skipping fit"
        
    # Limit degree by number of points
    actual_degree = min(degree, len(x) - 1)
    
    # Polynomial fit
    z = np.polyfit(x, y, actual_degree)
    p = np.poly1d(z)
    
    # Find max in range [min(x), max(x)]
    x_range = np.linspace(min(x), max(x), 1000)
    y_range = p(x_range)
    idx_max = np.argmax(y_range)
    
    opt_val = x_range[idx_max]
    opt_pnl = y_range[idx_max]
    
    # Calculate R-squared
    y_pred = p(x)
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "opt_val": opt_val,
        "opt_pnl": opt_pnl,
        "degree": actual_degree,
        "r2": r_squared,
        "x": x.tolist(),
        "y": y.tolist()
    }, None


def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters from extracted PnL data")
    parser.add_argument(
        "--product", type=str, default="TOMATOES",
        help="Product name to optimize for (default: TOMATOES)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to results_[product].csv (default: in tuning/)",
    )
    parser.add_argument(
        "--degree", type=int, default=3,
        help="Degree of the polynomial fit (default: 3)",
    )
    args = parser.parse_args()
    
    product = args.product
    input_path = Path(args.input) if args.input else Path(str(DEFAULT_RESULTS_TEMPLATE).format(product=product.lower()))
    
    if not input_path.exists():
        print(f"Error: Results file not found: {input_path}")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"Error: {input_path} is empty.")
        sys.exit(1)
        
    pnl_col = f"{product.lower()}_pnl"
    if pnl_col not in df.columns:
        print(f"Error: Could not find PnL column '{pnl_col}' in {input_path}")
        sys.exit(1)
        
    param_cols = [c for c in df.columns if c != pnl_col]
    
    print(f"{'='*60}")
    print(f"{product} PnL Optimizer")
    print(f"{'='*60}")
    print(f"Input:       {input_path}")
    print(f"Data points: {len(df)}")
    print(f"Parameters:  {param_cols}")
    print(f"Poly degree: {args.degree}")
    print(f"{'='*60}")
    
    try:
        from tqdm import tqdm
        iterator = tqdm(param_cols, desc="Fitting parameters", unit="param")
    except ImportError:
        iterator = param_cols
        
    results = {}
    warnings = []
    
    for param in iterator:
        res, warn = fit_and_optimize(df, param, pnl_col, degree=args.degree)
        if warn:
            warnings.append(warn)
        else:
            results[param] = res
            
            # Detailed logs for each param
            print(f"\n  == {param} ==")
            print(f"    Unique values tested: {res['x']}")
            # Use ASCII for PnL list to avoid Windows encoding issues
            print(f"    Avg PnL at each:      {[f'{val:.1f}' for val in res['y']]}")
            print(f"    Poly degree (actual): {res['degree']}")
            print(f"    R^2:                  {res['r2']:.4f}")
            print(f"    Optimal (from fit):    {res['opt_val']:.4f}  (predicted PnL: {res['opt_pnl']:.1f})")

    # Final recommendation
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION RESULTS: {product} (degree={args.degree})")
    print(f"{'='*60}")
    
    # Recommendation based on Poly Fit
    recommendation = {}
    for param in param_cols:
        if param in results:
            recommendation[param] = results[param]["opt_val"]
        else:
            # Fallback to the mean value for single-value params
            recommendation[param] = df[param].mean()
            
    # Raw Best
    raw_best_idx = df[pnl_col].idxmax()
    raw_best_row = df.loc[raw_best_idx]
    
    # Printing Comparison Table
    print(f"{'Parameter':>20}  {'Poly Fit':>12}  {'Raw Best':>12}")
    print(f"{'-'*20}  {'-'*12}  {'-'*12}")
    for param in param_cols:
        print(f"{param:>20}  {recommendation[param]:>12.4f}  {raw_best_row[param]:>12.4f}")
    
    print(f"{'PnL':>20}  {' (estimated)':>12}  {raw_best_row[pnl_col]:>12.1f}")
    
    if warnings:
        print(f"\nNotes:")
        for w in warnings:
            print(f"  [WARN] {w}")
            
    print(f"\nCompleted optimization for {product}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
