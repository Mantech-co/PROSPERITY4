"""
PnL Extractor for hyperparameter tuning.

Scans all .log files in backtests/tuning_runs/, parses the activitiesLog
to extract the final PnL for a specified product, and decodes the
hyperparameter combination from the filename.

Outputs a CSV: tuning/results_[product].csv
"""

import sys
import os
import csv
import json
import re
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "backtests" / "tuning_runs"
DEFAULT_OUTPUT_TEMPLATE = PROJECT_ROOT / "tuning" / "results_{product}.csv"


def parse_params_from_filename(filename: str) -> dict:
    """
    Extract hyperparameter values from a log filename.
    
    Expected format: [PRODUCT]--alpha=0.500--m_slope=2.log
    Pairs are separated by '--', keys and values by '='.
    """
    stem = Path(filename).stem
    params = {}
    
    # Split on '--'
    parts = stem.split("--")
    
    # First part is product name, ignore for params dict
    for pair in parts[1:]:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        
        # Reverse the shortening
        if key == "st":
            key = "slope_threshold"
        elif key == "th":
            key = "threshold"
        
        try:
            if '.' in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            params[key] = value
    
    return params


def extract_pnl(log_path: Path, target_product: str) -> float | None:
    """Parse a backtest log file and extract the final PnL for the target product."""
    try:
        with open(log_path, "r") as f:
            raw = f.read()
        
        data = json.loads(raw)
        activities_log = data.get("activitiesLog", "")
        
        if not activities_log:
            return None
        
        lines = activities_log.strip().split("\n")
        if len(lines) < 2:
            return None
        
        header = lines[0].split(";")
        pnl_idx = header.index("profit_and_loss")
        product_idx = header.index("product")
        
        last_pnl = None
        for line in lines[1:]:
            fields = line.split(";")
            if len(fields) > max(pnl_idx, product_idx):
                if fields[product_idx] == target_product:
                    try:
                        last_pnl = float(fields[pnl_idx])
                    except (ValueError, IndexError):
                        pass
        
        return last_pnl
    
    except Exception as e:
        print(f"  [WARN] Error parsing {log_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract PnL from tuning log files")
    parser.add_argument(
        "--log-dir", type=str, default=str(DEFAULT_LOG_DIR),
        help=f"Directory containing .log files (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--product", type=str, default="TOMATOES",
        help="Product name to extract PnL for (default: TOMATOES)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: tuning/results_[product].csv)",
    )
    args = parser.parse_args()
    
    product = args.product
    log_dir = Path(args.log_dir)
    output_path = Path(args.output) if args.output else Path(str(DEFAULT_OUTPUT_TEMPLATE).format(product=product.lower()))
    
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        sys.exit(1)
    
    # Filter logs that start with our product name
    log_files = sorted([f for f in log_dir.glob("*.log") if f.name.startswith(f"{product}--")])
    if not log_files:
        print(f"No .log files found for product '{product}' in {log_dir}")
        sys.exit(1)
    
    total = len(log_files)
    print(f"{'='*60}")
    print(f"PnL Extractor: {product}")
    print(f"{'='*60}")
    print(f"Log directory: {log_dir}")
    print(f"Found {total} relevant log files")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    results = []
    start = time.time()
    
    try:
        from tqdm import tqdm
        iterator = tqdm(log_files, desc="Extracting PnL", unit="file")
    except ImportError:
        iterator = log_files
    
    pnl_key = f"{product.lower()}_pnl"
    
    for i, log_path in enumerate(iterator):
        pnl = extract_pnl(log_path, product)
        if pnl is not None:
            params = parse_params_from_filename(log_path.name)
            params[pnl_key] = pnl
            results.append(params)
    
    if not results:
        print("\nNo results extracted. Check log files.")
        sys.exit(1)
    
    # Determine CSV columns
    param_keys = sorted([k for k in results[0].keys() if k != pnl_key])
    fieldnames = param_keys + [pnl_key]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Extraction complete in {elapsed:.1f}s")
    print(f"  Results: {len(results)}/{total} files processed successfully")
    print(f"  Saved to: {output_path}")
    
    # Summary
    pnls = [r[pnl_key] for r in results]
    print(f"\n  PnL Summary for {product}:")
    print(f"    Min:  {min(pnls):.1f}")
    print(f"    Max:  {max(pnls):.1f}")
    print(f"    Mean: {sum(pnls)/len(pnls):.1f}")
    
    best = max(results, key=lambda r: r[pnl_key])
    print(f"\n  Best combination (raw max):")
    for k in param_keys:
        print(f"    {k}: {best[k]}")
    print(f"    PnL: {best[pnl_key]:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
