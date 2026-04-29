#!/usr/bin/env python3
"""
Usage:
  python midplot.py "EXPR" [--days 2 3 4] [--roll N]

EXPR uses product names as variables. Examples:
  "PEBBLES_L"
  "PEBBLES_L - PEBBLES_S"
  "PEBBLES_L * PEBBLES_M - PEBBLES_S"
  "COCONUT_COUPON / COCONUT"

Product names with digits or special chars: wrap in quotes isn't needed,
they're substituted as variable names internally.
"""

import argparse
import glob
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "../data"


def load_prices(days: list[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        pattern = str(DATA_DIR / f"prices_round_*_day_{day}.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"warn: no file for day {day}", file=sys.stderr)
            continue
        for f in files:
            df = pd.read_csv(f, sep=";")
            frames.append(df)
    if not frames:
        sys.exit("no data files found")
    return pd.concat(frames, ignore_index=True)


def pivot_mid(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["day", "timestamp", "product", "mid_price"]].copy()
    df["t"] = df["day"].astype(str) + "_" + df["timestamp"].astype(str)
    pivot = df.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price")
    pivot = pivot.sort_index()
    pivot = pivot.ffill()
    return pivot


def find_products(expr: str, available: list[str]) -> list[str]:
    # sort by length desc so longer names match first
    found = []
    for p in sorted(available, key=len, reverse=True):
        if p in expr:
            found.append(p)
    return found


def eval_expr(expr: str, pivot: pd.DataFrame) -> pd.Series:
    products = find_products(expr, list(pivot.columns))
    if not products:
        sys.exit(f"no products found in expression: {expr}")

    # build safe namespace
    ns = {p: pivot[p] for p in products if p in pivot.columns}
    missing = [p for p in products if p not in pivot.columns]
    if missing:
        sys.exit(f"products not in data: {missing}")

    # replace product names with safe variable names
    safe_expr = expr
    mapping = {}
    for p in sorted(products, key=len, reverse=True):
        safe = "__p_" + re.sub(r"\W", "_", p)
        safe_expr = safe_expr.replace(p, safe)
        mapping[safe] = ns[p]

    mapping["np"] = np
    try:
        result = eval(safe_expr, {"__builtins__": {}}, mapping)
    except Exception as e:
        sys.exit(f"eval error: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Plot derived mid-price expression")
    parser.add_argument("expr", help="expression using product names")
    parser.add_argument("--days", nargs="+", type=int, default=None,
                        help="day numbers to include (default: all found)")
    parser.add_argument("--roll", type=int, default=None,
                        help="rolling mean window size")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if args.days is None:
        # auto-detect days
        files = glob.glob(str(DATA_DIR / "prices_round_*_day_*.csv"))
        days = sorted({int(re.search(r"day_(\d+)", f).group(1)) for f in files})
        if not days:
            sys.exit("no price files in data/")
    else:
        days = args.days

    print(f"loading days: {days}")
    df = load_prices(days)
    pivot = pivot_mid(df)
    print(f"products: {sorted(pivot.columns.tolist())}")

    series = eval_expr(args.expr, pivot)

    # x axis: flatten multi-index to sequential integers
    x = np.arange(len(series))
    y = series.values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, y, lw=0.8, alpha=0.6, color="steelblue", label="raw")

    if args.roll:
        rolled = pd.Series(y).rolling(args.roll, min_periods=1).mean().values
        ax.plot(x, rolled, lw=1.5, color="tomato", label=f"roll({args.roll})")
        ax.legend()

    # day boundary lines
    day_col = [d for d, _ in pivot.index]
    boundaries = [0] + [i for i in range(1, len(day_col)) if day_col[i] != day_col[i-1]]
    for b in boundaries[1:]:
        ax.axvline(b, color="gray", lw=0.8, ls="--", alpha=0.5)
    for b, d in zip(boundaries, days):
        ax.text(b + 5, ax.get_ylim()[1], f"day {d}", fontsize=7, color="gray", va="top")

    title = args.title or args.expr
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
