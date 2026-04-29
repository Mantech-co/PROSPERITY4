#!/usr/bin/env python3
"""
Usage:
  python overlayplot.py "EXPR1" "EXPR2" [--days 2 3 4] [--roll N] [--auto-lag]
                        [--max-offset N] [--invert VALUE]

Overlay two arbitrary mid-price expressions with interactive sliders.

Free variables:
  f1   - scalar in EXPR1, gets its own slider
  f2   - scalar in EXPR2, gets its own slider

Overlay sliders (always present):
  offset - shift EXPR2 in time (samples); range controlled by --max-offset
  scale  - multiply EXPR2 amplitude
  bias   - add constant to EXPR2

Options:
  --days 2 3 4     load specific days (default: all days found in data/)
  --roll N         overlay rolling mean with window N
  --auto-lag       initialise offset slider to peak cross-correlation lag
  --max-offset N   half-range of offset slider and cross-corr panel (default: 2000)
  --invert VALUE   invert expr2 about VALUE before overlaying; use 'mean' for nanmean

Examples:
  "PEBBLES_L - f1 * PEBBLES_S"  "KELP"
  "COCONUT_COUPON / COCONUT"     "np.log(SQUID_INK)"
  "PEBBLES_L * f1 - PEBBLES_M"  "PEBBLES_S * f2"
  "KELP" "SQUID_INK" --max-offset 5000 --auto-lag
"""

import argparse
import glob
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from pathlib import Path
from scipy.signal import correlate

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
    pivot = df.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price")
    pivot = pivot.sort_index()
    pivot = pivot.ffill()
    return pivot


def find_products(expr: str, available: list[str]) -> list[str]:
    found = []
    for p in sorted(available, key=len, reverse=True):
        if p in expr:
            found.append(p)
    return found


def has_free_var(expr: str, products: list[str], varname: str) -> bool:
    test = expr
    for p in sorted(products, key=len, reverse=True):
        test = test.replace(p, "")
    return bool(re.search(rf'\b{re.escape(varname)}\b', test))


def eval_expr(expr: str, pivot: pd.DataFrame, f1: float = 1.0, f2: float = 1.0) -> np.ndarray:
    products = find_products(expr, list(pivot.columns))
    if not products:
        sys.exit(f"no products found in expression: {expr}")

    missing = [p for p in products if p not in pivot.columns]
    if missing:
        sys.exit(f"products not in data: {missing}")

    safe_expr = expr
    mapping = {}
    for p in sorted(products, key=len, reverse=True):
        safe = "__p_" + re.sub(r"\W", "_", p)
        safe_expr = safe_expr.replace(p, safe)
        mapping[safe] = pivot[p].values.astype(float)

    mapping["np"] = np
    mapping["f1"] = f1
    mapping["f2"] = f2
    try:
        result = eval(safe_expr, {"__builtins__": {}}, mapping)
    except Exception as e:
        sys.exit(f"eval error: {e}")
    if isinstance(result, pd.Series):
        return result.values.astype(float)
    return np.asarray(result, dtype=float)


def apply_offset(y: np.ndarray, offset: int) -> np.ndarray:
    if offset == 0:
        return y.copy()
    shifted = np.full_like(y, np.nan)
    if offset > 0:
        shifted[offset:] = y[:-offset]
    else:
        shifted[:offset] = y[-offset:]
    return shifted


def best_lag(y1: np.ndarray, y2: np.ndarray, max_lag: int = 500) -> int:
    a = np.nan_to_num(y1 - np.nanmean(y1))
    b = np.nan_to_num(y2 - np.nanmean(y2))
    corr = correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    mask = np.abs(lags) <= max_lag
    best = lags[mask][np.argmax(corr[mask])]
    return int(best)


def main():
    parser = argparse.ArgumentParser(description="Overlay two mid-price expressions")
    parser.add_argument("expr1", help="first expression (may use f1)")
    parser.add_argument("expr2", help="second expression (may use f2)")
    parser.add_argument("--days", nargs="+", type=int, default=None)
    parser.add_argument("--roll", type=int, default=None)
    parser.add_argument("--max-offset", type=int, default=2000)
    parser.add_argument("--invert", default=None, metavar="VALUE",
                        help="invert expr2 about VALUE before overlaying: y = 2*VALUE - y  (use 'mean' for nanmean of expr2)")
    parser.add_argument("--auto-lag", action="store_true",
                        help="init offset to peak cross-correlation lag")
    args = parser.parse_args()

    if args.days is None:
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

    products1 = find_products(args.expr1, list(pivot.columns))
    products2 = find_products(args.expr2, list(pivot.columns))
    use_f1 = has_free_var(args.expr1, products1, "f1")
    use_f2 = has_free_var(args.expr2, products2, "f2")

    state = {"f1": 1.0, "f2": 1.0}

    y1 = eval_expr(args.expr1, pivot, f1=state["f1"], f2=state["f2"])
    y2 = eval_expr(args.expr2, pivot, f1=state["f1"], f2=state["f2"])
    x = np.arange(len(y1))

    std1 = np.nanstd(y1)
    std2 = np.nanstd(y2)
    init_scale = std1 / std2 if std2 != 0 else 1.0
    init_bias  = np.nanmean(y1) - init_scale * np.nanmean(y2)
    init_offset = best_lag(y1, y2, args.max_offset) if args.auto_lag else 0
    if args.auto_lag:
        print(f"auto lag: {init_offset}")

    # layout: how many slider rows we need
    n_extra = int(use_f1) + int(use_f2)
    # 3 overlay sliders + n_extra expression sliders + reset button row
    slider_rows = 3 + n_extra
    bottom_margin = 0.06 + slider_rows * 0.055
    fig, (ax_main, ax_corr) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(bottom=bottom_margin, hspace=0.35)

    if args.invert is None:
        invert_val = None
    elif args.invert.lower() == "mean":
        invert_val = np.nanmean(y2)
    else:
        try:
            invert_val = float(args.invert)
        except ValueError:
            sys.exit(f"--invert: expected a number or 'mean', got '{args.invert}'")

    def make_y2t(offset, scale, bias):
        base = 2 * invert_val - y2 if invert_val is not None else y2
        return apply_offset(base, int(round(offset))) * scale + bias

    y2t = make_y2t(init_offset, init_scale, init_bias)

    (line1,) = ax_main.plot(x, y1,  lw=0.9, color="steelblue", label=args.expr1, alpha=0.85)
    (line2,) = ax_main.plot(x, y2t, lw=0.9, color="tomato",    label=args.expr2, alpha=0.85)
    line1_roll = line2_roll = None
    if args.roll:
        r1 = pd.Series(y1).rolling(args.roll, min_periods=1).mean().values
        r2 = pd.Series(y2t).rolling(args.roll, min_periods=1).mean().values
        (line1_roll,) = ax_main.plot(x, r1, lw=1.8, color="steelblue", alpha=0.45, ls="--")
        (line2_roll,) = ax_main.plot(x, r2, lw=1.8, color="tomato",    alpha=0.45, ls="--")

    ax_main.legend(fontsize=8)
    ax_main.set_title(f"{args.expr1}   vs   {args.expr2}")
    ax_main.set_xlabel("timestep")

    day_col = [d for d, _ in pivot.index]
    boundaries = [0] + [i for i in range(1, len(day_col)) if day_col[i] != day_col[i - 1]]
    for b in boundaries[1:]:
        ax_main.axvline(b, color="gray", lw=0.7, ls="--", alpha=0.4)
    for b, d in zip(boundaries, days):
        ax_main.text(b + 5, ax_main.get_ylim()[1], f"day {d}", fontsize=7, color="gray", va="top")

    # cross-correlation panel
    max_lag = args.max_offset
    lags = np.arange(-max_lag, max_lag + 1)
    y2_base = 2 * invert_val - y2 if invert_val is not None else y2
    a = np.nan_to_num(y1 - np.nanmean(y1))
    b = np.nan_to_num(y2_base - np.nanmean(y2_base))
    corr_full = correlate(a, b, mode="full")
    center = len(a) - 1
    corr_slice = corr_full[center - max_lag: center + max_lag + 1]
    corr_norm = corr_slice / (np.max(np.abs(corr_slice)) + 1e-12)
    (line_corr,) = ax_corr.plot(lags, corr_norm, lw=0.9, color="purple")
    (vline_corr,) = ax_corr.plot([init_offset, init_offset], [-1.1, 1.1], color="red", lw=1.2, ls="--")
    ax_corr.set_xlim(-max_lag, max_lag)
    ax_corr.set_ylim(-1.15, 1.15)
    ax_corr.set_title("cross-correlation  (normalized)", fontsize=9)
    ax_corr.set_xlabel("lag  →  expr2 shifted right by N samples")
    ax_corr.axhline(0, color="gray", lw=0.5)

    stats_text = ax_main.text(
        0.01, 0.01, "", transform=ax_main.transAxes,
        fontsize=8, va="bottom", family="monospace",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5)
    )

    def update_stats(offset, scale, bias):
        y2t_ = make_y2t(offset, scale, bias)
        diff = y1 - y2t_
        valid = ~np.isnan(diff)
        if valid.sum() < 2:
            stats_text.set_text("insufficient data")
            return
        corr_val = np.corrcoef(y1[valid], y2t_[valid])[0, 1]
        rmse = np.sqrt(np.nanmean(diff ** 2))
        f_str = ""
        if use_f1: f_str += f"  f1={state['f1']:.4f}"
        if use_f2: f_str += f"  f2={state['f2']:.4f}"
        stats_text.set_text(
            f"offset={int(round(offset)):+d}  scale={scale:.4f}  bias={bias:.2f}{f_str}\n"
            f"corr={corr_val:.4f}  rmse={rmse:.4f}"
        )

    update_stats(init_offset, init_scale, init_bias)

    # build sliders bottom-up
    row = 0
    def slider_axes(row):
        bottom = 0.025 + row * 0.055
        return fig.add_axes([0.12, bottom, 0.75, 0.025])

    ax_bi  = slider_axes(row); row += 1
    ax_sc  = slider_axes(row); row += 1
    ax_off = slider_axes(row); row += 1

    sl_bias   = mwidgets.Slider(ax_bi,  "bias  (expr2)",
                                 init_bias - 3 * std1, init_bias + 3 * std1,
                                 valinit=init_bias)
    sl_scale  = mwidgets.Slider(ax_sc,  "scale (expr2)",
                                 -5.0, 5.0, valinit=init_scale, valstep=0.0001)
    sl_offset = mwidgets.Slider(ax_off, "offset (expr2)",
                                 -max_lag, max_lag, valinit=init_offset, valstep=1)

    sl_f1 = sl_f2 = None
    if use_f1:
        ax_f1 = slider_axes(row); row += 1
        sl_f1 = mwidgets.Slider(ax_f1, "f1  (expr1)", 0.0, 5.0, valinit=1.0, valstep=0.001)
    if use_f2:
        ax_f2 = slider_axes(row); row += 1
        sl_f2 = mwidgets.Slider(ax_f2, "f2  (expr2)", 0.0, 5.0, valinit=1.0, valstep=0.001)

    def recompute_corr():
        b_raw = 2 * invert_val - y2 if invert_val is not None else y2
        a_ = np.nan_to_num(y1 - np.nanmean(y1))
        b_ = np.nan_to_num(b_raw - np.nanmean(b_raw))
        cf = correlate(a_, b_, mode="full")
        cs = cf[center - max_lag: center + max_lag + 1]
        cn = cs / (np.max(np.abs(cs)) + 1e-12)
        line_corr.set_ydata(cn)

    def on_overlay(_):
        off = sl_offset.val
        sc  = sl_scale.val
        bi  = sl_bias.val
        y2t_ = make_y2t(off, sc, bi)
        line2.set_ydata(y2t_)
        if line2_roll is not None:
            line2_roll.set_ydata(pd.Series(y2t_).rolling(args.roll, min_periods=1).mean().values)
        vline_corr.set_xdata([off, off])
        ax_main.relim(); ax_main.autoscale_view()
        update_stats(off, sc, bi)
        fig.canvas.draw_idle()

    def on_f_change(_):
        nonlocal y1, y2, std1, std2
        if sl_f1 is not None:
            state["f1"] = sl_f1.val
        if sl_f2 is not None:
            state["f2"] = sl_f2.val
        y1 = eval_expr(args.expr1, pivot, f1=state["f1"], f2=state["f2"])
        y2 = eval_expr(args.expr2, pivot, f1=state["f1"], f2=state["f2"])
        std1 = np.nanstd(y1)
        std2 = np.nanstd(y2)
        line1.set_ydata(y1)
        if line1_roll is not None:
            line1_roll.set_ydata(pd.Series(y1).rolling(args.roll, min_periods=1).mean().values)
        recompute_corr()
        on_overlay(None)

    sl_offset.on_changed(on_overlay)
    sl_scale.on_changed(on_overlay)
    sl_bias.on_changed(on_overlay)
    if sl_f1 is not None: sl_f1.on_changed(on_f_change)
    if sl_f2 is not None: sl_f2.on_changed(on_f_change)

    ax_reset = fig.add_axes([0.87, 0.025 + row * 0.055, 0.08, 0.03])
    btn_reset = mwidgets.Button(ax_reset, "Reset")

    def on_reset(_):
        sl_offset.set_val(0)
        sl_scale.set_val(init_scale)
        sl_bias.set_val(init_bias)
        if sl_f1 is not None: sl_f1.set_val(1.0)
        if sl_f2 is not None: sl_f2.set_val(1.0)

    btn_reset.on_clicked(on_reset)
    fig._widgets = [sl_offset, sl_scale, sl_bias, sl_f1, sl_f2, btn_reset]

    plt.show()


if __name__ == "__main__":
    main()
