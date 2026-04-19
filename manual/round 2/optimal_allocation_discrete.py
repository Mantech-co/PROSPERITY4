"""
Optimal Research/Scale/Speed allocation given a DISCRETE distribution
of competitor Speed bids.

UNITS — the one thing that bites you every time
------------------------------------------------
Internally, r, s, sp are all in PERCENTAGE POINTS on [0, 100].
Your input can be in either convention; tell the function which:

    units='percent'   bid values are on [0, 100]   e.g. 22.5 means 22.5%
    units='fraction'  bid values are on [0, 1]     e.g. 0.225 means 22.5%

Input formats
-------------
  1. bids — flat list, one entry per competitor
       optimal_allocation(bids=[0.40, 0.35, 0.30], units='fraction')

  2. bid_weights — dict {bid_value: weight}; weights can be fractional
     (probability mass, histogram density, or player counts; all equivalent
     since only ratios matter for ranking)
       optimal_allocation(bid_weights={0.00: 378.9, 0.05: 63.8, 0.10: 110.2, ...},
                          units='fraction')

Mechanics
---------
  * With weighted bids, my rank-fraction collapses to a clean quantile:
        fraction_above = (total weight of competitors with bid > sp) / total_weight
        m(sp)          = 0.9 - 0.8 * fraction_above
    Ties (competitors bidding exactly sp) don't count as "above" — they share
    rank with me, which matches the original spec ("ties share the higher rank").

  * Only a finite set of sp values can ever be optimal:
        {0} ∪ {unique competitor bid levels}
    Any sp strictly between two bid levels pays more for the same quantile.

  * Given sp and its multiplier m, the inner (r, s) split is the Lagrangian
    shape s = (1+r)·ln(1+r) with r + s = 100 - sp. Full investment dominates.
"""

import numpy as np
from scipy.optimize import brentq

LOG101   = np.log(101)
K_GROSS  = 14_000 / LOG101     # Research * Scale gross coeff: K * ln(1+r) * s
PER_PCT  = 500                 # XIRECs per 1 percentage point (50_000 / 100)


# ---------- core helpers ---------- #

def _lagrangian_rs(T):
    """Optimal (r, s) in percentage points when r + s = T. Independent of m."""
    if T <= 1e-9:
        return 0.0, 0.0
    r = brentq(lambda r_: r_ + (1 + r_) * np.log(1 + r_) - T, 0, T)
    return r, T - r


def _pnl_full(sp, m):
    """PnL when spending the full 100% of budget, with given sp% and multiplier m."""
    T = 100.0 - sp
    r, s = _lagrangian_rs(T)
    gross = K_GROSS * m * np.log(1 + r) * s
    return gross - PER_PCT * 100, r, s


# ---------- public API ---------- #

def optimal_allocation(bids=None, bid_weights=None, units='percent', verbose=False):
    """
    Parameters
    ----------
    bids        : list of competitor bid values (one per competitor)
    bid_weights : dict {bid_value: weight}; weights are relative mass (don't
                  need to sum to 1 — only ratios matter)
    units       : 'percent' if bid values are in [0, 100],
                  'fraction' if bid values are in [0, 1]
    verbose     : if True, print the full rank ladder

    Returns
    -------
    dict with keys:
      r, s, sp    : optimal allocations in PERCENTAGE POINTS (0-100)
      pnl         : expected PnL in XIRECs
      quantile    : my rank-fraction (0 = top, 1 = bottom)
      multiplier  : resulting Speed multiplier
      action      : 'invest' or 'walk away'
      ladder      : list of dicts (one per tie-point) for inspection
    """
    if (bids is None) == (bid_weights is None):
        raise ValueError("Provide exactly one of `bids` or `bid_weights`.")
    if units not in ('percent', 'fraction'):
        raise ValueError("units must be 'percent' or 'fraction'.")

    # --- normalize input to (bid_pct, weight) pairs ---
    if bids is not None:
        pairs = [(float(b), 1.0) for b in bids]
    else:
        pairs = [(float(b), float(w)) for b, w in bid_weights.items() if float(w) > 0]

    if units == 'fraction':
        pairs = [(b * 100.0, w) for b, w in pairs]

    # Sanity: warn if bids drift outside [0, 100]
    max_bid = max(b for b, _ in pairs)
    if max_bid > 100.5:
        raise ValueError(
            f"Bid value {max_bid} > 100. Did you mean units='fraction'?"
        )

    total_weight = sum(w for _, w in pairs)
    candidates = sorted({0.0} | {b for b, _ in pairs})

    # --- walk-away baseline (always 0 in this problem, but kept explicit) ---
    best = {
        'r': 0.0, 's': 0.0, 'sp': 0.0, 'pnl': 0.0,
        'quantile': 1.0, 'multiplier': 0.1, 'action': 'walk away',
    }
    ladder = []

    for sp in candidates:
        weight_above = sum(w for b, w in pairs if b > sp)
        fraction_above = weight_above / total_weight        # 0 = top, 1 = bottom
        m = 0.9 - 0.8 * fraction_above
        pnl, r, s = _pnl_full(sp, m)
        ladder.append({
            'sp': round(sp, 3),
            'quantile': round(fraction_above, 3),
            'multiplier': round(m, 3),
            'r': round(r, 3),
            's': round(s, 3),
            'pnl': round(pnl, 0),
        })
        if pnl > best['pnl']:
            best = {
                'r': round(r, 3), 's': round(s, 3), 'sp': round(sp, 3),
                'pnl': round(pnl, 0),
                'quantile': round(fraction_above, 3),
                'multiplier': round(m, 3),
                'action': 'invest',
            }

    best['ladder'] = ladder

    if verbose:
        print(f"Total competitor weight = {total_weight:.3f}")
        print(f"{'sp%':>9} {'q-above':>9} {'m':>7} {'r%':>8} {'s%':>8} {'PnL':>12}")
        for row in ladder:
            mark = "  <-- BEST" if row['sp'] == best['sp'] and best['action'] == 'invest' else ""
            print(f"{row['sp']:>9.3f} {row['quantile']:>9.3f} "
                  f"{row['multiplier']:>7.3f} {row['r']:>8.3f} {row['s']:>8.3f} "
                  f"{row['pnl']:>12,.0f}{mark}")

    return best


# ---------- demo with the user's actual distribution ---------- #

if __name__ == "__main__":
    import csv, os

    csv_path = os.path.join(os.path.dirname(__file__), "distribution_xw.csv")
    weights = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights[float(row["x"])] = float(row["weight"])

    print("=" * 72)
    print(f"Loaded {len(weights)} bins from {csv_path}")
    print("=" * 72)
    result = optimal_allocation(bid_weights=weights, units='fraction', verbose=True)
    print()
    print(f"→ allocate r = {result['r']:.3f}%, s = {result['s']:.3f}%, sp = {result['sp']:.3f}%")
    print(f"→ quantile-above = {result['quantile']:.3f} "
          f"(I beat {(1-result['quantile'])*100:.3f}% of competitor mass)")
    print(f"→ multiplier m = {result['multiplier']:.3f}")
    print(f"→ expected PnL = {result['pnl']:,.0f} XIRECs")
