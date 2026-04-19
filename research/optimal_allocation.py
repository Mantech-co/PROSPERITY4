"""
Optimal Research/Scale/Speed allocation given a continuous distribution
of competitor Speed bids.

Key insight: under a continuous competitor distribution F, my speed multiplier is
    m(sp) = 0.1 + 0.8 * F(sp)
because my rank fraction = my quantile position. The total player count cancels.

The 3D problem (r, s, sp) then collapses to 1D:
  1. For a candidate sp, the inner (r, s) split is fixed by the Lagrangian:
        s = (1+r) * ln(1+r),   r + s = 100 - sp
  2. Search sp to maximize PnL.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

LOG101 = np.log(101)
K_GROSS = 14_000 / LOG101      # Research * Scale = K * ln(1+r) * s
PER_PCT = 500                  # XIRECs per 1% of allocation (50_000 / 100)


def optimal_allocation(pdf, support=(0, 100), verbose=False):
    """
    Given a competitor Speed-bid density `pdf(x)` on the given `support`,
    return the optimal (r%, s%, sp%) allocation and resulting PnL.

    pdf:     callable, density at bid level x (does NOT need to be normalized).
    support: (low, high) domain of x.
    """
    total_mass, _ = quad(pdf, *support, limit=200)

    def cdf(x):
        if x <= support[0]: return 0.0
        if x >= support[1]: return 1.0
        m, _ = quad(pdf, support[0], x, limit=200)
        return m / total_mass

    def speed_mult(sp):
        """m(sp) = 0.1 + 0.8 * F(sp)  — my quantile mapped to [0.1, 0.9]."""
        return 0.1 + 0.8 * cdf(sp)

    def pnl_for_sp(sp):
        """Max PnL over (r, s) for this sp, with r+s = 100-sp (full investment)."""
        T = 100.0 - sp
        if T <= 1e-9:
            return -PER_PCT * sp
        m = speed_mult(sp)
        # Lagrangian interior: r + (1+r)ln(1+r) = T, s = T - r
        r = brentq(lambda r_: r_ + (1 + r_) * np.log(1 + r_) - T, 0, T)
        s = T - r
        gross = K_GROSS * m * np.log(1 + r) * s
        pnl_full  = gross - PER_PCT * 100
        pnl_nothing_rs = -PER_PCT * sp            # invest only sp (sanity floor)
        return max(pnl_full, pnl_nothing_rs)

    # Grid search to avoid local-min traps with weird (e.g. bimodal) distributions
    grid = np.linspace(0, 99.5, 500)
    pnls = np.array([pnl_for_sp(sp) for sp in grid])
    i_best = int(np.argmax(pnls))

    # Refine around the best grid point
    lo = max(0.0, grid[i_best] - (grid[1] - grid[0]))
    hi = min(99.5, grid[i_best] + (grid[1] - grid[0]))
    res = minimize_scalar(lambda sp: -pnl_for_sp(sp), bounds=(lo, hi),
                          method='bounded', options={'xatol': 1e-5})
    best_sp = float(res.x)
    best_pnl = float(-res.fun)

    # Walk-away check (PnL=0 beats everything if best_pnl < 0)
    if best_pnl <= 0:
        return {'r': 0.0, 's': 0.0, 'sp': 0.0, 'pnl': 0.0,
                'multiplier': None, 'quantile': None, 'action': 'walk away'}

    T = 100.0 - best_sp
    r = brentq(lambda r_: r_ + (1 + r_) * np.log(1 + r_) - T, 0, T)
    s = T - r
    return {
        'r':          round(r, 2),
        's':          round(s, 2),
        'sp':         round(best_sp, 2),
        'pnl':        round(best_pnl, 0),
        'multiplier': round(speed_mult(best_sp), 3),
        'quantile':   round(cdf(best_sp), 3),
        'action':     'invest',
    }
