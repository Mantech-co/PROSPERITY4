import numpy as np
import matplotlib.pyplot as plt
import distribution

def research(x): 
    return 200_000 * np.log(1 + x) / np.log(1 + 100)


def scale(x):
    return 7*(x/100)

_i_vals = np.arange(1, 10001) / 100
_speed_func_vals = distribution.mixture_cdf(_i_vals) * 0.8 + 0.1

def speed_func(i_hundredths_idx):
    return _speed_func_vals[i_hundredths_idx - 1]
"""
Transcendental equation solver for:

    (t - x) / (1 + x) = ln(1 + x)

where t is a real parameter and x > -1 is the unknown.

Equivalently, find the root of:
    f(x; t) = (t - x)/(1 + x) - ln(1 + x) = 0
"""

import math
from scipy.optimize import brentq, newton


# ---------------------------------------------------------------------------
# Core equation
# ---------------------------------------------------------------------------
def f(x, t):
    """Residual of the equation. Root => equation is satisfied."""
    if x <= -1.0:
        raise ValueError("x must be > -1 (domain of ln(1+x)).")
    return (t - x) / (1.0 + x) - math.log1p(x)


def fprime(x, t):
    """Analytic derivative: f'(x) = -(2 + t + x) / (1+x)^2."""
    return -(2.0 + t + x) / (1.0 + x) ** 2


# ---------------------------------------------------------------------------
# Root-bracketing
# ---------------------------------------------------------------------------
def bracket_root(t, x0=0.0, step=1.0, max_expand=200):
    """
    Find an interval [a, b] (with a > -1) such that f changes sign.

    Strategy: start near x=0 and expand geometrically outward. The function
    is monotone decreasing (for generic t), with f -> +inf as x -> -1+
    and f -> -inf as x -> +inf, so a sign change is guaranteed.
    """
    # Keep `a` safely inside the domain (a > -1).
    a = max(x0 - step, -1.0 + 1e-12)
    b = x0 + step

    fa = f(a, t)
    fb = f(b, t)

    for _ in range(max_expand):
        if fa * fb < 0.0:
            return a, b

        # Expand whichever side hasn't flipped sign yet.
        if fa > 0.0:  # need to push b further right to find negative f
            b += step
            fb = f(b, t)
        else:  # fa < 0: push a further left (toward -1)
            a = max(-1.0 + (1.0 + a) * 0.5, -1.0 + 1e-15)
            fa = f(a, t)

        step *= 1.5  # geometric expansion

    raise RuntimeError(f"Could not bracket a root for t = {t}")


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def solve(t, method="brentq", tol=1e-12, x0=None):
    """
    Solve (t - x)/(1+x) = ln(1+x) for x.

    Parameters
    ----------
    t      : float   - the parameter in the equation
    method : str     - 'brentq' (robust bracketing) or 'newton' (needs x0)
    tol    : float   - convergence tolerance
    x0     : float   - initial guess (for Newton); ignored by brentq

    Returns
    -------
    x : float - the root
    """
    if method == "brentq":
        a, b = bracket_root(t)
        return brentq(f, a, b, args=(t,), xtol=tol, rtol=tol)

    elif method == "newton":
        if x0 is None:
            x0 = 0.0 if t >= 0 else -0.5
        return newton(f, x0, fprime=fprime, args=(t,), tol=tol, maxiter=100)

    else:
        raise ValueError(f"Unknown method: {method!r}")

speed = []
best_pnls = []
worst_pnls = []
risktoreward = []
prediction = []

for i in range(1, 10001):
    iv = i / 100
    x = solve(iv)
    worst_pnl = 0.1*research(x)*scale(iv-x) - 50000
    best_pnl = 0.9*research(x)*scale(iv-x) - 50000
    predicted = speed_func(100-i)*research(x)*scale(iv-x) - 50000

    speed.append(100-iv)
    best_pnls.append(best_pnl)
    worst_pnls.append(worst_pnl)
    risktoreward.append(max(0, predicted/(-1*worst_pnl)))
    prediction.append(predicted)

print(speed_func(27)*scale(73-solve(73))*research(solve(73))-50000)

speed_arr = np.array(speed)
best_arr = np.array(best_pnls)
worst_arr = np.array(worst_pnls)

def zero_crossing(x, y):
    idx = np.where(np.diff(np.sign(y)))[0][0]
    return x[idx] + (x[idx+1] - x[idx]) * (-y[idx] / (y[idx+1] - y[idx]))

best_zero = zero_crossing(speed_arr, best_arr)
worst_zero = zero_crossing(speed_arr, worst_arr)

rr_arr = np.array(risktoreward)
rr_zero = zero_crossing(speed_arr, rr_arr)
rr_max_x = speed_arr[np.argmax(rr_arr)]

h_val = worst_arr[speed_arr == 0][0]

diff = best_arr - h_val
idx = np.where(np.diff(np.sign(diff)))[0][0]
x_h_best = speed_arr[idx] + (speed_arr[idx+1] - speed_arr[idx]) * (-diff[idx] / (diff[idx+1] - diff[idx]))

pred_arr = np.array(prediction)
prediction_zero = zero_crossing(speed_arr, pred_arr)
pred_max_idx = np.argmax(pred_arr)
pred_max_x = speed_arr[pred_max_idx]
pred_max_y = pred_arr[pred_max_idx]

fig, ax1 = plt.subplots()
ax1.plot(speed, best_pnls, label="Best PnL")
ax1.plot(speed, worst_pnls, label="Worst PnL")
ax1.plot(speed, prediction, label="Predicted PnL")
ax1.axhline(y=0, color='r', linestyle='-')
ax1.set_ylabel("PnL")
ax1.axhline(y=h_val, color='gray', linestyle='--', label=f"Worst PnL @ speed=0")
ax1.axvline(x=x_h_best, color='gray', linestyle='-', label="Best PnL = Worst PnL @ speed=0")
ax1.axvline(x=best_zero, color='red', linestyle='--', label="Best PnL = 0")
ax1.axvline(x=worst_zero, color='red', linestyle=':', label="Worst PnL = 0")
ax1.axvline(x=rr_max_x, color='green', linestyle='--', label="Max Risk/Reward")
ax1.axvline(x=rr_zero, color='hotpink', linestyle='-', label="Risk/Reward = 0")
ax1.axvline(x=pred_max_x, color='darkorange', linestyle='--', label='Max Predicted PnL')
ax1.annotate(f"({pred_max_x:.2f}, {pred_max_y:.0f})",
             xy=(pred_max_x, pred_max_y),
             xytext=(pred_max_x + 2, pred_max_y - 5000),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
x_ints = np.arange(25, 44)
y_ints = np.interp(x_ints, speed_arr[::-1], pred_arr[::-1])
ax1.scatter(x_ints, y_ints, color='gray', zorder=5, s=30, label='Integer speed 25–43')
print(f"Best PnL = 0:              speed={best_zero:.4f}")
print(f"Worst PnL = 0:             speed={worst_zero:.4f}")
print(f"Best PnL = Worst@speed=0:  speed={x_h_best:.4f}")
print(f"Predicted PnL = 0:         speed={prediction_zero:.4f}")
print(f"Max Predicted PnL:         speed={pred_max_x:.4f}")
ax1.legend()
plt.show()

# ── Reverse propagation ──────────────────────────────────────────────────────
target_pnl = h_val  # PnL at speed=0: multiplier=0.1 (worst case)

product_arr = (worst_arr + 50000) / 0.1          # research(x) * scale(iv-x) at each point
required_mult_arr = (target_pnl + 50000) / product_arr
required_rank_arr = (required_mult_arr - 0.1) / 0.8

# predicted PnL at worst_zero: product=500000 there (worst_pnl=0), multiplier from loop formula
# speed_func(100-i) at speed s = mixture_cdf(1+s)*0.8+0.1 (valid for s < 99)
mult_at_worst_zero = distribution.mixture_cdf(1 + worst_zero) * 0.8 + 0.1
target_pnl2 = mult_at_worst_zero * 500000 - 50000
required_mult_arr2 = (target_pnl2 + 50000) / product_arr
required_rank_arr2 = (required_mult_arr2 - 0.1) / 0.8

x_dist = np.linspace(0, 100, 4001)
cdf_vals = distribution.mixture_cdf(x_dist)
pdf_vals = distribution.mixture_pdf(x_dist)

fig2, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 10))

ax_pdf.plot(x_dist, pdf_vals, color='black', lw=2, label='Mixture PDF')
ax_pdf.set_xlim(0, 100); ax_pdf.set_ylim(bottom=0)
ax_pdf.set_xlabel('Speed'); ax_pdf.set_ylabel('Density')
ax_pdf.set_title('Bid Distribution PDF')
ax_pdf.grid(True, alpha=0.3); ax_pdf.legend()

ax_cdf.plot(x_dist, cdf_vals, color='black', lw=2, label='Mixture CDF')
ax_cdf.plot(speed_arr, required_rank_arr, color='red', lw=2,
            label=f'Required Rank — Target PnL = {target_pnl:.0f} (worst @ speed=0)')
ax_cdf.plot(speed_arr, required_rank_arr2, color='blue', lw=2, linestyle='--',
            label=f'Required Rank — Target PnL = {target_pnl2:.0f} (predicted @ worst_zero={worst_zero:.2f})')
ax_cdf.axhline(0, color='gray', lw=0.8, linestyle='--')
ax_cdf.axhline(1, color='gray', lw=0.8, linestyle='--')
ax_cdf.set_xlim(0, 100); ax_cdf.set_ylim(-0.05, 1.1)
ax_cdf.set_xlabel('Speed'); ax_cdf.set_ylabel('CDF / Required Rank')
ax_cdf.set_title('Mixture CDF vs Required Rank')
ax_cdf.grid(True, alpha=0.3); ax_cdf.legend()

plt.tight_layout()
plt.show()

# ── Contour plot: Target PnL as parameter ────────────────────────────────────
s_grid = np.linspace(0, 100, 600)
r_grid = np.linspace(0, 1, 400)
S, R = np.meshgrid(s_grid, r_grid)

prod_interp = np.interp(s_grid, speed_arr[::-1], product_arr[::-1])
PROD = np.tile(prod_interp, (len(r_grid), 1))
Z = (R * 0.8 + 0.1) * PROD - 50000

# ── Locus of maxima across contours ──────────────────────────────────────────
# For each contour (fixed p): maximise mixture_cdf(s) - r(s) over s.
# FOC: mixture_pdf(s) + (p+50000)/0.8 * product'(s)/product(s)^2 = 0
# Eliminating p gives r_locus(s) = (-pdf(s)*product(s)/product'(s) - 0.1) / 0.8
sp = speed_arr[::-1]
pr = product_arr[::-1]
prod_deriv = np.gradient(pr, sp)              # d(product)/d(speed) — negative

r_locus = (-distribution.mixture_pdf(sp) * pr / prod_deriv - 0.1) / 0.8
valid = (r_locus >= 0) & (r_locus <= 1)
sp_v, r_v = sp[valid], r_locus[valid]

# ── Contour plot: Target PnL as parameter ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(13, 7))
levels = np.linspace(Z.min(), Z.max(), 40)
cf = ax3.contourf(S, R, Z, levels=levels, cmap='RdYlGn')
cs = ax3.contour(S, R, Z, levels=20, colors='k', linewidths=0.3, alpha=0.5)
plt.colorbar(cf, ax=ax3, label='Target PnL')
ax3.clabel(cs, inline=True, fontsize=6, fmt='%.0f')

ax3.plot(x_dist, cdf_vals, color='white', lw=2.5, label='Mixture CDF')
ax3.plot(speed_arr, required_rank_arr, color='red', lw=2,
         label=f'Target PnL = {target_pnl:.0f} (worst @ speed=0)')
ax3.plot(speed_arr, required_rank_arr2, color='blue', lw=2, linestyle='--',
         label=f'Target PnL = {target_pnl2:.0f} (predicted @ worst_zero={worst_zero:.2f})')
ax3.plot(sp_v, r_v, color='magenta', lw=2.5, label='Locus of max (mixture_cdf − contour y)')

ax3.set_xlim(0, 100); ax3.set_ylim(0, 1)
ax3.set_xlabel('Speed'); ax3.set_ylabel('Required Rank (CDF)')
ax3.set_title('Contour: Target PnL as a function of Speed & Required Rank')
ax3.legend()
plt.tight_layout()
plt.show()

# ── Contour plot: mixture_cdf(x) - y ─────────────────────────────────────────
cdf_at_s = distribution.mixture_cdf(s_grid)
CDF_S = np.tile(cdf_at_s, (len(r_grid), 1))
Z2 = CDF_S - R

fig4, ax4 = plt.subplots(figsize=(13, 7))
levels2 = np.linspace(-1, 1, 41)
cf2 = ax4.contourf(S, R, Z2, levels=levels2, cmap='RdYlGn')
cs2 = ax4.contour(S, R, Z2, levels=levels2, colors='k', linewidths=0.3, alpha=0.5)
plt.colorbar(cf2, ax=ax4, label='mixture_cdf(x) − y')
ax4.clabel(cs2, inline=True, fontsize=6, fmt='%.2f')

ax4.plot(x_dist, cdf_vals, color='white', lw=2.5, label='Mixture CDF (zero line of this plot)')
ax4.plot(speed_arr, required_rank_arr, color='red', lw=2,
         label=f'Target PnL = {target_pnl:.0f}')
ax4.plot(speed_arr, required_rank_arr2, color='blue', lw=2, linestyle='--',
         label=f'Target PnL = {target_pnl2:.0f}')
ax4.plot(sp_v, r_v, color='magenta', lw=2.5, label='Locus of max (mixture_cdf − contour y)')

ax4.set_xlim(0, 100); ax4.set_ylim(0, 1)
ax4.set_xlabel('Speed'); ax4.set_ylabel('y (rank)')
ax4.set_title('Contour: mixture_cdf(x) − y  [positive = natural rank exceeds y]')
ax4.legend()
plt.tight_layout()
plt.show()

