"""
Maximize  g(x, y, m) = (920 - y) * f(y) + (x - y) * f(x) * min(1, (920-m)^3 / (920-x)^3)
over integers x, y in [670, 920], for m varying from 670 to 925 in steps of 0.1.

f(x) = CDF(x - 1), where the CDF is the empirical CDF of values 670, 675, ..., 920.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Build the empirical CDF
# ---------------------------------------------------------------------------
values = np.arange(670, 925, 5)          # 670, 675, ..., 920
n = len(values)
cdf = np.arange(1, n + 1) / n            # 1/n, 2/n, ..., 1.0

def f(t):
    """Empirical CDF evaluated at t-1: F(t-1) = (# values <= t-1) / n.
    Vectorized: works for scalar or numpy array input."""
    t = np.asarray(t)
    # count of values <= (t - 1)
    counts = np.searchsorted(values, t - 1, side='right')
    return counts / n

# ---------------------------------------------------------------------------
# 2. Pre-compute f on the integer grid [670, 920]
# ---------------------------------------------------------------------------
xs = np.arange(670, 921)                 # integer grid for x and y
fx = f(xs)                               # f(x) for every candidate

# Pre-build matrices that don't depend on m:
#   A[i, j] = (920 - y_j) * f(y_j)        -> depends only on y (column)
#   B[i, j] = (x_i - y_j) * f(x_i)        -> depends on both
X, Y = np.meshgrid(xs, xs, indexing='ij')          # X[i,j]=xs[i], Y[i,j]=xs[j]
FX = fx[:, None] * np.ones_like(X, dtype=float)    # f(x_i) broadcast
FY = fx[None, :] * np.ones_like(X, dtype=float)    # f(y_j) broadcast

term_y = (920 - Y) * FY                  # constant across m
diff   = (X - Y) * FX                    # constant across m
denom  = (920 - X).astype(float) ** 3    # (920 - x)^3, has zeros when x = 920

# ---------------------------------------------------------------------------
# 3. Sweep m and find the optimum at each m
# ---------------------------------------------------------------------------
ms = np.round(np.arange(670.0, 925.0 + 1e-9, 0.1), 1)

best_x   = np.empty_like(ms)
best_y   = np.empty_like(ms)
best_obj = np.empty_like(ms)

# Helper: safely compute min(1, num/denom) elementwise.
# When denom == 0 (only at x = 920), the ratio is undefined; the standard reading
# is that the multiplier saturates at 1 there (no down-weighting).
def multiplier(num, denom_):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(denom_ == 0, np.inf, num / denom_)
    # min(1, ratio) — but allow it to go negative when m > 920 (num < 0)
    return np.minimum(1.0, ratio)

for k, m in enumerate(ms):
    num = (920 - m) ** 3
    mult = multiplier(num, denom)                # shape (251, 251) via broadcasting on x
    # mult depends only on x, so it's actually a row vector along axis 0
    # but denom is shape (251,251) already since X is. Fine.
    G = term_y + diff * mult
    idx = np.unravel_index(np.argmax(G), G.shape)
    best_x[k]   = xs[idx[0]]
    best_y[k]   = xs[idx[1]]
    best_obj[k] = G[idx]

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(ms, best_obj, color='#1f77b4', lw=1.4)
axes[0].set_ylabel('max objective')
axes[0].set_title('Optimal objective vs m')
axes[0].grid(alpha=0.3)

axes[1].plot(ms, best_x, color='#d62728', lw=1.2, label='x*')
axes[1].set_ylabel('x*')
axes[1].grid(alpha=0.3)

axes[2].plot(ms, best_y, color='#2ca02c', lw=1.2, label='y*')
axes[2].set_ylabel('y*')
axes[2].set_xlabel('m')
axes[2].grid(alpha=0.3)

plt.tight_layout()
out_path = './optimum_vs_m.png'
plt.savefig(out_path, dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# 5. Summary print
# ---------------------------------------------------------------------------
print(f"m range: {ms[0]:.1f} -> {ms[-1]:.1f}, step 0.1, {len(ms)} values")
print(f"x,y grid: integers in [{xs[0]}, {xs[-1]}]  ({len(xs)} each, {len(xs)**2} pairs)")
print()
print("Sample of results (every 25th m):")
print(f"{'m':>7} | {'x*':>4} | {'y*':>4} | {'objective':>12}")
print("-" * 38)
for k in range(0, len(ms), 250):
    print(f"{ms[k]:7.1f} | {int(best_x[k]):4d} | {int(best_y[k]):4d} | {best_obj[k]:12.4f}")
# also print global maximum across all m
kmax = int(np.argmax(best_obj))
print()
print(f"Global maximum over the m-sweep: m={ms[kmax]:.1f}, x*={int(best_x[kmax])}, "
      f"y*={int(best_y[kmax])}, obj={best_obj[kmax]:.4f}")
print(f"\nPlot saved to: {out_path}")