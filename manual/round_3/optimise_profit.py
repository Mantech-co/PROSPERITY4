import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt

values = np.arange(670, 925, 5)
n = len(values)
cdf_vals = np.arange(1, n + 1) / n

N = 4022
m0 = 836.66  # mean b2 of all other participants

def cdf(x, left_limit=False):
    side = 'left' if left_limit else 'right'
    idx = np.searchsorted(values, x, side=side)
    return cdf_vals[idx - 1] if idx > 0 else 0.0

def effective_m(b2):
    return ((N - 1) * m0 + b2) / N

def profit(b1, b2):
    m = effective_m(b2)
    c1 = cdf(b1, left_limit=True)
    c2 = cdf(b2, left_limit=True)
    fill_ratio = min(1.0, ((920 - m) / (920 - b2)) ** 3) if b2 < 920 else 1.0
    return c1 * (920 - b1) + (c2 - c1) * (920 - b2) * fill_ratio

b1_candidates = values + 0.001
b2_candidates = values

best_profit = -np.inf
best_b1, best_b2 = None, None

for b1 in b1_candidates:
    for b2 in b2_candidates:
        if b2 <= b1:
            continue
        p = profit(b1, b2)
        if p > best_profit:
            best_profit = p
            best_b1, best_b2 = b1, b2

print(f"m0 (others' mean bid) = {m0:.2f}")
print(f"effective m at optimal b2 = {effective_m(best_b2):.4f}")
print(f"Grid search  -> b1 just above {best_b1 - 0.001:.0f}, b2 = {best_b2:.0f}, profit = {best_profit:.4f}")

res = differential_evolution(
    lambda x: -profit(x[0], x[1]),
    bounds=[(670, 919), (671, 920)],
    constraints=NonlinearConstraint(lambda x: x[1] - x[0], 1, np.inf),
    seed=0, tol=1e-8, maxiter=10000,
)
print(f"Continuous   -> b1 = {res.x[0]:.2f}, b2 = {res.x[1]:.2f}, profit = {-res.fun:.4f}")
print(f"effective m at continuous b2 = {effective_m(res.x[1]):.4f}")

# --- profit landscape heatmap ---
Z = np.zeros((len(b1_candidates), len(b2_candidates)))
for i, b1 in enumerate(b1_candidates):
    for j, b2 in enumerate(b2_candidates):
        if b2 > b1:
            Z[i, j] = profit(b1, b2)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(b2_candidates, values, Z, levels=50, cmap="viridis")
plt.colorbar(im, ax=ax, label="Profit")
ax.scatter([best_b2], [best_b1 - 0.001], color="red", zorder=5,
           label=f"Optimal b1>{best_b1-0.001:.0f}, b2={best_b2:.0f}")
ax.set_xlabel("b2")
ax.set_ylabel("b1 step (bid just above)")
ax.set_title(f"Profit Landscape (m endogenous, m0={m0:.1f}, N={N})")
ax.legend()
plt.tight_layout()
plt.savefig("manual/round_3/profit_landscape.png", dpi=150)
plt.show()
