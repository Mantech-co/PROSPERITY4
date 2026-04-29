import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

R1, R2 = 2.0, 0.5  # returns for product 1 and 2

a1 = np.linspace(0, 1, 300)
a2 = np.linspace(0, 1, 300)
A1, A2 = np.meshgrid(a1, a2)

PNL = A1 * abs(R1) - A1**2 + A2 * abs(R2) - A2**2

# mask infeasible region (a1 + a2 > 1)
PNL[A1 + A2 > 1] = np.nan

# optimal point
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from portfolio_optimizer import optimize_portfolio
allocs, dirs, pnl = optimize_portfolio({"P1": R1, "P2": R2})
a1_opt, a2_opt = allocs["P1"], allocs["P2"]
pnl_opt = a1_opt * abs(R1) - a1_opt**2 + a2_opt * abs(R2) - a2_opt**2

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(A1, A2, PNL, cmap=cm.viridis, alpha=0.85, linewidth=0)
ax.scatter(a1_opt, a2_opt, pnl_opt, color="red", s=80, zorder=10, label=f"Optimum ({a1_opt*100:.1f}%, {a2_opt*100:.1f}%)\nPnL={pnl_opt*100:.1f}%")

ax.set_xlabel("Allocation P1 (r=+2.0)")
ax.set_ylabel("Allocation P2 (r=+0.5)")
ax.set_zlabel("PnL")
ax.set_title(f"Portfolio PnL surface  —  feasible region: a1+a2 ≤ 1")
ax.legend(loc="upper left")
fig.colorbar(surf, ax=ax, shrink=0.5, label="PnL")

plt.tight_layout()
plt.savefig("manual/r5/portfolio_3d.png", dpi=150)
plt.show()
print("saved: manual/r5/portfolio_3d.png")
