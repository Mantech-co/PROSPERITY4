import numpy as np
import matplotlib.pyplot as plt

returns = [0.2, 0.5, 1.0, 2.0, -0.5, -1.5]
a = np.linspace(0, 1, 500)

fig, ax = plt.subplots(figsize=(9, 6))

for r in returns:
    eff_r = abs(r)
    pnl = a * eff_r - a**2
    label = f"r = {r:+.1f} ({'SELL' if r < 0 else 'BUY'})"
    line, = ax.plot(a * 100, pnl * 100, label=label)
    a_opt = min(eff_r / 2, 1.0)
    pnl_opt = a_opt * eff_r - a_opt**2
    ax.scatter(a_opt * 100, pnl_opt * 100, color=line.get_color(), zorder=5, s=60)

ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_xlabel("Allocation (%)")
ax.set_ylabel("PnL (%)")
ax.set_title("Allocation vs PnL  —  PnL = a·|r| − a²")
ax.legend(title="Return")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("manual/r5/allocation_vs_pnl.png", dpi=150)
plt.show()
print("saved: manual/r5/allocation_vs_pnl.png")
