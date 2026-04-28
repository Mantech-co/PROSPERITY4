
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# 1. Alpha relative to spread
# alpha_buy  = Theoretical - Ask  (positive = cheap to buy)
# alpha_sell = Bid - Theoretical  (positive = expensive, good to sell)
#                         [    u,      p35,     p40,     p45,     p50,     c50,     c60,  chooser,     ko,      bp]
alpha_buy  = np.array([ -0.025,  -0.0021, -0.0058,  0.0228, -0.0730, -0.0730, -0.0603, -0.5251,  0.0470, -0.4169])
alpha_sell = np.array([ -0.025,  -0.0179, -0.0442, -0.0728,  0.0230,  0.0230,  0.0103,  0.4251, -0.0720,  0.3169])

# 2. Load exotic payout curves from CSV
def load_payout_curve(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

chooser_spots, chooser_payoffs = load_payout_curve("chooser_payout_grouped.csv")
ko_spots,      ko_payoffs      = load_payout_curve("ko_payout_B35_grouped.csv")

# Discrete spot levels present in CSVs
S_range = np.union1d(chooser_spots, ko_spots)
pdf = lognorm.pdf(S_range, s=0.2, scale=50)

# 3. Payoff function
# x = [v_u, v_p35, v_p40, v_p45, v_p50, v_c50, v_c60, w_chooser, w_ko, w_bp]
def get_payoff(S, x):
    v_u, v_p35, v_p40, v_p45, v_p50, v_c50, v_c60, w_chooser, w_ko, w_bp = x
    res  = v_u  * S
    res += v_p35 * np.maximum(0, 35 - S)
    res += v_p40 * np.maximum(0, 40 - S)
    res += v_p45 * np.maximum(0, 45 - S)
    res += v_p50 * np.maximum(0, 50 - S)
    res += v_c50 * np.maximum(0, S - 50)
    res += v_c60 * np.maximum(0, S - 60)
    res += w_chooser * np.interp(S, chooser_spots, chooser_payoffs,
                                 left=chooser_payoffs[0], right=chooser_payoffs[-1])
    res += w_ko      * np.interp(S, ko_spots, ko_payoffs,
                                 left=ko_payoffs[0], right=ko_payoffs[-1])
    res += w_bp      * np.where(S < 40, 10.0, 0.0)
    return res

# 4. Piecewise reward from mispricing
def calculate_reward(x):
    reward = 0
    for i in range(len(x)):
        if x[i] > 0:
            reward += x[i] * alpha_buy[i]
        elif x[i] < 0:
            reward += abs(x[i]) * alpha_sell[i]
    return reward

# 5. Objective: risk - lambda * reward, over CSV spot levels only
def objective(x, lam=500):
    payouts = get_payoff(S_range, x)
    risk    = np.sum(pdf * (payouts ** 2))
    reward  = calculate_reward(x)
    return risk - lam * reward

# 6. Bounds: Underlying 500, vanilla 50, exotics 50
bounds = [(-500, 500)] + [(-50, 50)] * 6 + [(-50, 50)] * 3

res = differential_evolution(
    objective,
    bounds=bounds,
    integrality=np.ones(10),
    seed=42,
    maxiter=10000,
    tol=1e-9,
)

# 7. Output
labels = ['Underlying', '35 Put', '40 Put', '45 Put', '50 Put', '50 Call', '60 Call',
          'Chooser K=50', 'KO Put B=35', 'Binary Put K=40']
for label, vol in zip(labels, res.x):
    print(f"{label:<16}: {int(vol):8d}")

# 8. Plots
S_plot      = np.linspace(10, 100, 1000)
payout_smooth = get_payoff(S_plot, res.x)
payout_disc   = get_payoff(S_range, res.x)
weighted      = payout_disc * pdf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

# Payout curve
ax1.plot(S_plot, payout_smooth, color="steelblue", linewidth=2)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax1.axvline(50, color="orange", linewidth=0.8, linestyle="--", label="S₀=50")
ax1.fill_between(S_plot, payout_smooth, 0, where=(payout_smooth >= 0), alpha=0.15, color="steelblue")
ax1.fill_between(S_plot, payout_smooth, 0, where=(payout_smooth < 0),  alpha=0.15, color="tomato")
ax1.set_ylabel("Portfolio payout")
ax1.set_title("Payout curve")
ax1.legend()

# Probability-weighted payout at discrete CSV levels
ax2.bar(S_range, weighted,
        width=np.diff(S_range, prepend=S_range[0] - (S_range[1]-S_range[0])) * 0.6,
        color=np.where(weighted >= 0, "steelblue", "tomato"), alpha=0.8)
ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.axvline(50, color="orange", linewidth=0.8, linestyle="--", label="S₀=50")
ax2.set_xlabel("Spot price at expiry")
ax2.set_ylabel("Payout × PDF")
ax2.set_title(f"Probability-weighted payout at CSV levels  (EV ≈ {weighted.sum():.4f})")
ax2.legend()

plt.tight_layout()
plt.show()
