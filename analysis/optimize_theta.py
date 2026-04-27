import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize, Bounds

DATA_DIR = Path(__file__).parent.parent / "data" / "iv"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TRADING_DAYS_PER_YEAR = 252

# ── USER-CONFIGURABLE CONSTRAINTS ──────────────────────────────────────────────
underlying_limit = 200       # max abs position in VELVETFRUIT_EXTRACT
option_limit     = 300       # max abs contracts per strike

S_min      = 5200.0          # terminal price lower bound (fill in your view)
S_max      = 5300.0          # terminal price upper bound (fill in your view)
loss_floor = 0.0             # minimum acceptable P/L at each endpoint
# ───────────────────────────────────────────────────────────────────────────────


def bs_theta(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) / TRADING_DAYS_PER_YEAR


# --- load last snapshot from IV files ---
rows = []
for K in STRIKES:
    iv_file = DATA_DIR / f"iv_underlying_mid_{K}.csv"
    df = pd.read_csv(iv_file).dropna(subset=["ask_iv", "bid_iv"])
    if df.empty:
        continue
    last = df.iloc[-1]
    mid_iv    = (last["ask_iv"] + last["bid_iv"]) / 2
    mid_price = (last["ask_price"] + last["bid_price"]) / 2
    rows.append({
        "K":         K,
        "mid_price": mid_price,
        "mid_iv":    mid_iv,
        "theta":     bs_theta(last["S_mid"], K, last["tte_years"], mid_iv),
        "S0":        last["S_mid"],
        "T":         last["tte_years"],
    })

data = pd.DataFrame(rows)
print("Market snapshot (last timestamp, day 2):")
print(data.to_string(index=False))

S0            = data["S0"].iloc[-1]
strikes       = data["K"].values.astype(float)
current_prices = data["mid_price"].values
thetas        = data["theta"].values
n             = len(strikes)

print(f"\nS0={S0:.1f}  S_min={S_min}  S_max={S_max}  loss_floor={loss_floor}\n")


def terminal_pl(x, test_S):
    return (x[0] * (test_S - S0)
            + np.sum(x[1:] * (np.maximum(0, test_S - strikes) - current_prices)))


cons = [
    {"type": "ineq", "fun": lambda x: terminal_pl(x, S_min) - loss_floor},
    {"type": "ineq", "fun": lambda x: terminal_pl(x, S_max) - loss_floor},
]

bounds = Bounds(
    [-underlying_limit] + [-option_limit] * n,
    [+underlying_limit] + [+option_limit] * n,
)

res = minimize(
    lambda x: -np.sum(x[1:] * thetas),
    np.zeros(1 + n),
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"maxiter": 10_000, "ftol": 1e-10},
)

print("--- Optimized Positions ---")
if not res.success:
    print(f"(solver: {res.message})")

x = res.x
print(f"Underlying:  {x[0]:+.2f} units")
for i, K in enumerate(strikes):
    print(f"Call K={int(K):5d}: {x[1+i]:+8.2f} contracts  theta={thetas[i]:+.4f}/day")

print(f"\nTotal Portfolio Theta : {np.sum(x[1:] * thetas):+.4f} per day")
print(f"P/L at S_min={S_min:.0f}    : {terminal_pl(x, S_min):+.2f}")
print(f"P/L at S_max={S_max:.0f}    : {terminal_pl(x, S_max):+.2f}")
