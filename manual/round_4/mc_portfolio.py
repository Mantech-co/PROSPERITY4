"""
Monte Carlo portfolio pricer for all contracts in ui.py.

Instead of the binomial-tree expected value used in ui.py, this simulates
N_PATHS full GBM paths and averages the portfolio payout across them.

Payout rules mirror _payout() in ui.py exactly.
Exotic options (chooser, knockout) are simulated from first principles —
no CSV lookup.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ── contracts (mirrors ui.py) ─────────────────────────────────────────────────
# Added: expiry_day, choose_day, barrier — derived from contract semantics.
CONTRACTS = [
    {"name": "AC",        "expiry": "N/A",     "kind": "underlying", "K": 50,  "bid": 49.975, "ask": 50.025, "max": 200, "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_50_P",   "expiry": "T+21",    "kind": "put",        "K": 50,  "bid": 12,     "ask": 12.05,  "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_50_C",   "expiry": "T+21",    "kind": "call",       "K": 50,  "bid": 12,     "ask": 12.05,  "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_35_P",   "expiry": "T+21",    "kind": "put",        "K": 35,  "bid": 4.33,   "ask": 4.35,   "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_40_P",   "expiry": "T+21",    "kind": "put",        "K": 40,  "bid": 6.5,    "ask": 6.55,   "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_45_P",   "expiry": "T+21",    "kind": "put",        "K": 45,  "bid": 9.05,   "ask": 9.1,    "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_60_C",   "expiry": "T+21",    "kind": "call",       "K": 60,  "bid": 8.8,    "ask": 8.85,   "max": 50,  "aon": None, "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_50_P_2", "expiry": "T+14",    "kind": "put",        "K": 50,  "bid": 9.7,    "ask": 9.75,   "max": 50,  "aon": None, "expiry_day": 10, "choose_day": None, "barrier": None},
    {"name": "AC_50_C_2", "expiry": "T+14",    "kind": "call",       "K": 50,  "bid": 9.7,    "ask": 9.75,   "max": 50,  "aon": None, "expiry_day": 10, "choose_day": None, "barrier": None},
    {"name": "AC_50_CO",  "expiry": "T+14/21", "kind": "chooser",    "K": 50,  "bid": 22.2,   "ask": 22.3,   "max": 50,  "aon": None, "expiry_day": 15, "choose_day": 10,   "barrier": None},
    {"name": "AC_40_BP",  "expiry": "T+21",    "kind": "aon_put",    "K": 40,  "bid": 5.0,    "ask": 5.1,    "max": 50,  "aon": 10,   "expiry_day": 15, "choose_day": None, "barrier": None},
    {"name": "AC_45_KO",  "expiry": "T+21",    "kind": "ko_put",     "K": 45,  "bid": 0.15,   "ask": 0.175,  "max": 500, "aon": None, "expiry_day": 15, "choose_day": None, "barrier": 35},
]

CONTRACT_MAP = {c["name"]: c for c in CONTRACTS}


def _simulate_paths(S0, sigma, s, N_max, n_paths, rng):
    """
    Simulate n_paths GBM paths using binomial increments (same model as ui.py).
    Returns S: shape (n_paths, N_max*s + 1).
    """
    total_steps = N_max * s
    dt = 1.0 / s
    u = np.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)

    up = rng.random((n_paths, total_steps)) < p
    moves = np.where(up, u, d)

    S = np.empty((n_paths, total_steps + 1))
    S[:, 0] = S0
    for t in range(total_steps):
        S[:, t + 1] = S[:, t] * moves[:, t]

    return S


def _contract_pnl(c, S, s, side, vol):
    """
    Compute per-path PnL for one contract.
    S: full path array (n_paths, N_max*s+1).
    s: steps per trading day.
    """
    expiry_step = c["expiry_day"] * s
    S_T = S[:, expiry_step]

    prem = c["ask"] if side == "Buy" else c["bid"]
    qty  = vol if side == "Buy" else -vol
    kind = c["kind"]

    if kind == "call":
        raw = np.maximum(S_T - c["K"], 0.0)

    elif kind == "put":
        raw = np.maximum(c["K"] - S_T, 0.0)

    elif kind == "aon_put":
        raw = np.where(S_T < c["K"], float(c["aon"]), 0.0)

    elif kind == "underlying":
        raw = S_T

    elif kind == "chooser":
        choose_step = c["choose_day"] * s
        S_choose = S[:, choose_step]
        is_call = S_choose > c["K"]
        raw = np.where(
            is_call,
            np.maximum(S_T - c["K"], 0.0),
            np.maximum(c["K"] - S_T, 0.0),
        )

    elif kind == "ko_put":
        # down-and-out: zero payout if S ever touches barrier
        alive = np.all(S[:, :expiry_step + 1] > c["barrier"], axis=1)
        raw = np.where(alive, np.maximum(c["K"] - S_T, 0.0), 0.0)

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return qty * (raw - prem)


def mc_portfolio(
    positions,
    S0=50.0,
    sigma=0.1582,
    s=4,
    N_max=15,
    n_paths=200_000,
    seed=42,
    verbose=True,
):
    """
    Monte Carlo portfolio pricer.

    Parameters
    ----------
    positions : dict  {contract_name: (side, volume)}
                side  : "Buy" or "Sell"
                volume: float  (clamped to contract max)
    S0        : initial spot price
    sigma     : volatility (same convention as ui.py — per sqrt(day))
    s         : steps per day
    N_max     : trading days to simulate (T+21 → 15, T+14 → 10)
    n_paths   : number of Monte Carlo paths
    seed      : RNG seed (None for random)
    verbose   : print per-contract breakdown

    Returns
    -------
    ev        : portfolio expected value
    stderr    : Monte Carlo standard error
    pnl       : per-path portfolio pnl array (n_paths,)
    """
    rng = np.random.default_rng(seed)
    S = _simulate_paths(S0, sigma, s, N_max, n_paths, rng)

    portfolio_pnl = np.zeros(n_paths)

    for name, (side, vol) in positions.items():
        c = CONTRACT_MAP[name]
        vol = min(vol, c["max"])
        pnl = _contract_pnl(c, S, s, side, vol)
        portfolio_pnl += pnl

        if verbose:
            ev_c = pnl.mean()
            se_c = pnl.std() / math.sqrt(n_paths)
            print(f"  {side:4s} {vol:>6.0f}x {name:<12s}  EV = {ev_c:+9.4f}  ±{1.96*se_c:.4f}")

    ev     = portfolio_pnl.mean()
    stderr = portfolio_pnl.std() / math.sqrt(n_paths)
    return ev, stderr, portfolio_pnl


def plot_pnl_distribution(pnl, ev, n_paths, out_path="portfolio_payout_dist.png"):
    BG = "#1a1a2e"
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    pos_mask = pnl >= 0
    ax.hist(pnl[pos_mask],  bins=120, color="#00ff88", alpha=0.75, label="Profit")
    ax.hist(pnl[~pos_mask], bins=120, color="#ff4d6d", alpha=0.75, label="Loss")
    ax.axvline(ev, color="#ffd700", linewidth=2.0, label=f"EV = {ev:+.4f}")
    ax.axvline(0,  color="white",   linewidth=1.0, linestyle="--", label="Break-even")

    p_profit = pos_mask.mean() * 100
    ax.set_xlabel("Portfolio Payout", color="#e0e0e0")
    ax.set_ylabel("Paths", color="#e0e0e0")
    ax.set_title(
        f"Portfolio Payout Distribution  |  n={n_paths:,}  |  P(profit)={p_profit:.1f}%",
        color="#e0e0e0", fontsize=10,
    )
    ax.tick_params(colors="#e0e0e0")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.show()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # ── edit positions here ───────────────────────────────────────────────────
    positions = {
        "AC_50_CO": ("Sell", 50),   # sell 50x chooser  (overpriced per fair values.txt)
        "AC_45_KO": ("Buy",  500),  # buy 500x KO put   (underpriced per fair values.txt)
        "AC_40_BP": ("Sell", 50),   # sell 50x AON put  (overpriced per fair values.txt)
    }
    # ─────────────────────────────────────────────────────────────────────────

    N_PATHS = 200_000

    print("=== Monte Carlo Portfolio Pricer ===")
    print(f"S0=50  sigma=0.1582  s=4  N_max=15  n_paths={N_PATHS:,}\n")

    ev, se, pnl = mc_portfolio(
        positions,
        S0=50.0,
        sigma=0.1582,
        s=4,
        N_max=15,
        n_paths=N_PATHS,
        seed=42,
        verbose=True,
    )

    print(f"\nPortfolio EV = {ev:+.4f}  ±{1.96*se:.4f}  (95% CI)")
    print(f"95% CI       [{ev - 1.96*se:.4f},  {ev + 1.96*se:.4f}]")
    print(f"P(profit)    {(pnl >= 0).mean()*100:.1f}%")
    print(f"P(loss)      {(pnl <  0).mean()*100:.1f}%")
    print(f"Max profit   {pnl.max():.4f}")
    print(f"Max loss     {pnl.min():.4f}")

    plot_pnl_distribution(pnl, ev, N_PATHS)
