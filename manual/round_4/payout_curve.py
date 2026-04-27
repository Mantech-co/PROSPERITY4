import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import gammaln


class PayoutCurveBuilder:
    def __init__(self, S_min=None, S_max=None, n_points=500):
        self._positions = []
        self.S_min = S_min
        self.S_max = S_max
        self.n_points = n_points

    def add_call(self, K: float, qty: float = 1, premium: float = 0):
        self._positions.append(("call", K, qty, premium))

    def add_put(self, K: float, qty: float = 1, premium: float = 0):
        self._positions.append(("put", K, qty, premium))

    def add_aon_put(self, K: float, payout: float = 10, qty: float = 1, premium: float = 0):
        """All-or-nothing put: pays `payout` if S < K, else 0."""
        self._positions.append(("aon_put", K, payout, qty, premium))

    def add_underlying(self, qty: float = 1, premium: float = 0):
        self._positions.append(("underlying", qty, premium))

    def add_from_csv(self, csv_path: str, label: str, qty: float = 1, premium: float = 0):
        """Load expected payout curve from a grouped CSV (spot, expected_payoff, n_paths)."""
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self._positions.append(("csv", data[:, 0], data[:, 1], label, qty, premium))

    def _S_range(self):
        Ks     = [p[1] for p in self._positions if p[0] in ("call", "put", "aon_put")]
        csv_lo = [p[1].min() for p in self._positions if p[0] == "csv"]
        csv_hi = [p[1].max() for p in self._positions if p[0] == "csv"]
        lo = min(Ks + csv_lo) if (Ks or csv_lo) else 0
        hi = max(Ks + csv_hi) if (Ks or csv_hi) else 100
        S_min = self.S_min if self.S_min is not None else lo * 0.5
        S_max = self.S_max if self.S_max is not None else hi * 1.5
        return np.linspace(S_min, S_max, self.n_points)

    def _binomial_probs(self, S0, N, sigma, s):
        total_steps = N * s
        dt  = 1.0 / s
        u   = np.exp(sigma * math.sqrt(dt))
        d   = 1.0 / u
        p   = (1.0 - d) / (u - d)
        j   = np.arange(total_steps + 1)
        # log-space for numerical stability
        log_prob = (gammaln(total_steps + 1) - gammaln(j + 1) - gammaln(total_steps - j + 1)
                    + j * math.log(p) + (total_steps - j) * math.log(1 - p))
        probs    = np.exp(log_prob)
        net      = 2 * j - total_steps
        levels   = np.where(net >= 0, S0 * u**net, S0 * d**(-net))
        return levels, probs

    def plot(self, title: str = "Payout curve", S0=None, N=None, sigma=None, s=None):
        S = self._S_range()
        total = np.zeros(len(S))

        has_ev = all(x is not None for x in (S0, N, sigma, s))
        fig, axes = plt.subplots(2 if has_ev else 1, 1,
                                 figsize=(11, 9 if has_ev else 6),
                                 sharex=True)
        ax = axes[0] if has_ev else axes

        for pos in self._positions:
            kind = pos[0]
            if kind == "call":
                _, K, qty, premium = pos
                payout = qty * (np.maximum(S - K, 0) - premium)
                label  = f"{'Long' if qty > 0 else 'Short'} call  K={K}  prem={premium}  ×{abs(qty)}"
            elif kind == "put":
                _, K, qty, premium = pos
                payout = qty * (np.maximum(K - S, 0) - premium)
                label  = f"{'Long' if qty > 0 else 'Short'} put   K={K}  prem={premium}  ×{abs(qty)}"
            elif kind == "aon_put":
                _, K, p, qty, premium = pos
                payout = qty * (np.where(S < K, float(p), 0.0) - premium)
                label  = f"{'Long' if qty > 0 else 'Short'} AoN put  K={K}  pays={p}  prem={premium}  ×{abs(qty)}"
            elif kind == "underlying":
                qty, premium = pos[1], pos[2]
                payout = qty * (S - premium)
                label  = f"{'Long' if qty > 0 else 'Short'} underlying  prem={premium}  ×{abs(qty)}"
            elif kind == "csv":
                _, spots, payoffs, csv_label, qty, premium = pos
                payout = qty * (np.interp(S, spots, payoffs,
                                          left=payoffs[0], right=payoffs[-1]) - premium)
                label  = f"{'Long' if qty > 0 else 'Short'} {csv_label}  prem={premium}  ×{abs(qty)}"

            ax.plot(S, payout, linewidth=1.2, linestyle="--", alpha=0.75, label=label)
            total += payout

        ax.plot(S, total, linewidth=2.2, color="black", label="Total payout")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("Payout")
        ax.set_title(title)
        ax.legend(fontsize=8)

        if has_ev:
            levels, probs = self._binomial_probs(S0, N, sigma, s)
            payout_at_levels = np.interp(levels, S, total)
            ev_contrib = payout_at_levels * probs

            ax2 = axes[1]
            ax2.bar(levels, ev_contrib, width=(levels[1] - levels[0]) * 0.8,
                    color=np.where(ev_contrib >= 0, "steelblue", "tomato"),
                    alpha=0.8)
            ax2.axhline(0, color="gray", linewidth=0.8)
            ax2.set_xlabel("Spot price at expiry")
            ax2.set_ylabel("Payout × P(S)")
            ax2.set_title(f"Expected value contributions  (total EV = {ev_contrib.sum():.4f})")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    b = PayoutCurveBuilder(S_min=20, S_max=90)

    b.add_put(K=50, qty=-1, premium=12)
    b.add_call(K=50, qty=-1, premium=12)
    b.add_call(K=60, qty=0, premium=8.78)
    b.add_put(K=45, qty=1, premium=9.1)
    b.add_put(K=35, qty=1, premium=4.35)
    b.add_put(K=40, qty=1, premium=6.55)
    b.add_aon_put(K=50, payout=10, qty=-1, premium=5)
    b.add_from_csv("chooser_payout_grouped.csv",  label="chooser",  qty=-1, premium=22.2)
    b.add_from_csv("ko_payout_B35_grouped.csv",   label="KO put B=35", qty=10, premium=0.175)
    b.add_underlying(qty=1, premium=49.975)
    # mult = 50
    # b.add_put(K=50, qty=mult, premium=9.7)
    # b.add_call(K=50, qty=-mult, premium=9.75)


    sigma = 2.51 / math.sqrt(252)
    b.plot(title="Sample portfolio payout", S0=50, N=15, sigma=sigma, s=4)
