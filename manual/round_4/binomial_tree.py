import numpy as np
import math
from typing import Callable


def binomial_tree_price(
    S0: float,
    N: int,
    sigma: float,
    s: int,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    p: float = None,
) -> float:
    """
    Recombining binomial tree (fast).
    payoff_fn(S_final: np.ndarray) -> np.ndarray  (vectorised over all terminal prices)
    O(N*s) nodes.
    """
    total_steps = N * s
    dt = 1.0 / s
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    if p is None:
        p = (1.0 - d) / (u - d)
    q = 1.0 - p

    j = np.arange(total_steps + 1)
    S_final = S0 * (u ** j) * (d ** (total_steps - j))
    V = payoff_fn(S_final).astype(float)

    for _ in range(total_steps):
        V = p * V[1:] + q * V[:-1]

    return float(V[0])


def binomial_tree_price_full(
    S0: float,
    N: int,
    sigma: float,
    s: int,
    payoff_fn: Callable,
    p: float = None,
    path_aware: bool = False,
) -> float:
    """
    Full non-recombining binomial tree (exact).
    path_aware=False: payoff_fn(S_final) -> scalar
    path_aware=True:  payoff_fn(path: np.ndarray) -> scalar, path = [S0, ..., S_T]
    O(2^(N*s)) time — keep N*s <= ~20.
    """
    total_steps = N * s
    dt = 1.0 / s
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    if p is None:
        p = (1.0 - d) / (u - d)
    q = 1.0 - p

    path = [S0]

    def recurse(S, step):
        if step == total_steps:
            arg = np.array(path) if path_aware else path[-1]
            return float(payoff_fn(arg))
        path.append(S * u)
        up_val = recurse(S * u, step + 1)
        path.pop()
        path.append(S * d)
        dn_val = recurse(S * d, step + 1)
        path.pop()
        return p * up_val + q * dn_val

    return recurse(S0, 0)


def ko_put_binomial(
    S0: float,
    K: float,
    B: float,
    N: int,
    sigma: float,
    s: int,
    p: float = None,
) -> float:
    """
    Knock-out put: pays max(K-S_T, 0) unless S ever drops below barrier B.

    Hybrid full/recombining tree with two pruning rules:
      1. S < B at any node -> value = 0, stop branch
      2. S * d^remaining > B -> barrier unreachable from here, switch to
         fast recombining subtree
      Otherwise: full DFS
    """
    total_steps = N * s
    dt = 1.0 / s
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    if p is None:
        p = (1.0 - d) / (u - d)
    q = 1.0 - p

    d_pow = d ** np.arange(total_steps + 1)  # precompute d^k for all k

    def recombining_put(S, remaining):
        j = np.arange(remaining + 1)
        S_T = S * (u ** j) * (d ** (remaining - j))
        V = np.maximum(K - S_T, 0.0)
        for _ in range(remaining):
            V = p * V[1:] + q * V[:-1]
        return float(V[0])

    def recurse(S, step):
        if S < B:                               # pruning 1: knocked out
            return 0.0
        remaining = total_steps - step
        if remaining == 0:
            return max(K - S, 0.0)
        if S * d_pow[remaining] > B:            # pruning 2: barrier safe, go recombining
            return recombining_put(S, remaining)
        return p * recurse(S * u, step + 1) + q * recurse(S * d, step + 1)

    return recurse(S0, 0)


# --- payoffs ---

def call(K: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda S: np.maximum(S - K, 0.0)

def put(K: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda S: np.maximum(K - S, 0.0)


if __name__ == "__main__":
    S0    = 50
    K     = 50
    N     = 15
    sigma = 2.51 / math.sqrt(252)
    s     = 4

    
    
    ko = ko_put_binomial(S0, 45, 35, N, sigma, s)
    print(f"ko put B=35: {ko:.4f}")
