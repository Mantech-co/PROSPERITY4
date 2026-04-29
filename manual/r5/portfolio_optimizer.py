"""
Portfolio optimizer: maximize PnL given known product returns.

For each product, choose to BUY (long) or SELL (short):
  - Long: profit from positive return
  - Short: profit from negative return (sell before price drops)
Direction chosen automatically to maximize profit from |r_i|.

PnL = sum_i [ a_i * |r_i| - a_i^2 ]
  a_i = allocation fraction [0, 1]  (always positive, budget is consumed either way)
  fee = a_i^2

Constraint: sum(a_i) <= 1, a_i >= 0

Analytical KKT solution:
  Optimal unconstrained: a_i* = |r_i| / 2
  If sum of unconstrained <= 1: done
  Else with binding budget constraint and Lagrange multiplier λ:
    a_i = max((|r_i| - λ) / 2, 0)
    Find λ such that sum(a_i) = 1
"""

from __future__ import annotations


def optimize_portfolio(returns: dict[str, float]) -> tuple[dict[str, float], dict[str, str], float]:
    """
    returns: product -> return fraction (e.g. 0.20 for +20%, -0.10 for -10%)
    Returns: (allocations dict, directions dict, expected PnL)
    """
    # effective return is |r|; direction chosen to exploit it
    effective = {p: abs(r) for p, r in returns.items() if r != 0}
    directions = {p: ("BUY" if returns[p] > 0 else "SELL") for p in effective}

    if not effective:
        return {p: 0.0 for p in returns}, {}, 0.0

    sorted_products = sorted(effective, key=lambda p: effective[p], reverse=True)
    r_vals = [effective[p] for p in sorted_products]

    # unconstrained optimum
    if sum(r / 2 for r in r_vals) <= 1.0:
        allocs = {p: effective[p] / 2 for p in sorted_products}
    else:
        # KKT: find λ such that sum(max((r_i - λ)/2, 0)) = 1
        lam = 0.0
        for k in range(1, len(r_vals) + 1):
            lam_k = (sum(r_vals[:k]) - 2) / k
            if r_vals[k - 1] > lam_k:
                lam = lam_k

        allocs = {p: max((r_vals[i] - lam) / 2, 0.0) for i, p in enumerate(sorted_products)}

    for p in returns:
        if p not in allocs:
            allocs[p] = 0.0
            directions[p] = "SKIP"

    pnl = sum(allocs[p] * effective.get(p, 0) - allocs[p] ** 2 for p in returns)
    return allocs, directions, pnl


def print_result(returns: dict[str, float]) -> None:
    allocs, directions, pnl = optimize_portfolio(returns)
    print(f"\nReturns: {returns}")
    for p in sorted(allocs, key=lambda p: allocs[p], reverse=True):
        if allocs[p] > 1e-9:
            print(f"  {p}: {directions[p]} {allocs[p]*100:.2f}%")
    print(f"Total alloc: {sum(allocs.values())*100:.2f}%")
    print(f"Expected PnL: {pnl*100:.2f}%")


if __name__ == "__main__":
    print_result({"A": 0.2, "B": 0.04})
