import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_distribution import generate_weights
from optimal_allocation_discrete import optimal_allocation
from collections import Counter

results = []
for seed in range(100):
    weights = generate_weights(seed=seed, bin_w=0.001)
    res = optimal_allocation(bid_weights=weights, units='fraction')
    results.append({'seed': seed, 'sp': res['sp'], 'pnl': res['pnl'],
                    'r': res['r'], 's': res['s']})

sp_vals = [r['sp'] for r in results]
counter = Counter(sp_vals)

print(f"{'Seed':>5} {'sp%':>8} {'r%':>8} {'s%':>8} {'PnL':>12}")
for r in results:
    print(f"{r['seed']:>5} {r['sp']:>8.3f} {r['r']:>8.3f} {r['s']:>8.3f} {r['pnl']:>12,.0f}")

print()
print("sp distribution across 100 runs:")
for sp, cnt in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"  sp={sp:>7.3f}%   {cnt:>3}/100 runs")

best = max(results, key=lambda x: x['pnl'])
most_common = counter.most_common(1)[0][0]
print(f"\nMost common optimal sp : {most_common:.3f}%")
print(f"Best single-run PnL    : {best['pnl']:,.0f}  (sp={best['sp']:.3f}%, seed={best['seed']})")
