import sys
import os
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

data = [{"speed":0,"players":453},{"speed":1.00,"players":125},{"speed":2.00,"players":84},{"speed":3.00,"players":52},{"speed":4.00,"players":27},{"speed":5.00,"players":129},{"speed":6.00,"players":27},{"speed":7.00,"players":31},{"speed":8.00,"players":30},{"speed":9.00,"players":12},{"speed":10.0,"players":249},{"speed":11.00,"players":53},{"speed":12.00,"players":29},{"speed":13.00,"players":17},{"speed":14.00,"players":5},{"speed":15.00,"players":137},{"speed":16.00,"players":35},{"speed":17.00,"players":22},{"speed":18.00,"players":23},{"speed":19.00,"players":13},{"speed":20.0,"players":281},{"speed":21.00,"players":78},{"speed":22.00,"players":44},{"speed":23.00,"players":33},{"speed":24.00,"players":22},{"speed":25.00,"players":172},{"speed":26.00,"players":63},{"speed":27.00,"players":54},{"speed":28.00,"players":22},{"speed":29.00,"players":14},{"speed":30.0,"players":242},{"speed":31.00,"players":57},{"speed":32.00,"players":42},{"speed":33.00,"players":68},{"speed":34.00,"players":99},{"speed":34.0100,"players":1},{"speed":35.00,"players":155},{"speed":36.00,"players":184},{"speed":37.00,"players":118},{"speed":38.00,"players":75},{"speed":39.00,"players":37},{"speed":40.0,"players":227},{"speed":41.00,"players":139},{"speed":42.00,"players":119},{"speed":42.100,"players":1},{"speed":43.00,"players":89},{"speed":44.00,"players":40},{"speed":45.00,"players":100},{"speed":46.00,"players":69},{"speed":47.00,"players":42},{"speed":48.00,"players":20},{"speed":49.00,"players":14},{"speed":50.0,"players":86},{"speed":51.00,"players":62},{"speed":52.00,"players":56},{"speed":53.00,"players":40},{"speed":54.00,"players":22},{"speed":55.00,"players":33},{"speed":56.00,"players":21},{"speed":57.00,"players":17},{"speed":58.00,"players":17},{"speed":59.00,"players":4},{"speed":60.0,"players":30},{"speed":61.00,"players":14},{"speed":61.100,"players":1},{"speed":62.00,"players":3},{"speed":63.00,"players":7},{"speed":64.00,"players":8},{"speed":65.00,"players":9},{"speed":66.00,"players":5},{"speed":67.00,"players":4},{"speed":68.00,"players":3},{"speed":69.00,"players":3},{"speed":70.0,"players":8},{"speed":71.00,"players":9},{"speed":72.00,"players":2},{"speed":74.00,"players":1},{"speed":75.00,"players":1},{"speed":76.00,"players":1},{"speed":77.00,"players":2},{"speed":78.00,"players":1},{"speed":79.00,"players":1},{"speed":80.0,"players":4},{"speed":81.00,"players":2},{"speed":85.00,"players":1},{"speed":90.0,"players":1},{"speed":91.00,"players":1},{"speed":100,"players":2}]

speeds = [d['speed'] for d in data]
players = [d['players'] for d in data]

# ── Speed multiplier calculation ─────────────────────────────────────────────
N = sum(players)
data_sorted = sorted(data, key=lambda x: x['speed'], reverse=True)
# max_rank = rank of the lowest-speed group (they share a rank, not N)
players_with_min_speed = data_sorted[-1]['players']
max_rank = N - players_with_min_speed + 1  # rank of last group
print(f"Total players: {N}, max_rank: {max_rank}")
print(f"{'Speed':>8} {'Players':>8} {'Rank':>8} {'Multiplier':>12}")
players_above = 0
multipliers = {}
for d in data_sorted:
    rank = players_above + 1
    mult = 0.9 - (rank - 1) / (max_rank - 1) * 0.8
    multipliers[d['speed']] = mult
    print(f"{d['speed']:>8.2f} {d['players']:>8} {rank:>8} {mult:>12.4f}")
    players_above += d['players']

# ── Fig 1: Speed distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(speeds, players, width=0.8, color='steelblue', alpha=0.85, edgecolor='none')
ax.set_xlabel('Speed', fontsize=13)
ax.set_ylabel('Players', fontsize=13)
ax.set_title('Player Speed Distribution — Round 2', fontsize=15, fontweight='bold')
ax.set_xlim(-1, max(speeds) + 1)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
for p, s in sorted(zip(players, speeds), reverse=True)[:5]:
    ax.annotate(f'{p}', xy=(s, p), xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=9, color='black')
plt.tight_layout()
plt.show()

# ── Fig 2: Speed multiplier distribution ─────────────────────────────────────
mult_vals = [multipliers[d['speed']] for d in data]
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.bar(mult_vals, players, width=0.002, color='darkorange', alpha=0.85, edgecolor='none')
ax2.set_xlabel('Speed Multiplier', fontsize=13)
ax2.set_ylabel('Players', fontsize=13)
ax2.set_title('Player Speed Multiplier Distribution — Round 2', fontsize=15, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

# ── Optimal research/scale for each speed bucket ─────────────────────────────
def research(x):
    return 200_000 * math.log(1 + x) / math.log(101)

def scale_mult(x):
    return 7 * (x / 100)

def _f(x, t):
    return (t - x) / (1.0 + x) - math.log1p(x)

def solve(t):
    if t <= 0:
        return 0.0
    a, b = -1.0 + 1e-12, t + 1.0
    fa, fb = _f(a, t), _f(b, t)
    for _ in range(300):
        if fa * fb < 0:
            break
        if fb > 0:
            b += 1.0; fb = _f(b, t)
        else:
            a = max(-1.0 + (1.0 + a) * 0.5, -1.0 + 1e-15); fa = _f(a, t)
    return brentq(_f, a, b, args=(t,), xtol=1e-12)

def best_integer_alloc(budget):
    """Return (research_int, scale_int) maximising research(r)*scale_mult(s) with r+s=budget, r,s>=0 integers."""
    if budget <= 0:
        return 0, 0
    budget_int = round(budget)  # for non-integer budgets round to nearest int
    x_cont = solve(budget)
    best_prod, best_r, best_s = -1, 0, budget_int
    for r in [math.floor(x_cont), math.ceil(x_cont)]:
        r = max(0, min(r, budget_int))
        s = budget_int - r
        prod = (research(r) if r > 0 else 0.0) * (scale_mult(s) if s > 0 else 0.0)
        if prod > best_prod:
            best_prod, best_r, best_s = prod, r, s
    return best_r, best_s

bucket_results = []
print(f"\n{'Speed':>8} {'Budget':>8} {'Research':>10} {'Scale':>8} {'Res$':>12} {'ScMult':>8} {'SpeedMult':>10} {'NetPnL':>12}")
for d in sorted(data, key=lambda x: x['speed']):
    spd = d['speed']
    cnt = d['players']
    mult = multipliers[spd]
    budget = 100 - spd
    x_opt, sc = best_integer_alloc(budget)
    r = research(x_opt) if x_opt > 0 else 0.0
    sm = scale_mult(sc) if sc > 0 else 0.0
    net_pnl = mult * r * sm - 50000
    bucket_results.append({
        'speed': spd, 'players': cnt, 'multiplier': mult,
        'research': x_opt, 'scale': sc,
        'research_profit': r, 'scale_mult': sm, 'net_pnl': net_pnl
    })
    print(f"{spd:>8.2f} {budget:>8.2f} {x_opt:>10} {sc:>8} {r:>12.0f} {sm:>8.4f} {mult:>10.4f} {net_pnl:>12.0f}")

# ── Fig 3: Net PnL distribution ───────────────────────────────────────────────
pnl_expanded = []
for b in bucket_results:
    pnl_expanded.extend([b['net_pnl']] * b['players'])

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.hist(pnl_expanded, bins=100, color='mediumseagreen', alpha=0.85, edgecolor='none')
ax3.axvline(0, color='red', linestyle='--', linewidth=1.5, label='PnL = 0')
ax3.set_xlabel('Net PnL', fontsize=13)
ax3.set_ylabel('Players', fontsize=13)
ax3.set_title('Net PnL Distribution (optimal alloc, actual speed mult) — Round 2', fontsize=15, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.spines[['top', 'right']].set_visible(False)
ax3.legend()
pct_positive = sum(1 for p in pnl_expanded if p > 0) / len(pnl_expanded) * 100
ax3.annotate(f'{pct_positive:.1f}% positive', xy=(0.98, 0.95), xycoords='axes fraction',
             ha='right', va='top', fontsize=11)
plt.tight_layout()
plt.show()

# ── Leaderboard: infer speed allocation ──────────────────────────────────────
lb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../leaderboard_master.csv')
df = pd.read_csv(lb_path)

# r2 manual pnl = cumulative r2 manual - cumulative r1 manual
df['r2_manual_pnl'] = pd.to_numeric(df['r2_manual_score'], errors='coerce') - \
                      pd.to_numeric(df['r1_manual_score'], errors='coerce')

# For each speed bucket, we have a predicted pnl (assuming optimal alloc)
# Sort buckets by net_pnl for bisect
pnl_curve = sorted(bucket_results, key=lambda x: x['net_pnl'])
pnl_vals_sorted = np.array([b['net_pnl'] for b in pnl_curve])
spd_vals_sorted = np.array([b['speed'] for b in pnl_curve])

def infer_speed(actual_pnl):
    idx = np.argmin(np.abs(pnl_vals_sorted - actual_pnl))
    return spd_vals_sorted[idx]

speed_to_expected_pnl = {b['speed']: b['net_pnl'] for b in bucket_results}

missing_teams = []
inferred_speeds = []
expected_pnls = []
for _, row in df.iterrows():
    if pd.isna(row['r2_manual_pnl']):
        missing_teams.append(row.get('name', 'unknown'))
        inferred_speeds.append(np.nan)
        expected_pnls.append(np.nan)
    else:
        spd = infer_speed(row['r2_manual_pnl'])
        inferred_speeds.append(spd)
        expected_pnls.append(speed_to_expected_pnl[spd])

df['inferred_speed'] = inferred_speeds
df['expected_pnl'] = expected_pnls
df['pnl_deviation'] = df['r2_manual_pnl'] - df['expected_pnl']

df['suboptimal'] = df['pnl_deviation'] < -1
df['overoptimal'] = df['pnl_deviation'] > 1

print(f"\nMissing data ({len(missing_teams)} teams): {missing_teams}")

deviators = df[df['suboptimal']].sort_values('pnl_deviation')
overopt = df[df['overoptimal']].sort_values('pnl_deviation', ascending=False)
print(f"\nSuboptimal (deviation < -1): {len(deviators)} teams")
print(deviators[['name', 'r2_overall_pos', 'r2_manual_pnl', 'inferred_speed', 'expected_pnl', 'pnl_deviation']].to_string(index=False))
print(f"\nOveroptimal (deviation > +1): {len(overopt)} teams")
print(overopt[['name', 'r2_overall_pos', 'r2_manual_pnl', 'inferred_speed', 'expected_pnl', 'pnl_deviation']].to_string(index=False))

print(f"\nLeaderboard — top 20 by r2 overall rank:")
print_cols = ['name', 'r2_overall_pos', 'r2_manual_pnl', 'inferred_speed', 'expected_pnl', 'pnl_deviation', 'suboptimal', 'overoptimal']
top20 = df.sort_values('r2_overall_pos').head(20)[print_cols]
print(top20.to_string(index=False))

# ── Fig 4: PnL deviation histogram ───────────────────────────────────────────
devs = df['pnl_deviation'].dropna()

drop_cols = [c for c in ['r2_manual_pnl', 'expected_pnl', 'pnl_deviation'] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
df.to_csv(lb_path, index=False)

# ── Repeat for altogether_phase_1.csv ────────────────────────────────────────
def process_leaderboard(path):
    d = pd.read_csv(path)
    d['r2_manual_pnl'] = pd.to_numeric(d['r2_manual_score'], errors='coerce') - \
                         pd.to_numeric(d['r1_manual_score'], errors='coerce')
    # teams with identical r1/r2 manual didn't participate in round 2 manual
    no_entry = d['r2_manual_score'] == d['r1_manual_score']
    d.loc[no_entry, 'r2_manual_pnl'] = np.nan
    missing, inferred, expected = [], [], []
    for _, row in d.iterrows():
        if pd.isna(row['r2_manual_pnl']):
            missing.append(row.get('name', 'unknown'))
            inferred.append(np.nan); expected.append(np.nan)
        else:
            spd = infer_speed(row['r2_manual_pnl'])
            inferred.append(spd); expected.append(speed_to_expected_pnl[spd])
    d['inferred_speed'] = inferred
    d['expected_pnl'] = expected
    d['pnl_deviation'] = d['r2_manual_pnl'] - d['expected_pnl']
    d['suboptimal'] = d['pnl_deviation'] < -1
    d['overoptimal'] = d['pnl_deviation'] > 1
    print(f"\n{path} — missing: {len(missing)}, suboptimal: {d['suboptimal'].sum()}, overoptimal: {d['overoptimal'].sum()}")
    if missing:
        print(f"  Missing: {missing[:20]}{'...' if len(missing)>20 else ''}")
    d.drop(columns=[c for c in ['r2_manual_pnl','expected_pnl','pnl_deviation'] if c in d.columns], inplace=True)
    d.to_csv(path, index=False)

p1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../altogether_phase_1.csv')
process_leaderboard(p1_path)

# ── Fig 5: inferred_speed vs true distribution (leaderboard_master) ──────────
lbm = pd.read_csv(lb_path)
lbm_r2pnl = pd.to_numeric(lbm['r2_manual_score'], errors='coerce') - \
            pd.to_numeric(lbm['r1_manual_score'], errors='coerce')
no_entry_lbm = lbm['r2_manual_score'] == lbm['r1_manual_score']
lbm_inferred = lbm['inferred_speed'].copy()
lbm_inferred[no_entry_lbm | lbm_r2pnl.isna()] = np.nan
inferred_lbm = lbm_inferred.dropna()

fig5, ax5 = plt.subplots(figsize=(14, 6))
total_true = sum(players)
ax5.bar(speeds, [p/total_true for p in players], width=0.8,
        color='steelblue', alpha=0.5, label='True distribution (normalised)', edgecolor='none')
inf_counts_lbm = inferred_lbm.value_counts().sort_index()
ax5.bar(inf_counts_lbm.index, inf_counts_lbm.values/len(inferred_lbm), width=0.8,
        color='darkorange', alpha=0.5, label=f'Inferred — leaderboard_master (n={len(inferred_lbm)})', edgecolor='none')
ax5.set_xlabel('Speed', fontsize=13)
ax5.set_ylabel('Density', fontsize=13)
ax5.set_title('Inferred vs True Speed Distribution — leaderboard_master', fontsize=15, fontweight='bold')
ax5.set_xlim(-1, max(speeds) + 1)
ax5.grid(axis='y', alpha=0.3)
ax5.spines[['top', 'right']].set_visible(False)
ax5.legend(fontsize=11)
plt.tight_layout()
plt.show()

# ── Fig 6: inferred_speed vs true distribution (altogether_phase_1) ──────────
p1 = pd.read_csv(p1_path)
inferred = p1['inferred_speed'].dropna()

fig5, ax5 = plt.subplots(figsize=(14, 6))

# true distribution — normalise to density
total_true = sum(players)
ax5.bar(speeds, [p/total_true for p in players], width=0.8,
        color='steelblue', alpha=0.5, label='True distribution (normalised)', edgecolor='none')

# inferred distribution — normalise to density
total_inf = len(inferred)
inf_counts = inferred[inferred != 85].value_counts().sort_index()
ax5.bar(inf_counts.index, inf_counts.values/total_inf, width=0.8,
        color='darkorange', alpha=0.5, label='Inferred distribution (normalised)', edgecolor='none')

ax5.set_xlabel('Speed', fontsize=13)
ax5.set_ylabel('Density', fontsize=13)
ax5.set_title('Inferred vs True Speed Distribution — altogether_phase_1', fontsize=15, fontweight='bold')
ax5.set_xlim(-1, max(speeds) + 1)
ax5.grid(axis='y', alpha=0.3)
ax5.spines[['top', 'right']].set_visible(False)
ax5.legend(fontsize=11)
plt.tight_layout()
plt.show()
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(devs, bins=100, color='steelblue', alpha=0.85, edgecolor='none')
axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('PnL Deviation (actual − expected)', fontsize=12)
axes[0].set_ylabel('Teams', fontsize=12)
axes[0].set_title('PnL Deviation Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].spines[['top', 'right']].set_visible(False)

# zoomed: exclude extreme outliers (beyond 3 std)
std = devs.std()
mean = devs.mean()
devs_zoom = devs[(devs > mean - 3*std) & (devs < mean + 3*std)]
axes[1].hist(devs_zoom, bins=80, color='darkorange', alpha=0.85, edgecolor='none')
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('PnL Deviation (actual − expected)', fontsize=12)
axes[1].set_ylabel('Teams', fontsize=12)
axes[1].set_title(f'PnL Deviation — Zoomed (±3σ), σ={std:.0f}', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].spines[['top', 'right']].set_visible(False)

axes[1].axvline(-1, color='red', linestyle=':', linewidth=1, label='suboptimal threshold')
axes[1].axvline(1, color='green', linestyle=':', linewidth=1, label='overoptimal threshold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
