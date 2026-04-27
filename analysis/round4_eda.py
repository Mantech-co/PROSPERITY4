import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis/round4_plots/"
import os; os.makedirs(OUT, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
DATA = "/media/manukrishnan/Mk/prosperity_4/round_4/"

price_dfs, trade_dfs = [], []
for day in [1, 2, 3]:
    p = pd.read_csv(f"{DATA}prices_round_4_day_{day}.csv", sep=";")
    p['day'] = day
    price_dfs.append(p)
    t = pd.read_csv(f"{DATA}trades_round_4_day_{day}.csv", sep=";")
    t['day'] = day
    trade_dfs.append(t)

prices = pd.concat(price_dfs, ignore_index=True)
trades = pd.concat(trade_dfs, ignore_index=True)

# global timestamp
prices['gts'] = (prices['day'] - 1) * 1_000_000 + prices['timestamp']
trades['gts'] = (trades['day'] - 1) * 1_000_000 + trades['timestamp']

# mid price from order book
prices['mid_calc'] = (prices['ask_price_1'] + prices['bid_price_1']) / 2
prices['spread'] = prices['ask_price_1'] - prices['bid_price_1']

# merge mid into trades
mid_lookup = prices[['gts', 'product', 'mid_calc', 'spread']].rename(
    columns={'product': 'symbol'})
trades = trades.merge(mid_lookup, on=['gts', 'symbol'], how='left')
trades['trade_spread'] = trades['price'] - trades['mid_calc']

TRADERS = sorted(trades['buyer'].unique().tolist() +
                 [s for s in trades['seller'].unique() if s not in trades['buyer'].unique()])
TRADERS = sorted(set(trades['buyer'].tolist() + trades['seller'].tolist()))
PRODUCTS = sorted(trades['symbol'].unique())
OPTION_PRODUCTS = [p for p in PRODUCTS if p.startswith('VEV_')]
SPOT_PRODUCTS = [p for p in PRODUCTS if not p.startswith('VEV_')]
DAYS = [1, 2, 3]

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
TRADER_COLOR = {t: COLORS[i % len(COLORS)] for i, t in enumerate(TRADERS)}
PRODUCT_COLOR = {p: COLORS[i % len(COLORS)] for i, p in enumerate(PRODUCTS)}

print(f"Prices: {len(prices)} rows | Trades: {len(trades)} rows")
print(f"Traders: {TRADERS}")
print(f"Products: {PRODUCTS}")


def save(fig, name):
    fig.savefig(f"{OUT}{name}.png", dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MID PRICE TIME SERIES — all products
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(18, 3 * len(PRODUCTS)), sharex=False)
for ax, prod in zip(axes, PRODUCTS):
    sub = prices[prices['product'] == prod]
    for d in DAYS:
        ds = sub[sub['day'] == d]
        ax.plot(ds['gts'], ds['mid_calc'], label=f'Day {d}', lw=0.8)
    ax.set_title(f"Mid Price — {prod}")
    ax.set_ylabel("Price")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle("Mid Price Time Series (all products)", fontsize=14, y=1.001)
fig.tight_layout()
save(fig, "01_mid_price_all_products")

# ══════════════════════════════════════════════════════════════════════════════
# 2. BID-ASK SPREAD OVER TIME — all products
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(18, 3 * len(PRODUCTS)), sharex=False)
for ax, prod in zip(axes, PRODUCTS):
    sub = prices[prices['product'] == prod]
    for d in DAYS:
        ds = sub[sub['day'] == d]
        ax.plot(ds['gts'], ds['spread'], label=f'Day {d}', lw=0.6, alpha=0.8)
    ax.set_title(f"Bid-Ask Spread — {prod}")
    ax.set_ylabel("Spread")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle("Bid-Ask Spread Over Time", fontsize=14, y=1.001)
fig.tight_layout()
save(fig, "02_bidask_spread_time")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRADE ARRIVAL TIMESTAMPS — histogram per day
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, d in zip(axes, DAYS):
    sub = trades[trades['day'] == d]
    ax.hist(sub['timestamp'], bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_title(f"Trade Arrival — Day {d}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
fig.suptitle("Trade Timestamp Distribution", fontsize=14)
fig.tight_layout()
save(fig, "03_trade_arrival_histogram")

# ══════════════════════════════════════════════════════════════════════════════
# 4. INTER-ARRIVAL TIMES — global and per trader
# ══════════════════════════════════════════════════════════════════════════════
trades_sorted = trades.sort_values('gts')
trades_sorted['iat'] = trades_sorted['gts'].diff().clip(0)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
# global
axes[0].hist(trades_sorted['iat'].dropna(), bins=80, color='grey', edgecolor='none', alpha=0.8)
axes[0].set_title("IAT — All Traders")
axes[0].set_xlabel("Δt")
for i, trader in enumerate(TRADERS, start=1):
    if i >= len(axes): break
    sub = trades_sorted[
        (trades_sorted['buyer'] == trader) | (trades_sorted['seller'] == trader)
    ].sort_values('gts')
    iat = sub['gts'].diff().clip(0).dropna()
    axes[i].hist(iat, bins=50, color=TRADER_COLOR[trader], edgecolor='none', alpha=0.8)
    axes[i].set_title(f"IAT — {trader}")
    axes[i].set_xlabel("Δt")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Inter-Arrival Times", fontsize=14)
fig.tight_layout()
save(fig, "04_inter_arrival_times")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BOT vs BOT HEATMAP — trade count matrix (buyer × seller)
# ══════════════════════════════════════════════════════════════════════════════
for metric, label, fname in [
    ('quantity', 'Total Volume', '05a_bot_bot_volume_heatmap'),
    ('price',    'Trade Count',  '05b_bot_bot_count_heatmap'),
]:
    if metric == 'price':
        mat = trades.groupby(['buyer', 'seller']).size().unstack(fill_value=0)
    else:
        mat = trades.groupby(['buyer', 'seller'])['quantity'].sum().unstack(fill_value=0)
    mat = mat.reindex(index=TRADERS, columns=TRADERS, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title(f"Bot×Bot {label} Matrix")
    ax.set_xlabel("Seller")
    ax.set_ylabel("Buyer")
    fig.tight_layout()
    save(fig, fname)

# per product
for prod in PRODUCTS:
    sub = trades[trades['symbol'] == prod]
    mat = sub.groupby(['buyer', 'seller'])['quantity'].sum().unstack(fill_value=0)
    mat = mat.reindex(index=TRADERS, columns=TRADERS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='Blues', ax=ax, linewidths=0.5)
    ax.set_title(f"Bot×Bot Volume — {prod}")
    ax.set_xlabel("Seller"); ax.set_ylabel("Buyer")
    fig.tight_layout()
    save(fig, f"05c_botbot_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRADE VOLUME per trader (buy / sell / net)
# ══════════════════════════════════════════════════════════════════════════════
buy_vol  = trades.groupby('buyer')['quantity'].sum().reindex(TRADERS, fill_value=0)
sell_vol = trades.groupby('seller')['quantity'].sum().reindex(TRADERS, fill_value=0)
net_vol  = buy_vol - sell_vol

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x = range(len(TRADERS))
axes[0].bar(x, buy_vol, color=[TRADER_COLOR[t] for t in TRADERS])
axes[0].set_title("Buy Volume per Trader"); axes[0].set_xticks(x); axes[0].set_xticklabels(TRADERS, rotation=30)
axes[1].bar(x, sell_vol, color=[TRADER_COLOR[t] for t in TRADERS])
axes[1].set_title("Sell Volume per Trader"); axes[1].set_xticks(x); axes[1].set_xticklabels(TRADERS, rotation=30)
axes[2].bar(x, net_vol, color=['green' if v >= 0 else 'red' for v in net_vol])
axes[2].set_title("Net Volume (Buy−Sell) per Trader"); axes[2].set_xticks(x); axes[2].set_xticklabels(TRADERS, rotation=30)
axes[2].axhline(0, color='black', lw=1)
for ax in axes: ax.grid(True, alpha=0.3, axis='y')
fig.suptitle("Trade Volume per Trader", fontsize=14)
fig.tight_layout()
save(fig, "06_volume_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 7. VOLUME per PRODUCT per TRADER (stacked bars)
# ══════════════════════════════════════════════════════════════════════════════
for role, label in [('buyer', 'Buy'), ('seller', 'Sell')]:
    pvt = trades.groupby([role, 'symbol'])['quantity'].sum().unstack(fill_value=0)
    pvt = pvt.reindex(TRADERS, fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 6))
    pvt.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', edgecolor='none')
    ax.set_title(f"{label} Volume per Trader per Product")
    ax.set_xlabel("Trader"); ax.set_ylabel("Volume")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.set_xticklabels(TRADERS, rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, f"07_{label.lower()}_vol_product_stacked")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TRADE PRICE vs MID PRICE — scatter per product
# ══════════════════════════════════════════════════════════════════════════════
ncols = 3
nrows = int(np.ceil(len(PRODUCTS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = trades[trades['symbol'] == prod].dropna(subset=['mid_calc'])
    axes[i].scatter(sub['gts'], sub['price'], s=10, alpha=0.5, label='Trade Price', color='steelblue')
    axes[i].scatter(sub['gts'], sub['mid_calc'], s=4, alpha=0.3, label='Mid Price', color='orange')
    axes[i].set_title(prod); axes[i].set_xlabel("gts"); axes[i].set_ylabel("Price")
    axes[i].legend(fontsize=7); axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Trade Price vs Mid Price", fontsize=14)
fig.tight_layout()
save(fig, "08_trade_vs_mid_price")

# ══════════════════════════════════════════════════════════════════════════════
# 9. TRADE SPREAD FROM MID — distribution per product
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = trades[trades['symbol'] == prod]['trade_spread'].dropna()
    axes[i].hist(sub, bins=60, color=PRODUCT_COLOR[prod], edgecolor='none', alpha=0.85)
    axes[i].axvline(0, color='black', lw=1)
    axes[i].set_title(f"Spread from Mid — {prod}")
    axes[i].set_xlabel("Trade Price − Mid"); axes[i].set_ylabel("Count")
    axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Trade Spread from Mid Price", fontsize=14)
fig.tight_layout()
save(fig, "09_trade_spread_from_mid")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CUMULATIVE POSITION per TRADER per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
for prod in PRODUCTS:
    sub = trades[trades['symbol'] == prod].sort_values('gts').copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    for trader in TRADERS:
        buys  = sub[sub['buyer']  == trader].set_index('gts')['quantity']
        sells = sub[sub['seller'] == trader].set_index('gts')['quantity']
        if buys.empty and sells.empty: continue
        buys_agg  = buys.groupby(level=0).sum()
        sells_agg = sells.groupby(level=0).sum()
        all_ts = pd.Index(sorted(set(buys_agg.index) | set(sells_agg.index)))
        if all_ts.empty: continue
        net = buys_agg.reindex(all_ts, fill_value=0) - sells_agg.reindex(all_ts, fill_value=0)
        cum = net.cumsum()
        ax.plot(cum.index, cum.values, label=trader, color=TRADER_COLOR[trader], lw=1.2)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(f"Cumulative Position — {prod}")
    ax.set_xlabel("Global Timestamp"); ax.set_ylabel("Position")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"10_cumpos_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. PORTFOLIO VALUE over TIME — pnl from prices file
# ══════════════════════════════════════════════════════════════════════════════
pnl = prices.groupby(['gts', 'product'])['profit_and_loss'].first().unstack(fill_value=0)
pnl['total'] = pnl.sum(axis=1)

fig, ax = plt.subplots(figsize=(16, 6))
for prod in pnl.columns:
    if prod == 'total': continue
    ax.plot(pnl.index, pnl[prod], lw=0.8, alpha=0.7, label=prod)
ax.plot(pnl.index, pnl['total'], lw=2, color='black', label='TOTAL')
ax.set_title("Portfolio PnL over Time")
ax.set_xlabel("Global Timestamp"); ax.set_ylabel("PnL")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "11_portfolio_pnl_time")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PROFIT per PRODUCT — bar
# ══════════════════════════════════════════════════════════════════════════════
final_pnl = prices.groupby('product')['profit_and_loss'].last()
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['green' if v >= 0 else 'red' for v in final_pnl.values]
ax.bar(final_pnl.index, final_pnl.values, color=colors)
ax.axhline(0, color='black', lw=1)
ax.set_title("Final PnL per Product")
ax.set_xlabel("Product"); ax.set_ylabel("PnL")
ax.set_xticklabels(final_pnl.index, rotation=30, ha='right')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save(fig, "12_final_pnl_per_product")

# ══════════════════════════════════════════════════════════════════════════════
# 13. PERCENTAGE CHANGE in TOTAL PnL
# ══════════════════════════════════════════════════════════════════════════════
pct = pnl['total'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
axes[0].plot(pct.index, pct.values, lw=0.6, color='steelblue', alpha=0.8)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_title("Pct Change — Total PnL"); axes[0].set_ylabel("Δ%"); axes[0].grid(True, alpha=0.3)
axes[1].hist(pct, bins=100, color='steelblue', edgecolor='none', alpha=0.85)
axes[1].set_title("Distribution of PnL Pct Change"); axes[1].set_xlabel("Δ%"); axes[1].grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "13_pct_change_pnl")

# ══════════════════════════════════════════════════════════════════════════════
# 14. ROLLING VOLATILITY — mid prices per product
# ══════════════════════════════════════════════════════════════════════════════
WINDOW = 500
fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(18, 3 * len(PRODUCTS)))
for ax, prod in zip(axes, PRODUCTS):
    sub = prices[prices['product'] == prod].set_index('gts')['mid_calc']
    ret = sub.pct_change().replace([np.inf, -np.inf], np.nan)
    roll_vol = ret.rolling(WINDOW).std()
    ax.plot(roll_vol.index, roll_vol.values, lw=0.8)
    ax.set_title(f"Rolling Vol (w={WINDOW}) — {prod}")
    ax.set_ylabel("σ"); ax.grid(True, alpha=0.3)
fig.suptitle("Rolling Volatility (mid price returns)", fontsize=14, y=1.001)
fig.tight_layout()
save(fig, "14_rolling_volatility")

# ══════════════════════════════════════════════════════════════════════════════
# 15. TRADE COUNT per TRADER per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
buy_ct  = trades.groupby(['buyer',  'symbol']).size().unstack(fill_value=0).reindex(TRADERS, fill_value=0)
sell_ct = trades.groupby(['seller', 'symbol']).size().unstack(fill_value=0).reindex(TRADERS, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, mat, title in [(axes[0], buy_ct, "Buy Count"), (axes[1], sell_ct, "Sell Count")]:
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='Greens', ax=ax, linewidths=0.4)
    ax.set_title(f"{title} — Trader × Product")
    ax.set_xlabel("Product"); ax.set_ylabel("Trader")
fig.suptitle("Trade Count per Trader per Product", fontsize=13)
fig.tight_layout()
save(fig, "15_trade_count_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# 16. TRADE SIZE DISTRIBUTION per TRADER
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
axes[0].hist(trades['quantity'], bins=40, color='grey', edgecolor='none')
axes[0].set_title("All Traders — Trade Size")
for i, trader in enumerate(TRADERS, start=1):
    sub = trades[(trades['buyer'] == trader) | (trades['seller'] == trader)]['quantity']
    axes[i].hist(sub, bins=30, color=TRADER_COLOR[trader], edgecolor='none')
    axes[i].set_title(f"{trader} — Trade Size")
for ax in axes: ax.grid(True, alpha=0.3)
fig.suptitle("Trade Size Distribution", fontsize=14)
fig.tight_layout()
save(fig, "16_trade_size_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# 17. VWAP vs MID PRICE — per product
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = trades[trades['symbol'] == prod].dropna(subset=['mid_calc']).copy()
    if sub.empty: axes[i].set_visible(False); continue
    sub = sub.sort_values('gts')
    sub['cum_vol'] = sub['quantity'].cumsum()
    sub['cum_pv'] = (sub['price'] * sub['quantity']).cumsum()
    sub['vwap'] = sub['cum_pv'] / sub['cum_vol']
    mid_sub = prices[prices['product'] == prod].sort_values('gts')
    axes[i].plot(mid_sub['gts'], mid_sub['mid_calc'], lw=0.8, alpha=0.6, label='Mid', color='orange')
    axes[i].plot(sub['gts'], sub['vwap'], lw=1.2, label='VWAP', color='steelblue')
    axes[i].set_title(f"VWAP vs Mid — {prod}")
    axes[i].legend(fontsize=7); axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("VWAP vs Mid Price", fontsize=14)
fig.tight_layout()
save(fig, "17_vwap_vs_mid")

# ══════════════════════════════════════════════════════════════════════════════
# 18. TRADE FREQUENCY HEATMAP — timestamp bucket × product
# ══════════════════════════════════════════════════════════════════════════════
trades['ts_bucket'] = (trades['timestamp'] // 10000) * 10000
heat = trades.groupby(['ts_bucket', 'symbol']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(heat.T, cmap='YlOrRd', ax=ax, linewidths=0.0)
ax.set_title("Trade Frequency — Time Bucket × Product")
ax.set_xlabel("Timestamp Bucket"); ax.set_ylabel("Product")
fig.tight_layout()
save(fig, "18_trade_freq_timebucket_product")

# ══════════════════════════════════════════════════════════════════════════════
# 19. ORDER FLOW IMBALANCE (OFI) per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(18, 3 * len(PRODUCTS)))
for ax, prod in zip(axes, PRODUCTS):
    sub = trades[trades['symbol'] == prod].sort_values('gts').copy()
    if sub.empty: ax.set_visible(False); continue
    sub['ofi'] = sub['quantity']  # buys positive (buyer side)
    sub_ofi = sub.groupby('gts')['ofi'].sum().cumsum()
    ax.plot(sub_ofi.index, sub_ofi.values, lw=0.9, color='purple')
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(f"Cumulative Order Flow Imbalance — {prod}")
    ax.set_ylabel("Cum OFI"); ax.grid(True, alpha=0.3)
fig.suptitle("Cumulative Order Flow Imbalance", fontsize=14, y=1.001)
fig.tight_layout()
save(fig, "19_order_flow_imbalance")

# ══════════════════════════════════════════════════════════════════════════════
# 20. NET POSITION per TRADER per DAY
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for day in DAYS:
    d = trades[trades['day'] == day]
    for trader in TRADERS:
        b = d[d['buyer']  == trader].groupby('symbol')['quantity'].sum()
        s = d[d['seller'] == trader].groupby('symbol')['quantity'].sum()
        net = b.subtract(s, fill_value=0)
        for sym, val in net.items():
            rows.append({'day': day, 'trader': trader, 'symbol': sym, 'net_pos': val})
net_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, d in zip(axes, DAYS):
    sub = net_df[net_df['day'] == d].pivot(index='trader', columns='symbol', values='net_pos').fillna(0)
    sub = sub.reindex(TRADERS, fill_value=0)
    sns.heatmap(sub, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax, linewidths=0.4)
    ax.set_title(f"Net Position — Day {d}")
    ax.set_xlabel("Product"); ax.set_ylabel("Trader")
fig.suptitle("Net Position per Trader per Product per Day", fontsize=13)
fig.tight_layout()
save(fig, "20_net_position_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# 21. TRADER ACTIVITY TIME SERIES — rolling 5k trade count
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
for trader in TRADERS:
    sub = trades[(trades['buyer'] == trader) | (trades['seller'] == trader)].sort_values('gts')
    counts = sub.groupby('gts').size()
    roll = counts.rolling(50, min_periods=1).sum()
    ax.plot(roll.index, roll.values, label=trader, color=TRADER_COLOR[trader], lw=1.0, alpha=0.8)
ax.set_title("Trader Activity (Rolling Trade Count)")
ax.set_xlabel("Global Timestamp"); ax.set_ylabel("Count")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "21_trader_activity_rolling")

# ══════════════════════════════════════════════════════════════════════════════
# 22. PRICE RETURNS DISTRIBUTION per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = prices[prices['product'] == prod]['mid_calc'].pct_change().dropna()
    axes[i].hist(sub, bins=100, color=PRODUCT_COLOR[prod], edgecolor='none', alpha=0.85, density=True)
    mu, std = sub.mean(), sub.std()
    from scipy.stats import norm
    x = np.linspace(sub.min(), sub.max(), 200)
    axes[i].plot(x, norm.pdf(x, mu, std), 'k--', lw=1.2, label='Normal')
    axes[i].set_title(f"Returns — {prod}  μ={mu:.4f} σ={std:.4f}")
    axes[i].legend(fontsize=7); axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Mid Price Return Distributions", fontsize=14)
fig.tight_layout()
save(fig, "22_return_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# 23. CROSS-PRODUCT MID PRICE CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
pivot_mid = prices.pivot_table(index='gts', columns='product', values='mid_calc')
corr_mid = pivot_mid.corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_mid, dtype=bool))
sns.heatmap(corr_mid, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5)
ax.set_title("Cross-Product Mid Price Correlation")
fig.tight_layout()
save(fig, "23_cross_product_midprice_corr")

# returns correlation
pivot_ret = pivot_mid.pct_change().replace([np.inf, -np.inf], np.nan)
corr_ret = pivot_ret.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_ret, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5)
ax.set_title("Cross-Product Return Correlation")
fig.tight_layout()
save(fig, "23b_cross_product_return_corr")

# ══════════════════════════════════════════════════════════════════════════════
# 24. TRADER PAIR TRADE FLOW — time series of volume between each pair
# ══════════════════════════════════════════════════════════════════════════════
pairs = list(combinations(TRADERS, 2))
n = len(pairs)
ncols_p = 4; nrows_p = int(np.ceil(n / ncols_p))
fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(20, 4 * nrows_p))
axes = axes.flatten()
for i, (b, s) in enumerate(pairs):
    # both directions
    sub1 = trades[(trades['buyer'] == b) & (trades['seller'] == s)]
    sub2 = trades[(trades['buyer'] == s) & (trades['seller'] == b)]
    if not sub1.empty:
        v1 = sub1.groupby('gts')['quantity'].sum().cumsum()
        axes[i].plot(v1.index, v1.values, label=f"{b}→{s}", lw=1.2)
    if not sub2.empty:
        v2 = sub2.groupby('gts')['quantity'].sum().cumsum()
        axes[i].plot(v2.index, v2.values, label=f"{s}→{b}", lw=1.2, ls='--')
    axes[i].set_title(f"{b} ↔ {s}", fontsize=8)
    axes[i].legend(fontsize=6); axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Cumulative Volume — Each Trader Pair", fontsize=13)
fig.tight_layout()
save(fig, "24_trader_pair_volume_timeseries")

# ══════════════════════════════════════════════════════════════════════════════
# 25. TRADER DOMINANCE — fraction of total volume
# ══════════════════════════════════════════════════════════════════════════════
total_vol = trades['quantity'].sum()
buy_frac  = (buy_vol  / total_vol * 100).round(2)
sell_frac = (sell_vol / total_vol * 100).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].pie(buy_frac, labels=TRADERS, colors=[TRADER_COLOR[t] for t in TRADERS],
            autopct='%1.1f%%', startangle=140)
axes[0].set_title("Buy Volume Share")
axes[1].pie(sell_frac, labels=TRADERS, colors=[TRADER_COLOR[t] for t in TRADERS],
            autopct='%1.1f%%', startangle=140)
axes[1].set_title("Sell Volume Share")
fig.suptitle("Trader Volume Dominance", fontsize=14)
fig.tight_layout()
save(fig, "25_trader_volume_dominance_pie")

# ══════════════════════════════════════════════════════════════════════════════
# 26. PRICE IMPACT — trade size vs spread from mid
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = trades[trades['symbol'] == prod].dropna(subset=['trade_spread', 'quantity'])
    axes[i].scatter(sub['quantity'], sub['trade_spread'].abs(), alpha=0.4, s=10, color=PRODUCT_COLOR[prod])
    axes[i].set_title(f"Price Impact — {prod}")
    axes[i].set_xlabel("Trade Size"); axes[i].set_ylabel("|Spread from Mid|")
    axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Price Impact: Trade Size vs |Spread from Mid|", fontsize=14)
fig.tight_layout()
save(fig, "26_price_impact")

# ══════════════════════════════════════════════════════════════════════════════
# 27. AUTOCORRELATION of MID PRICE RETURNS per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for prod in PRODUCTS:
    sub = prices[prices['product'] == prod]['mid_calc'].pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 50: continue
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(sub,  ax=axes[0], lags=40, title=f"ACF — {prod}")
    plot_pacf(sub, ax=axes[1], lags=40, title=f"PACF — {prod}", method='ywm')
    fig.tight_layout()
    save(fig, f"27_acf_pacf_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 28. MARKET MAKING — traders who both buy and sell same product
# ══════════════════════════════════════════════════════════════════════════════
mm_rows = []
for trader in TRADERS:
    for prod in PRODUCTS:
        b = trades[(trades['buyer']  == trader) & (trades['symbol'] == prod)]['quantity'].sum()
        s = trades[(trades['seller'] == trader) & (trades['symbol'] == prod)]['quantity'].sum()
        if b > 0 and s > 0:
            mm_rows.append({'trader': trader, 'symbol': prod, 'buy_vol': b, 'sell_vol': s,
                            'turnover': b + s, 'imbalance': abs(b - s) / (b + s)})
mm_df = pd.DataFrame(mm_rows)

if not mm_df.empty:
    pvt_turn = mm_df.pivot(index='trader', columns='symbol', values='turnover').fillna(0)
    pvt_imb  = mm_df.pivot(index='trader', columns='symbol', values='imbalance').fillna(1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(pvt_turn, annot=True, fmt='.0f', cmap='Blues', ax=axes[0], linewidths=0.4)
    axes[0].set_title("Market Making Turnover (Buy+Sell)")
    sns.heatmap(pvt_imb, annot=True, fmt='.2f', cmap='Reds', ax=axes[1], linewidths=0.4)
    axes[1].set_title("Directional Imbalance |B−S|/(B+S)")
    fig.suptitle("Market Maker Analysis", fontsize=13)
    fig.tight_layout()
    save(fig, "28_market_maker_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 29. TRADE PRICE DEVIATION per TRADER (buyer premium / seller discount)
# ══════════════════════════════════════════════════════════════════════════════
buy_premium  = trades.groupby('buyer')['trade_spread'].mean().reindex(TRADERS)
sell_discount= trades.groupby('seller')['trade_spread'].mean().reindex(TRADERS)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_b = ['green' if v >= 0 else 'red' for v in buy_premium.fillna(0)]
colors_s = ['red' if v >= 0 else 'green' for v in sell_discount.fillna(0)]
axes[0].bar(buy_premium.index, buy_premium.fillna(0), color=colors_b)
axes[0].axhline(0, color='black', lw=1)
axes[0].set_title("Avg Buy Premium vs Mid"); axes[0].set_xlabel("Buyer")
axes[0].set_xticklabels(buy_premium.index, rotation=30)
axes[1].bar(sell_discount.index, sell_discount.fillna(0), color=colors_s)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_title("Avg Sell Deviation vs Mid"); axes[1].set_xlabel("Seller")
axes[1].set_xticklabels(sell_discount.index, rotation=30)
for ax in axes: ax.grid(True, alpha=0.3, axis='y')
fig.suptitle("Trade Price Deviation from Mid per Trader", fontsize=13)
fig.tight_layout()
save(fig, "29_trade_price_deviation_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 30. CUMULATIVE TRADE VOLUME per PRODUCT over TIME
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
for prod in PRODUCTS:
    sub = trades[trades['symbol'] == prod].sort_values('gts')
    cum = sub.groupby('gts')['quantity'].sum().cumsum()
    ax.plot(cum.index, cum.values, label=prod, lw=1.2)
ax.set_title("Cumulative Trade Volume per Product")
ax.set_xlabel("Global Timestamp"); ax.set_ylabel("Cumulative Volume")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "30_cumulative_volume_product")

# ══════════════════════════════════════════════════════════════════════════════
# 31. TRADER PAIR FREQUENCY OVER TIME (heatmap: time bucket × pair)
# ══════════════════════════════════════════════════════════════════════════════
trades['pair'] = trades['buyer'] + "→" + trades['seller']
pair_heat = trades.groupby(['ts_bucket', 'pair']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(16, max(6, pair_heat.shape[1] * 0.5)))
sns.heatmap(pair_heat.T, cmap='YlOrRd', ax=ax, linewidths=0.0)
ax.set_title("Trader Pair Trade Frequency — Time Bucket × Pair")
ax.set_xlabel("Timestamp Bucket"); ax.set_ylabel("Buyer→Seller")
fig.tight_layout()
save(fig, "31_pair_freq_timebucket")

# ══════════════════════════════════════════════════════════════════════════════
# 32. PnL PER DAY BAR
# ══════════════════════════════════════════════════════════════════════════════
daily_pnl = prices.groupby(['day', 'product'])['profit_and_loss'].last().unstack(fill_value=0)
daily_pnl['total'] = daily_pnl.sum(axis=1)
fig, ax = plt.subplots(figsize=(12, 5))
daily_pnl.drop(columns='total').plot(kind='bar', ax=ax, colormap='tab20', edgecolor='none')
ax.set_title("PnL per Product per Day")
ax.set_xlabel("Day"); ax.set_ylabel("PnL")
ax.set_xticklabels([f"Day {d}" for d in DAYS], rotation=0)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save(fig, "32_pnl_per_product_per_day")

# ══════════════════════════════════════════════════════════════════════════════
# 33. SPREAD FROM MID — time series per trader (as buyer and seller)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
axes[0].plot([], [])  # placeholder
axes[0].set_title("All — Spread from Mid")
all_s = trades.dropna(subset=['trade_spread'])
axes[0].scatter(all_s['gts'], all_s['trade_spread'], s=4, alpha=0.2, color='grey')
axes[0].axhline(0, color='black', lw=0.8)
for i, trader in enumerate(TRADERS, start=1):
    sub_b = trades[(trades['buyer']  == trader)].dropna(subset=['trade_spread'])
    sub_s = trades[(trades['seller'] == trader)].dropna(subset=['trade_spread'])
    if not sub_b.empty:
        axes[i].scatter(sub_b['gts'], sub_b['trade_spread'], s=6, alpha=0.5, color='green', label='as Buyer')
    if not sub_s.empty:
        axes[i].scatter(sub_s['gts'], sub_s['trade_spread'], s=6, alpha=0.5, color='red',   label='as Seller')
    axes[i].axhline(0, color='black', lw=0.7)
    axes[i].set_title(f"{trader} — Spread from Mid")
    axes[i].legend(fontsize=7); axes[i].grid(True, alpha=0.3)
fig.suptitle("Trade Spread from Mid — per Trader", fontsize=14)
fig.tight_layout()
save(fig, "33_spread_from_mid_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 34. OPTION STRIKE PROFILE — volume at each strike
# ══════════════════════════════════════════════════════════════════════════════
if OPTION_PRODUCTS:
    option_trades = trades[trades['symbol'].isin(OPTION_PRODUCTS)].copy()
    option_trades['strike'] = option_trades['symbol'].str.extract(r'VEV_(\d+)').astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    strike_vol = option_trades.groupby('strike')['quantity'].sum()
    strike_ct  = option_trades.groupby('strike').size()
    axes[0].bar(strike_vol.index, strike_vol.values, width=40, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_title("Option Volume by Strike"); axes[0].set_xlabel("Strike"); axes[0].set_ylabel("Volume")
    axes[1].bar(strike_ct.index,  strike_ct.values,  width=40, color='orange',   edgecolor='black', alpha=0.8)
    axes[1].set_title("Option Trade Count by Strike"); axes[1].set_xlabel("Strike"); axes[1].set_ylabel("Count")
    for ax in axes: ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle("Option Strike Profile", fontsize=13)
    fig.tight_layout()
    save(fig, "34_option_strike_profile")

    # volume per strike per trader
    pvt_opt = option_trades.groupby(['strike', 'buyer'])['quantity'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    pvt_opt.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', edgecolor='none')
    ax.set_title("Option Buy Volume per Strike per Trader")
    ax.set_xlabel("Strike"); ax.set_ylabel("Volume")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, "34b_option_strike_trader_volume")

# ══════════════════════════════════════════════════════════════════════════════
# 35. ROLLING TRADE PRICE MEAN per TRADER
# ══════════════════════════════════════════════════════════════════════════════
for prod in SPOT_PRODUCTS:
    sub_prod = trades[trades['symbol'] == prod].sort_values('gts')
    if sub_prod.empty: continue
    fig, ax = plt.subplots(figsize=(14, 5))
    mid_line = prices[prices['product'] == prod].sort_values('gts')
    ax.plot(mid_line['gts'], mid_line['mid_calc'], color='black', lw=1, alpha=0.5, label='Mid')
    for trader in TRADERS:
        sub_t = sub_prod[(sub_prod['buyer'] == trader) | (sub_prod['seller'] == trader)]
        if sub_t.empty: continue
        roll = sub_t.set_index('gts')['price'].rolling(10, min_periods=1).mean()
        ax.plot(roll.index, roll.values, label=trader, color=TRADER_COLOR[trader], lw=1.0, alpha=0.8)
    ax.set_title(f"Rolling Trade Price (w=10) per Trader — {prod}")
    ax.set_xlabel("gts"); ax.set_ylabel("Price")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"35_rolling_trade_price_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 36. TRADE TIMING — per-trader inter-arrival CDF
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
for trader in TRADERS:
    sub = trades[(trades['buyer'] == trader) | (trades['seller'] == trader)].sort_values('gts')
    iat = sub['gts'].diff().dropna().clip(0)
    if iat.empty: continue
    iat_sorted = np.sort(iat)
    cdf = np.arange(1, len(iat_sorted)+1) / len(iat_sorted)
    ax.plot(iat_sorted, cdf, label=trader, color=TRADER_COLOR[trader], lw=1.5)
ax.set_title("CDF of Inter-Arrival Times per Trader")
ax.set_xlabel("Inter-Arrival Time"); ax.set_ylabel("CDF")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xscale('log')
fig.tight_layout()
save(fig, "36_iat_cdf_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 37. SPREAD DISTRIBUTION per PRODUCT (book spread)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()
for i, prod in enumerate(PRODUCTS):
    sub = prices[prices['product'] == prod]['spread'].dropna()
    if sub.empty: axes[i].set_visible(False); continue
    axes[i].hist(sub, bins=60, color=PRODUCT_COLOR[prod], edgecolor='none', alpha=0.85)
    axes[i].set_title(f"Book Spread Dist — {prod}")
    axes[i].set_xlabel("Ask1 − Bid1"); axes[i].set_ylabel("Count")
    axes[i].grid(True, alpha=0.3)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Order Book Spread Distribution per Product", fontsize=14)
fig.tight_layout()
save(fig, "37_book_spread_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# 38. TRADE VOLUME per DAY per PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
day_prod_vol = trades.groupby(['day', 'symbol'])['quantity'].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 6))
day_prod_vol.plot(kind='bar', ax=ax, colormap='tab20', edgecolor='none')
ax.set_title("Trade Volume per Day per Product")
ax.set_xlabel("Day"); ax.set_ylabel("Volume")
ax.set_xticklabels([f"Day {d}" for d in DAYS], rotation=0)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save(fig, "38_volume_day_product")

# ══════════════════════════════════════════════════════════════════════════════
# 39. PnL INCREMENTAL (diff of PnL) — all products
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(18, 3 * len(PRODUCTS)))
for ax, prod in zip(axes, PRODUCTS):
    sub = prices[prices['product'] == prod].sort_values('gts')
    dpnl = sub.set_index('gts')['profit_and_loss'].diff().dropna()
    ax.plot(dpnl.index, dpnl.values, lw=0.7, color=PRODUCT_COLOR[prod])
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(f"PnL Increment — {prod}")
    ax.set_ylabel("ΔPNL"); ax.grid(True, alpha=0.3)
fig.suptitle("Incremental PnL per Product", fontsize=14, y=1.001)
fig.tight_layout()
save(fig, "39_incremental_pnl")

# ══════════════════════════════════════════════════════════════════════════════
# 40. NETWORK GRAPH — trader interaction (edge weight = volume)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(TRADERS)
    edge_vol = trades.groupby(['buyer', 'seller'])['quantity'].sum()
    for (b, s), vol in edge_vol.items():
        G.add_edge(b, s, weight=vol)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [5 * w / max_w for w in weights]
    node_colors = [TRADER_COLOR[n] for n in G.nodes()]

    nx.draw_networkx(G, pos, ax=axes[0], with_labels=True, node_color=node_colors,
                     node_size=800, width=edge_widths, edge_color='steelblue',
                     arrows=True, arrowsize=20, font_size=9, connectionstyle='arc3,rad=0.1')
    axes[0].set_title("Trader Interaction Network (Volume-weighted)")

    # per product for spot
    for prod in SPOT_PRODUCTS:
        sub = trades[trades['symbol'] == prod]
        G2 = nx.DiGraph()
        G2.add_nodes_from(TRADERS)
        ev2 = sub.groupby(['buyer', 'seller'])['quantity'].sum()
        for (b, s), vol in ev2.items():
            G2.add_edge(b, s, weight=vol)
        w2 = [G2[u][v]['weight'] for u, v in G2.edges()]
        mw2 = max(w2) if w2 else 1
        nx.draw_networkx(G2, pos, ax=axes[1], with_labels=True, node_color=node_colors,
                         node_size=800, width=[5*w/mw2 for w in w2], edge_color='orange',
                         arrows=True, arrowsize=20, font_size=9, connectionstyle='arc3,rad=0.1')
        axes[1].set_title(f"Trader Network — {prod}")
    fig.suptitle("Trader Interaction Networks", fontsize=13)
    fig.tight_layout()
    save(fig, "40_trader_network")
except ImportError:
    print("  networkx not installed, skipping network graph")

# ══════════════════════════════════════════════════════════════════════════════
# 41. PnL vs MID PRICE CHANGE SCATTER
# ══════════════════════════════════════════════════════════════════════════════
for prod in SPOT_PRODUCTS:
    sub_p = prices[prices['product'] == prod].sort_values('gts').copy()
    sub_p['mid_ret'] = sub_p['mid_calc'].pct_change()
    sub_p['pnl_diff'] = sub_p['profit_and_loss'].diff()
    sub_p = sub_p.dropna(subset=['mid_ret', 'pnl_diff'])
    if sub_p.empty: continue
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub_p['mid_ret'], sub_p['pnl_diff'], s=6, alpha=0.4, color=PRODUCT_COLOR[prod])
    ax.set_title(f"PnL Change vs Mid Return — {prod}")
    ax.set_xlabel("Mid Price Return"); ax.set_ylabel("PnL Increment")
    ax.axhline(0, color='black', lw=0.7); ax.axvline(0, color='black', lw=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"41_pnl_vs_midret_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 42. VOLUME-WEIGHTED AVERAGE SPREAD per TRADER
# ══════════════════════════════════════════════════════════════════════════════
vwas_b = trades.groupby('buyer').apply(
    lambda x: (x['trade_spread'].abs() * x['quantity']).sum() / x['quantity'].sum()
).reindex(TRADERS)
vwas_s = trades.groupby('seller').apply(
    lambda x: (x['trade_spread'].abs() * x['quantity']).sum() / x['quantity'].sum()
).reindex(TRADERS)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = range(len(TRADERS))
axes[0].bar(x, vwas_b.fillna(0), color=[TRADER_COLOR[t] for t in TRADERS])
axes[0].set_xticks(x); axes[0].set_xticklabels(TRADERS, rotation=30)
axes[0].set_title("VWAS as Buyer"); axes[0].set_ylabel("|Trade Spread| vol-weighted")
axes[1].bar(x, vwas_s.fillna(0), color=[TRADER_COLOR[t] for t in TRADERS])
axes[1].set_xticks(x); axes[1].set_xticklabels(TRADERS, rotation=30)
axes[1].set_title("VWAS as Seller")
for ax in axes: ax.grid(True, alpha=0.3, axis='y')
fig.suptitle("Volume-Weighted Average |Spread from Mid| per Trader", fontsize=13)
fig.tight_layout()
save(fig, "42_vwas_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 43. BID DEPTH vs ASK DEPTH over TIME
# ══════════════════════════════════════════════════════════════════════════════
for prod in PRODUCTS:
    sub = prices[prices['product'] == prod].sort_values('gts').copy()
    sub['bid_depth'] = sub[['bid_volume_1','bid_volume_2','bid_volume_3']].fillna(0).sum(axis=1)
    sub['ask_depth'] = sub[['ask_volume_1','ask_volume_2','ask_volume_3']].fillna(0).sum(axis=1)
    if sub['bid_depth'].max() == 0 and sub['ask_depth'].max() == 0: continue
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(sub['gts'], sub['bid_depth'], lw=0.8, color='green', label='Bid Depth')
    axes[0].plot(sub['gts'], sub['ask_depth'], lw=0.8, color='red',   label='Ask Depth')
    axes[0].set_title(f"Book Depth — {prod}"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(sub['gts'], sub['bid_depth'] - sub['ask_depth'], lw=0.8, color='purple')
    axes[1].axhline(0, color='black', lw=0.7, ls='--')
    axes[1].set_title("Bid−Ask Depth Imbalance"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"43_book_depth_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 44. TRADE COUNT per HOUR-EQUIVALENT BUCKET per TRADER
# ══════════════════════════════════════════════════════════════════════════════
trades['bucket'] = (trades['timestamp'] // 50000) * 50000
fig, ax = plt.subplots(figsize=(18, 7))
for trader in TRADERS:
    sub = trades[(trades['buyer'] == trader) | (trades['seller'] == trader)]
    ct = sub.groupby('bucket').size()
    ax.plot(ct.index, ct.values, label=trader, color=TRADER_COLOR[trader], lw=1.2, alpha=0.8)
ax.set_title("Trade Count per Time Bucket per Trader")
ax.set_xlabel("Timestamp Bucket"); ax.set_ylabel("Trade Count")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "44_trade_count_per_bucket_per_trader")

# ══════════════════════════════════════════════════════════════════════════════
# 45. BOX PLOTS: trade size per trader × product
# ══════════════════════════════════════════════════════════════════════════════
for prod in PRODUCTS:
    sub = trades[trades['symbol'] == prod].copy()
    sub['trader'] = sub.apply(lambda r: r['buyer'] if r['buyer'] else r['seller'], axis=1)
    data = [sub[(sub['buyer'] == t) | (sub['seller'] == t)]['quantity'].dropna().values
            for t in TRADERS]
    data = [d for d in data if len(d) > 0]
    labels = [t for t, d in zip(TRADERS, [sub[(sub['buyer']==t)|(sub['seller']==t)]['quantity'].values for t in TRADERS]) if len(d) > 0]
    if not data: continue
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    ax.set_title(f"Trade Size Distribution — {prod}")
    ax.set_xlabel("Trader"); ax.set_ylabel("Quantity"); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, f"45_tradesize_boxplot_{prod.replace(' ', '_')}")

# ══════════════════════════════════════════════════════════════════════════════
# 46. CORRELATION: TRADER PAIR TRADE VOLUME PER TIMESTAMP
# ══════════════════════════════════════════════════════════════════════════════
trader_vol_ts = {}
for trader in TRADERS:
    sub = trades[(trades['buyer'] == trader) | (trades['seller'] == trader)]
    trader_vol_ts[trader] = sub.groupby('gts')['quantity'].sum()

tv_df = pd.DataFrame(trader_vol_ts).fillna(0)
corr_tv = tv_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_tv, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation: Trader Activity Volume (per timestamp)")
fig.tight_layout()
save(fig, "46_trader_activity_correlation")

# ══════════════════════════════════════════════════════════════════════════════
# 47. OPTION PRICING: TRADE PRICE vs PRODUCT TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
if OPTION_PRODUCTS:
    fig, ax = plt.subplots(figsize=(16, 7))
    for prod in OPTION_PRODUCTS:
        sub_t = trades[trades['symbol'] == prod].sort_values('gts')
        if sub_t.empty: continue
        ax.scatter(sub_t['gts'], sub_t['price'], s=8, alpha=0.6, label=prod)
    ax.set_title("Option Trade Prices over Time")
    ax.set_xlabel("Global Timestamp"); ax.set_ylabel("Price")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "47_option_trade_prices_time")

# ══════════════════════════════════════════════════════════════════════════════
# 48. PAIR TRADE RATIO — who tends to buy vs sell in each pair
# ══════════════════════════════════════════════════════════════════════════════
pair_dir = trades.groupby(['buyer', 'seller'])['quantity'].sum()
pair_matrix = pair_dir.unstack(fill_value=0).reindex(index=TRADERS, columns=TRADERS, fill_value=0)
pair_ratio = pair_matrix / (pair_matrix + pair_matrix.T + 1e-9)  # fraction as buyer

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pair_ratio, annot=True, fmt='.2f', cmap='PiYG', center=0.5, ax=ax, linewidths=0.5,
            vmin=0, vmax=1)
ax.set_title("Pair Trade Ratio (fraction where row=Buyer, col=Seller)")
ax.set_xlabel("Seller"); ax.set_ylabel("Buyer")
fig.tight_layout()
save(fig, "48_pair_trade_ratio")

# ══════════════════════════════════════════════════════════════════════════════
# 49. CUMULATIVE PnL CHANGE per DAY
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, d in zip(axes, DAYS):
    sub = prices[prices['day'] == d].copy()
    for prod in PRODUCTS:
        ps = sub[sub['product'] == prod].sort_values('timestamp')
        if ps.empty: continue
        cum = ps.set_index('timestamp')['profit_and_loss']
        ax.plot(cum.index, cum.values, label=prod, lw=1.0)
    ax.set_title(f"PnL per Product — Day {d}")
    ax.set_xlabel("Timestamp"); ax.set_ylabel("PnL")
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
fig.suptitle("PnL per Product per Day", fontsize=14)
fig.tight_layout()
save(fig, "49_pnl_per_product_per_day_line")

# ══════════════════════════════════════════════════════════════════════════════
# 50. SUMMARY STATS TABLE — as figure
# ══════════════════════════════════════════════════════════════════════════════
stats_rows = []
for prod in PRODUCTS:
    sub_t = trades[trades['symbol'] == prod]
    sub_p = prices[prices['product'] == prod]
    stats_rows.append({
        'Product': prod,
        'N Trades': len(sub_t),
        'Total Vol': int(sub_t['quantity'].sum()),
        'Avg Size': round(sub_t['quantity'].mean(), 1) if len(sub_t) else 0,
        'Mid μ': round(sub_p['mid_calc'].mean(), 2),
        'Mid σ': round(sub_p['mid_calc'].std(), 2),
        'Spread μ': round(sub_p['spread'].mean(), 2),
        'Final PnL': round(sub_p['profit_and_loss'].iloc[-1] if len(sub_p) else 0, 1),
    })
stats_df = pd.DataFrame(stats_rows)

fig, ax = plt.subplots(figsize=(18, max(4, len(stats_df) * 0.6 + 1)))
ax.axis('off')
tbl = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(True)
ax.set_title("Summary Statistics per Product", fontsize=14, pad=20)
fig.tight_layout()
save(fig, "50_summary_stats_table")

print(f"\nAll plots saved to {OUT}")
print(f"Total files: {len(os.listdir(OUT))}")
