import json
import os

NOTEBOOK_PATH = r"m:\prosperity 4\dataviz\data_visualizer.ipynb"

code_source = """
import os
import re
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & Helpers ---
DATA_DIR = r"F:\\PROSPERITY4\\Tutorial Round\\TUTORIAL_ROUND_1"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "." # Fallback

# Hardcoded selections
TARGET_ROUND = "0"
TARGET_PRODUCT = "AMETHYSTS"
TARGET_DAY = "All"  # Set to "All" or a specific day string like "-1"
SHOW_SPREAD = True
SHOW_MIDPRICE_OVERLAY = False
VIEWS_TO_SHOW = ["View 1", "View 2", "View 3"] # Remove items to hide views

DEFAULT_COLUMN_MAP = {
    'prices': {
        'product': 'product',
        'day': 'day',
        'timestamp': 'timestamp',
        'mid_price': 'mid_price',
        'bid_price_1': 'bid_price_1',
        'ask_price_1': 'ask_price_1',
        'bid_volume_1': 'bid_volume_1',
        'ask_volume_1': 'ask_volume_1',
        'bid_price_prefix': 'bid_price_',
        'ask_price_prefix': 'ask_price_',
        'bid_volume_prefix': 'bid_volume_',
        'ask_volume_prefix': 'ask_volume_'
    },
    'trades': {
        'symbol': 'symbol',
        'day': 'day',
        'timestamp': 'timestamp',
        'price': 'price'
    }
}

def scan_directory(data_dir):
    rounds = {}
    price_pattern = re.compile(r"prices_round_(\d+)_day_(-?\d+)\\.csv")
    trade_pattern = re.compile(r"trades_round_(\d+)_day_(-?\d+)\\.csv")

    if not os.path.exists(data_dir):
        return rounds

    for f in os.listdir(data_dir):
        if not f.endswith('.csv'): continue
        
        p_match = price_pattern.match(f)
        if p_match:
            rnd = p_match.group(1)
            rounds.setdefault(rnd, {'prices': [], 'trades': []})['prices'].append(f)
            continue
            
        t_match = trade_pattern.match(f)
        if t_match:
            rnd = t_match.group(1)
            rounds.setdefault(rnd, {'prices': [], 'trades': []})['trades'].append(f)
            
    return rounds

def get_column(section, col):
    return DEFAULT_COLUMN_MAP[section][col]

def pc(col): return get_column('prices', col)
def tc(col): return get_column('trades', col)

print(f"Scanning {DATA_DIR} for round {TARGET_ROUND}...")
rounds_map = scan_directory(DATA_DIR)

if TARGET_ROUND not in rounds_map:
    print(f"Error: Round {TARGET_ROUND} not found in {DATA_DIR}.")
else:
    files = rounds_map[TARGET_ROUND]
    p_dfs, t_dfs = [], []
    
    # Load Prices
    for f in files['prices']:
        p_dfs.append(pl.read_csv(os.path.join(DATA_DIR, f), separator=";"))
    prices_df = pl.concat(p_dfs) if p_dfs else None

    # Load Trades
    for f in files['trades']:
        day_match = re.match(r"trades_round_\\d+_day_(-?\\d+)\\.csv", f)
        if day_match:
            day_val = int(day_match.group(1))
            df = pl.read_csv(os.path.join(DATA_DIR, f), separator=";")
            df = df.with_columns(pl.lit(day_val).alias("day"))
            t_dfs.append(df)
    trades_df = pl.concat(t_dfs) if t_dfs else None

    if prices_df is None or len(prices_df) == 0:
        print("No price data loaded.")
    else:
        print(f"Loaded {len(prices_df)} price records and {len(trades_df) if trades_df is not None else 0} trade records.")

        # Filter Polars
        p_f = prices_df.filter(pl.col(pc('product')) == TARGET_PRODUCT)
        t_f = pl.DataFrame()
        if trades_df is not None and len(trades_df) > 0:
            t_f = trades_df.filter(pl.col(tc('symbol')) == TARGET_PRODUCT)

        if TARGET_DAY == "All":
            p_f = p_f.sort([pc('day'), pc('timestamp')])
            if len(t_f) > 0: t_f = t_f.sort([tc('day'), tc('timestamp')])
            
            # Continuous time across days
            DAY_LEN = 1_000_000
            min_day = p_f[pc('day')].min()
            p_f = p_f.with_columns(
                (pl.col(pc('timestamp')) + (pl.col(pc('day')) - min_day) * DAY_LEN).alias("plot_time")
            )
            if len(t_f) > 0:
                t_f = t_f.with_columns(
                    (pl.col(tc('timestamp')) + (pl.col(tc('day')) - min_day) * DAY_LEN).alias("plot_time")
                )
        else:
            d_v = int(TARGET_DAY)
            p_f = p_f.filter(pl.col(pc('day')) == d_v).sort(pc('timestamp'))
            p_f = p_f.with_columns(pl.col(pc('timestamp')).alias("plot_time"))
            if len(t_f) > 0:
                t_f = t_f.filter(pl.col(tc('day')) == d_v).sort(tc('timestamp'))
                t_f = t_f.with_columns(pl.col(tc('timestamp')).alias("plot_time"))

        if len(p_f) == 0:
            print(f"No data matched Product={TARGET_PRODUCT}, Day={TARGET_DAY}.")
        else:
            print("Crunching data and building Plotly figures...")

            # Extract NumPy arrays
            plot_time = p_f['plot_time'].to_numpy()
            mid = p_f[pc('mid_price')].to_numpy()
            bid = p_f[pc('bid_price_1')].to_numpy()
            ask = p_f[pc('ask_price_1')].to_numpy()
            
            bid_vol = p_f[pc('bid_volume_1')].to_numpy()
            ask_vol = p_f[pc('ask_volume_1')].to_numpy()
            total_vol = bid_vol + ask_vol
            obi = np.divide(bid_vol - ask_vol, total_vol, out=np.zeros_like(bid_vol, dtype=float), where=total_vol!=0)
            
            t_time = t_f['plot_time'].to_numpy() if len(t_f) > 0 else np.array([])
            t_price = t_f[tc('price')].to_numpy() if len(t_f) > 0 else np.array([])

            show_v1 = "View 1" in VIEWS_TO_SHOW
            show_v2 = "View 2" in VIEWS_TO_SHOW
            show_v3 = "View 3" in VIEWS_TO_SHOW
            
            num_rows = sum([show_v1, show_v2, show_v3])
                
            if num_rows > 0:
                titles = []
                if show_v1: titles.append("Market Microstructure (Dual-Tone Spread)")
                if show_v2: titles.append("Order Book Liquidity & OBI")
                if show_v3: titles.append("Order Book Heatmap")
                
                fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
                
                row_idx = 1
                
                # Decimation to avoid lag on massive datasets
                def decimate(x, y, max_pts=10000):
                    if len(x) <= max_pts: return x, y
                    step = max(1, len(x) // max_pts)
                    return x[::step], y[::step]
                    
                time_dec, mid_dec = decimate(plot_time, mid)
                _, bid_dec = decimate(plot_time, bid)
                _, ask_dec = decimate(plot_time, ask)

                # --- View 1 ---
                if show_v1:
                    if SHOW_SPREAD:
                        fig.add_trace(go.Scatter(x=time_dec, y=ask_dec, line=dict(color='#FF2D55', width=1), name='Ask', legendgroup='Spread'), row=row_idx, col=1)
                        fig.add_trace(go.Scatter(x=time_dec, y=bid_dec, fill='tonexty', fillcolor='rgba(255, 45, 85, 0.2)', line=dict(color='rgba(255,45,85,0)'), name='Ask Fill', hoverinfo='skip', showlegend=False, legendgroup='Spread'), row=row_idx, col=1)
                        
                        fig.add_trace(go.Scatter(x=time_dec, y=mid_dec, line=dict(color='#00BFFF', width=2), name='Mid Price'), row=row_idx, col=1)
                        
                        fig.add_trace(go.Scatter(x=time_dec, y=bid_dec, fill='tonexty', fillcolor='rgba(57, 255, 20, 0.2)', line=dict(color='#39FF14', width=1), name='Bid', legendgroup='Spread'), row=row_idx, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=time_dec, y=mid_dec, line=dict(color='#00BFFF', width=2), name='Mid Price'), row=row_idx, col=1)

                    if len(t_time) > 0:
                        tt_dec, tp_dec = decimate(t_time, t_price)
                        fig.add_trace(go.Scatter(x=tt_dec, y=tp_dec, mode='markers', marker=dict(color='#FF1493', size=5, symbol='x'), name='Executions'), row=row_idx, col=1)
                        
                    row_idx += 1

                # --- View 2 ---
                if show_v2:
                    _, bv_dec = decimate(plot_time, bid_vol)
                    _, av_dec = decimate(plot_time, ask_vol)
                    _, obi_dec = decimate(plot_time, obi)
                    
                    max_vol = max(np.max(bv_dec), np.max(av_dec))
                    scaled_obi = obi_dec * (max_vol if max_vol > 0 else 1)
                    
                    fig.add_trace(go.Scatter(x=time_dec, y=bv_dec, fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.3)', line=dict(color='#39FF14', width=1), name='Bid Vol'), row=row_idx, col=1)
                    fig.add_trace(go.Scatter(x=time_dec, y=-av_dec, fill='tozeroy', fillcolor='rgba(255, 45, 85, 0.3)', line=dict(color='#FF2D55', width=1), name='Ask Vol'), row=row_idx, col=1)
                    fig.add_trace(go.Scatter(x=time_dec, y=scaled_obi, line=dict(color='#FFD700', width=2), name='OBI (Scaled)'), row=row_idx, col=1)
                    
                    row_idx += 1

                # --- View 3 (Heatmap) ---
                if show_v3:
                    # Build sparse heatmap from Polars
                    b_pref = pc('bid_price_prefix'); a_pref = pc('ask_price_prefix')
                    bv_pref = pc('bid_volume_prefix'); av_pref = pc('ask_volume_prefix')
                    
                    b_cols = sorted([c for c in p_f.columns if c.startswith(b_pref)], key=lambda x: int(x.rsplit("_", 1)[1]))
                    a_cols = sorted([c for c in p_f.columns if c.startswith(a_pref)], key=lambda x: int(x.rsplit("_", 1)[1]))
                    
                    b_levels = [(c, f"{bv_pref}{c.rsplit('_', 1)[1]}") for c in b_cols if f"{bv_pref}{c.rsplit('_', 1)[1]}" in p_f.columns]
                    a_levels = [(c, f"{av_pref}{c.rsplit('_', 1)[1]}") for c in a_cols if f"{av_pref}{c.rsplit('_', 1)[1]}" in p_f.columns]
                    
                    all_price_arrays = []
                    for p_col, v_col in b_levels + a_levels:
                        p_arr = p_f[p_col].to_numpy()
                        v_arr = p_f[v_col].to_numpy()
                        valid = (p_arr > 0) & (v_arr > 0)
                        if np.any(valid): all_price_arrays.append(p_arr[valid])
                        
                    if all_price_arrays:
                        price_levels = np.unique(np.concatenate(all_price_arrays))
                        
                        n_t = len(time_dec)
                        step = max(1, len(plot_time) // 10000)
                        
                        y_bins = np.linspace(price_levels.min(), price_levels.max(), min(50, len(price_levels)))
                        heatmap_z = np.zeros((len(y_bins)-1, n_t))
                        
                        for p_col, v_col in b_levels:
                            p_arr = p_f[p_col].to_numpy()[::step][:n_t]
                            v_arr = p_f[v_col].to_numpy()[::step][:n_t]
                            
                            y_idx = np.digitize(p_arr, y_bins) - 1
                            valid = (y_idx >= 0) & (y_idx < len(y_bins)-1) & (v_arr > 0)
                            heatmap_z[y_idx[valid], np.arange(n_t)[valid]] -= v_arr[valid]
                            
                        for p_col, v_col in a_levels:
                            p_arr = p_f[p_col].to_numpy()[::step][:n_t]
                            v_arr = p_f[v_col].to_numpy()[::step][:n_t]
                            
                            y_idx = np.digitize(p_arr, y_bins) - 1
                            valid = (y_idx >= 0) & (y_idx < len(y_bins)-1) & (v_arr > 0)
                            heatmap_z[y_idx[valid], np.arange(n_t)[valid]] += v_arr[valid]

                        fig.add_trace(go.Heatmap(
                            x=time_dec, 
                            y=(y_bins[:-1] + y_bins[1:])/2, 
                            z=heatmap_z,
                            colorscale=[[0, 'blue'], [0.5, 'black'], [1, 'red']],
                            zmid=0,
                            showscale=False,
                            name='Order Book',
                            hoverinfo='none'
                        ), row=row_idx, col=1)

                    if SHOW_MIDPRICE_OVERLAY:
                        fig.add_trace(go.Scatter(x=time_dec, y=mid_dec, line=dict(color='#00BFFF', width=2), name='Mid (OB overlay)'), row=row_idx, col=1)
                        
                    row_idx += 1

                fig.update_layout(
                    height=350 * num_rows,
                    margin=dict(l=40, r=40, t=40, b=40),
                    template="plotly_dark",
                    hovermode="x unified",
                    dragmode='zoom'
                )
                
                fig.show()
"""

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Prosperity 4 - Smart Alpha Scanner\n",
            "Configure variables such as `TARGET_PRODUCT` below and run the cell to plot."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in code_source.split("\\n")]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Created notebook at {NOTEBOOK_PATH}")
