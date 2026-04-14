import numpy as np
import polars as pl
from PyQt6.QtCore import QRectF

def build_ob_heatmap(p_df, product, day, continuous_ts=False, buy_palette=None, sell_palette=None):
    flt = p_df.filter(pl.col('product') == product)
    min_day = p_df['day'].min() if 'day' in p_df.columns else 0
    try:
        if day != 'All': flt = flt.filter(pl.col('day') == int(day))
    except (ValueError, TypeError):
        pass
    flt = flt.sort(['day', 'timestamp'])
    
    if day == 'All' and 'day' in flt.columns and not continuous_ts:
        times = (flt['timestamp'] + (flt['day'] - min_day) * 1_000_000).to_numpy()
    else:
        times = flt['timestamp'].to_numpy()
        
    bid_p_cols = sorted([c for c in flt.columns if 'bid_price_' in c], key=lambda x: int(x.split('_')[-1]))
    ask_p_cols = sorted([c for c in flt.columns if 'ask_price_' in c], key=lambda x: int(x.split('_')[-1]))

    all_p, all_v = [], []
    for side in ['bid', 'ask']:
        p_cols = bid_p_cols if side == 'bid' else ask_p_cols
        for pc in p_cols:
            vc = f"{side}_volume_{pc.split('_')[-1]}"
            if vc not in flt.columns: continue
            pa = flt[pc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            va = flt[vc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            valid = np.isfinite(pa) & (pa > 0) & np.isfinite(va) & (va > 0)
            all_p.append(pa[valid]); all_v.append(va[valid])

    if not all_p: return None
    concat_p = np.concatenate(all_p)
    if len(concat_p) == 0: return None
    max_vol = max(np.max(np.concatenate(all_v)), 1.0)
    
    p_min_raw, p_max_raw = np.min(concat_p), np.max(concat_p)
    prices_are_int = np.all(concat_p == np.floor(concat_p))
    if prices_are_int:
        p_min = float(int(p_min_raw))
        p_max = float(int(p_max_raw))
        price_levels = np.arange(p_min, p_max + 1, dtype=float)
        step = 1.0
    else:
        N_BINS = 500
        price_levels = np.linspace(p_min_raw, p_max_raw, N_BINS)
        p_min = p_min_raw
        step = (p_max_raw - p_min_raw) / (N_BINS - 1) if N_BINS > 1 else 1.0

    w, h = len(times), len(price_levels)
    raw_vol = np.zeros((h, w), dtype=float)
    red_l, blue_l = np.zeros(h * w, np.uint8), np.zeros(h * w, np.uint8)

    for side in ['bid', 'ask']:
        p_cols = bid_p_cols if side == 'bid' else ask_p_cols
        for pc in p_cols:
            vc = f"{side}_volume_{pc.split('_')[-1]}"
            if vc not in flt.columns: continue
            pa = flt[pc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            va = flt[vc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            mask = np.isfinite(pa) & (pa > 0)
            v_idx = np.where(mask)[0]
            if len(v_idx) == 0: continue
            y_idxs = np.clip(np.round((pa[mask] - p_min) / step).astype(np.int64), 0, h - 1)
            flat_idxs = y_idxs * w + v_idx.astype(np.int64)
            
            lvl = np.clip(va[mask] / max_vol * 10, 0, 9).astype(np.uint8)
            np.maximum.at(red_l if side == 'ask' else blue_l, flat_idxs, lvl)
            raw_vol.flat[flat_idxs] += va[mask]

    img = np.zeros((h, w, 4), np.uint8)
    red_img_l = red_l.reshape(h, w)
    blue_img_l = blue_l.reshape(h, w)
    
    for l in range(10):
        mask_r = (red_img_l == l) & (red_img_l > 0)
        if np.any(mask_r):
            img[mask_r] = sell_palette[l] + [200]
            
        mask_b = (blue_img_l == l) & (blue_img_l > 0)
        if np.any(mask_b):
            img[mask_b] = buy_palette[l] + [200]

    return {
        'img': img, 
        'times': times, 
        'levels': price_levels, 
        'raw_vol': raw_vol, 
        'max_vol': max_vol,
        'rect': [times[0], times[-1], price_levels[0], price_levels[-1]]
    }

def build_order_placement_heatmap(orders, product, day, times, price_levels, continuous_ts=False, min_day=0, buy_palette=None, sell_palette=None):
    try:
        if day != 'All':
            day_val = int(day)
            flt = [o for o in orders if (o['product'] == product or o['product'] == '') and o['day'] == day_val]
        else:
            flt = [o for o in orders if (o['product'] == product or o['product'] == '')]
    except:
        flt = [o for o in orders if (o['product'] == product or o['product'] == '')]

    if not flt or len(times) == 0 or len(price_levels) == 0:
        return None

    p_min = price_levels[0]
    h = len(price_levels)
    step = (price_levels[-1] - price_levels[0]) / (h - 1) if h > 1 else 1.0
    w = len(times)
    img = np.zeros((h, w, 4), np.uint8)

    vols = [o['qty'] for o in flt]
    max_vol = max(vols) if vols else 1.0
    ts_to_idx = {ts: i for i, ts in enumerate(times)}

    for o in flt:
        ts = o['ts']
        if day == 'All' and not continuous_ts:
            ts += (o['day'] - min_day) * 1_000_000

        if ts not in ts_to_idx:
            continue

        x = ts_to_idx[ts]
        y = int(np.clip(round((o['price'] - p_min) / step), 0, h - 1))

        side = o['side']
        vol = o['qty']
        lvl = np.clip(int(vol / max_vol * 10), 0, 9)

        color = (buy_palette[lvl] if side == 'BUY' else sell_palette[lvl]) + [220]
        img[y, x] = color

    return {'img': img}

def get_rect(times, levels):
    if len(times) < 1 or len(levels) < 1:
        return QRectF(0, 0, 1, 1)
    x_min, x_max = times[0], times[-1]
    y_min, y_max = levels[0], levels[-1]
    x_step = (times[1] - times[0]) if len(times) > 1 else 100
    # Align pixels: the coordinate (ts, price) should be the CENTER of the pixel.
    # So left edge is ts - 0.5 * step, right edge is ts + 0.5 * step.
    # Total width = (num_pixels) * x_step.
    return QRectF(x_min - 0.5 * x_step, y_min - 0.5, len(times) * x_step, len(levels))
