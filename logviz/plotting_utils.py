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
    red_l = np.zeros(h * w, np.uint8)
    blue_l = np.zeros(h * w, np.uint8)
    has_r = np.zeros(h * w, bool)
    has_b = np.zeros(h * w, bool)

    for side in ['bid', 'ask']:
        p_cols = bid_p_cols if side == 'bid' else ask_p_cols
        for pc in p_cols:
            vc = f"{side}_volume_{pc.split('_')[-1]}"
            if vc not in flt.columns: continue
            pa = flt[pc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            va = flt[vc].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
            mask = np.isfinite(pa) & (pa > 0) & np.isfinite(va) & (va > 0)
            v_idx = np.where(mask)[0]
            if len(v_idx) == 0: continue
            y_idxs = np.clip(np.round((pa[mask] - p_min) / step).astype(np.int64), 0, h - 1)
            flat_idxs = y_idxs * w + v_idx.astype(np.int64)

            lvl = np.clip(va[mask] / max_vol * 10, 0, 9).astype(np.uint8)
            if side == 'ask':
                np.maximum.at(red_l, flat_idxs, lvl)
                has_r[flat_idxs] = True
            else:
                np.maximum.at(blue_l, flat_idxs, lvl)
                has_b[flat_idxs] = True
            np.add.at(raw_vol.reshape(-1), flat_idxs, va[mask])

    img = np.zeros((h, w, 4), np.uint8)
    red_img_l = red_l.reshape(h, w)
    blue_img_l = blue_l.reshape(h, w)
    has_r_2d = has_r.reshape(h, w)
    has_b_2d = has_b.reshape(h, w)

    for l in range(10):
        mask_r = (red_img_l == l) & has_r_2d
        if np.any(mask_r):
            img[mask_r] = sell_palette[l] + [200]

        mask_b = (blue_img_l == l) & has_b_2d
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

def build_order_placement_heatmap(orders, product, day, times, continuous_ts=False, min_day=0, buy_palette=None, sell_palette=None):
    try:
        if day != 'All':
            day_val = int(day)
            flt = [o for o in orders if (o['product'] == product or o['product'] == '') and o['day'] == day_val]
        else:
            flt = [o for o in orders if (o['product'] == product or o['product'] == '')]
    except:
        flt = [o for o in orders if (o['product'] == product or o['product'] == '')]

    if not flt or len(times) == 0:
        return None

    prices = [o['price'] for o in flt]
    p_min_raw, p_max_raw = min(prices), max(prices)
    prices_are_int = all(p == int(p) for p in prices)
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

    h = len(price_levels)
    w = len(times)
    img = np.zeros((h, w, 4), np.uint8)

    vols = [o['qty'] for o in flt]
    max_vol = max(vols) if vols else 1.0
    times_i64 = times.astype(np.int64)

    for o in flt:
        ts = int(o['ts'])
        if day == 'All' and not continuous_ts:
            ts += (int(o.get('day', 0)) - int(min_day)) * 1_000_000

        # Find nearest timestamp index
        idx = np.searchsorted(times_i64, ts)
        if idx > 0 and (idx == len(times_i64) or abs(ts - times_i64[idx-1]) < abs(ts - times_i64[idx])):
            idx -= 1
        
        if idx < 0 or idx >= w or abs(ts - times_i64[idx]) > 5000: # Max 5s deviation
            continue

        x = idx
        y = int(round((o['price'] - p_min) / step))
        if y < 0 or y >= h:
            continue

        side = o['side']
        vol = o['qty']
        lvl = np.clip(int(vol / max_vol * 10), 0, 9)

        color = (buy_palette[lvl] if side == 'BUY' else sell_palette[lvl]) + [220]
        img[y, x] = color

    return {'img': img, 'levels': price_levels}

def get_rect(times, levels):
    if len(times) < 1 or len(levels) < 1:
        return QRectF(0, 0, 1, 1)
    
    x_min = times[0]
    if len(times) > 1:
        x_step = (times[-1] - times[0]) / (len(times) - 1)
    else:
        x_step = 100.0
        
    y_min = levels[0]
    if len(levels) > 1:
        y_step = (levels[-1] - levels[0]) / (len(levels) - 1)
    else:
        y_step = 1.0

        
    # Align pixels: the coordinate (ts, price) should be the CENTER of the pixel.
    # So left edge is ts - 0.5 * step, right edge is ts + 0.5 * step.
    return QRectF(x_min - 0.5 * x_step, y_min - 0.5 * y_step, len(times) * x_step, len(levels) * y_step)
