"""
GPU-vectorized parameter sweep backtester.
Uses PyTorch: all N combos run simultaneously, one tensor op per timestamp.
Each combo sees the full independent orderbook (no cross-combo interference).
"""
import os, sys, csv
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "round5")
DAYS = [2, 3, 4]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preload_product(product: str) -> np.ndarray:
    """Load all days for one product → (n_ts, 13) float64.
    cols: bid1 bv1 bid2 bv2 bid3 bv3 ask1 av1 ask2 av2 ask3 av3 timestamp"""
    rows = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row["product"] != product:
                    continue
                def g(k):
                    v = row.get(k, "")
                    return float(v) if v else 0.0
                rows.append([
                    g("bid_price_1"), g("bid_volume_1"),
                    g("bid_price_2"), g("bid_volume_2"),
                    g("bid_price_3"), g("bid_volume_3"),
                    g("ask_price_1"), g("ask_volume_1"),
                    g("ask_price_2"), g("ask_volume_2"),
                    g("ask_price_3"), g("ask_volume_3"),
                    float(row["timestamp"]),
                ])
    return np.array(rows, dtype=np.float64)


def _fill_buys(order_px, qty, BA1, AV1, BA2, AV2, BA3, AV3, zeros):
    """Match buy orders (N,) against up to 3 ask levels. Returns (cash_delta, pos_delta)."""
    rem = qty.clone()
    cd, pd = zeros.clone(), zeros.clone()
    if AV1 > 0 and BA1 > 0:
        mask = rem > 0
        f = torch.where(mask & (BA1 <= order_px), rem.clamp(max=AV1), zeros)
        cd -= BA1 * f; pd += f; rem = (rem - f).clamp(min=0)
    if AV2 > 0 and BA2 > 0:
        mask = rem > 0
        f = torch.where(mask & (BA2 <= order_px), rem.clamp(max=AV2), zeros)
        cd -= BA2 * f; pd += f; rem = (rem - f).clamp(min=0)
    if AV3 > 0 and BA3 > 0:
        mask = rem > 0
        f = torch.where(mask & (BA3 <= order_px), rem.clamp(max=AV3), zeros)
        cd -= BA3 * f; pd += f
    return cd, pd


def _fill_sells(order_px, qty, BB1, BV1, BB2, BV2, BB3, BV3, zeros):
    """Match sell orders (N,) against up to 3 bid levels. Returns (cash_delta, pos_delta)."""
    rem = qty.clone()
    cd, pd = zeros.clone(), zeros.clone()
    if BV1 > 0 and BB1 > 0:
        mask = rem > 0
        f = torch.where(mask & (BB1 >= order_px), rem.clamp(max=BV1), zeros)
        cd += BB1 * f; pd -= f; rem = (rem - f).clamp(min=0)
    if BV2 > 0 and BB2 > 0:
        mask = rem > 0
        f = torch.where(mask & (BB2 >= order_px), rem.clamp(max=BV2), zeros)
        cd += BB2 * f; pd -= f; rem = (rem - f).clamp(min=0)
    if BV3 > 0 and BB3 > 0:
        mask = rem > 0
        f = torch.where(mask & (BB3 >= order_px), rem.clamp(max=BV3), zeros)
        cd += BB3 * f; pd -= f
    return cd, pd


def run_sweep(product: str, combos: list, n_tiers: int, fixed_params: dict) -> np.ndarray:
    """
    Vectorized sweep across all combos on GPU/CPU.
    Returns cash per combo as np.ndarray (n_combos,).
    """
    price_data = preload_product(product)
    N = len(combos)
    zeros = torch.zeros(N, dtype=torch.float64, device=DEVICE)

    def _t(vals):
        return torch.tensor(vals, dtype=torch.float64, device=DEVICE)

    # Tier params: shape (N,) each
    dists = []  # list of (dist_tensor, frac_tensor) per tier
    if n_tiers == 2:
        dists = [
            (_t([c["d1"] for c in combos]), _t([c["f1"] for c in combos])),
            (_t([c["d2"] for c in combos]), torch.ones(N, dtype=torch.float64, device=DEVICE)),
        ]
    else:
        dists = [
            (_t([c["d1"] for c in combos]), _t([c["f1"] for c in combos])),
            (_t([c["d2"] for c in combos]), _t([c["f2"] for c in combos])),
            (_t([c["d3"] for c in combos]), torch.ones(N, dtype=torch.float64, device=DEVICE)),
        ]

    base_px   = float(fixed_params["base"])
    alpha     = 2.0 / (float(fixed_params["ema_period"]) + 1.0)
    obi_w     = float(fixed_params["obi_weight"])
    skew_f    = float(fixed_params["skew_factor"])
    p_lim     = float(fixed_params["position_limit"])
    g_lim     = float(fixed_params["grid_limit"])

    pos  = zeros.clone()
    cash = zeros.clone()
    ema  = -1.0  # scalar EMA (same ema_period for all combos)

    n_ts = price_data.shape[0]

    for t in range(n_ts):
        row = price_data[t]
        BB1, BV1 = row[0], row[1]
        BB2, BV2 = row[2], row[3]
        BB3, BV3 = row[4], row[5]
        BA1, AV1 = row[6], row[7]
        BA2, AV2 = row[8], row[9]
        BA3, AV3 = row[10], row[11]
        ts = row[12]

        if BA1 <= 0 or BB1 <= 0:
            continue

        mid = (BB1 + BA1) / 2.0
        ema = mid if ema < 0 else mid * alpha + ema * (1.0 - alpha)

        bvol = BV1 + BV2 + BV3
        avol = AV1 + AV2 + AV3
        obi  = (bvol - avol) / (bvol + avol + 1e-9)
        fair = ema + obi * obi_w  # scalar

        epl = p_lim if ts < 900000 else float(int(p_lim) // 2)
        egl = g_lim if ts < 900000 else float(int(g_lim) // 2)
        mm_lim = epl - egl

        orig_pos = pos.clone()

        # ── End-of-day wind-down ─────────────────────────────────────────
        sim_bid = orig_pos.clone()
        sim_ask = orig_pos.clone()
        if ts >= 900000:
            # Forced sell: pos > epl
            eod_sell_qty = (orig_pos - epl).clamp(min=0)
            eod_sell_px  = torch.full_like(pos, max(BA1 - 1.0, float(int(ema + 0.9999))))
            cd, pd = _fill_sells(eod_sell_px, eod_sell_qty, BB1, BV1, BB2, BV2, BB3, BV3, zeros)
            cash += cd; pos += pd; sim_ask -= eod_sell_qty

            # Forced buy: pos < -epl
            eod_buy_qty = (-orig_pos - epl).clamp(min=0)
            eod_buy_px  = torch.full_like(pos, min(BB1 + 1.0, float(int(ema))))
            cd, pd = _fill_buys(eod_buy_px, eod_buy_qty, BA1, AV1, BA2, AV2, BA3, AV3, zeros)
            cash += cd; pos += pd; sim_bid += eod_buy_qty

        # ── Grid simulation ───────────────────────────────────────────────
        for dist_t, frac_t in dists:
            tgt = torch.floor(torch.full_like(pos, egl) * frac_t)
            ba = (tgt - sim_bid).clamp(min=0)
            sa = (sim_ask + tgt).clamp(min=0)
            sim_bid = sim_bid + ba
            sim_ask = sim_ask - sa

            # Grid buys at base_px - dist
            gpx_buy = (torch.full_like(pos, base_px) - dist_t)
            cd, pd = _fill_buys(gpx_buy, ba, BA1, AV1, BA2, AV2, BA3, AV3, zeros)
            cash += cd; pos += pd

            # Grid sells at base_px + dist
            gpx_sell = (torch.full_like(pos, base_px) + dist_t)
            cd, pd = _fill_sells(gpx_sell, sa, BB1, BV1, BB2, BV2, BB3, BV3, zeros)
            cash += cd; pos += pd

        # ── MM orders ────────────────────────────────────────────────────
        gbv = sim_bid - orig_pos
        gav = orig_pos - sim_ask

        buy_cap  = (torch.full_like(pos, epl) - orig_pos).clamp(min=0)
        sell_cap = (torch.full_like(pos, epl) + orig_pos).clamp(min=0)

        mm_bq = torch.minimum(torch.full_like(pos, mm_lim), (buy_cap - gbv).clamp(min=0))
        mm_aq = torch.minimum(torch.full_like(pos, mm_lim), (sell_cap - gav).clamp(min=0))

        sfair    = fair - orig_pos * skew_f
        mm_bpx   = torch.minimum(torch.full_like(sfair, BB1 + 1.0), torch.floor(sfair) - 1.0)
        mm_apx   = torch.maximum(torch.full_like(sfair, BA1 - 1.0), torch.ceil(sfair) + 1.0)

        cd, pd = _fill_buys(mm_bpx, mm_bq, BA1, AV1, BA2, AV2, BA3, AV3, zeros)
        cash += cd; pos += pd

        cd, pd = _fill_sells(mm_apx, mm_aq, BB1, BV1, BB2, BV2, BB3, BV3, zeros)
        cash += cd; pos += pd

    return cash.cpu().numpy()
