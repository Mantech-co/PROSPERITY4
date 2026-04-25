import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from datamodel import OrderDepth, TradingState, Order

# ── Black-Scholes helpers ─────────────────────────────────────────────────────
_TS_SPAN = 1_000_000
_TDAY    = 252


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
    return S * _ncdf(d1) - K * _ncdf(d1 - sigma * sq)


def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


def _implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    if T < 1e-9 or price <= 0 or price <= max(S - K, 0.0) + 1e-6:
        return None
    lo, hi = 1e-4, 20.0
    if _bs_call(S, K, T, hi) < price:
        return None
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) * 0.5


def _tte_years(day: int, timestamp: int) -> float:
    return max((8 - day) - timestamp / _TS_SPAN, 1e-9) / _TDAY


def _median(lst: List[float]) -> float:
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) * 0.5


# ── Params ────────────────────────────────────────────────────────────────────
_PARAMS_FILE = Path(__file__).parent / "params.json"
if _PARAMS_FILE.exists():
    with open(_PARAMS_FILE) as _f:
        TUNING_PARAMS = json.load(_f)
else:
    TUNING_PARAMS = {}


# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    def _emit(self, line: str) -> None:
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)

    def log(self, **series: float) -> None:
        clean: dict[str, float] = {}
        for k, v in series.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        if clean:
            self._emit(self.PREFIX + json.dumps(clean, separators=(',', ':')))

    def log_order(self, product: str, side: str, price: int, qty: int, tag: str) -> None:
        self._emit(f'LOGORDER:{product}:{side}:{price}:{qty}:{tag}')

    def flush(self) -> str:
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out


# ── Options Arb ───────────────────────────────────────────────────────────────
class OptionsArb:
    """
    Each tick, for each active strike:
      - Bid IV is computed using underlying ask as S  (cost to hedge long option by selling und)
      - Ask IV is computed using underlying bid as S  (cost to hedge short option by buying und)

    Signal:
      - median_bid_iv - bid_iv > threshold  → BUY option at ask, SELL underlying at bid
      - ask_iv - median_ask_iv > threshold  → SELL option at bid, BUY underlying at ask

    Sizing: whichever leg (option or underlying) hits its volume/limit ceiling first
    determines the trade size; the other leg is scaled down to match delta neutrality.
    """
    UNDERLYING = "VELVETFRUIT_EXTRACT"
    UND_LIMIT  = 200
    STRIKES    = [5200]

    def __init__(self, logger: Logger, params: Dict[str, Any]):
        self.logger            = logger
        self.iv_window         = int(params.get("iv_window", 100))
        self.iv_diff_threshold = float(params.get("iv_diff_threshold", 0.002))
        self.option_limit      = int(params.get("option_limit", 200))

    def _track_day(self, state: TradingState, sd: Dict[str, Any]) -> int:
        if "opt_prev_ts" not in sd:
            sd["opt_day"] = 0
        elif state.timestamp < sd["opt_prev_ts"]:
            sd["opt_day"] = sd.get("opt_day", 0) + 1
        sd["opt_prev_ts"] = state.timestamp
        return sd["opt_day"]

    def _size(self, opt_avail: int, opt_cap: int,
              und_avail: int, und_cap: int, delta: float):
        """
        Returns (opt_qty, und_qty) such that und_qty = round(opt_qty * delta),
        scaled so neither leg exceeds its available volume or position capacity.
        The smaller constraint exhausts first; the other is adjusted to match.
        """
        if delta < 1e-6:
            return 0, 0
        opt_max = min(opt_avail, opt_cap)
        und_max = min(und_avail, und_cap)
        if opt_max <= 0 or und_max <= 0:
            return 0, 0

        desired_und = round(opt_max * delta)
        if desired_und <= und_max:
            # option is the binding constraint
            return opt_max, desired_und
        else:
            # underlying is the binding constraint — scale option down
            und_qty = und_max
            opt_qty = math.floor(und_qty / delta)
            und_qty = round(opt_qty * delta)
            return opt_qty, und_qty

    def run(self, state: TradingState, sd: Dict[str, Any]) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        day = self._track_day(state, sd)
        T   = _tte_years(day, state.timestamp)

        und_od = state.order_depths.get(self.UNDERLYING)
        if not und_od or not und_od.buy_orders or not und_od.sell_orders:
            return result

        S_bid = max(und_od.buy_orders)
        S_ask = min(und_od.sell_orders)

        und_pos      = int(state.position.get(self.UNDERLYING, 0))
        und_buy_cap  = self.UND_LIMIT - und_pos
        und_sell_cap = self.UND_LIMIT + und_pos

        und_orders: List[Order] = []

        for K in self.STRIKES:
            prod = f"VEV_{K}"
            od   = state.order_depths.get(prod)
            if not od:
                continue

            best_bid = max(od.buy_orders,  default=None)
            best_ask = min(od.sell_orders, default=None)

            # IV: bid uses S_ask, ask uses S_bid (matches analysis plot)
            bid_iv = _implied_vol(best_bid, S_ask, K, T) if best_bid is not None else None
            ask_iv = _implied_vol(best_ask, S_bid, K, T) if best_ask is not None else None

            b_buf: List[float] = sd.get(f"biv_{K}", [])
            a_buf: List[float] = sd.get(f"aiv_{K}", [])
            if bid_iv is not None:
                b_buf.append(bid_iv)
                b_buf = b_buf[-self.iv_window:]
            if ask_iv is not None:
                a_buf.append(ask_iv)
                a_buf = a_buf[-self.iv_window:]
            sd[f"biv_{K}"] = b_buf
            sd[f"aiv_{K}"] = a_buf

            med_bid_iv = _median(b_buf) if len(b_buf) >= 5 else None
            med_ask_iv = _median(a_buf) if len(a_buf) >= 5 else None

            self.logger.log(**{
                f"bid_iv_{K}":     bid_iv     if bid_iv     is not None else float("nan"),
                f"ask_iv_{K}":     ask_iv     if ask_iv     is not None else float("nan"),
                f"med_bid_iv_{K}": med_bid_iv if med_bid_iv is not None else float("nan"),
                f"med_ask_iv_{K}": med_ask_iv if med_ask_iv is not None else float("nan"),
            })

            pos          = int(state.position.get(prod, 0))
            opt_buy_cap  = self.option_limit - pos
            opt_sell_cap = self.option_limit + pos
            opt_orders: List[Order] = []

            # ── BUY option (bid IV below median) + SELL underlying ───────────
            if (bid_iv is not None and med_bid_iv is not None
                    and best_ask is not None
                    and med_bid_iv - bid_iv > self.iv_diff_threshold):

                delta     = _bs_delta(S_ask, K, T, med_bid_iv)
                opt_avail = abs(od.sell_orders[best_ask])
                und_avail = abs(und_od.buy_orders.get(S_bid, 0))

                opt_qty, und_qty = self._size(
                    opt_avail, opt_buy_cap, und_avail, und_sell_cap, delta)

                if opt_qty > 0:
                    opt_orders.append(Order(prod, best_ask, opt_qty))
                    self.logger.log_order(prod, "BUY", best_ask, opt_qty, "IV_CHEAP")
                if und_qty > 0:
                    und_orders.append(Order(self.UNDERLYING, S_bid, -und_qty))
                    self.logger.log_order(self.UNDERLYING, "SELL", S_bid, und_qty, "DELTA_HEDGE")
                    und_sell_cap -= und_qty

            # ── SELL option (ask IV above median) + BUY underlying ──────────
            if (ask_iv is not None and med_ask_iv is not None
                    and best_bid is not None
                    and ask_iv - med_ask_iv > self.iv_diff_threshold):

                delta     = _bs_delta(S_bid, K, T, med_ask_iv)
                opt_avail = abs(od.buy_orders[best_bid])
                und_avail = abs(und_od.sell_orders.get(S_ask, 0))

                opt_qty, und_qty = self._size(
                    opt_avail, opt_sell_cap, und_avail, und_buy_cap, delta)

                if opt_qty > 0:
                    opt_orders.append(Order(prod, best_bid, -opt_qty))
                    self.logger.log_order(prod, "SELL", best_bid, opt_qty, "IV_RICH")
                if und_qty > 0:
                    und_orders.append(Order(self.UNDERLYING, S_ask, und_qty))
                    self.logger.log_order(self.UNDERLYING, "BUY", S_ask, und_qty, "DELTA_HEDGE")
                    und_buy_cap -= und_qty

            if opt_orders:
                result[prod] = opt_orders

        if und_orders:
            merged: Dict[int, int] = {}
            for o in und_orders:
                merged[o.price] = merged.get(o.price, 0) + o.quantity
            result[self.UNDERLYING] = [
                Order(self.UNDERLYING, p, q) for p, q in merged.items() if q != 0
            ]

        return result


# ── Trader ────────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self):
        self.logger = Logger()
        self.arb    = OptionsArb(self.logger, TUNING_PARAMS)

    def run(self, state: TradingState):
        try:
            sd = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            sd = {}

        result = self.arb.run(state, sd)
        return result, 0, json.dumps(sd)
