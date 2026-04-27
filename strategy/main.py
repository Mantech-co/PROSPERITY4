import json
import math
from typing import List, Dict, Optional
from datamodel import OrderDepth, TradingState, Order

UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

SHORT_SYMS = ["VEV_5200", "VEV_5300"]   # sell to -limit
LONG_SYMS  = ["VEV_5400", "VEV_5500"]   # buy to +limit (delta offset)

_DEFAULTS = {
    "option_limit":     300,
    "underlying_limit": 200,
    "sigma":            0.20,
    "r":                0.0,
    "total_expiry_days": 8,
    "start_day":        2,
    "rebalance_ticks":  100,
}

TS_PER_DAY = 1_000_000


def _load_params() -> dict:
    import os
    path = os.path.join(os.path.dirname(__file__), "params.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return {**_DEFAULTS, **data}
    except Exception:
        return dict(_DEFAULTS)


TUNING_PARAMS = _load_params()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


class Logger:
    PREFIX = "LOGVIZ:"

    def log(self, **series: float) -> None:
        clean = {k: float(v) for k, v in series.items() if v == v}
        if clean:
            print(self.PREFIX + json.dumps(clean, separators=(",", ":")))

    def log_order(self, product: str, side: str, price: int, qty: int, tag: str) -> None:
        print(f"LOGORDER:{product}:{side}:{price}:{qty}:{tag}")


class CondorDeltaHedge:
    def __init__(self, logger: Logger, params: dict):
        self.logger = logger
        self.p = params

    def _mid(self, od: OrderDepth) -> Optional[float]:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _tte(self, ts: dict, timestamp: int) -> float:
        prev = ts.get("prev_ts", -1)
        day  = ts.get("day", self.p["start_day"])
        if 0 <= timestamp < prev:
            day += 1
            ts["day"] = day
        ts["prev_ts"] = timestamp
        tte_days = max(self.p["total_expiry_days"] - day - timestamp / TS_PER_DAY, 1e-9)
        return tte_days / 252.0

    def _sweep_buy(self, sym: str, od: OrderDepth, want: int) -> List[Order]:
        orders, rem = [], want
        for price in sorted(od.sell_orders):
            if rem <= 0:
                break
            qty = min(rem, -od.sell_orders[price])
            orders.append(Order(sym, price, qty))
            self.logger.log_order(sym, "BUY", price, qty, "FILL")
            rem -= qty
        return orders

    def _sweep_sell(self, sym: str, od: OrderDepth, want: int) -> List[Order]:
        orders, rem = [], want
        for price in sorted(od.buy_orders, reverse=True):
            if rem <= 0:
                break
            qty = min(rem, od.buy_orders[price])
            orders.append(Order(sym, price, -qty))
            self.logger.log_order(sym, "SELL", price, qty, "FILL")
            rem -= qty
        return orders

    def run(self, state: TradingState, ts: dict) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        und_od = state.order_depths.get(UNDERLYING)
        if und_od is None:
            return result
        S = self._mid(und_od)
        if S is None:
            return result

        tte   = self._tte(ts, state.timestamp)
        sigma = self.p["sigma"]
        r     = self.p["r"]
        olim  = self.p["option_limit"]
        ulim  = self.p["underlying_limit"]

        # 1. sell ATM calls (5200, 5300) to -limit
        for sym in SHORT_SYMS:
            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders:
                continue
            pos  = int(state.position.get(sym, 0))
            need = olim + pos              # units still shortable
            if need <= 0:
                continue
            orders = self._sweep_sell(sym, od, need)
            if orders:
                result[sym] = orders

        # 2. buy OTM calls (5400, 5500) to +limit
        for sym in LONG_SYMS:
            od = state.order_depths.get(sym)
            if od is None or not od.sell_orders:
                continue
            pos  = int(state.position.get(sym, 0))
            need = olim - pos              # units still buyable
            if need <= 0:
                continue
            orders = self._sweep_buy(sym, od, need)
            if orders:
                result[sym] = orders

        # 3. rebalance underlying to complete delta neutral
        tick = ts.get("tick", 0)
        ts["tick"] = tick + 1
        if tick % self.p["rebalance_ticks"] == 0:
            net_delta = 0.0
            for sym in SHORT_SYMS + LONG_SYMS:
                K   = STRIKES[sym]
                pos = int(state.position.get(sym, 0))
                net_delta += pos * bs_delta(S, K, r, sigma, tte)

            target_und = max(-ulim, min(ulim, -round(net_delta)))
            und_pos    = int(state.position.get(UNDERLYING, 0))
            diff       = target_und - und_pos
            if diff > 0:
                orders = self._sweep_buy(UNDERLYING, und_od, diff)
                if orders:
                    result[UNDERLYING] = orders
            elif diff < 0:
                orders = self._sweep_sell(UNDERLYING, und_od, -diff)
                if orders:
                    result[UNDERLYING] = orders

            self.logger.log(S=S, net_delta=net_delta, und_pos=und_pos,
                            target_und=target_und, tte=tte)

        return result


class Trader:
    def __init__(self):
        self.logger   = Logger()
        self.strategy = CondorDeltaHedge(self.logger, TUNING_PARAMS)

    def run(self, state: TradingState):
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        result = self.strategy.run(state, ts)
        return result, 0, json.dumps(ts)
