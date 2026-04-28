import json
import math
from typing import List, Dict, Any, Optional
from datamodel import OrderDepth, TradingState, Order, Trade
from boilerplate import Logger, BaseStrategy, TUNING_PARAMS

TUNING_PARAMS = {
    "position_limit": 50,
    "ema_span": 545,
    "z_threshold": 2.3,

}

MARK38 = "Mark 38"


class HydrogelM38Strategy(BaseStrategy):
    PRODUCT = "HYDROGEL_PACK"

    def __init__(self, logger: Logger, params: Dict[str, Any]):
        super().__init__(logger, params)
        self.ema_span = params.get("ema_span", 545)
        self.z_threshold = params.get("z_threshold", 2.3)
        self._ema: Optional[float] = None
        self._m2: float = 0.0   # running sum of squared deviations (Welford)
        self._n: int = 0        # number of mid prices seen
        self._std: float = 0.0
        self._alpha = 2.0 / (self.ema_span + 1)

    def _update_ema_std(self, mid: float):
        if self._ema is None:
            self._ema = mid
            self._n = 1
            return
        prev_ema = self._ema
        self._ema = self._alpha * mid + (1 - self._alpha) * prev_ema
        self._n += 1
        # online std via Welford on (mid - ema) residuals
        dev = mid - self._ema
        self._m2 += dev * dev
        self._std = math.sqrt(self._m2 / self._n) if self._n > 1 else 0.0

    def run(self, state_dict: Dict[str, Any]) -> List[Order]:
        # restore persistent state
        sd = state_dict.get(self.PRODUCT, {})
        if sd:
            self._ema  = sd.get("ema")
            self._m2   = sd.get("m2", 0.0)
            self._n    = sd.get("n", 0)
            self._std  = sd.get("std", 0.0)

        od = self.order_depth
        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return []

        mid = (best_bid + best_ask) / 2.0
        self._update_ema_std(mid)

        # look for Mark 38 in market trades
        m38_trades: List[Trade] = [
            t for t in self.state.market_trades.get(self.PRODUCT, [])
            if t.buyer == MARK38 or t.seller == MARK38
        ]

        if m38_trades and self._ema is not None and self._std > 0:
            for trade in m38_trades:
                z = (trade.price - self._ema) / self._std
                if abs(z) < self.z_threshold:
                    continue
                if z < -self.z_threshold:
                    self.bid(best_ask, self.buy_capacity, tag="M38_BUY")
                else:
                    self.ask(best_bid, self.sell_capacity, tag="M38_SELL")

        self.logger.log(
            ema=self._ema or 0,
            std=self._std,
            mid=mid,
            pos=self.current_pos,
        )

        # persist state
        state_dict[self.PRODUCT] = {
            "ema": self._ema,
            "m2":  self._m2,
            "n":   self._n,
            "std": self._std,
        }

        return self._consolidate()


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.params = TUNING_PARAMS
        self.strategies: Dict[str, BaseStrategy] = {
            "HYDROGEL_PACK": HydrogelM38Strategy(self.logger, self.params),
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        try:
            state_dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            state_dict = {}

        for product, strategy in self.strategies.items():
            if product in state.order_depths:
                strategy.reset(state)
                result[product] = strategy.run(state_dict)

        return result, 0, json.dumps(state_dict)
