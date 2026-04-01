"""
logger.py — Prosperity Structured Logger
=========================================
Drop this file next to your main trading bot file and import it.

The Logger class encodes arbitrary key→value data into a single-line JSON
string prefixed with "LOGVIZ:" and printed to stdout.  The log visualizer
(log_visualizer.py) picks up every line that starts with "LOGVIZ:" from the
lambdaLog field of the sandbox output and automatically plots each key as a
separate time-series.

Quick-start
-----------
    from logger import Logger

    class Trader:
        def __init__(self):
            self.logger = Logger()

        def run(self, state: TradingState) -> tuple[dict, list, list]:
            # log whatever numeric values you want plotted
            self.logger.log(state.timestamp,
                spread     = ask1 - bid1,
                position   = state.position.get('TOMATOES', 0),
                fair_value = my_computed_fair,
            )
            # ... rest of your logic ...
            return orders, conversions, self.logger.flush()

    # In the actual entry-point that Prosperity calls:
    trader = Trader()
    def run(state: TradingState):
        orders, conversions, logs = trader.run(state)
        # Prosperity captures anything printed during run() as lambdaLog
        return orders, conversions, logs

Notes
-----
* Values MUST be numeric (int or float).  The visualizer ignores non-numeric.
* All keys are plotted on a shared time-axis so different keys can be
  visually compared.
* Calling flush() returns the accumulated log string for the current
  timestamp and resets the internal buffer.  Pass it as the third return
  value from your trader if Prosperity expects a log string; otherwise just
  call self.logger.log() and the print happens immediately via auto_print.
* auto_print=True (default) prints immediately inside log(); set to False if
  you prefer to collect and return via flush().
"""

import json
from typing import Any


class Logger:
    """
    Emit LOGVIZ-encoded lines that log_visualizer.py can decode and plot.

    Parameters
    ----------
    auto_print : bool
        If True (default), each call to log() prints immediately.
        Set to False to accumulate lines and retrieve via flush().
    """

    PREFIX = 'LOGVIZ:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, timestamp: int, **series: float) -> None:
        """
        Record one data point for each keyword argument.

        Example
        -------
        logger.log(state.timestamp,
                   spread=ask - bid,
                   pos=position,
                   pnl=current_pnl)
        """
        if not series:
            return
        # Convert everything to float; skip non-numeric silently
        clean: dict[str, float] = {}
        for k, v in series.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        if not clean:
            return

        line = self.PREFIX + json.dumps(clean, separators=(',', ':'))
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)

    def flush(self) -> str:
        """
        Return all buffered lines joined by '\\n' and clear the buffer.
        Use this as the third return value from your trader when
        auto_print=False.
        """
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out

    def reset(self) -> None:
        """Discard the buffer without returning it."""
        self._buffer.clear()


# ─────────────────────────── usage example ───────────────────────────────────
#
#   from datamodel import TradingState, Order
#   from logger import Logger
#
#   class Trader:
#       def __init__(self):
#           self.logger = Logger()          # auto_print=True by default
#
#       def run(self, state: TradingState):
#           for product, ob in state.order_depths.items():
#               if not ob.buy_orders or not ob.sell_orders:
#                   continue
#               best_bid = max(ob.buy_orders)
#               best_ask = min(ob.sell_orders)
#               mid = (best_bid + best_ask) / 2
#               spread = best_ask - best_bid
#               pos = state.position.get(product, 0)
#
#               # This line gets captured in lambdaLog and decoded by the viz
#               self.logger.log(
#                   state.timestamp,
#                   **{f'{product}_spread': spread,
#                      f'{product}_pos':    pos,
#                      f'{product}_mid':    mid}
#               )
#
#           orders = {}   # fill in your trading logic
#           return orders, 0, ""
