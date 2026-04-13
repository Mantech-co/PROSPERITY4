"""
logger.py — Prosperity Structured Logger
=========================================
Drop this file next to your main trading bot file and import it.

This module provides the `Logger` class, which helps you emit structured logs
that the Prosperity Log Visualizer (log_visualizer.py) can parse and display.

It supports two types of logging:
1.  **Time-series Logging (`log`)**: For numeric data (spreads, positions, mid
    prices, etc.) that you want to plot as line charts.
2.  **Debug Logging (`debug`)**: For text messages, warnings, or errors that
    you want to see in a searchable, color-coded table (Logcat-style).

Usage Example:
--------------
    from logger import Logger

    class Trader:
        def __init__(self):
            self.logger = Logger()

        def run(self, state: TradingState):
            # 1. Log numeric data for plotting (Market View / PnL / Custom tabs)
            self.logger.log(state.timestamp,
                mid_price = compute_mid(state),
                position  = state.position.get('AMETHYSTS', 0)
            )

            # 2. Log text messages for the "Logs" tab
            if something_wrong:
                self.logger.debug(state.timestamp, "Price gap too high!", tag="WARN", product="AMETHYSTS")

            # ... logic ...
            
            # Use self.logger.flush() as the 3rd return value if auto_print=False
            return orders, conversions, ""
"""

import json
from typing import Any


class Logger:
    """
    Handles structured logging for Prosperity.
    
    If `auto_print` is True, logs are printed immediately to stdout, which
    Prosperity captures in the `lambdaLog` field. If False, logs are buffered
    and can be retrieved (and cleared) using `flush()`.
    """

    LOGVIZ_PREFIX = 'LOGVIZ:'
    LOGDBG_PREFIX = 'LOGDBG:'

    def __init__(self, auto_print: bool = True):
        self._auto_print = auto_print
        self._buffer: list[str] = []

    def log(self, **series: float) -> None:
        """
        Records numeric data points for the given timestamp.
        
        Each keyword argument becomes a separate plot in the visualizer.
        Non-numeric values are silently ignored.
        
        Args:
            **series:  Key-value pairs of numeric data to plot.
        """
        if not series:
            return
        
        clean: dict[str, float] = {}
        for k, v in series.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                pass
        
        if not clean:
            return

        line = self.LOGVIZ_PREFIX + json.dumps(clean, separators=(',', ':'))
        self._emit(line)

    def debug(self, msg: str, tag: str = 'DBG', product: str = '') -> None:
        """
        Logs a text message visible in the "Logs" tab of the visualizer.
        
        Args:
            msg:       The message text to display.
            tag:       Severity tag: 'INFO' (cyan), 'WARN' (gold), 'ERR' (red), 'DBG' (dim).
            product:   Optional product name (e.g. 'PEARLS'). If provided, the visualizer 
                       will automatically show the position and PnL at that timestamp.
        """
        # Format: LOGDBG:tag:product:message
        line = f"{self.LOGDBG_PREFIX}{tag}:{product}:{msg}"
        self._emit(line)

    def flush(self) -> str:
        """Returns all buffered logs joined by newlines and clears the buffer."""
        out = '\n'.join(self._buffer)
        self._buffer.clear()
        return out

    def reset(self) -> None:
        """Clears the internal log buffer."""
        self._buffer.clear()

    def _emit(self, line: str) -> None:
        if self._auto_print:
            print(line)
        else:
            self._buffer.append(line)
