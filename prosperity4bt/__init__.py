from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from prosperity4bt.back_tester import BackTester
from prosperity4bt.models.test_options import TestOptions, TradeMatchingMode

def run_backtest(
    trader_class: Any = None,
    round_day: list[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
    algorithm_path: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    print_output: bool = False,
    match_trades: TradeMatchingMode = TradeMatchingMode.worse,
    show_progress: bool = True,
    merge_pnl: bool = True,
    show_vis: bool = False,
    original_timestamps: bool = False,
):
    """
    Run the prosperity4bt backtester programmatically.

    Args:
        trader_class: The Trader class to backtest.
        round_day: List of strings in format 'round-day' or 'round'.
        data_dir: Path to the data directory.
        algorithm_path: Path to the algorithm file (if trader_class is not provided).
        output_file: Path to save the output log.
        print_output: Whether to print trader output to stdout.
        match_trades: Trade matching mode ('all', 'worse', 'none').
        show_progress: Whether to show progress bars.
        merge_pnl: Whether to merge PnL across days.
        show_vis: Whether to open the visualizer after the backtest.
        original_timestamps: Whether to preserve original timestamps.
    """
    if round_day is None:
        raise ValueError("round_day must be provided (e.g. ['1-1', '1-2'])")

    if algorithm_path is not None:
        algorithm_path = Path(algorithm_path).resolve()

    if output_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = Path.cwd() / "backtests" / f"{timestamp}.log"
    else:
        output_file = Path(output_file).resolve()

    options = TestOptions(algorithm_path, round_day, output_file)
    options.trader_class = trader_class
    options.back_data_dir = Path(data_dir).resolve() if data_dir else None
    options.print_output = print_output
    options.trade_matching_mode = match_trades
    options.show_progress = show_progress
    options.merge_profit_loss = merge_pnl
    options.show_visualizer = show_vis
    options.merge_timestamps = not original_timestamps

    back_tester = BackTester(options)
    back_tester.run()
