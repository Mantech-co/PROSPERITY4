import sys
from pathlib import Path

# Add the project root to sys.path so we can import prosperity4bt and strategy
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Also add strategy folder to sys.path because main.py might expect to import datamodel directly
# as it does: from datamodel import OrderDepth, UserId, TradingState, Order
sys.path.append(str(project_root / "prosperity4bt"))

try:
    from strategy.main import Trader
    from prosperity4bt import run_backtest
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting programmatic backtest...")
    
    # Run backtest for round 0, day -1
    # round_day format: 'round-day'
    # For round 0 day -1, use '0--1'
    try:
        run_backtest(
            trader_class=Trader,
            round_day=["4-2"],
            data_dir=project_root / "data",
            print_output=False,
        )
        print("\nBacktest completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during backtest: {e}")
        import traceback
        traceback.print_exc()
