import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

try:
    import prosperity4bt
    print(f"Successfully imported prosperity4bt from {prosperity4bt.__file__}")
    from prosperity4bt import run_backtest
    print("Successfully imported run_backtest")
except Exception as e:
    print(f"Failed to import: {e}")
    import traceback
    traceback.print_exc()
