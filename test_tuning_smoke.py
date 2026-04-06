"""Quick smoke test for the tuning pipeline."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prosperity4bt"))

# 1. Verify param loading
params = {"alpha": 0.5, "n_sma": 20, "m_slope": 2, "slope_threshold": 0.1, "position_limit": 80}
with open(ROOT / "strategy" / "tomato_params.json", "w") as f:
    json.dump(params, f)

from strategy.main import Trader, TOMATO_PARAMS
print("1. TOMATO_PARAMS loaded:", TOMATO_PARAMS)
assert TOMATO_PARAMS["alpha"] == 0.5, "alpha mismatch"
assert TOMATO_PARAMS["n_sma"] == 20, "n_sma mismatch"
print("   OK - Param loading works\n")

# 2. Run a single backtest with custom output
from prosperity4bt import run_backtest

test_output = ROOT / "backtests" / "tuning_runs" / "_smoke_test.log"
test_output.parent.mkdir(parents=True, exist_ok=True)

print("2. Running single backtest...")
run_backtest(
    trader_class=Trader,
    round_day=["0--1"],
    output_file=str(test_output),
    print_output=False,
    show_progress=False,
)
assert test_output.exists(), "Log file not created"
print(f"   OK - Log created: {test_output.name} ({test_output.stat().st_size} bytes)\n")

# 3. Test PnL extraction on the smoke test log
sys.path.insert(0, str(ROOT / "tuning"))
from pnl_extractor import extract_tomato_pnl, parse_params_from_filename

pnl = extract_tomato_pnl(test_output)
print(f"3. Extracted TOMATOES PnL: {pnl}")
assert pnl is not None, "PnL extraction failed"
print("   OK - PnL extraction works\n")

# 4. Test filename parsing
test_name = "alpha=0.500--m_slope=2--n_sma=20--st=0.100.log"
parsed = parse_params_from_filename(test_name)
print(f"4. Parsed filename: {parsed}")
assert parsed["alpha"] == 0.5
assert parsed["slope_threshold"] == 0.1
print("   OK - Filename parsing works\n")

# Clean up smoke test file
test_output.unlink()
print("All smoke tests passed!")
