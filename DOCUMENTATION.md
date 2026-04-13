# Prosperity 4: Tooling & Strategy Documentation

This document provides a comprehensive guide to the tools available in this repository for backtesting, hyperparameter tuning, and data visualization.

---

## 1. Quick Start: Backtesting
The fastest way to test your strategy logic is using the programmatic backtester.

### 🏃 Running a Single Backtest
Use `run_my_strategy.py` to execute your `Trader` class against a specific round and day.
```bash
python run_my_strategy.py
```
*   **Logic**: Imports `Trader` from `strategy/main.py` and runs it via `prosperity4bt.run_backtest`.
*   **Configuration**: Edit `run_my_strategy.py` to change the `round_day` (e.g., `["0--1"]` for Round 0, Day -1).

### 🛠️ Backtester Core (`prosperity4bt`)
The backtester can also be run as a module:
```bash
python -m prosperity4bt --help
```
*   **Programmatic API**:
    ```python
    from prosperity4bt import run_backtest
    run_backtest(trader_class=Trader, round_day=["0--1"], output_file="test.log")
    ```

---

## 2. Hyperparameter Tuning Suite (`tuning/`)
A generalized pipeline to optimize any commodity's performance.

### 🧪 Grid Search: `hyper_parameter_tester.py`
Sweeps a range of parameters and runs backtests in parallel.
```bash
# Tune TOMATOES (default)
python tuning/hyper_parameter_tester.py --mode parallel

# Tune a specific product with custom logging
python tuning/hyper_parameter_tester.py --product EMERALDS --mode sequential
```
*   **Modes**: `parallel` (faster, uses multiple cores) or `sequential`.
*   **Injection**: Automatically writes current parameters to `strategy/params.json` before each run.
*   **Logs**: Saves results to `backtests/tuning_runs/` with descriptive names: `[PRODUCT]--[params].log`.

### 📊 PnL Extraction: `pnl_extractor.py`
Parses tuning logs and consolidates final PnL into a CSV.
```bash
python tuning/pnl_extractor.py --product TOMATOES
```
*   **Output**: Creates `tuning/results_tomatoes.csv`.
*   **Filter**: Only processes logs starting with the specified product prefix.

### 🎯 Fit-Based Optimization: `pnl_optimizer.py`
Uses polynomial regression to find the "mathematical" optimal parameter set.
```bash
python tuning/pnl_optimizer.py --product TOMATOES --degree 3
```
*   **Analysis**: Fits an Nth-degree polynomial to each parameter's PnL curve.
*   **Recommendation**: Provides a comparison between the "Raw Best" (max found) and "Poly Fit" (predicted peak).

### 📈 Visualization: `pnl_plotter.py`
Visualizes the PnL landscape for intuitive analysis.
```bash
# 2-Parameter Comparison
python tuning/pnl_plotter.py --product TOMATOES --mode selection --x m_slope --y slope_threshold

# Dimensionality Reduction (3+ Parameters)
python tuning/pnl_plotter.py --product TOMATOES --mode pca
```
*   **Output**: Generates `tuning/pnl_visualization_[product].png` (3D surface + 2D heatmap).

---

## 3. Visualization Tools

### 🖥️ Desktop Data Visualizer: `dataviz/`
A high-performance PyQt6 dashboard for deep market analysis.
```bash
python dataviz/data_visualizer.py
```
*   **Views**:
    1.  **Market Price & Trades**: Mid-price, L1 spread, and execution points.
    2.  **Liquidity**: Order book depth (Bid/Ask volume) and OBI (Order Book Imbalance).
    3.  **Heatmap**: Time-series visualization of order book depth (Blue=Buy, Red=Sell).
*   **Controls**: Filter by Round, Product, and Day. Use `X` or `Y` keys for constrained zooming.

### 🔍 Log Analyzer: `logviz/`
An interactive tool for exploring and debugging backtest session logs.
```bash
python logviz/log_visualizer.py
```
*   **Dashboard View**: A high-level overview showing key performance metrics:
    *   **Sharpe Ratio**: Reward-to-risk ratio of strategy performance.
    *   **Win Rate**: Percentage of profitable position closures.
    *   **Total Volume**: Aggregate quantity traded.
    *   **Trade Counts**: Separation of "My Trades" vs "Bot Trades" on the platform.
*   **Performance Plots**:
    *   **Cumulative PnL**: View total profit growth over time.
    *   **Drawdown (Absolute/%)**: Track drawdown from high-water marks with a toggle for percentage view.
*   **Filtering**: Analyze "Overall" session performance or drill down into individual products.
*   **Market View**: Interactive heatmap and price plot with trade markers.
*   **Debug Logs**: Integrated viewer for your strategy's `LOGDBG:` output with product context.
*   **Data Import**: Use the "Import Data" button to automatically load and visualize all raw market data CSVs directly from the `dataviz/` directory.

---

## 4. Strategy Structure (`strategy/`)

### 🧠 The Trader: `main.py`
The core trading logic. All parameters are dynamically loaded.
```python
from strategy.main import TUNING_PARAMS, TOMATO_PARAMS
```
*   **Default Logic**: Product-specific defaults are defined in code.
*   **Overrides**: If `strategy/params.json` exists, its values override the defaults. This ensures you can swap between "Submission mode" and "Tuning mode" seamlessly.

### ⚙️ Shared Config: `params.json`
A simple JSON file used by the Tuning Suite to inject parameters during sweeps.
```json
{
    "m_slope": 30,
    "slope_threshold": 0.2
}
```

---

## 5. Required Directory Structure

To ensure all tools, visualizers, and tuners work harmoniously, your repository should be structured as follows. **Pay special attention to data folders**, as the tooling assumes specific locations for your files.

| Directory / File | Purpose | Notes |
| :--- | :--- | :--- |
| `dataviz/` | Market Data visualization & **Raw Data Storage** | **🛑 IMPORTANT**: Store your raw platform data CSVs (`prices_*.csv`, `trades_*.csv`) here. Both `data_visualizer.py` and the `log_visualizer.py` "Import Data" feature automatically read from this directory. |
| `logviz/` | Log Analysis GUI | Launch `log_visualizer.py` to analyze logs or import raw dataviz data. |
| `backtests/` | Session Output & Tuning Log Storage | Tuning sweeps automatically save logs to `backtests/tuning_runs/`. |
| `prosperity4bt/`| Core Backtester framework | Contains algorithmic matching logic. |
| `tuning/` | Hyperparameter Optimization Suite | Scripts like `hyper_parameter_tester.py` and `pnl_extractor.py`. |
| `strategy/` | Your Trading Logic | Contains the `Trader` class in `main.py` & hyperparameter states in `params.json`. |
