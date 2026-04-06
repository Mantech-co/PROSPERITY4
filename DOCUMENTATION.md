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
*   **Functionality**: Parses `activitiesLog`, `traderLog`, and `sandboxLog` sections separately.
*   **PnL Graph**: Visualizes PnL growth over time for each product.
*   **Debug Logs**: Integrated viewer for your strategy's `Logger.log()` output.

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

## Directory Summary (Non-Data Files)

| Directory | Purpose | Key Tool |
| :--- | :--- | :--- |
| `prosperity4bt/` | Core Backtester | `back_tester.py` |
| `tuning/` | Optimization Suite | `hyper_parameter_tester.py` |
| `dataviz/` | Market Data GUI | `data_visualizer.py` |
| `logviz/` | Log Analysis GUI | `log_visualizer.py` |
| `strategy/` | Trading Logic | `main.py` |
| `backtests/` | Output Storage | - |
