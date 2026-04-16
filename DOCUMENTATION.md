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
*   **Configuration**: Edit `run_my_strategy.py` to change the `round_day` (e.g., `["1-0"]` for Round 1, Day 0).

### 🛠️ Backtester Core (`prosperity4bt`)
The backtester can also be run as a module:
```bash
python -m prosperity4bt --help
```

---

## 2. Hyperparameter Tuning Suite (`tuning/`)
A generalized pipeline to optimize any commodity's performance.

### 🧪 Grid Search: `hyper_parameter_tester.py`
Sweeps a range of parameters and runs backtests in parallel.
```bash
python tuning/hyper_parameter_tester.py --product ASH_COATED_OSMIUM --mode parallel
```
*   **Injection**: Automatically writes current parameters to `strategy/params.json` before each run.
*   **Logs**: Saves results to `backtests/tuning_runs/` with descriptive names.

### 🎯 Fit-Based Optimization
1.  **Extract**: `python tuning/pnl_extractor.py --product ASH_COATED_OSMIUM` (creates CSV).
2.  **Optimize**: `python tuning/pnl_optimizer.py --product ASH_COATED_OSMIUM --degree 3` (polynomial fit).

---

## 3. Visualization Tools

### 🔍 Log Analyzer: `logviz/`
The ultimate interactive tool for exploring and debugging backtest session logs.
```bash
python logviz/log_visualizer.py
```

#### 🛠️ Core Features
*   **Refresh Backtest**: Rerun `run_my_strategy.py` and reload the latest log directly from the GUI (one-click iteration).
*   **Import Raw Data**: Directly visualize `data/*.csv` market data without a log file.
*   **Data Setup (S)**: Configure which `LOGVIZ` metrics appear on the **Main Pane** (overlaying price) vs. the **Generic Pane** (separate axis below).

#### 📈 Market View Tab
*   **Order Book Heatmap**: Visualizes depth (volume) at every price level over time.
*   **Order Placement Heatmap**: Shows your strategy's active orders (Bids/Asks) as a heatmap overlay.
*   **Interactive Legend**: Click any item (Heatmap, Mid Price, My Buy/Sell, Bot Trades, Custom) to toggle visibility.
*   **Markup Mode (M)**: Left-click the chart to place persistent crosshairs with (Timestamp, Price) labels for precise measurement.
*   **Coordinate Strip**: Real-time display of TS, Mid Price, PnL, and Volume at the mouse cursor.

#### 📊 Dashboard Tab
*   **Performance Metrics**: Real-time Sharpe Ratio, Win Rate, Total Volume, and Trade Counts (My vs. Bot).
*   **Cumulative PnL**: Continuous profit tracking across all selected days/products.
*   **Drawdown Plot**: Visualizes capital drawdown with a toggle for absolute vs. percentage (%) view.
*   **Filtering**: Analyze "Overall" session performance or drill down into specific Products/Days.

#### 📝 Debugging Tabs
*   **Logs Tab**: Searchable, color-coded table for `LOGDBG` messages. Automatically shows Position and PnL at the log's timestamp.
*   **SandboxLog Tab**: Consolidates and counts engine-level messages or errors, with a timestamp history for each unique message.
*   **Custom Tab**: Individual large-scale plots for every unique key passed to `logger.log()`.
*   **Position Tab**: Multi-product position tracking across the entire session.

#### ⌨️ Keybindings
| Key | Action |
| :--- | :--- |
| **X / Y** | Lock zoom to X or Y axis |
| **Z** | Reset zoom to both axes (XY) |
| **A** | Autoscale all plots to data |
| **M** | Toggle Markup mode (click to place) |
| **Shift + M** | Toggle secondary Generic Pane visibility |
| **C** | Clear all markup lines |
| **S** | Open Data Setup dialog |
| **?** | Focus Keyboard Shortcuts tab |

---

## 4. Team Performance Tracker (`strategy_tracker.py`)

A centralized dashboard for tracking team strategy performance via GitHub.
```bash
python strategy_tracker.py
```
*   **Leaderboard**: Real-time ranking by PnL, Sharpe, or Drawdown.
*   **Artifacts**: Automatically uploads strategy `.zip` (code + logs + config) to GitHub for team review.

---

## 5. Strategy Structure (`strategy/`)

### 🧠 Core Strategies (`main.py`)
*   **`OsmiumStrategy`**: Z-score mean reversion + tiered MM.
*   **`PepperRootStrategy`**: Microprice MM + inventory accumulation + empty-book exploit.

### 📝 Structured Logging (`logger.py`)
```python
self.logger.log(mid=mid, my_metric=1.5) # Plotting
self.logger.log_order(prod, side, price, qty, tag="ALPHA") # Trade tagging
self.logger.debug("Logic check", tag="WARN", product=prod) # Logcat
```
