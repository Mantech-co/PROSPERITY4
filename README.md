# Prosperity 4 Data Visualizer

A fast desktop market-data visualizer built with Python, PyQt6, Polars, NumPy, and PyQtGraph.

The app is designed for responsive pan/zoom on large time-series data, with multiple analysis views and an order-book heatmap.

## Features

- Round-aware CSV loader
  - Automatically scans a folder for files matching:
    - `prices_round_<round>_day_<day>.csv`
    - `trades_round_<round>_day_<day>.csv`
  - Groups files by round and loads all days for the selected round.

- Product and day filtering
  - Select product (symbol) and day from dropdowns.
  - Supports `All` days by stitching day timestamps into one timeline.

- Three view modes
  - View 1: Mid price, trade executions, and L1 spread.
  - View 2: Bid/ask liquidity and scaled OBI.
  - View 3: Order-book heatmap over time.

- Order-book heatmap (View 3)
  - Buy-side liquidity rendered in blue.
  - Sell-side liquidity rendered in red.
  - Pixel intensity scales from 0 to 255 based on relative volume.
  - Optional mid-price overlay toggle.

- Responsive rendering
  - Persistent plot items (`setData` updates) instead of full redraws.
  - Downsampling and clipping enabled for faster interaction.
  - Optional OpenGL-backed rendering path for smoother panning.
  - Trade point decimation for heavy datasets.

- View refresh
  - `Refresh` button reprocesses and repaints current round/product/day.
  - View switching also repaints using latest cached data.

- Zoom mode toggle (keyboard)
  - Press `X` to toggle X-only zoom.
  - Press `Y` to toggle Y-only zoom.
  - Press the same key again to return to normal XY zoom.
  - Current zoom mode is shown in subtle text at bottom-right.

## UI Controls

Top control bar:

- `Round`: Select round from detected CSV sets.
- `Product`: Select instrument/product.
- `Day`: Select day or `All`.
- `View`: Switch between the three chart layouts.
- `Refresh`: Recompute and repaint current selection.
- `Show L1 Spread`: Show/hide spread shading in View 1.
- `Overlay Midprice on Heatmap`: Show/hide mid line in View 3.

## Data Expectations

Price CSVs are expected to include at least:

- `day`, `timestamp`, `product`
- `mid_price`
- `bid_price_1`, `ask_price_1`
- `bid_volume_1`, `ask_volume_1`

Heatmap logic will also use deeper levels when present:

- `bid_price_n`, `bid_volume_n`
- `ask_price_n`, `ask_volume_n`

Trade CSVs are expected to include:

- `timestamp`, `symbol`, `price`

Note: trade files do not need a `day` column; day is inferred from filename.

## Installation

Install dependencies in your Python environment:

```bash
pip install pyqt6 pyqtgraph polars numpy
```

## Run

From workspace root:

```bash
python dataviz/data_visualizer.py
```

The script tries a hardcoded data path first, then falls back to the current directory (`.`).

## Performance Notes

- For very large datasets, keep View 3 mid overlay off unless needed.
- If your system has OpenGL/driver issues, the app falls back to software rendering.
- If VS Code shows unresolved imports but app runs, your editor interpreter may differ from runtime interpreter.

## File Layout

- `dataviz/data_visualizer.py` - main application
- `dataviz/prices_round_...csv` - input price data
- `dataviz/trades_round_...csv` - input trade data
