import sys, os, json, re
from io import StringIO
import numpy as np
import polars as pl
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit,
    QScrollArea, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QFont, QColor, QBrush

# --- Styling & Colors ---
BG, PANEL_BG, BORDER, TEXT, DIM = '#0d0f14', '#12151c', '#1e2330', '#c8d0e0', '#4a5068'
ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE = \
    '#00d4ff', '#39ff6e', '#ff3d5a', '#ffd700', '#ffffff', '#b06dff'
CUSTOM_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f', '#b06dff', '#ff9f43']

# ── Heatmap Volume Coloring (10-Band Contrasting Hues) ─────────────────────────
# Format: [R, G, B] (0-255). Lowest volume at index 0, Highest at index 9.
BUY_VOLUME_COLORS = [
    [0, 170, 255],   # 1 electric blue
    [0, 51, 102],    # 2 navy
    [0, 255, 102],   # 3 neon green
    [204, 255, 0],   # 4 lime
    [0, 68, 34],     # 5 dark green
    [0, 255, 255],   # 6 cyan
    [51, 0, 153],    # 7 indigo
    [255, 238, 0],   # 8 yellow
    [0, 119, 85],    # 9 emerald
    [255, 255, 255], # 10 white
]
SELL_VOLUME_COLORS = [
    [255, 215, 0],   # 1 gold
    [255, 68, 0],    # 2 red-orange
    [43, 0, 89],     # 3 deep violet
    [255, 170, 0],   # 4 amber
    [170, 0, 0],     # 5 deep red
    [255, 0, 136],   # 6 magenta-pink
    [0, 204, 255],   # 7 sky blue
    [136, 68, 0],    # 8 burnt sienna
    [232, 232, 232], # 9 light gray
    [204, 68, 255],  # 10 violet
]

# Dedicated palette for bot trade markers (volume only, no buy/sell split)
# 20 colors — only as many are used as there are unique quantile buckets.
TRADE_VOLUME_COLORS = [
    [50, 205, 50],    # 1  lime green
    [0, 255, 127],    # 2  spring green
    [0, 230, 180],    # 3  mint
    [0, 200, 255],    # 4  sky blue
    [0, 140, 255],    # 5  azure
    [30, 60, 255],    # 6  cobalt
    [100, 0, 255],    # 7  violet
    [160, 0, 255],    # 8  purple
    [200, 0, 200],    # 9  magenta
    [255, 0, 160],    # 10 hot pink
    [255, 0, 80],     # 11 rose
    [255, 40, 0],     # 12 red-orange
    [255, 100, 0],    # 13 orange
    [255, 160, 0],    # 14 amber
    [255, 210, 0],    # 15 gold
    [255, 240, 80],   # 16 yellow
    [200, 255, 100],  # 17 lime-yellow
    [100, 255, 200],  # 18 aquamarine
    [200, 200, 255],  # 19 lavender
    [255, 255, 255],  # 20 white-hot
]

APP_STYLE = f"""
QMainWindow, QWidget {{ 
    background-color: {BG}; 
    color: {TEXT}; 
    font-family: 'JetBrains Mono', 'Consolas', monospace; 
    font-size: 9pt; 
}}
QComboBox {{ 
    background: {PANEL_BG}; 
    border: 1px solid {BORDER}; 
    padding: 2px 8px; 
    border-radius: 3px; 
    color: {TEXT}; 
    min-width: 100px; 
}}
QPushButton {{ 
    background: {PANEL_BG}; 
    border: 1px solid {BORDER}; 
    padding: 3px 12px; 
    border-radius: 3px; 
    color: {TEXT}; 
}}
QPushButton:hover {{ border-color: {ACCENT_CYAN}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; }}
QTabBar::tab {{ 
    background: {PANEL_BG}; 
    color: {DIM}; 
    padding: 6px 18px; 
    border: 1px solid {BORDER}; 
    border-bottom: none; 
    margin-right: 2px;
}}
QTabBar::tab:selected {{ 
    color: {ACCENT_CYAN}; 
    border-bottom: 2px solid {ACCENT_CYAN}; 
    background: {BG}; 
}}
#DataStrip {{ 
    background-color: {PANEL_BG}; 
    border-top: 1px solid {BORDER}; 
    color: {ACCENT_CYAN}; 
    padding: 4px 15px; 
    font-size: 9pt;
}}
QTableWidget {{
    background-color: {BG};
    color: {TEXT};
    gridline-color: {BORDER};
    border: none;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 8.5pt;
}}
QTableWidget::item {{
    padding: 2px 6px;
    border-bottom: 1px solid {BORDER};
}}
QTableWidget::item:selected {{
    background-color: {BORDER};
}}
QHeaderView::section {{
    background-color: {PANEL_BG};
    color: {ACCENT_CYAN};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    font-weight: bold;
    font-size: 8.5pt;
}}
QLineEdit {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    padding: 3px 8px;
    border-radius: 3px;
    color: {TEXT};
}}
"""

class InteractiveLegendItem(pg.LegendItem):
    """Refined legend that supports proxy items for toggle logic."""
    def addItem(self, item, name, toggle_target=None):
        try:
            # If toggle_target is provided, it's the actual object (like ImageItem)
            # while 'item' is the proxy with .opts for the legend to draw.
            target = toggle_target if toggle_target else item
            super().addItem(item, name)
            label = self.items[-1][1]
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.mouseClickEvent = lambda ev: self._toggle_item(target, label)
        except Exception as e:
            print(f"Legend Error: {e}")

    def _toggle_item(self, target, label):
        visible = not target.isVisible()
        target.setVisible(visible)
        label.setAttr('color', TEXT if visible else DIM)

# --- Legend Helper ---
class HeatmapLegend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 6, 15, 6)
        self.main_layout.setSpacing(2)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")

        self.row_layout = QHBoxLayout()
        self.row_layout.setSpacing(25)

        self.buy_ranges = []
        self.sell_ranges = []
        self.trade_ranges = []
        # Store swatch+label containers so we can hide unused buckets
        self.trade_cells = []  # list of (swatch QLabel, rlbl QLabel)

        self.row_layout.addLayout(self._create_side("BUY BOOK",  BUY_VOLUME_COLORS,  self.buy_ranges))
        self.row_layout.addLayout(self._create_side("SELL BOOK", SELL_VOLUME_COLORS, self.sell_ranges))
        self.row_layout.addLayout(self._create_trade_side())

        self.row_layout.addStretch()
        self.main_layout.addLayout(self.row_layout)

    def _create_side(self, title, colors, label_list):
        side_layout = QHBoxLayout()
        side_layout.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 7.5pt;")
        side_layout.addWidget(title_lbl)
        for i, color in enumerate(colors):
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch = QLabel()
            swatch.setFixedSize(18, 10)
            swatch.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #333;")
            vbox.addWidget(swatch)
            rlbl = QLabel("0")
            rlbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rlbl.setStyleSheet(f"font-size: 6.5pt; color: {DIM};")
            label_list.append(rlbl)
            vbox.addWidget(rlbl)
            side_layout.addLayout(vbox)
        return side_layout

    def _create_trade_side(self):
        self._trade_layout = QHBoxLayout()
        self._trade_layout.setSpacing(6)
        self._trade_title = QLabel("BOT TRADES")
        self._trade_title.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 7.5pt;")
        self._trade_layout.addWidget(self._trade_title)
        for i, color in enumerate(TRADE_VOLUME_COLORS):
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch = QLabel()
            swatch.setFixedSize(18, 10)
            swatch.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #333;")
            vbox.addWidget(swatch)
            rlbl = QLabel("")
            rlbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rlbl.setStyleSheet(f"font-size: 6.5pt; color: {DIM};")
            self.trade_ranges.append(rlbl)
            vbox.addWidget(rlbl)
            # Wrap in a container widget so we can show/hide the whole cell
            cell = QWidget()
            cell.setLayout(vbox)
            self.trade_cells.append(cell)
            self._trade_layout.addWidget(cell)
        return self._trade_layout

    def update_ranges(self, max_vol, quantile_edges):
        """Update range labels.
        quantile_edges: sorted list of volume thresholds defining the trade buckets.
        """
        # --- Order-book bands (always 10) ---
        def update_ob(lbl_list, mv):
            for i in range(10):
                lower = int(np.floor(i * mv / 10))
                upper = int(np.floor((i + 1) * mv / 10))
                lbl_list[i].setText(f"{lower}-{upper}" if i < 9 else f">{lower}")
        update_ob(self.buy_ranges,  max_vol)
        update_ob(self.sell_ranges, max_vol)

        # --- Trade bands (quantile-based, dynamic count) ---
        n_buckets = len(quantile_edges) - 1  # edges define n_buckets intervals
        for i, (cell, rlbl) in enumerate(zip(self.trade_cells, self.trade_ranges)):
            if i < n_buckets:
                pal_idx = int(round(i * (len(TRADE_VOLUME_COLORS) - 1) / max(n_buckets - 1, 1)))
                c = TRADE_VOLUME_COLORS[pal_idx]
                swatch = cell.findChild(QLabel)
                swatch.setStyleSheet(f"background-color: rgb({c[0]}, {c[1]}, {c[2]}); border: 1px solid #333;")
                lo, hi = int(quantile_edges[i]), int(quantile_edges[i + 1])
                rlbl.setText(f"{lo}-{hi}" if i < n_buckets - 1 else f">{lo}")
                cell.setVisible(True)
            else:
                cell.setVisible(False)
                rlbl.setText("")

# --- Data Engine ---
def build_ob_heatmap(p_df: pl.DataFrame, product: str, day):
    flt = p_df.filter(pl.col('product') == product)
    min_day = p_df['day'].min() if 'day' in p_df.columns else 0
    if day != 'All': flt = flt.filter(pl.col('day') == int(day))
    flt = flt.sort(['day', 'timestamp'])
    
    # Use relative day offset (plot_time) for multi-day views
    if day == 'All' and 'day' in flt.columns:
        times = (flt['timestamp'] + (flt['day'] - min_day) * 1_000_000).to_numpy()
    else:
        times = flt['timestamp'].to_numpy()
    bid_p_cols = sorted([c for c in flt.columns if 'bid_price_' in c], key=lambda x: int(x.split('_')[-1]))
    ask_p_cols = sorted([c for c in flt.columns if 'ask_price_' in c], key=lambda x: int(x.split('_')[-1]))

    all_p, all_v = [], []
    for side in ['bid', 'ask']:
        p_cols = bid_p_cols if side == 'bid' else ask_p_cols
        for pc in p_cols:
            vc = f"{side}_volume_{pc.split('_')[-1]}"
            if vc not in flt.columns: continue
            pa, va = flt[pc].to_numpy(), flt[vc].to_numpy()
            valid = np.isfinite(pa) & (pa > 0) & np.isfinite(va) & (va > 0)
            all_p.append(pa[valid]); all_v.append(va[valid])

    if not all_p: return None
    concat_p = np.concatenate(all_p)
    if len(concat_p) == 0: return None
    price_levels = np.unique(concat_p)
    max_vol = max(np.max(np.concatenate(all_v)), 1.0)
    
    w, h = len(times), len(price_levels)
    
    raw_vol = np.zeros((h, w), dtype=float)
    
    # Store levels (0-9) instead of raw intensity
    red_l, blue_l = np.zeros(h * w, np.uint8), np.zeros(h * w, np.uint8)

    for side in ['bid', 'ask']:
        p_cols = bid_p_cols if side == 'bid' else ask_p_cols
        for pc in p_cols:
            vc = f"{side}_volume_{pc.split('_')[-1]}"
            if vc not in flt.columns: continue
            pa, va = flt[pc].to_numpy(), flt[vc].to_numpy()
            mask = np.isfinite(pa) & (pa > 0)
            v_idx = np.where(mask)[0]
            if len(v_idx) == 0: continue
            y_idxs = np.searchsorted(price_levels, pa[mask])
            flat_idxs = y_idxs.astype(np.int64) * w + v_idx.astype(np.int64)
            
            # Map volume to 10 discrete bands (0-9)
            lvl = np.clip(va[mask] / max_vol * 10, 0, 9).astype(np.uint8)
            np.maximum.at(red_l if side == 'ask' else blue_l, flat_idxs, lvl)
            raw_vol.flat[flat_idxs] += va[mask]

    # Construct RGB image using palettes
    img = np.zeros((h, w, 3), np.uint8)
    
    # Reconstruct 2D level grids
    red_img_l = red_l.reshape(h, w)
    blue_img_l = blue_l.reshape(h, w)
    
    # Map levels to colors
    for l in range(10):
        mask_r = (red_img_l == l) & (red_img_l > 0)
        if np.any(mask_r):
            img[mask_r] = SELL_VOLUME_COLORS[l]
            
        mask_b = (blue_img_l == l) & (blue_img_l > 0)
        if np.any(mask_b):
            # If they overlap, let one win or blend? Last one wins for now.
            img[mask_b] = BUY_VOLUME_COLORS[l]

    return {'img': img, 'times': times, 'levels': price_levels, 'raw_vol': raw_vol, 'max_vol': max_vol}

class LogVisualizer(QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.setWindowTitle('Prosperity Sandbox Visualizer')
        self.setGeometry(50, 50, 1600, 920)
        self.data, self.current_df, self.ob_res = None, None, None
        self.custom_curves = {}
        pg.setConfigOptions(useOpenGL=True, imageAxisOrder='row-major')
        self._build_ui()
        if log_path: self._load_file(log_path)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        # Header Bar
        controls = QHBoxLayout(); controls.setContentsMargins(12, 12, 12, 12); controls.setSpacing(12)
        btn_open = QPushButton("📂 Open Log"); btn_open.clicked.connect(self._open_dialog)
        controls.addWidget(btn_open)
        
        btn_import = QPushButton("📊 Import Data")
        btn_import.clicked.connect(self._import_dataviz_data)
        controls.addWidget(btn_import)
        
        controls.addWidget(QLabel("Product:"))
        self.cb_prod = QComboBox(); controls.addWidget(self.cb_prod)
        
        controls.addWidget(QLabel("Day:"))
        self.cb_day = QComboBox(); controls.addWidget(self.cb_day)
        
        self.cb_prod.currentTextChanged.connect(self._process_selection)
        self.cb_day.currentTextChanged.connect(self._process_selection)
        
        controls.addStretch()
        self.lbl_zoom = QLabel("Mode: XY"); self.lbl_zoom.setStyleSheet(f"color: {DIM};")
        controls.addWidget(self.lbl_zoom)

        btn_export = QPushButton("💾 Export Custom CSV")
        btn_export.setToolTip("Export all custom LOGVIZ data to a CSV file")
        btn_export.clicked.connect(self._export_custom_csv)
        controls.addWidget(btn_export)

        main_layout.addLayout(controls)

        # Tabs
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        
        # Build Dashboard Tab
        self._build_dashboard_tab()
        
        # Market View
        market_container = QWidget()
        market_layout = QVBoxLayout(market_container)
        market_layout.setContentsMargins(0, 0, 0, 0)
        market_layout.setSpacing(0)

        # Add Heatmap Legend at the top of the Market View
        self.hm_legend = HeatmapLegend()
        market_layout.addWidget(self.hm_legend)

        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG)
        market_layout.addWidget(self.gw_m)
        
        self.tabs.addTab(market_container, "Market View")
        self.p_m = self.gw_m.addPlot(); self.p_m.showGrid(x=True, y=True, alpha=0.3)
        
        self.img_item = pg.ImageItem(); self.img_item.setZValue(0); self.p_m.addItem(self.img_item)
        self.curve_mid = self.p_m.plot(pen=pg.mkPen(ACCENT_CYAN, width=2), name="Mid Price")
        self.sc_bot = pg.ScatterPlotItem(symbol='x', size=7, brush=ACCENT_WHITE, name="Bot Trades")
        self.sc_buy = pg.ScatterPlotItem(symbol='t1', size=10, brush=ACCENT_GREEN, name="My Buy")
        self.sc_sell = pg.ScatterPlotItem(symbol='t', size=10, brush=ACCENT_RED, name="My Sell")
        for item in [self.sc_bot, self.sc_buy, self.sc_sell]: self.p_m.addItem(item)
        
        # Legend with Toggles
        self.leg_m = InteractiveLegendItem(offset=(10, 10))
        self.leg_m.setParentItem(self.p_m.graphicsItem())
        
        # Use a PlotDataItem proxy for the Heatmap so Legend doesn't crash
        self._heatmap_proxy = pg.PlotDataItem(pen=None, brush=pg.mkBrush(ACCENT_PURPLE))
        self.leg_m.addItem(self._heatmap_proxy, "Heatmap", toggle_target=self.img_item)
        self.leg_m.addItem(self.curve_mid, "Mid Price")
        self.leg_m.addItem(self.sc_bot, "Bot Trades")
        self.leg_m.addItem(self.sc_buy, "My Buy")
        self.leg_m.addItem(self.sc_sell, "My Sell")

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.p_m.addItem(self.v_line, ignoreBounds=True); self.p_m.addItem(self.h_line, ignoreBounds=True)
        self.p_m.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # PnL Tab
        pnl_container = QWidget()
        pnl_layout = QVBoxLayout(pnl_container); pnl_layout.setContentsMargins(0, 0, 0, 0); pnl_layout.setSpacing(0)
        
        pnl_ctrl = QHBoxLayout(); pnl_ctrl.setContentsMargins(12, 8, 12, 8)
        pnl_ctrl.addWidget(QLabel("Calculation Method:"))
        self.cb_pnl_type = QComboBox()
        self.cb_pnl_type.addItems(["Log PnL", "Realized PnL", "Valuation PnL"])
        self.cb_pnl_type.currentTextChanged.connect(self._process_selection)
        pnl_ctrl.addWidget(self.cb_pnl_type)
        pnl_ctrl.addStretch()
        pnl_layout.addLayout(pnl_ctrl)

        self.gw_p = pg.GraphicsLayoutWidget(); self.gw_p.setBackground(BG)
        pnl_layout.addWidget(self.gw_p)
        self.tabs.addTab(pnl_container, "PnL")
        
        self.p_pnl = self.gw_p.addPlot(); self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_pnl = self.p_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2))

        self.gw_pos = pg.GraphicsLayoutWidget(); self.gw_pos.setBackground(BG)
        self.tabs.addTab(self.gw_pos, "Position")
        self.p_pos = self.gw_pos.addPlot(title="Position vs Timestamp")
        self.p_pos.showGrid(x=True, y=True, alpha=0.3)
        self.p_pos.addLegend()
        self.p_pos.setLabel('left', 'Position'); self.p_pos.setLabel('bottom', 'Timestamp')
        self.pos_curves = {}

        self.gw_c = pg.GraphicsLayoutWidget(); self.gw_c.setBackground(BG)
        self.tabs.addTab(self.gw_c, "Custom")

        # Logs Tab (logcat-style)
        logs_container = QWidget()
        logs_layout = QVBoxLayout(logs_container)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.setSpacing(0)
        
        # Filter bar
        logs_filter_bar = QHBoxLayout()
        logs_filter_bar.setContentsMargins(8, 6, 8, 6)
        self.log_filter_input = QLineEdit()
        self.log_filter_input.setPlaceholderText("Filter logs...")
        self.log_filter_input.textChanged.connect(self._filter_logs_table)
        logs_filter_bar.addWidget(QLabel("🔍"))
        logs_filter_bar.addWidget(self.log_filter_input)
        logs_layout.addLayout(logs_filter_bar)
        
        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(6)
        self.logs_table.setHorizontalHeaderLabels(['Timestamp', 'Tag', 'Product', 'Position', 'PnL', 'Message'])
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.logs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.logs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.logs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.logs_table.verticalHeader().setVisible(False)
        self.logs_table.setAlternatingRowColors(True)
        logs_layout.addWidget(self.logs_table)
        self.tabs.addTab(logs_container, "Logs")

        # Fixed Data Strip
        self.data_strip = QLabel("Ready")
        self.data_strip.setObjectName("DataStrip")
        self.data_strip.setFixedHeight(32)
        main_layout.addWidget(self.data_strip)

        QShortcut(QKeySequence("X"), self).activated.connect(lambda: self._set_zoom("x"))
        QShortcut(QKeySequence("Y"), self).activated.connect(lambda: self._set_zoom("y"))
        QShortcut(QKeySequence("Z"), self).activated.connect(lambda: self._set_zoom("xy"))

    def _build_dashboard_tab(self):
        dash_container = QWidget()
        dash_layout = QVBoxLayout(dash_container)
        dash_layout.setContentsMargins(12, 12, 12, 12)
        dash_layout.setSpacing(12)

        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Dashboard Product:"))
        self.cb_dash_prod = QComboBox()
        self.cb_dash_prod.currentTextChanged.connect(self._update_dashboard)
        filter_layout.addWidget(self.cb_dash_prod)
        filter_layout.addStretch()
        dash_layout.addLayout(filter_layout)

        # Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"background-color: {BG};")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(15)

        # Metrics Panel
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(15, 15, 15, 15)
        metrics_layout.setSpacing(10)
        
        self.lbl_sharpe = QLabel("Sharpe Ratio: --")
        self.lbl_sharpe.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 11pt;")
        self.lbl_winrate = QLabel("Win Rate: --")
        self.lbl_volume = QLabel("Total Volume Traded: --")
        self.lbl_my_trades = QLabel("My Trades: --")
        self.lbl_bot_trades = QLabel("Bot Trades: --")
        
        metrics_layout.addWidget(self.lbl_sharpe)
        
        h_metrics = QHBoxLayout()
        h_metrics.addWidget(self.lbl_winrate)
        h_metrics.addWidget(self.lbl_volume)
        h_metrics.addWidget(self.lbl_my_trades)
        h_metrics.addWidget(self.lbl_bot_trades)
        h_metrics.addStretch()
        metrics_layout.addLayout(h_metrics)
        
        scroll_layout.addWidget(metrics_frame)

        # Plots
        self.gw_dash_pnl = pg.GraphicsLayoutWidget()
        self.gw_dash_pnl.setBackground(BG)
        self.gw_dash_pnl.setFixedHeight(300)
        self.p_dash_pnl = self.gw_dash_pnl.addPlot(title="Cumulative PnL")
        self.p_dash_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_dash_pnl = self.p_dash_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2))
        scroll_layout.addWidget(self.gw_dash_pnl)

        # Drawdown 
        dd_container = QWidget()
        dd_layout = QVBoxLayout(dd_container)
        dd_layout.setContentsMargins(0, 0, 0, 0)
        dd_ctrl = QHBoxLayout()
        self.chk_dd_pct = QCheckBox("Show Percentage %")
        self.chk_dd_pct.stateChanged.connect(self._update_dashboard)
        dd_ctrl.addWidget(self.chk_dd_pct)
        dd_ctrl.addStretch()
        dd_layout.addLayout(dd_ctrl)

        self.gw_dash_dd = pg.GraphicsLayoutWidget()
        self.gw_dash_dd.setBackground(BG)
        self.gw_dash_dd.setFixedHeight(300)
        self.p_dash_dd = self.gw_dash_dd.addPlot(title="Drawdown")
        self.p_dash_dd.showGrid(x=True, y=True, alpha=0.3)
        self.p_dash_dd.setXLink(self.p_dash_pnl)
        self.curve_dash_dd = self.p_dash_dd.plot(pen=pg.mkPen(ACCENT_RED, width=2, fillLevel=0, brush=(255, 61, 90, 50)))
        dd_layout.addWidget(self.gw_dash_dd)
        
        scroll_layout.addWidget(dd_container)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        dash_layout.addWidget(scroll_area)

        self.tabs.insertTab(0, dash_container, "Dashboard")
        self.tabs.setCurrentIndex(0)

    def _on_mouse_moved(self, pos):
        if not self.p_m.sceneBoundingRect().contains(pos) or self.current_df is None: return
        mouse_point = self.p_m.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        ts_array = self.current_df['timestamp'].to_numpy()
        idx = np.clip(np.searchsorted(ts_array, x), 0, len(ts_array) - 1)
        row = self.current_df.row(idx, named=True)
        ts_val = int(row['timestamp'])
        
        mid, pnl = row.get('mid_price',0), row.get('profit_and_loss',0)
        ask1, bid1 = row.get('ask_price_1', 0), row.get('bid_price_1', 0)
        spread = ask1 - bid1 if (ask1 and bid1) else 0

        info = f"TS: {ts_val}  |  MID: {mid:,.1f}  |  PnL: {pnl:,.0f}  |  SPREAD: {spread:,.1f}"

        if self.ob_res:
            y_levels = self.ob_res['levels']
            y_idx = np.searchsorted(y_levels, y)
            if 0 <= y_idx < len(y_levels):
                vol = self.ob_res['raw_vol'][y_idx, idx]
                if vol > 0: info += f"  |  VOL @ {y_levels[y_idx]:.0f}: {vol:,.0f}"

        trades = [t for t in self.data['trades'] if t['timestamp'] == ts_val and t['symbol'] == self.cb_prod.currentText()]
        if trades:
            t = trades[0]
            b = "YOU" if t.get('buyer') == 'SUBMISSION' else "BOT"
            s = "YOU" if t.get('seller') == 'SUBMISSION' else "BOT"
            info += f"  |  TRADE: {t['quantity']} @ {t['price']} ({b} > {s})"

        self.v_line.setPos(ts_val); self.h_line.setPos(y)
        self.data_strip.setText(info)

    def _set_zoom(self, mode):
        self.lbl_zoom.setText(f"Mode: {mode.upper()}")
        self.p_m.setMouseEnabled(x=(mode in ['x', 'xy']), y=(mode in ['y', 'xy']))

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Log", "", "Log (*.log *.json)")
        if path: self._load_file(path)

    def _export_custom_csv(self):
        custom = (self.data or {}).get('custom', {})
        if not custom:
            QMessageBox.information(self, "No Data", "No custom LOGVIZ data to export.\nMake sure your strategy prints LOGVIZ: lines.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Custom Data", "custom_data.csv", "CSV (*.csv)")
        if not path:
            return
        # Build unified rows: timestamp + one column per key
        all_ts = sorted(set(ts for pts in custom.values() for ts, _ in pts))
        rows = ['timestamp,' + ','.join(custom.keys())]
        ts_map = {k: dict(pts) for k, pts in custom.items()}
        for ts in all_ts:
            row = [str(int(ts))] + [str(ts_map[k].get(ts, '')) for k in custom]
            rows.append(','.join(row))
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('\n'.join(rows))
        QMessageBox.information(self, "Exported", f"Custom data saved to:\n{path}")

    def _import_dataviz_data(self):
        dataviz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataviz")
        if not os.path.exists(dataviz_dir):
            QMessageBox.warning(self, "Error", f"Dataviz directory not found at: {dataviz_dir}")
            return
            
        p_dfs = []
        t_dicts = []
        
        for f in os.listdir(dataviz_dir):
            if not f.endswith('.csv'): continue
            filepath = os.path.join(dataviz_dir, f)
            if f.startswith("prices_"):
                p_dfs.append(pl.read_csv(filepath, separator=";", null_values=['', 'nan']))
            elif f.startswith("trades_"):
                day_match = re.search(r"day_(-?\d+)", f)
                df = pl.read_csv(filepath, separator=";", null_values=['', 'nan'])
                if day_match:
                    day_val = int(day_match.group(1))
                    df = df.with_columns(pl.lit(day_val).alias("day"))
                t_dicts.extend(df.to_dicts())
                
        if not p_dfs:
            QMessageBox.warning(self, "Error", "No price CSVs found in dataviz.")
            return
            
        df = pl.concat(p_dfs)
        self.data = {'prices_df': df, 'trades': t_dicts, 'custom': {}, 'debug': []}
        
        self.cb_prod.blockSignals(True)
        products = df['product'].unique().sort().to_list()
        self.cb_prod.clear()
        self.cb_prod.addItems(products)
        self.cb_day.clear()
        self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False)
        
        if hasattr(self, 'cb_dash_prod'):
            self.cb_dash_prod.blockSignals(True)
            self.cb_dash_prod.clear()
            self.cb_dash_prod.addItems(['Overall'] + products)
            self.cb_dash_prod.blockSignals(False)
            
        self._build_custom_plots()
        self._build_position_plot()
        self._build_logs_table()
        self._process_selection()
        if hasattr(self, 'cb_dash_prod'):
            self._update_dashboard()

    def _load_file(self, path):
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        csv_str = raw.get('activitiesLog', '').replace('\\n', '\n')
        df = pl.read_csv(StringIO(csv_str), separator=';', null_values=['', 'nan'])
        df = df.rename({c: c.strip() for c in df.columns})
        
        custom = {}
        for entry in raw.get('logs', []):
            ts, log = entry.get('timestamp', 0), entry.get('lambdaLog', '') or ''
            for line in log.split('\n'):
                if line.startswith('LOGVIZ:'):
                    try:
                        payload = json.loads(line[7:])
                        for k, v in payload.items(): custom.setdefault(k, []).append((ts, float(v)))
                    except: pass

        # Parse debug messages: LOGDBG:timestamp:tag:product:message
        debug_msgs = []
        for entry in raw.get('logs', []):
            entry_ts = entry.get('timestamp', 0)
            log = entry.get('lambdaLog', '') or ''
            for line in log.split('\n'):
                if line.startswith('LOGDBG:'):
                    rest = line[7:]
                    # Format: LOGDBG:tag:product:message
                    parts = rest.split(':', 2)
                    ts = entry_ts
                    if len(parts) == 3:
                        tag, prod_ctx, msg = parts
                    elif len(parts) == 2:
                        tag, msg = parts
                        prod_ctx = ''
                    else:
                        tag, msg, prod_ctx = 'DBG', rest, ''
                    
                    debug_msgs.append({
                        'ts': ts, 
                        'tag': tag.strip(), 
                        'product': prod_ctx.strip(), 
                        'msg': msg.strip()
                    })

        self.data = {'prices_df': df, 'trades': raw.get('tradeHistory', []), 'custom': custom, 'debug': debug_msgs}
        self.cb_prod.blockSignals(True)
        products = df['product'].unique().sort().to_list()
        self.cb_prod.clear(); self.cb_prod.addItems(products)
        self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False)
        
        if hasattr(self, 'cb_dash_prod'):
            self.cb_dash_prod.blockSignals(True)
            self.cb_dash_prod.clear()
            self.cb_dash_prod.addItems(['Overall'] + products)
            self.cb_dash_prod.blockSignals(False)
            
        self._build_custom_plots()
        self._build_position_plot()
        self._build_logs_table()
        self._process_selection()
        if hasattr(self, 'cb_dash_prod'):
            self._update_dashboard()

    def _build_logs_table(self):
        debug_msgs = self.data.get('debug', [])
        df = self.data.get('prices_df')
        trades = self.data.get('trades', [])

        # Pre-compute cumulative position per product at each timestamp
        pos_at = {}  # (product, timestamp) -> cumulative position
        pos_state = {}
        sorted_trades = sorted(
            [t for t in trades if str(t.get('buyer', '')).upper() == 'SUBMISSION' or str(t.get('seller', '')).upper() == 'SUBMISSION'],
            key=lambda t: t.get('timestamp', 0)
        )
        for tr in sorted_trades:
            sym = tr.get('symbol', '')
            is_buy = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            delta = tr.get('quantity', 0) if is_buy else -tr.get('quantity', 0)
            pos_state[sym] = pos_state.get(sym, 0) + delta
            pos_at[(sym, tr['timestamp'])] = pos_state[sym]

        # Build a lookup for PnL and mid per (product, timestamp)
        pnl_mid = {}
        if df is not None and len(df) > 0:
            for row in df.iter_rows(named=True):
                key = (row.get('product', ''), row.get('timestamp', 0))
                pnl_mid[key] = (row.get('profit_and_loss', ''), row.get('mid_price', ''))

        # Closest position at or before a given timestamp for a product
        def get_position(prod, ts):
            best_ts, best_pos = None, 0
            for (s, t), p in pos_at.items():
                if s == prod and t <= ts:
                    if best_ts is None or t > best_ts:
                        best_ts, best_pos = t, p
            return best_pos

        TAG_COLORS = {
            'ERR': QColor('#ff3d5a'),
            'WARN': QColor('#ffd700'),
            'INFO': QColor('#00d4ff'),
            'DBG': QColor('#4a5068'),
        }

        self.logs_table.setRowCount(len(debug_msgs))
        self._logs_data = debug_msgs  # Keep for filtering

        for i, entry in enumerate(debug_msgs):
            ts = entry['ts']
            tag = entry['tag'].upper()
            prod = entry.get('product', '')
            msg = entry['msg']

            pnl_val, mid_val = '', ''
            if prod and (prod, ts) in pnl_mid:
                pnl_val, mid_val = pnl_mid[(prod, ts)]
            pos_val = get_position(prod, ts) if prod else ''

            row_color = TAG_COLORS.get(tag, QColor(TEXT))

            items = [
                str(ts),
                tag,
                prod,
                str(pos_val) if pos_val != '' else '',
                f'{pnl_val:.1f}' if isinstance(pnl_val, (int, float)) else str(pnl_val),
                msg
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(QBrush(row_color))
                self.logs_table.setItem(i, col, item)

        self.logs_table.scrollToBottom()

    def _filter_logs_table(self, text):
        text = text.lower()
        for row in range(self.logs_table.rowCount()):
            match = False
            for col in range(self.logs_table.columnCount()):
                item = self.logs_table.item(row, col)
                if item and text in item.text().lower():
                    match = True
                    break
            self.logs_table.setRowHidden(row, not match)

    def _build_custom_plots(self):
        self.gw_c.clear()
        custom = self.data.get('custom', {})
        anchor = None
        for i, (name, pts) in enumerate(custom.items()):
            p = self.gw_c.addPlot(row=i, col=0, title=name)
            if anchor: p.setXLink(anchor)
            else: anchor = p
            p.plot([x[0] for x in pts], [x[1] for x in pts], pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2))

    def _build_position_plot(self):
        # Clear old curves
        for c in self.pos_curves.values():
            self.p_pos.removeItem(c)
        self.pos_curves.clear()

        trades = self.data.get('trades', [])
        if not trades:
            return

        # Group self-trades by symbol
        pos_by_sym = {}  # symbol -> sorted list of (timestamp, delta)
        for tr in trades:
            is_buyer = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_sell = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if not is_buyer and not is_sell:
                continue
            sym = tr.get('symbol', '')
            qty = tr.get('quantity', 0)
            ts = tr.get('timestamp', 0)
            if 'day' in tr:
                ts += tr['day'] * 1000000
            delta = qty if is_buyer else -qty
            pos_by_sym.setdefault(sym, []).append((ts, delta))

        products = sorted(pos_by_sym.keys())
        colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_PURPLE, ACCENT_WHITE] + CUSTOM_COLORS

        for i, sym in enumerate(products):
            events = sorted(pos_by_sym[sym], key=lambda x: x[0])
            ts_list, pos_list = [], []
            cum = 0
            for ts, delta in events:
                cum += delta
                ts_list.append(ts)
                pos_list.append(cum)
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            curve = self.p_pos.plot(ts_list, pos_list, pen=pen, name=sym, stepMode='right')
            self.pos_curves[sym] = curve

        self.p_pos.autoRange()

    def _process_selection(self):
        if not self.data or not self.cb_prod.currentText(): return
        prod, day = self.cb_prod.currentText(), self.cb_day.currentText()
        self.current_df = self.data['prices_df'].filter(pl.col('product') == prod)
        if day != 'All': self.current_df = self.current_df.filter(pl.col('day') == int(day))
        
        has_day = 'day' in self.current_df.columns
        self.current_df = self.current_df.sort(['day', 'timestamp'] if has_day else ['timestamp'])
        
        min_day = self.data['prices_df']['day'].min() if 'day' in self.data['prices_df'].columns else 0
        t = self.current_df['timestamp'].to_numpy()
        if day == 'All' and has_day:
            t = t + (self.current_df['day'].to_numpy() - min_day) * 1000000

        mid = self.current_df['mid_price'].to_numpy()
        self.curve_mid.setData(t, mid)
        
        # PnL Calculation
        pnl_type = self.cb_pnl_type.currentText()
        if pnl_type == "Log PnL":
            pnl_data = self.current_df['profit_and_loss'].to_numpy() if 'profit_and_loss' in self.current_df.columns else np.zeros(len(t))
        else:
            # Calculate from trades
            if day != 'All':
                prod_trades = sorted([tr for tr in self.data['trades'] if tr.get('symbol') == prod and tr.get('day', int(day)) == int(day)], key=lambda x: x['timestamp'])
            else:
                prod_trades = []
                for tr in self.data['trades']:
                    if tr.get('symbol') == prod:
                        tr_c = tr.copy()
                        if 'day' in tr_c: tr_c['timestamp'] += (tr_c['day'] - min_day) * 1000000
                        prod_trades.append(tr_c)
                prod_trades.sort(key=lambda x: x['timestamp'])
            realized, cash, pos, avg_cost = 0.0, 0.0, 0, 0.0
            pnl_array = []
            trade_idx = 0
            
            # Map timestamps to pnl
            for i, ts in enumerate(t):
                while trade_idx < len(prod_trades) and prod_trades[trade_idx]['timestamp'] <= ts:
                    tr = prod_trades[trade_idx]
                    p, q = float(tr['price']), int(tr['quantity'])
                    is_buy = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
                    
                    if is_buy:
                        if pos >= 0: # adding to long
                            avg_cost = (avg_cost * pos + p * q) / (pos + q)
                        else: # reducing short
                            closing = min(q, abs(pos))
                            realized += closing * (avg_cost - p)
                            if q > abs(pos): avg_cost = p # flipped to long
                        pos += q; cash -= p * q
                    else: # sell
                        if pos <= 0: # adding to short
                            avg_cost = (avg_cost * abs(pos) + p * q) / (abs(pos) + q)
                        else: # reducing long
                            closing = min(q, pos)
                            realized += closing * (p - avg_cost)
                            if q > pos: avg_cost = p # flipped to short
                        pos -= q; cash += p * q
                    trade_idx += 1
                
                if pnl_type == "Realized PnL":
                    pnl_array.append(realized)
                else: # Valuation PnL
                    pnl_array.append(cash + pos * mid[i])
            pnl_data = np.array(pnl_array)

        self.curve_pnl.setData(t, pnl_data)

        mb_t, mb_p, ms_t, ms_p = [], [], [], []
        b_t, b_p, b_brushes, b_sizes = [], [], [], []
        bot_raw = []  # (ts, price, vol)
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and tr.get('day', int(day)) != int(day): continue
            
            ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr:
                ts += (tr['day'] - min_day) * 1000000

            is_buy = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_sell = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if is_buy:
                mb_t.append(ts); mb_p.append(tr['price'])
            elif is_sell:
                ms_t.append(ts); ms_p.append(tr['price'])
            else:
                bot_raw.append((ts, tr['price'], int(tr.get('quantity', 1))))

        if bot_raw:
            vols = np.array([v for _, _, v in bot_raw])
            unique_vols = np.unique(vols)
            n_buckets = min(len(TRADE_VOLUME_COLORS), len(unique_vols))
            if n_buckets <= 1:
                quantile_edges = np.array([unique_vols[0] - 0.5, unique_vols[-1] + 0.5])
            else:
                pcts = np.linspace(0, 100, n_buckets + 1)
                quantile_edges = np.unique(np.percentile(vols, pcts))
                if len(quantile_edges) < 2:
                    quantile_edges = np.linspace(vols.min(), vols.max() + 1, n_buckets + 1)
            n_buckets = len(quantile_edges) - 1
        else:
            quantile_edges = np.array([0, 1])
            n_buckets = 1

        for ts, pr, vol in bot_raw:
            b_t.append(ts); b_p.append(pr)
            bucket = min(np.searchsorted(quantile_edges[1:], vol, side='right'), n_buckets - 1)
            pal_idx = int(round(bucket * (len(TRADE_VOLUME_COLORS) - 1) / max(n_buckets - 1, 1)))
            c = TRADE_VOLUME_COLORS[pal_idx]
            b_brushes.append(pg.mkBrush(c))
            b_sizes.append(6 + bucket * (14 / max(n_buckets - 1, 1)))

        self.sc_buy.setData(x=mb_t, y=mb_p)
        self.sc_sell.setData(x=ms_t, y=ms_p)
        self.sc_bot.setData(x=b_t, y=b_p, brush=b_brushes, size=b_sizes)

        self.ob_res = build_ob_heatmap(self.data['prices_df'], prod, day)
        ob_max_vol = self.ob_res['max_vol'] if self.ob_res else 1.0
        if self.ob_res:
            self.img_item.setImage(self.ob_res['img'], autoLevels=False)
            x_min, x_max = t[0], t[-1]
            y_min, y_max = self.ob_res['levels'][0], self.ob_res['levels'][-1]
            x_step = (t[1] - t[0]) if len(t) > 1 else 100
            self.img_item.setRect(QRectF(x_min - 0.5 * x_step, y_min - 0.5, (len(t)) * x_step, y_max - y_min + 1))
            self.img_item.setVisible(self.img_item.isVisible())
        else:
            self.img_item.setVisible(False)
        self.hm_legend.update_ranges(ob_max_vol, quantile_edges)

        custom_data = self.data.get('custom', {})
        if len(t) > 0:
            t_min, t_max = t[0], t[-1]
            for i, (name, pts) in enumerate(custom_data.items()):
                if name not in self.custom_curves:
                    color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
                    curve = self.p_m.plot(pen=pg.mkPen(color, width=1.5), name=f"[C] {name}")
                    curve.setVisible(False)
                    self.custom_curves[name] = curve
                    self.leg_m.addItem(curve, f"[C] {name}")
                    label = self.leg_m.items[-1][1]
                    label.setAttr('color', DIM)
                
                curve = self.custom_curves[name]
                pts_filtered = [p for p in pts if t_min <= p[0] <= t_max]
                if pts_filtered:
                    curve.setData([p[0] for p in pts_filtered], [p[1] for p in pts_filtered])
                else:
                    curve.setData([], [])

        self.p_m.autoRange()
        self.p_pnl.autoRange()

    def _update_dashboard(self):
        if not self.data or not hasattr(self, 'cb_dash_prod'): return
        prod = self.cb_dash_prod.currentText()
        day = self.cb_day.currentText()
        df = self.data['prices_df']
        if prod != 'Overall':
            df = df.filter(pl.col('product') == prod)
        
        trades = self.data.get('trades', [])
        my_trades_cnt = 0
        bot_trades_cnt = 0
        volume_traded = 0
        realized_pnl_trades = []
        
        if prod != 'Overall':
            prod_trades = [tr for tr in trades if tr.get('symbol') == prod and (day == 'All' or tr.get('day', int(day)) == int(day))]
        else:
            prod_trades = [tr for tr in trades if (day == 'All' or tr.get('day', int(day)) == int(day))]

        pos_map = {}
        cost_map = {}

        for tr in sorted(prod_trades, key=lambda x: (x.get('day', 0), x.get('timestamp', 0))):
            is_buyer = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_seller = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            qty = int(tr.get('quantity', 0))
            price = float(tr.get('price', 0))
            sym = tr.get('symbol', 'unknown')

            if not is_buyer and not is_seller:
                bot_trades_cnt += 1
                continue

            my_trades_cnt += 1
            volume_traded += qty
            
            p_pos = pos_map.get(sym, 0)
            p_cost = cost_map.get(sym, 0.0)

            if is_buyer:
                if p_pos >= 0:
                    cost_map[sym] = (p_cost * p_pos + price * qty) / (p_pos + qty)
                else:
                    closing = min(qty, abs(p_pos))
                    realized = closing * (p_cost - price)
                    realized_pnl_trades.append(realized)
                    if qty > abs(p_pos):
                        cost_map[sym] = price
                pos_map[sym] = p_pos + qty
            else:
                if p_pos <= 0:
                    cost_map[sym] = (p_cost * abs(p_pos) + price * qty) / (abs(p_pos) + qty)
                else:
                    closing = min(qty, p_pos)
                    realized = closing * (price - p_cost)
                    realized_pnl_trades.append(realized)
                    if qty > p_pos:
                        cost_map[sym] = price
                pos_map[sym] = p_pos - qty

        winning_trades = sum(1 for r in realized_pnl_trades if r > 0)
        win_rate = (winning_trades / len(realized_pnl_trades) * 100) if realized_pnl_trades else 0.0

        if day == 'All' and 'day' in df.columns:
            t_col = 'continuous_ts'
            min_day = self.data['prices_df']['day'].min() if 'day' in self.data['prices_df'].columns else 0
            df = df.with_columns((pl.col('timestamp') + (pl.col('day') - min_day) * 1000000).alias(t_col))
        else:
            t_col = 'timestamp'

        df = df.sort(t_col)
        
        if len(df) > 0:
            agg_df = df.group_by(t_col).agg(pl.col('profit_and_loss').sum().alias('pnl'))
            agg_df = agg_df.sort(t_col)
            t = agg_df[t_col].to_numpy()
            pnl_arr = agg_df['pnl'].to_numpy()
        else:
            t = np.array([])
            pnl_arr = np.array([])

        if len(pnl_arr) > 1:
            pnl_deltas = np.diff(pnl_arr)
            mean_delta = np.mean(pnl_deltas)
            std_delta = np.std(pnl_deltas)
            # Use 1000 multiplier to roughly normalize the typical small tick movements
            sharpe = (mean_delta / std_delta * 1000) if std_delta > 0 else 0
        else:
            sharpe = 0.0

        if len(pnl_arr) > 0:
            peak = np.maximum.accumulate(pnl_arr)
            dd_abs = peak - pnl_arr
            dd_pct = np.zeros_like(dd_abs)
            valid = peak > 0
            dd_pct[valid] = (dd_abs[valid] / peak[valid]) * 100.0
            
            show_pct = self.chk_dd_pct.isChecked()
            dd_arr = dd_pct if show_pct else dd_abs
            max_dd = np.max(dd_arr)
        else:
            dd_arr = np.array([])
            max_dd = 0.0
            show_pct = False

        self.lbl_sharpe.setText(f"Sharpe Ratio: {sharpe:,.3f}")
        self.lbl_winrate.setText(f"Win Rate: {win_rate:,.1f}% ({winning_trades}/{len(realized_pnl_trades)})")
        self.lbl_volume.setText(f"Total Volume Traded: {volume_traded:,}")
        self.lbl_my_trades.setText(f"My Trades: {my_trades_cnt:,}")
        self.lbl_bot_trades.setText(f"Bot Trades: {bot_trades_cnt:,}")

        self.curve_dash_pnl.setData(t, pnl_arr)
        self.curve_dash_dd.setData(t, dd_arr)

        if show_pct:
            self.p_dash_dd.setLabel('left', 'Drawdown (%)')
        else:
            self.p_dash_dd.setLabel('left', 'Drawdown (Shells)')

        self.p_dash_pnl.autoRange()
        self.p_dash_dd.autoRange()

def _auto_detect_log(script_dir: str):
    """Return path if exactly one .log file exists next to the script, else None."""
    logs = [f for f in os.listdir(script_dir) if f.endswith('.log')]
    return os.path.join(script_dir, logs[0]) if len(logs) == 1 else None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    # Resolve startup log path:
    #   1. CLI argument
    #   2. Single .log in script directory (auto-detect)
    #   3. Open file dialog
    if len(sys.argv) > 1:
        startup_log = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        startup_log = _auto_detect_log(script_dir)
        if startup_log is None:
            # Multiple or zero logs found — let user pick
            startup_log, _ = QFileDialog.getOpenFileName(
                None, "Open Log File", script_dir, "Log (*.log *.json)"
            )
            startup_log = startup_log or None  # empty string → None

    win = LogVisualizer(startup_log)
    win.show()
    sys.exit(app.exec())