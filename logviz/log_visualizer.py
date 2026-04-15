import sys, os, json, re, subprocess
from io import StringIO
import numpy as np
import polars as pl
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit,
    QScrollArea, QCheckBox, QDialog, QRadioButton, QButtonGroup, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QFont, QColor, QBrush

from plotting_utils import build_ob_heatmap, build_order_placement_heatmap, get_rect

# --- Styling & Colors ---
BG, PANEL_BG, BORDER, TEXT, DIM = '#0d0f14', '#12151c', '#1e2330', '#c8d0e0', '#4a5068'
ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE, ACCENT_ORANGE = \
    '#00d4ff', '#39ff6e', '#ff3d5a', '#ffd700', '#ffffff', '#b06dff', '#ff9f43'
CUSTOM_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f', '#b06dff', '#ff9f43']

# ── Heatmap Volume Coloring (10-Band Gradient Hues) ─────────────────────────
# Format: [R, G, B]. Buy = Cyan/Teal tones, Sell = Gold/Orange tones.
BUY_VOLUME_COLORS = [
    [0, 100, 100], [0, 130, 130], [0, 160, 160], [0, 190, 190], [0, 220, 220],
    [0, 255, 255], [100, 255, 255], [150, 255, 255], [200, 255, 255], [255, 255, 255]
]
SELL_VOLUME_COLORS = [
    [100, 60, 0], [130, 80, 0], [160, 100, 0], [190, 120, 0], [220, 140, 0],
    [255, 160, 0], [255, 180, 50], [255, 200, 100], [255, 220, 150], [255, 255, 255]
]

# ── Order Placement Heatmap Coloring ─────────────────────────────────────────
ORDER_BUY_COLORS = [[0, 0, int(100 + i * 15.5)] for i in range(10)]
ORDER_SELL_COLORS = [[int(100 + i * 15.5), 0, 0] for i in range(10)]

# Palette for bot trade markers
TRADE_VOLUME_COLORS = [
    [0, 255, 255], [0, 230, 230], [0, 200, 200], [0, 170, 170],
    [255, 215, 0], [255, 180, 0], [255, 140, 0], [255, 100, 0],
    [255, 60, 0], [255, 0, 0]
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
QRadioButton {{
    spacing: 8px;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {BORDER};
    border-radius: 11px;
    background: {PANEL_BG};
}}
QRadioButton::indicator:checked {{
    background: {ACCENT_CYAN};
    border-color: {ACCENT_WHITE};
}}
"""

class InteractiveLegendItem(pg.LegendItem):
    """Refined legend that supports proxy items for toggle logic."""
    def addItem(self, item, name, toggle_target=None):
        try:
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
        self.trade_cells = []

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
            cell = QWidget()
            cell.setLayout(vbox)
            self.trade_cells.append(cell)
            self._trade_layout.addWidget(cell)
        return self._trade_layout

    def update_ranges(self, max_vol, quantile_edges):
        def update_ob(lbl_list, mv):
            for i in range(10):
                lower = int(np.floor(i * mv / 10))
                upper = int(np.floor((i + 1) * mv / 10))
                lbl_list[i].setText(f"{lower}-{upper}" if i < 9 else f">{lower}")
        update_ob(self.buy_ranges,  max_vol)
        update_ob(self.sell_ranges, max_vol)

        n_buckets = len(quantile_edges) - 1
        for i, (cell, rlbl) in enumerate(zip(self.trade_cells, self.trade_ranges)):
            if i < n_buckets:
                pal_idx = int(round(i * (len(TRADE_VOLUME_COLORS) - 1) / max(n_buckets - 1, 1)))
                c = TRADE_VOLUME_COLORS[pal_idx]
                swatch = cell.findChild(QLabel)
                swatch.setStyleSheet(f"background-color: rgb({c[0]}, {c[1]}, {c[2]}); border: 1px solid #333;")
                lo, hi = quantile_edges[i], quantile_edges[i + 1]
                fmt = lambda v: f"{v:.0f}" if v == int(v) else f"{v:.2f}"
                rlbl.setText(f"{fmt(lo)}-{fmt(hi)}" if i < n_buckets - 1 else f">{fmt(lo)}")
                cell.setVisible(True)
            else:
                cell.setVisible(False)
                rlbl.setText("")

def _is_timestamps_continuous(df):
    if 'day' not in df.columns: return False
    days = df['day'].unique().sort().to_list()
    if len(days) <= 1: return False
    for i in range(len(days) - 1):
        d1_max = df.filter(pl.col('day') == days[i])['timestamp'].max()
        d2_min = df.filter(pl.col('day') == days[i + 1])['timestamp'].min()
        if d2_min > d1_max:
            return True
    return False

class BacktestRunner(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, script_path, backtests_dir):
        super().__init__()
        self.script_path = script_path
        self.backtests_dir = backtests_dir

    def run(self):
        try:
            subprocess.run([sys.executable, self.script_path], cwd=os.path.dirname(self.script_path), check=True)
            logs = [os.path.join(self.backtests_dir, f) for f in os.listdir(self.backtests_dir) if f.endswith('.log')]
            if logs:
                self.finished.emit(max(logs, key=os.path.getmtime))
            else:
                self.error.emit("No .log files found in backtests dir")
        except Exception as e:
            self.error.emit(str(e))

# --- Data Settings Dialog ---
class DataSetupDialog(QDialog):
    def __init__(self, keys, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Setup"); self.setMinimumWidth(400)
        self.settings = settings; layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll_content = QWidget(); self.grid = QGridLayout(scroll_content)
        self.grid.setColumnStretch(0, 1)
        self.grid.addWidget(QLabel("<b>Data Key</b>"), 0, 0)
        self.grid.addWidget(QLabel("<b>Main Pane</b>"), 0, 1)
        self.grid.addWidget(QLabel("<b>Generic Pane</b>"), 0, 2)
        self.groups = {}
        for i, key in enumerate(keys):
            self.grid.addWidget(QLabel(key), i+1, 0)
            bm, bg = QRadioButton(), QRadioButton()
            grp = QButtonGroup(self); grp.addButton(bm); grp.addButton(bg)
            self.grid.addWidget(bm, i+1, 1, Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(bg, i+1, 2, Qt.AlignmentFlag.AlignCenter)
            if self.settings.get(key, "generic") == "main": bm.setChecked(True)
            else: bg.setChecked(True)
            self.groups[key] = grp
        scroll.setWidget(scroll_content); scroll.setWidgetResizable(True); layout.addWidget(scroll)
        btn = QPushButton("Apply"); btn.clicked.connect(self.accept); layout.addWidget(btn)
    def get_results(self): return {k: ("main" if g.buttons()[0].isChecked() else "generic") for k, g in self.groups.items()}

class LogVisualizer(QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.setWindowTitle('Prosperity Sandbox Visualizer')
        self.setGeometry(50, 50, 1600, 920)
        self.data, self.current_df, self.ob_res = None, None, None
        self.custom_curves = {}
        self.data_settings = {}
        self.markup_lines = []
        self.markup_enabled = False
        self.sandbox_msgs = {}
        self._current_log_path = None
        self._gen_pane_minimized = False
        self._backtest_runner = None
        pg.setConfigOptions(useOpenGL=True, imageAxisOrder='row-major')
        self._build_ui()
        self._restore_window_state()
        if log_path: self._load_file(log_path)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)
        controls = QHBoxLayout(); controls.setContentsMargins(12, 12, 12, 12); controls.setSpacing(12)
        btn_open = QPushButton("📂 Open Log"); btn_open.clicked.connect(self._open_dialog); controls.addWidget(btn_open)
        self.btn_backtest = QPushButton("⟳ Refresh Backtest"); self.btn_backtest.clicked.connect(self._run_backtest); self.btn_backtest.setVisible(False); controls.addWidget(self.btn_backtest)
        btn_import = QPushButton("📊 Import Data"); btn_import.clicked.connect(self._import_dataviz_data); controls.addWidget(btn_import)
        controls.addWidget(QLabel("Product:")); self.cb_prod = QComboBox(); controls.addWidget(self.cb_prod)
        controls.addWidget(QLabel("Day:")); self.cb_day = QComboBox(); controls.addWidget(self.cb_day)
        self.cb_prod.currentTextChanged.connect(self._process_selection); self.cb_day.currentTextChanged.connect(self._process_selection)
        controls.addStretch()
        btn_setup = QPushButton("⚙️ Data Setup [S]"); btn_setup.clicked.connect(self._open_data_setup); controls.addWidget(btn_setup)
        self.lbl_markup = QLabel("MARKUP: OFF"); self.lbl_markup.setStyleSheet(f"color: {DIM}; font-weight: bold;"); controls.addWidget(self.lbl_markup)
        self.lbl_zoom = QLabel("Mode: XY"); self.lbl_zoom.setStyleSheet(f"color: {DIM};"); controls.addWidget(self.lbl_zoom)
        btn_export = QPushButton("💾 Export Custom CSV"); btn_export.clicked.connect(self._export_custom_csv); controls.addWidget(btn_export)
        main_layout.addLayout(controls)
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self._build_dashboard_tab()
        market_container = QWidget(); market_layout = QVBoxLayout(market_container); market_layout.setContentsMargins(0, 0, 0, 0); market_layout.setSpacing(0)
        self.hm_legend = HeatmapLegend(); market_layout.addWidget(self.hm_legend)
        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG); market_layout.addWidget(self.gw_m)
        self.tabs.addTab(market_container, "Market View")
        self.p_m = self.gw_m.addPlot(row=0, col=0); self.p_m.showGrid(x=True, y=True, alpha=0.3); self.p_m.setDownsampling(auto=True, mode='peak')
        self.p_gen = self.gw_m.addPlot(row=1, col=0); self.p_gen.showGrid(x=True, y=True, alpha=0.3); self.p_gen.setFixedHeight(200); self.p_gen.setXLink(self.p_m); self.p_gen.hideAxis('bottom'); self.p_gen.addLegend()
        self.img_item = pg.ImageItem(); self.img_item.setZValue(0); self.p_m.addItem(self.img_item)
        self.img_orders = pg.ImageItem(); self.img_orders.setZValue(1); self.p_m.addItem(self.img_orders); self.img_orders.setVisible(False)
        self.curve_mid = self.p_m.plot(pen=pg.mkPen(ACCENT_CYAN, width=2), name="Mid Price", clipToView=True)
        self.sc_bot = pg.ScatterPlotItem(symbol='x', size=7, brush=ACCENT_WHITE, name="Bot Trades")
        self.sc_buy = pg.ScatterPlotItem(symbol='t1', size=10, brush=ACCENT_CYAN, name="My Buy")
        self.sc_sell = pg.ScatterPlotItem(symbol='t', size=10, brush=ACCENT_ORANGE, name="My Sell")
        for item in [self.sc_bot, self.sc_buy, self.sc_sell]: self.p_m.addItem(item)
        self.leg_m = InteractiveLegendItem(offset=(10, 10)); self.leg_m.setParentItem(self.p_m.graphicsItem())
        self._heatmap_proxy = pg.PlotDataItem(pen=None, brush=pg.mkBrush(ACCENT_PURPLE))
        self.leg_m.addItem(self._heatmap_proxy, "Heatmap", toggle_target=self.img_item)
        self._orders_proxy = pg.PlotDataItem(pen=None, brush=pg.mkBrush(ACCENT_WHITE))
        self.leg_m.addItem(self._orders_proxy, "Order Placement", toggle_target=self.img_orders)
        self.leg_m.addItem(self.curve_mid, "Mid Price"); self.leg_m.addItem(self.sc_bot, "Bot Trades"); self.leg_m.addItem(self.sc_buy, "My Buy"); self.leg_m.addItem(self.sc_sell, "My Sell")
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.p_m.addItem(self.v_line, ignoreBounds=True); self.p_m.addItem(self.h_line, ignoreBounds=True)
        self.p_m.scene().sigMouseMoved.connect(self._on_mouse_moved); self.p_m.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # PnL Tab
        pnl_container = QWidget()
        pnl_layout = QVBoxLayout(pnl_container); pnl_layout.setContentsMargins(0, 0, 0, 0); pnl_layout.setSpacing(0)
        pnl_ctrl = QHBoxLayout(); pnl_ctrl.setContentsMargins(12, 8, 12, 8)
        self.cb_pnl_type = QComboBox(); self.cb_pnl_type.addItems(["Log PnL", "Realized PnL", "Valuation PnL"])
        self.cb_pnl_type.currentTextChanged.connect(self._process_selection)
        pnl_ctrl.addWidget(QLabel("PnL Method:")); pnl_ctrl.addWidget(self.cb_pnl_type); pnl_ctrl.addStretch()
        pnl_layout.addLayout(pnl_ctrl)
        self.gw_p = pg.GraphicsLayoutWidget(); self.gw_p.setBackground(BG); pnl_layout.addWidget(self.gw_p)
        self.tabs.addTab(pnl_container, "PnL")
        self.p_pnl = self.gw_p.addPlot(); self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_pnl = self.p_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2), clipToView=True)

        self.gw_pos = pg.GraphicsLayoutWidget(); self.gw_pos.setBackground(BG); self.tabs.addTab(self.gw_pos, "Position")
        self.p_pos = self.gw_pos.addPlot(); self.p_pos.showGrid(x=True, y=True, alpha=0.3); self.p_pos.addLegend()
        self.pos_curves = {}

        self.gw_c = pg.GraphicsLayoutWidget(); self.gw_c.setBackground(BG); self.tabs.addTab(self.gw_c, "Custom")

        # Logs Tab
        logs_container = QWidget(); logs_layout = QVBoxLayout(logs_container); logs_layout.setContentsMargins(0, 0, 0, 0); logs_layout.setSpacing(0)
        logs_filter_bar = QHBoxLayout(); logs_filter_bar.setContentsMargins(8, 6, 8, 6)
        self.log_filter_input = QLineEdit(); self.log_filter_input.setPlaceholderText("Filter logs..."); self.log_filter_input.textChanged.connect(self._filter_logs_table)
        logs_filter_bar.addWidget(QLabel("🔍")); logs_filter_bar.addWidget(self.log_filter_input); logs_layout.addLayout(logs_filter_bar)
        self.logs_table = QTableWidget(); self.logs_table.setColumnCount(6); self.logs_table.setHorizontalHeaderLabels(['TS', 'Tag', 'Product', 'Pos', 'PnL', 'Msg'])
        self.logs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch); self.logs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.logs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.logs_table.verticalHeader().setVisible(False); logs_layout.addWidget(self.logs_table)
        self.tabs.addTab(logs_container, "Logs")

        self.data_strip = QLabel("Ready"); self.data_strip.setObjectName("DataStrip"); self.data_strip.setFixedHeight(32); main_layout.addWidget(self.data_strip)

        QShortcut(QKeySequence("X"), self).activated.connect(lambda: self._set_zoom("x"))
        QShortcut(QKeySequence("Y"), self).activated.connect(lambda: self._set_zoom("y"))
        QShortcut(QKeySequence("Z"), self).activated.connect(lambda: self._set_zoom("xy"))
        QShortcut(QKeySequence("A"), self).activated.connect(self._autoscale_all)
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_markup)
        QShortcut(QKeySequence("Shift+M"), self).activated.connect(self._toggle_gen_pane)
        QShortcut(QKeySequence("C"), self).activated.connect(self._clear_markup)
        QShortcut(QKeySequence("S"), self).activated.connect(self._open_data_setup)
        QShortcut(QKeySequence("?"), self).activated.connect(self._focus_keybinds_tab)

        self._build_sandbox_tab()
        self._build_keybinds_tab()

    def _open_data_setup(self):
        if not self.data: return
        dlg = DataSetupDialog(sorted(self.data.get('custom',{}).keys()), self.data_settings, self)
        if dlg.exec():
            self.data_settings = dlg.get_results()
            self._save_data_settings()
            for c in self.custom_curves.values(): self.p_m.removeItem(c); self.p_gen.removeItem(c)
            self.custom_curves = {}
            self._process_selection()

    def _settings_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.logviz_settings.json')

    def _read_config(self):
        try:
            p = self._settings_path()
            if os.path.exists(p):
                with open(p) as f:
                    data = json.load(f)
                    # migrate old flat data_settings format
                    if data and not any(k in data for k in ('data_settings', 'window')):
                        return {'data_settings': data}
                    return data
        except Exception as e:
            print(f"Config read error: {e}")
        return {}

    def _write_config(self, cfg):
        try:
            with open(self._settings_path(), 'w') as f: json.dump(cfg, f)
        except Exception as e:
            print(f"Config write error: {e}")

    def _save_data_settings(self):
        cfg = self._read_config()
        cfg['data_settings'] = self.data_settings
        self._write_config(cfg)

    def _load_data_settings(self):
        return self._read_config().get('data_settings', {})

    def _save_window_state(self):
        g = self.geometry()
        cfg = self._read_config()
        cfg['window'] = {'x': g.x(), 'y': g.y(), 'w': g.width(), 'h': g.height(), 'tab': self.tabs.currentIndex()}
        self._write_config(cfg)

    def _restore_window_state(self):
        w = self._read_config().get('window')
        if not w: return
        self.setGeometry(w.get('x', 50), w.get('y', 50), w.get('w', 1600), w.get('h', 920))
        tab = w.get('tab', 0)
        if 0 <= tab < self.tabs.count():
            self.tabs.setCurrentIndex(tab)

    def closeEvent(self, event):
        self._save_window_state()
        super().closeEvent(event)

    def _build_keybinds_tab(self):
        keybinds = [
            ("X",       "Zoom X axis only"),
            ("Y",       "Zoom Y axis only"),
            ("Z",       "Zoom XY (reset)"),
            ("A",       "Autoscale all plots"),
            ("M",       "Toggle markup mode"),
            ("Shift+M", "Toggle secondary (gen) pane"),
            ("C",       "Clear markup lines"),
            ("S",       "Open Data Setup"),
            ("?",       "Go to this tab"),
        ]
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(24, 24, 24, 24); layout.setSpacing(0)
        table = QTableWidget(len(keybinds), 2); table.setHorizontalHeaderLabels(["Key", "Action"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for i, (key, desc) in enumerate(keybinds):
            k_item = QTableWidgetItem(key); k_item.setForeground(QBrush(QColor(ACCENT_CYAN)))
            k_item.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
            k_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 0, k_item); table.setItem(i, 1, QTableWidgetItem(desc))
        table.resizeRowsToContents(); table.setMaximumWidth(500); layout.addWidget(table); layout.addStretch()
        self._keybinds_tab_index = self.tabs.addTab(container, "Keys")

    def _focus_keybinds_tab(self):
        self.tabs.setCurrentIndex(self._keybinds_tab_index)

    def _apply_saved_settings(self):
        saved = self._load_data_settings()
        if not saved or not self.data: return
        current_keys = set(self.data.get('custom', {}).keys())
        for k, v in saved.items():
            if k in current_keys:
                self.data_settings[k] = v

    def _toggle_gen_pane(self):
        self._gen_pane_minimized = not self._gen_pane_minimized
        if self._gen_pane_minimized:
            self.p_gen.setFixedHeight(0)
            self.p_gen.setVisible(False)
        else:
            self.p_gen.setFixedHeight(200)
            if self.data:
                has_generic = any(self.data_settings.get(k, "generic") == "generic" for k in self.data.get('custom', {}))
                self.p_gen.setVisible(has_generic)

    def _is_backtest_log(self, path):
        if not path: return False
        return 'backtests' in os.path.normpath(path).split(os.sep)

    def _run_backtest(self):
        self.btn_backtest.setEnabled(False)
        self.btn_backtest.setText("⟳ Running...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(project_root, 'run_my_strategy.py')
        backtests_dir = os.path.join(project_root, 'backtests')
        self._backtest_runner = BacktestRunner(script, backtests_dir)
        self._backtest_runner.finished.connect(self._on_backtest_done)
        self._backtest_runner.error.connect(self._on_backtest_error)
        self._backtest_runner.start()

    def _on_backtest_done(self, log_path):
        self.btn_backtest.setEnabled(True)
        self.btn_backtest.setText("⟳ Refresh Backtest")
        self._load_file(log_path)

    def _on_backtest_error(self, msg):
        self.btn_backtest.setEnabled(True)
        self.btn_backtest.setText("⟳ Refresh Backtest")
        QMessageBox.warning(self, "Backtest Error", msg)

    def _toggle_markup(self):
        self.markup_enabled = not self.markup_enabled
        self.lbl_markup.setText(f"MARKUP: {'ON' if self.markup_enabled else 'OFF'}")
        self.lbl_markup.setStyleSheet(f"color: {ACCENT_GOLD if self.markup_enabled else DIM}; font-weight: bold;")

    def _clear_markup(self):
        for item in self.markup_lines:
            self.p_m.removeItem(item)
        self.markup_lines = []

    def _on_mouse_clicked(self, event):
        if not self.markup_enabled or event.button() != Qt.MouseButton.LeftButton: return
        pos = event.scenePos()
        if not self.p_m.sceneBoundingRect().contains(pos): return
        mouse_point = self.p_m.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        vl = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
        hl = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
        txt = pg.TextItem(text=f"({x:.0f}, {y:.1f})", color=ACCENT_GOLD, anchor=(0, 1))
        txt.setPos(x, y)
        
        self.p_m.addItem(vl); self.p_m.addItem(hl); self.p_m.addItem(txt)
        self.markup_lines.extend([vl, hl, txt])
        print(f"Markup point added: X={x:.0f}, Y={y:.1f}")

    def _build_dashboard_tab(self):
        dash_container = QWidget()
        dash_layout = QVBoxLayout(dash_container); dash_layout.setContentsMargins(12, 12, 12, 12); dash_layout.setSpacing(12)
        filter_layout = QHBoxLayout(); filter_layout.addWidget(QLabel("Dashboard Product:")); self.cb_dash_prod = QComboBox()
        self.cb_dash_prod.currentTextChanged.connect(self._update_dashboard); filter_layout.addWidget(self.cb_dash_prod); filter_layout.addStretch(); dash_layout.addLayout(filter_layout)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setFrameShape(QFrame.Shape.NoFrame); scroll_area.setStyleSheet(f"background-color: {BG};")
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content); scroll_layout.setContentsMargins(0, 0, 0, 0); scroll_layout.setSpacing(15)
        metrics_frame = QFrame(); metrics_frame.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;"); metrics_layout = QVBoxLayout(metrics_frame)
        self.lbl_sharpe = QLabel("Sharpe Ratio: --"); self.lbl_sharpe.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 11pt;"); metrics_layout.addWidget(self.lbl_sharpe)
        h_metrics = QHBoxLayout(); self.lbl_winrate = QLabel("Win Rate: --"); self.lbl_volume = QLabel("Total Volume Traded: --")
        self.lbl_my_trades = QLabel("My Trades: --"); self.lbl_bot_trades = QLabel("Bot Trades: --")
        for l in [self.lbl_winrate, self.lbl_volume, self.lbl_my_trades, self.lbl_bot_trades]: h_metrics.addWidget(l)
        h_metrics.addStretch(); metrics_layout.addLayout(h_metrics); scroll_layout.addWidget(metrics_frame)
        self.gw_dash_pnl = pg.GraphicsLayoutWidget(); self.gw_dash_pnl.setBackground(BG); self.gw_dash_pnl.setFixedHeight(300)
        self.p_dash_pnl = self.gw_dash_pnl.addPlot(title="Cumulative PnL"); self.curve_dash_pnl = self.p_dash_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2)); scroll_layout.addWidget(self.gw_dash_pnl)
        dd_container = QWidget(); dd_layout = QVBoxLayout(dd_container); dd_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_dd_pct = QCheckBox("Show Percentage %"); self.chk_dd_pct.stateChanged.connect(self._update_dashboard); dd_layout.addWidget(self.chk_dd_pct)
        self.gw_dash_dd = pg.GraphicsLayoutWidget(); self.gw_dash_dd.setBackground(BG); self.gw_dash_dd.setFixedHeight(300)
        self.p_dash_dd = self.gw_dash_dd.addPlot(title="Drawdown"); self.p_dash_dd.setXLink(self.p_dash_pnl); self.curve_dash_dd = self.p_dash_dd.plot(pen=pg.mkPen(ACCENT_RED, width=2, fillLevel=0, brush=(255, 61, 90, 50))); dd_layout.addWidget(self.gw_dash_dd)
        scroll_layout.addWidget(dd_container); scroll_layout.addStretch(); scroll_area.setWidget(scroll_content); dash_layout.addWidget(scroll_area)
        self.tabs.insertTab(0, dash_container, "Dashboard"); self.tabs.setCurrentIndex(0)

    def _build_sandbox_tab(self):
        sandbox_container = QWidget(); sandbox_layout = QVBoxLayout(sandbox_container); sandbox_layout.setContentsMargins(0, 0, 0, 0); sandbox_layout.setSpacing(0)
        sb_filter_bar = QHBoxLayout(); sb_filter_bar.setContentsMargins(8, 6, 8, 6); self.sb_filter_input = QLineEdit()
        self.sb_filter_input.setPlaceholderText("Filter sandbox logs..."); self.sb_filter_input.textChanged.connect(self._filter_sandbox_table)
        sb_filter_bar.addWidget(QLabel("🔍")); sb_filter_bar.addWidget(self.sb_filter_input); sandbox_layout.addLayout(sb_filter_bar)
        sb_content_layout = QHBoxLayout(); self.sb_table = QTableWidget(); self.sb_table.setColumnCount(2); self.sb_table.setHorizontalHeaderLabels(['Message', 'Occurrences'])
        self.sb_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch); self.sb_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sb_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.sb_table.verticalHeader().setVisible(False); self.sb_table.itemSelectionChanged.connect(self._on_sb_selection_changed)
        sb_content_layout.addWidget(self.sb_table, stretch=2); self.sb_detail_panel = QFrame(); self.sb_detail_panel.setStyleSheet(f"background-color: {PANEL_BG}; border-left: 1px solid {BORDER};"); self.sb_detail_panel.setFixedWidth(300)
        detail_layout = QVBoxLayout(self.sb_detail_panel); detail_layout.addWidget(QLabel("<b>Occurrences</b>")); self.sb_detail_text = QLabel("Select message")
        self.sb_detail_text.setWordWrap(True); detail_scroll = QScrollArea(); detail_scroll.setWidgetResizable(True); detail_scroll.setWidget(self.sb_detail_text)
        detail_layout.addWidget(detail_scroll); sb_content_layout.addWidget(self.sb_detail_panel); sandbox_layout.addLayout(sb_content_layout)
        self.tabs.addTab(sandbox_container, "SandboxLog")

    def _on_sb_selection_changed(self):
        items = self.sb_table.selectedItems()
        if not items: return
        msg = items[0].text()
        if msg in self.sandbox_msgs:
            ts_list = self.sandbox_msgs[msg]; txt = f"<b>Total: {len(ts_list)}</b><br><br>" + "<br>".join([str(ts) for ts in ts_list[:50]])
            if len(ts_list) > 50: txt += "<br>..."
            self.sb_detail_text.setText(txt)

    def _filter_sandbox_table(self, text):
        for row in range(self.sb_table.rowCount()):
            item = self.sb_table.item(row, 0)
            if item: self.sb_table.setRowHidden(row, text.lower() not in item.text().lower())

    def _on_mouse_moved(self, pos):
        if not self.p_m.sceneBoundingRect().contains(pos) or self.current_df is None: return
        mouse_point = self.p_m.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        ts_array = self.current_df['timestamp'].to_numpy()
        idx = np.clip(np.searchsorted(ts_array, x), 0, len(ts_array) - 1)
        row = self.current_df.row(idx, named=True); ts_val = int(row['timestamp'])
        info = f"TS: {ts_val}  |  MID: {row.get('mid_price',0):,.1f}  |  PnL: {row.get('profit_and_loss',0):,.0f}"
        if self.ob_res:
            y_levels = self.ob_res['levels']; y_idx = np.searchsorted(y_levels, y)
            if 0 <= y_idx < len(y_levels):
                vol = self.ob_res['raw_vol'][y_idx, idx]
                if vol > 0: info += f"  |  VOL @ {y_levels[y_idx]:.0f}: {vol:,.0f}"
        self.v_line.setPos(ts_val); self.h_line.setPos(y); self.data_strip.setText(info)

    def _set_zoom(self, mode):
        self.lbl_zoom.setText(f"Mode: {mode.upper()}"); self.p_m.setMouseEnabled(x=(mode in ['x', 'xy']), y=(mode in ['y', 'xy']))

    def _autoscale_all(self):
        for p in [self.p_m, self.p_pnl, self.p_pos, self.p_dash_pnl, self.p_dash_dd]:
            if p: p.autoRange()

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Log", "", "Log (*.log *.json)")
        if path: self._load_file(path)

    def _export_custom_csv(self):
        custom = (self.data or {}).get('custom', {})
        if not custom: return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "custom.csv", "CSV (*.csv)")
        if not path: return
        all_ts = sorted(set(ts for pts in custom.values() for ts, _ in pts))
        rows = ['timestamp,' + ','.join(custom.keys())]
        ts_map = {k: dict(pts) for k, pts in custom.items()}
        for ts in all_ts: rows.append(','.join([str(int(ts))] + [str(ts_map[k].get(ts, '')) for k in custom]))
        with open(path, 'w') as f: f.write('\n'.join(rows))

    def _import_dataviz_data(self):
        dataviz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataviz")
        if not os.path.exists(dataviz_dir): return
        p_dfs, t_dicts = [], []
        for f in os.listdir(dataviz_dir):
            if not f.endswith('.csv'): continue
            filepath = os.path.join(dataviz_dir, f)
            if f.startswith("prices_"): p_dfs.append(pl.read_csv(filepath, separator=";", null_values=['', 'nan']))
            elif f.startswith("trades_"):
                day_match = re.search(r"day_(-?\d+)", f); df = pl.read_csv(filepath, separator=";", null_values=['', 'nan'])
                if day_match: df = df.with_columns(pl.lit(int(day_match.group(1))).alias("day"))
                t_dicts.extend(df.to_dicts())
        if not p_dfs: return
        df = pl.concat(p_dfs, how='diagonal_relaxed').sort(['day', 'timestamp'])
        self.data = {'prices_df': df, 'trades': t_dicts, 'custom': {}, 'debug': [], 'orders': [], '_continuous_ts': _is_timestamps_continuous(df), '_min_day': df['day'].min() if 'day' in df.columns else 0}
        self.cb_prod.blockSignals(True); products = df['product'].unique().sort().to_list()
        self.cb_prod.clear(); self.cb_prod.addItems(products); self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False); self.cb_dash_prod.clear(); self.cb_dash_prod.addItems(['Overall'] + products)
        self._build_custom_plots(); self._build_position_plot(); self._build_logs_table(); self._process_selection(); self._update_dashboard()

    def _load_file(self, path):
        self._current_log_path = path
        with open(path, encoding='utf-8') as f: raw = json.load(f)
        csv_str = raw.get('activitiesLog', '').replace('\\n', '\n')
        df = pl.read_csv(StringIO(csv_str), separator=';', null_values=['', 'nan'])
        df = df.rename({c: c.strip() for c in df.columns}).sort(['day', 'timestamp'])
        # ts→day lookup so LOGORDER gets the actual day, not ts//1000000
        ts_to_day = dict(df.select(['timestamp', 'day']).unique().iter_rows()) if 'day' in df.columns else {}
        custom, orders, sandbox_msgs, debug_msgs = {}, [], {}, []
        for entry in raw.get('logs', []):
            ts, log = entry.get('timestamp', 0), entry.get('lambdaLog', '') or ''
            sb_log = (entry.get('sandboxLog', '') or '').strip()
            if sb_log: sandbox_msgs.setdefault(sb_log, []).append(ts)
            for line in log.split('\n'):
                if line.startswith('LOGVIZ:'):
                    try:
                        p = json.loads(line[7:])
                        for k, v in p.items(): custom.setdefault(k, []).append((ts, float(v)))
                    except: pass
                elif line.startswith('LOGORDER:'):
                    parts = line.split(':')
                    if len(parts) >= 5:
                        orders.append({'ts': ts, 'day': ts_to_day.get(ts, ts//1000000), 'product': parts[1] if len(parts)==6 else '', 'side': parts[2] if len(parts)==6 else parts[1],
                                     'price': float(parts[3] if len(parts)==6 else parts[2]), 'qty': int(parts[4] if len(parts)==6 else parts[3]), 'tag': parts[-1]})
                elif line.startswith('LOGDBG:'):
                    parts = line[7:].split(':', 2)
                    debug_msgs.append({'ts': ts, 'tag': parts[0], 'product': parts[1] if len(parts)>2 else '', 'msg': parts[-1]})
        if raw.get('error'): sandbox_msgs.setdefault(str(raw['error']).strip(), []).append("N/A")
        self.sandbox_msgs = sandbox_msgs
        min_day = df['day'].min() if 'day' in df.columns else 0
        self.data = {'prices_df': df, 'trades': raw.get('tradeHistory', []), 'custom': custom, 'debug': debug_msgs, 'orders': orders, '_continuous_ts': _is_timestamps_continuous(df), '_min_day': min_day}
        self.cb_prod.blockSignals(True); products = df['product'].unique().sort().to_list(); self.cb_prod.clear(); self.cb_prod.addItems(products)
        self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()]); self.cb_prod.blockSignals(False)
        self.cb_dash_prod.clear(); self.cb_dash_prod.addItems(['Overall'] + products)
        for c in self.custom_curves.values():
            try: self.p_m.removeItem(c)
            except: pass
            try: self.p_gen.removeItem(c)
            except: pass
        self.custom_curves = {}
        self.data_settings = {}
        self._apply_saved_settings()
        self.btn_backtest.setVisible(self._is_backtest_log(path))
        self._build_custom_plots(); self._build_position_plot(); self._build_logs_table(); self._update_sandbox_table(); self._process_selection(); self._update_dashboard()

    def _update_sandbox_table(self):
        self.sb_table.setRowCount(len(self.sandbox_msgs))
        for i, (msg, ts_list) in enumerate(sorted(self.sandbox_msgs.items(), key=lambda x: len(x[1]), reverse=True)):
            m_item = QTableWidgetItem(msg); m_item.setForeground(QBrush(QColor(ACCENT_RED)))
            c_item = QTableWidgetItem(str(len(ts_list))); c_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sb_table.setItem(i, 0, m_item); self.sb_table.setItem(i, 1, c_item)

    def _build_logs_table(self):
        debug_msgs = self.data.get('debug', []); df = self.data.get('prices_df'); trades = self.data.get('trades', [])
        self.logs_table.setRowCount(len(debug_msgs))
        for i, entry in enumerate(debug_msgs):
            tag = entry['tag'].upper(); row_color = {'ERR':QColor(ACCENT_RED),'WARN':QColor(ACCENT_GOLD),'INFO':QColor(ACCENT_CYAN)}.get(tag, QColor(TEXT))
            items = [str(entry['ts']), tag, entry.get('product',''), '', '', entry['msg']]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text); item.setForeground(QBrush(row_color)); self.logs_table.setItem(i, col, item)

    def _filter_logs_table(self, text):
        for row in range(self.logs_table.rowCount()):
            match = any(text.lower() in (self.logs_table.item(row, c).text() if self.logs_table.item(row, c) else '').lower() for c in range(6))
            self.logs_table.setRowHidden(row, not match)

    def _build_custom_plots(self):
        self.gw_c.clear(); custom = self.data.get('custom', {}); anchor = None
        for i, (name, pts) in enumerate(custom.items()):
            p = self.gw_c.addPlot(row=i, col=0, title=name); p.setDownsampling(auto=True, mode='peak')
            if anchor: p.setXLink(anchor)
            else: anchor = p
            p.plot([x[0] for x in pts], [x[1] for x in pts], pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2))

    def _build_position_plot(self):
        for c in self.pos_curves.values(): self.p_pos.removeItem(c)
        self.pos_curves.clear(); trades = self.data.get('trades', [])
        if not trades: return
        pos_by_sym = {}; cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        for tr in trades:
            is_b = str(tr.get('buyer','')).upper()=='SUBMISSION'; is_s = str(tr.get('seller','')).upper()=='SUBMISSION'
            if not is_b and not is_s: continue
            sym, qty, ts = tr.get('symbol',''), tr.get('quantity',0), tr.get('timestamp',0)
            pts = ts + (tr.get('day', min_day) - min_day)*1000000 if 'day' in tr and not cont_ts else ts
            pos_by_sym.setdefault(sym, []).append((tr.get('day', ts//1000000), pts, qty if is_b else -qty))
        for i, sym in enumerate(sorted(pos_by_sym.keys())):
            events = sorted(pos_by_sym[sym], key=lambda x: x[1]); ts_list, pos_list, cum, last_day = [], [], 0, events[0][0]
            for day, pts, delta in events:
                if day != last_day: cum = 0; last_day = day
                cum += delta; ts_list.append(pts); pos_list.append(cum)
            self.pos_curves[sym] = self.p_pos.plot(ts_list, pos_list, pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2), name=sym, stepMode='right')

    def _process_selection(self):
        if not self.data or not self.cb_prod.currentText(): return
        prod, day = self.cb_prod.currentText(), self.cb_day.currentText()
        self.current_df = self.data['prices_df'].filter(pl.col('product') == prod)
        try:
            if day != 'All': self.current_df = self.current_df.filter(pl.col('day') == int(day))
        except: pass
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t = self.current_df['timestamp'].to_numpy()
        if day == 'All' and 'day' in self.current_df.columns and not cont_ts: t = t + (self.current_df['day'].to_numpy() - min_day) * 1000000
        mid = self.current_df['mid_price'].to_numpy(); self.curve_mid.setData(t, mid)
        pnl_data = self.current_df['profit_and_loss'].to_numpy() if 'profit_and_loss' in self.current_df.columns else np.zeros(len(t))
        self.curve_pnl.setData(t, pnl_data)

        mb_t, mb_p, ms_t, ms_p, bot_raw = [], [], [], [], []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
            ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr and not cont_ts: ts += (tr['day'] - min_day) * 1000000
            is_b = str(tr.get('buyer','')).upper()=='SUBMISSION'; is_s = str(tr.get('seller','')).upper()=='SUBMISSION'
            if is_b: mb_t.append(ts); mb_p.append(tr['price'])
            elif is_s: ms_t.append(ts); ms_p.append(tr['price'])
            else: bot_raw.append((ts, tr['price'], int(tr.get('quantity', 1))))
        
        vols = np.array([v for _, _, v in bot_raw]) if bot_raw else np.array([1])
        q_edges = np.unique(np.percentile(vols, np.linspace(0, 100, 11))) if len(bot_raw)>1 else np.array([0, 1000000])
        n_buckets = len(q_edges)-1
        b_t, b_p, b_brushes, b_sizes = [], [], [], []
        for ts, pr, vol in bot_raw:
            b_t.append(ts); b_p.append(pr); buck = min(np.searchsorted(q_edges[1:], vol), n_buckets-1)
            pal_idx = int(buck * (len(TRADE_VOLUME_COLORS)-1) / max(n_buckets-1, 1))
            b_brushes.append(pg.mkBrush(TRADE_VOLUME_COLORS[pal_idx])); b_sizes.append(6 + buck * 2)
        self.sc_buy.setData(x=mb_t, y=mb_p); self.sc_sell.setData(x=ms_t, y=ms_p); self.sc_bot.setData(x=b_t, y=b_p, brush=b_brushes, size=b_sizes)

        self.ob_res = build_ob_heatmap(self.data['prices_df'], prod, day, cont_ts, BUY_VOLUME_COLORS, SELL_VOLUME_COLORS)
        if self.ob_res:
            self.img_item.setImage(self.ob_res['img'], autoLevels=False)
            self.img_item.setRect(get_rect(t, self.ob_res['levels']))
            self.img_item.setVisible(True)
            o_res = build_order_placement_heatmap(self.data.get('orders',[]), prod, day, t, self.ob_res['levels'], cont_ts, min_day, ORDER_BUY_COLORS, ORDER_SELL_COLORS)
            if o_res:
                self.img_orders.setImage(o_res['img'], autoLevels=False)
                self.img_orders.setRect(get_rect(t, self.ob_res['levels']))
            else: self.img_orders.clear()
            self.hm_legend.update_ranges(self.ob_res['max_vol'], q_edges)
        else: self.img_item.setVisible(False); self.img_orders.clear()

        # Update Custom Data Curves
        custom_data = self.data.get('custom', {})
        t_min, t_max = (t[0], t[-1]) if len(t) > 0 else (0, 1)
        
        has_generic_data = False
        for name, pts in custom_data.items():
            pane = self.data_settings.get(name, "generic")
            target_plot = self.p_m if pane == "main" else self.p_gen
            if pane == "generic": has_generic_data = True
            
            if name not in self.custom_curves:
                color = CUSTOM_COLORS[len(self.custom_curves) % len(CUSTOM_COLORS)]
                curve = target_plot.plot(pen=pg.mkPen(color, width=1.5), name=name)
                self.custom_curves[name] = curve
                if pane == "main":
                    self.leg_m.addItem(curve, f"[C] {name}")
            
            curve = self.custom_curves[name]
            pts_f = [p for p in pts if t_min <= p[0] <= t_max]
            if pts_f:
                curve.setData([p[0] for p in pts_f], [p[1] for p in pts_f])
            else:
                curve.setData([], [])

        self.p_gen.setVisible(has_generic_data and not self._gen_pane_minimized)
        self.p_m.autoRange(); self.p_pnl.autoRange()

    def _update_dashboard(self):
        if not self.data: return
        p, d, df = self.cb_dash_prod.currentText(), self.cb_day.currentText(), self.data['prices_df']
        if p != 'Overall': df = df.filter(pl.col('product') == p)
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t_col = 'cts' if (d=='All' and 'day' in df.columns and not cont_ts) else 'timestamp'
        if t_col == 'cts': df = df.with_columns((pl.col('timestamp') + (pl.col('day') - min_day) * 1000000).alias(t_col))
        df = df.sort(t_col); agg = df.group_by(t_col).agg(pl.col('profit_and_loss').sum().alias('pnl')).sort(t_col)
        t, pnl = agg[t_col].to_numpy(), agg['pnl'].to_numpy()
        self.curve_dash_pnl.setData(t, pnl)
        if len(pnl)>0:
            pk = np.maximum.accumulate(pnl); dd = pk - pnl
            if self.chk_dd_pct.isChecked() and np.any(pk>0): dd = (dd / np.where(pk>0, pk, 1)) * 100
            self.curve_dash_dd.setData(t, dd)
        self.p_dash_pnl.autoRange(); self.p_dash_dd.autoRange()

def _auto_detect_log(script_dir: str):
    search_dirs = [script_dir]
    project_root = os.path.dirname(script_dir)
    backtests_dir = os.path.join(project_root, 'backtests')
    if os.path.isdir(backtests_dir): search_dirs.append(backtests_dir)
    cwd_backtests = os.path.join(os.getcwd(), 'backtests')
    if os.path.isdir(cwd_backtests) and cwd_backtests not in search_dirs: search_dirs.append(cwd_backtests)
    all_logs = []
    for d in search_dirs:
        try:
            for f in os.listdir(d):
                if f.endswith('.log'): all_logs.append(os.path.join(d, f))
        except OSError: continue
    if not all_logs: return None
    return max(all_logs, key=os.path.getmtime)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(APP_STYLE)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    startup_log = sys.argv[1] if len(sys.argv)>1 else _auto_detect_log(script_dir)
    if startup_log is None:
        startup_log, _ = QFileDialog.getOpenFileName(None, "Open Log File", script_dir, "Log (*.log *.json)")
        startup_log = startup_log or None
    win = LogVisualizer(startup_log); win.show(); sys.exit(app.exec())