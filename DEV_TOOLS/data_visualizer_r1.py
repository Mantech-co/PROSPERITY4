# Consolidated Visualizer V3
import sys, os, json, re
from io import StringIO
import numpy as np
import polars as pl
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QFont, QColor, QBrush

# --- Styling & Colors ---
BG, PANEL_BG, BORDER, TEXT, DIM = '#0d0f14', '#12151c', '#1e2330', '#c8d0e0', '#4a5068'
ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE = \
    '#00d4ff', '#39ff6e', '#ff3d5a', '#ffd700', '#ffffff', '#b06dff'
CUSTOM_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f', '#b06dff', '#ff9f43']

APP_STYLE = f"""
QMainWindow, QWidget {{ 
    background-color: {BG}; 
    color: {TEXT}; 
    font-family: 'Segoe UI', Arial, sans-serif; 
    font-size: 10pt; 
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
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
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

# --- Data Engine ---

# Removed build_ob_heatmap

class LogVisualizer(QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.setStyleSheet(APP_STYLE)
        self.setWindowTitle('Prosperity Sandbox Visualizer')
        self.setGeometry(50, 50, 1600, 920)
        self.data, self.current_df = None, None
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
        
        controls.addWidget(QLabel("Product:"))
        self.cb_prod = QComboBox(); controls.addWidget(self.cb_prod)
        
        controls.addWidget(QLabel("Day:"))
        self.cb_day = QComboBox(); controls.addWidget(self.cb_day)
        
        self.cb_prod.currentTextChanged.connect(self._process_selection)
        self.cb_day.currentTextChanged.connect(self._process_selection)
        
        controls.addStretch()
        self.lbl_zoom = QLabel("Mode: XY"); self.lbl_zoom.setStyleSheet(f"color: {DIM};")
        controls.addWidget(self.lbl_zoom)

        lbl_marker_hint = QLabel(" (L-Click graph: Drop Point, R-Click: Clear) ")
        lbl_marker_hint.setStyleSheet(f"color: {DIM};")
        controls.addWidget(lbl_marker_hint)

        btn_export = QPushButton("💾 Export Custom CSV")
        btn_export.setToolTip("Export all custom LOGVIZ data to a CSV file")
        btn_export.clicked.connect(self._export_custom_csv)
        controls.addWidget(btn_export)

        main_layout.addLayout(controls)

        # Tabs
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        
        # Market View
        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG)
        self.tabs.addTab(self.gw_m, "Market View")
        self.p_m = self.gw_m.addPlot(); self.p_m.showGrid(x=True, y=True, alpha=0.3)
        
        self.curve_mid = self.p_m.plot(pen=pg.mkPen(ACCENT_CYAN, width=2), name="Mid Price")
        self.curve_ask = self.p_m.plot(pen=pg.mkPen(ACCENT_RED, width=1), name="Ask Price")
        self.curve_bid = self.p_m.plot(pen=pg.mkPen(ACCENT_GREEN, width=1), name="Bid Price")
        
        self.sc_bot = pg.ScatterPlotItem(symbol='x', size=7, brush=ACCENT_WHITE, name="Bot Trades")
        self.sc_buy = pg.ScatterPlotItem(symbol='t1', size=10, brush=ACCENT_GREEN, name="My Buy")
        self.sc_sell = pg.ScatterPlotItem(symbol='t', size=10, brush=ACCENT_RED, name="My Sell")
        for item in [self.sc_bot, self.sc_buy, self.sc_sell]: self.p_m.addItem(item)
        
        # Legend with Toggles
        self.leg_m = InteractiveLegendItem(offset=(10, 10))
        self.leg_m.setParentItem(self.p_m.graphicsItem())
        
        self.leg_m.addItem(self.curve_mid, "Mid Price")
        self.leg_m.addItem(self.curve_ask, "Ask Price")
        self.leg_m.addItem(self.curve_bid, "Bid Price")
        self.leg_m.addItem(self.sc_bot, "Bot Trades")
        self.leg_m.addItem(self.sc_buy, "My Buy")
        self.leg_m.addItem(self.sc_sell, "My Sell")

        # Initialize the new indicator curves
        self.curve_vwap = self.p_m.plot(pen=pg.mkPen('#FF00FF', width=2, style=Qt.PenStyle.DashLine), name="VWAP")
        self.curve_microprice = self.p_m.plot(pen=pg.mkPen('#FF8C00', width=2, style=Qt.PenStyle.DotLine), name="Microprice")
        self.curve_outermid = self.p_m.plot(pen=pg.mkPen('#FFD700', width=1.5, style=Qt.PenStyle.DashDotLine), name="Outer Mid")
        self.curve_vwworst = self.p_m.plot(pen=pg.mkPen('#00FFFF', width=1.5, style=Qt.PenStyle.DashDotLine), name="VW Worst")

        # Hide them by default and add them to the interactive legend
        for curve, name in [
            (self.curve_vwap, "VWAP"), 
            (self.curve_microprice, "Microprice"), 
            (self.curve_outermid, "Outer Mid"), 
            (self.curve_vwworst, "VW Worst")
        ]:
            curve.setVisible(False)
            self.leg_m.addItem(curve, name)
            self.leg_m.items[-1][1].setAttr('color', DIM) # Dim the text to show it is disabled
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        # Ensure infinite lines don't swallow mouse clicks
        self.v_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.h_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.v_line.setAcceptHoverEvents(False)
        self.h_line.setAcceptHoverEvents(False)
        self.p_m.addItem(self.v_line, ignoreBounds=True); self.p_m.addItem(self.h_line, ignoreBounds=True)
        self.p_m.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.p_m.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._dropped_markers = []

        # Other Tabs
        self.gw_p = pg.GraphicsLayoutWidget(); self.gw_p.setBackground(BG)
        self.tabs.addTab(self.gw_p, "PnL")
        self.p_pnl = self.gw_p.addPlot(); self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.p_pnl.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.curve_pnl = self.p_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2))

        self.gw_pos = pg.GraphicsLayoutWidget(); self.gw_pos.setBackground(BG)
        self.tabs.addTab(self.gw_pos, "Position")
        self.p_pos = self.gw_pos.addPlot(title="Position vs Timestamp")
        self.p_pos.showGrid(x=True, y=True, alpha=0.3)
        self.p_pos.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.p_pos.addLegend()
        self.p_pos.setLabel('left', 'Position'); self.p_pos.setLabel('bottom', 'Timestamp')
        self.pos_curves = {}

        self.gw_c = pg.GraphicsLayoutWidget(); self.gw_c.setBackground(BG)
        self.gw_c.scene().sigMouseClicked.connect(self._on_mouse_clicked)
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

    def _on_mouse_clicked(self, ev):
        if self.current_df is None: return
        pos = ev.scenePos()
        target_plot = None
        
        # Accurately determine the target plot by matching the emitting pyqtSignal scene
        scene = self.sender()
        if scene == self.p_m.scene(): target_plot = self.p_m
        elif scene == self.p_pnl.scene(): target_plot = self.p_pnl
        elif scene == self.p_pos.scene(): target_plot = self.p_pos
        elif hasattr(self, 'gw_c') and scene == self.gw_c.scene():
            for item in self.gw_c.scene().items():
                if isinstance(item, pg.PlotItem) and item.sceneBoundingRect().contains(pos):
                    target_plot = item
                    break
                    
        # Fallback in case sender mapping fails
        if not target_plot:
            for p in [self.p_m, self.p_pnl, self.p_pos]:
                if p.scene() is not None and p.sceneBoundingRect().contains(pos):
                    target_plot = p
                    break
                
        if not target_plot: return

        if ev.button() == Qt.MouseButton.RightButton:
            if hasattr(target_plot, '_dropped_markers'):
                for item in target_plot._dropped_markers:
                    target_plot.removeItem(item)
                target_plot._dropped_markers = []
            return

        if ev.button() == Qt.MouseButton.LeftButton:
            mouse_point = target_plot.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            if not hasattr(target_plot, '_dropped_markers'):
                target_plot._dropped_markers = []

            pt = pg.ScatterPlotItem(x=[x], y=[y], symbol='o', size=8, brush=ACCENT_GOLD)
            v_dash = pg.InfiniteLine(pos=x, angle=90, movable=False, pen=pg.mkPen(ACCENT_GOLD, style=Qt.PenStyle.DotLine))
            h_dash = pg.InfiniteLine(pos=y, angle=0, movable=False, pen=pg.mkPen(ACCENT_GOLD, style=Qt.PenStyle.DotLine))
            text = pg.TextItem(f"({int(x)}, {y:,.2f})", color=ACCENT_GOLD, anchor=(0, 1))
            text.setPos(x, y)

            for item in (pt, v_dash, h_dash, text):
                target_plot.addItem(item, ignoreBounds=True)
                target_plot._dropped_markers.append(item)

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

    def _load_file(self, path):
        with open(path, encoding='utf-8') as f:
            content = f.read()

        try:
            raw = json.loads(content)
            csv_str = raw.get('activitiesLog', '').replace('\\n', '\n')
            logs_list = raw.get('logs', [])
            trade_history = raw.get('tradeHistory', [])
        except json.JSONDecodeError:
            activities_idx = content.find("Activities log:")
            trade_idx = content.find("Trade History:")
            
            if activities_idx != -1:
                act_start = content.find('\n', activities_idx) + 1
                act_end = trade_idx if trade_idx != -1 else len(content)
                csv_str = content[act_start:act_end].strip()
            else:
                csv_str = ""
                
            if trade_idx != -1:
                th_start = content.find('\n', trade_idx) + 1
                th_str = content[th_start:].strip()
                th_str = re.sub(r',\s*\}', '}', th_str)
                th_str = re.sub(r',\s*\]', ']', th_str)
                try:
                    trade_history = json.loads(th_str)
                except:
                    trade_history = []
            else:
                trade_history = []
                
            sandbox_idx = content.find("Sandbox logs:")
            if sandbox_idx != -1:
                end_idx = activities_idx if activities_idx != -1 else len(content)
                sb_start = content.find('\n', sandbox_idx) + 1
                sb_str = content[sb_start:end_idx].strip()
                sb_str = re.sub(r'\}\s*\n\s*\{', '},\n{', sb_str)
                try:
                    logs_list = json.loads(f"[{sb_str}]")
                except:
                    logs_list = []
            else:
                logs_list = []

        df = pl.read_csv(StringIO(csv_str), separator=';', null_values=['', 'nan'])
        df = df.rename({c: c.strip() for c in df.columns})
        df = df.with_columns((pl.col("timestamp") % 1000000).alias("timestamp"))

        # Recompute profit_and_loss cumulatively across days
        if 'profit_and_loss' in df.columns:
            unique_days = df['day'].unique().sort().to_list()
            products = df['product'].unique().to_list()
            pnl_series = np.array(df['profit_and_loss'].to_numpy())
            for prod_item in products:
                cum_offset = 0
                for d in unique_days:
                    mask = (df['product'] == prod_item).to_numpy() & (df['day'] == d).to_numpy()
                    idx = np.where(mask)[0]
                    if len(idx) == 0: continue
                    pnl_series[idx] += cum_offset
                    cum_offset = pnl_series[idx[-1]]
            df = df.with_columns(pl.Series(name="profit_and_loss", values=pnl_series))
        
        custom = {}
        for entry in logs_list:
            ts, log = entry.get('timestamp', 0), entry.get('lambdaLog', '') or ''
            for line in log.split('\n'):
                if line.startswith('LOGVIZ:'):
                    try:
                        payload = json.loads(line[7:])
                        for k, v in payload.items(): custom.setdefault(k, []).append((ts, float(v)))
                    except: pass

        # Parse debug messages: LOGDBG:timestamp:tag:product:message
        debug_msgs = []
        for entry in logs_list:
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

        self.data = {'prices_df': df, 'trades': trade_history, 'custom': custom, 'debug': debug_msgs}
        
        # Inject day into trade history using monotonically increasing timestamps
        if trade_history and df is not None and not df.is_empty():
            unique_days = df['day'].unique().sort().to_list()
            for tr in trade_history:
                raw_ts = tr.get('timestamp', 0)
                # The jmerle offline backtester iteratively adds 1,000,000 per round appended
                day_offset = int(raw_ts // 1000000)
                tr['day'] = unique_days[min(day_offset, len(unique_days)-1)]
                # Normalize the timestamp to fit directly against activitiesLog coordinate bounds
                tr['timestamp'] = int(raw_ts % 1000000)
        
        self.cb_prod.blockSignals(True)
        self.cb_prod.clear(); self.cb_prod.addItems(df['product'].unique().sort().to_list())
        self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False)
        self._build_custom_plots()
        self._build_logs_table()
        self._process_selection()

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

    def _build_position_plot(self, prod, day):
        # Clear old curves
        for c in list(self.pos_curves.values()):
            self.p_pos.removeItem(c)
        self.pos_curves.clear()

        trades = self.data.get('trades', [])
        if not trades:
            return

        min_day = self.data['prices_df']['day'].min()
        DAY_LENGTH = 1_000_000

        # Group self-trades by symbol
        pos_by_sym = {}  # symbol -> sorted list of (day, timestamp, delta)
        for tr in trades:
            if day != 'All' and str(tr.get('day', '')) != str(day):
                continue
            if str(prod) != 'All' and str(tr.get('symbol', '')) != str(prod):
                continue
            is_buyer = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_seller = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if not is_buyer and not is_seller:
                continue
            sym = tr.get('symbol', '')
            qty = tr.get('quantity', 0)
            ts = tr.get('timestamp', 0)
            tr_day = tr.get('day', 0)
            delta = qty if is_buyer else -qty
            pos_by_sym.setdefault(sym, []).append((tr_day, ts, delta))

        products = sorted(pos_by_sym.keys())
        colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_PURPLE, ACCENT_WHITE] + CUSTOM_COLORS

        for i, sym in enumerate(products):
            # Sort chronologically by day, then timestamp
            events = sorted(pos_by_sym[sym], key=lambda x: (x[0], x[1]))
            step_x, step_y = [], []
            cum = 0
            current_day = None
            
            for tr_day, ts, delta in events:
                if current_day is not None and tr_day != current_day:
                    # Drop back to 0 abruptly at the round boundary end (999900)
                    last_plot_ts = 999900 + (current_day - min_day) * DAY_LENGTH if day == 'All' else 999900
                    step_x.extend([last_plot_ts, last_plot_ts])
                    step_y.extend([cum, 0])
                    cum = 0
                
                if current_day is None or tr_day != current_day:
                    # Anchor firmly at start of day (0)
                    init_plot_ts = 0 + (tr_day - min_day) * DAY_LENGTH if day == 'All' else 0
                    step_x.append(init_plot_ts)
                    step_y.append(0)
                
                current_day = tr_day
                plot_ts = ts + (tr_day - min_day) * DAY_LENGTH if day == 'All' else ts
                
                step_x.extend([plot_ts, plot_ts])
                step_y.extend([cum, cum + delta])
                cum += delta
                
            if events:
                final_plot_ts = 999900 + (current_day - min_day) * DAY_LENGTH if day == 'All' else 999900
                step_x.extend([final_plot_ts, final_plot_ts])
                step_y.extend([cum, cum])
                
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            curve = self.p_pos.plot(step_x, step_y, pen=pen, name=sym)
            self.pos_curves[sym] = curve

        self.p_pos.autoRange()

    def _process_selection(self):
        if not self.data or not self.cb_prod.currentText(): return
        prod, day = self.cb_prod.currentText(), self.cb_day.currentText()
        self.current_df = self.data['prices_df'].filter(pl.col('product') == prod)
        
        min_day = self.data['prices_df']['day'].min()
        DAY_LENGTH = 1_000_000
        
        if day != 'All': 
            self.current_df = self.current_df.filter(pl.col('day') == int(day))
            self.current_df = self.current_df.sort('timestamp')
            t = self.current_df['timestamp'].to_numpy()
        else:
            self.current_df = self.current_df.sort(['day', 'timestamp'])
            t = (self.current_df['timestamp'] + (self.current_df['day'] - min_day) * DAY_LENGTH).to_numpy()
            
        def safe_get(col_name):
            if col_name in self.current_df.columns:
                return self.current_df[col_name].to_numpy()
            return np.zeros(len(t))

        mid = safe_get('mid_price')
        ask = safe_get('ask_price_1')
        bid = safe_get('bid_price_1')
        ask_vol = safe_get('ask_volume_1')
        bid_vol = safe_get('bid_volume_1')

        self.curve_mid.setData(t, mid)
        self.curve_ask.setData(t, ask)
        self.curve_bid.setData(t, bid)

        # Calculate Advanced Indicators
        total_vol = bid_vol + ask_vol
        vwap = np.divide((bid * bid_vol) + (ask * ask_vol), total_vol, out=np.copy(mid), where=total_vol!=0)
        microprice = np.divide((bid * ask_vol) + (ask * bid_vol), total_vol, out=np.copy(mid), where=total_vol!=0)

        bp3, bp2 = safe_get('bid_price_3'), safe_get('bid_price_2')
        ap3, ap2 = safe_get('ask_price_3'), safe_get('ask_price_2')
        bv3, bv2 = safe_get('bid_volume_3'), safe_get('bid_volume_2')
        av3, av2 = safe_get('ask_volume_3'), safe_get('ask_volume_2')

        def safe_coalesce(a3, a2, a1):
            res = np.copy(a1).astype(float)
            if a2 is not None:
                mask2 = ~np.isnan(a2.astype(float))
                res = np.where(mask2, a2, res)
            if a3 is not None:
                mask3 = ~np.isnan(a3.astype(float))
                res = np.where(mask3, a3, res)
            return res

        worst_bid = safe_coalesce(bp3, bp2, bid)
        worst_ask = safe_coalesce(ap3, ap2, ask)
        worst_bid_vol = safe_coalesce(bv3, bv2, bid_vol)
        worst_ask_vol = safe_coalesce(av3, av2, ask_vol)

        outer_mid = (worst_bid + worst_ask) / 2.0
        worst_total_vol = worst_bid_vol + worst_ask_vol
        vw_worst = np.divide((worst_bid * worst_bid_vol) + (worst_ask * worst_ask_vol), worst_total_vol, out=np.copy(outer_mid), where=worst_total_vol!=0)

        # Plot new curves
        self.curve_vwap.setData(t, vwap)
        self.curve_microprice.setData(t, microprice)
        self.curve_outermid.setData(t, outer_mid)
        self.curve_vwworst.setData(t, vw_worst)
        
        self.curve_pnl.setData(t, self.current_df['profit_and_loss'].to_numpy() if 'profit_and_loss' in self.current_df.columns else np.zeros(len(t)))

        mb_t, mb_p, ms_t, ms_p, b_t, b_p = [], [], [], [], [], []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and str(tr.get('day', '')) != str(day): continue
            
            ts, pr = tr['timestamp'], tr['price']
            tr_day = tr.get('day', 0)
            
            plot_ts = ts + (tr_day - min_day) * DAY_LENGTH if day == 'All' else ts
            
            if tr.get('buyer') == 'SUBMISSION': mb_t.append(plot_ts); mb_p.append(pr)
            elif tr.get('seller') == 'SUBMISSION': ms_t.append(plot_ts); ms_p.append(pr)
            else: b_t.append(plot_ts); b_p.append(pr)
        self.sc_buy.setData(x=mb_t, y=mb_p); self.sc_sell.setData(x=ms_t, y=ms_p); self.sc_bot.setData(x=b_t, y=b_p)

        self._build_position_plot(prod, day)

        # Update Custom Overlays on Main Plot
        custom_data = self.data.get('custom', {})
        t_min, t_max = t[0], t[-1]
        
        for i, (name, pts) in enumerate(custom_data.items()):
            if name not in self.custom_curves:
                color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
                curve = self.p_m.plot(pen=pg.mkPen(color, width=1.5), name=f"[C] {name}")
                curve.setVisible(False) # Default disabled
                self.custom_curves[name] = curve
                self.leg_m.addItem(curve, f"[C] {name}")
                # Sync legend label color with hidden state
                label = self.leg_m.items[-1][1]
                label.setAttr('color', DIM)
            
            curve = self.custom_curves[name]
            # Filter custom points to fit the current time range
            pts_filtered = [p for p in pts if t_min <= p[0] <= t_max]
            if pts_filtered:
                curve.setData([p[0] for p in pts_filtered], [p[1] for p in pts_filtered])
            else:
                curve.setData([], [])

        self.p_m.autoRange(); self.p_pnl.autoRange()


class DataProcessor(QThread):
    data_ready = pyqtSignal(object, object)

    def __init__(self, prices_df, trades_df, product, day_val):
        super().__init__()
        self.prices_df = prices_df
        self.trades_df = trades_df
        self.product = product
        self.day_val = day_val

    def run(self):
        if self.prices_df is None or len(self.prices_df) == 0:
            self.data_ready.emit({'time': []}, {'time': [], 'quantity': []})
            return

        p_filtered = self.prices_df.filter(pl.col("product") == self.product)
        
        if self.trades_df is not None and "day" in self.trades_df.columns and len(self.trades_df) > 0:
            t_filtered = self.trades_df.filter(pl.col("symbol") == self.product)
        else:
            t_filtered = pl.DataFrame()

        if self.day_val == "All":
            p_filtered = p_filtered.sort(["day", "timestamp"])
            if len(t_filtered) > 0:
                t_filtered = t_filtered.sort(["day", "timestamp"])

            DAY_LENGTH = 1_000_000
            min_day = p_filtered["day"].min()

            p_filtered = p_filtered.with_columns(
                (pl.col("timestamp") + (pl.col("day") - min_day) * DAY_LENGTH).alias("plot_time")
            )
            
            if len(t_filtered) > 0:
                t_filtered = t_filtered.with_columns(
                    (pl.col("timestamp") + (pl.col("day") - min_day) * DAY_LENGTH).alias("plot_time")
                )
        else:
            p_filtered = p_filtered.filter(pl.col("day") == self.day_val).sort("timestamp")
            p_filtered = p_filtered.with_columns(pl.col("timestamp").alias("plot_time"))
            
            if len(t_filtered) > 0:
                t_filtered = t_filtered.filter(pl.col("day") == self.day_val).sort("timestamp")
                t_filtered = t_filtered.with_columns(pl.col("timestamp").alias("plot_time"))

        bid_price = p_filtered['bid_price_1'].to_numpy()
        ask_price = p_filtered['ask_price_1'].to_numpy()
        bid_vol = p_filtered['bid_volume_1'].to_numpy()
        ask_vol = p_filtered['ask_volume_1'].to_numpy()
        mid_price = p_filtered['mid_price'].to_numpy()

        total_vol = bid_vol + ask_vol
        
        # Standard Order Book Imbalance (OBI)
        obi = np.divide(bid_vol - ask_vol, total_vol, out=np.zeros_like(bid_vol, dtype=float), where=total_vol!=0)

        # OBI Exponential Moving Average (Smoothed Trend)
        try:
            # Use Polars native exponentially weighted moving average
            obi_ema = pl.Series(obi).ewm_mean(span=50).to_numpy()
        except Exception:
            # Fallback for version differences
            obi_ema = obi

        # Level 1 Book VWAP
        vwap = np.divide((bid_price * bid_vol) + (ask_price * ask_vol), total_vol, out=np.copy(mid_price), where=total_vol!=0)
        
        # Microprice - Interchanged weights
        microprice = np.divide((bid_price * ask_vol) + (ask_price * bid_vol), total_vol, out=np.copy(mid_price), where=total_vol!=0)

        def safe_get(col_name):
            return p_filtered[col_name].to_numpy() if col_name in p_filtered.columns else None

        bp3, bp2 = safe_get('bid_price_3'), safe_get('bid_price_2')
        ap3, ap2 = safe_get('ask_price_3'), safe_get('ask_price_2')
        bv3, bv2 = safe_get('bid_volume_3'), safe_get('bid_volume_2')
        av3, av2 = safe_get('ask_volume_3'), safe_get('ask_volume_2')

        def safe_coalesce(a3, a2, a1):
            res = np.copy(a1).astype(float)
            if a2 is not None:
                mask2 = ~np.isnan(a2.astype(float))
                res = np.where(mask2, a2, res)
            if a3 is not None:
                mask3 = ~np.isnan(a3.astype(float))
                res = np.where(mask3, a3, res)
            return res

        worst_bid = safe_coalesce(bp3, bp2, bid_price)
        worst_ask = safe_coalesce(ap3, ap2, ask_price)
        worst_bid_vol = safe_coalesce(bv3, bv2, bid_vol)
        worst_ask_vol = safe_coalesce(av3, av2, ask_vol)

        outer_mid = (worst_bid + worst_ask) / 2.0
        worst_total_vol = worst_bid_vol + worst_ask_vol
        vw_worst = np.divide((worst_bid * worst_bid_vol) + (worst_ask * worst_ask_vol), worst_total_vol, out=np.copy(outer_mid), where=worst_total_vol!=0)

        p_data = {
            'time': p_filtered['plot_time'].to_numpy(),
            'mid': mid_price,
            'bid': bid_price,
            'ask': ask_price,
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'obi': obi,
            'obi_ema': obi_ema,
            'vwap': vwap,
            'microprice': microprice,
            'outer_mid': outer_mid,
            'vw_worst': vw_worst
        }
        
        t_data = {
            'time': t_filtered['plot_time'].to_numpy() if len(t_filtered) > 0 else np.array([]),
            'price': t_filtered['price'].to_numpy() if len(t_filtered) > 0 else np.array([]),
            'quantity': t_filtered['quantity'].to_numpy() if len(t_filtered) > 0 and 'quantity' in t_filtered.columns else np.array([])
        }

        self.data_ready.emit(p_data, t_data)


class ProsperityVisualizer(QMainWindow):
    def __init__(self, data_dir="."):
        super().__init__()
        self.data_dir = data_dir
        self.rounds_map = self.scan_directory()
        
        self.prices_df = None
        self.trades_df = None
        self.has_plotted_data = False
        
        self.spread_fill_ask = None 
        self.spread_fill_bid = None 
        self.curve_ask = None 
        self.curve_bid = None 
        self.obi_curve = None
        
        # New indicator curves
        self.curve_vwap = None
        self.curve_microprice = None
        
        self.init_ui()

    def scan_directory(self):
        rounds = {}
        price_pattern = re.compile(r"prices_round_(\d+)_day_(-?\d+)\.csv")
        trade_pattern = re.compile(r"trades_round_(\d+)_day_(-?\d+)\.csv")

        dataviz_dir = os.path.join(self.data_dir, "dataviz")
        if not os.path.exists(dataviz_dir):
            return rounds

        for f in os.listdir(dataviz_dir):
            if not f.endswith('.csv'): continue
            
            p_match = price_pattern.match(f)
            if p_match:
                rnd = p_match.group(1)
                rounds.setdefault(rnd, {'prices': [], 'trades': []})['prices'].append(os.path.join("dataviz", f))
                continue
                
            t_match = trade_pattern.match(f)
            if t_match:
                rnd = t_match.group(1)
                rounds.setdefault(rnd, {'prices': [], 'trades': []})['trades'].append(os.path.join("dataviz", f))
                
        return rounds

    def init_ui(self):
        self.setWindowTitle("Prosperity 4 - Smart Alpha Scanner")
        self.setGeometry(100, 100, 1400, 900)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        controls_layout = QHBoxLayout()
        
        self.round_combo = QComboBox()
        available_rounds = sorted(list(self.rounds_map.keys()), key=int) if self.rounds_map else ["None"]
        self.round_combo.addItems(available_rounds)
        self.round_combo.currentTextChanged.connect(self.load_round_data)

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.start_processing)
        
        self.day_combo = QComboBox()
        self.day_combo.currentTextChanged.connect(self.start_processing)
        
        self.btn_open_log = QPushButton("📂 Open Algo Log")
        self.btn_open_log.setStyleSheet("background-color: #3A3A3A; font-weight: bold; padding: 4px; border-radius: 4px; color: #E0E0E0; border: 1px solid #444;")
        self.btn_open_log.clicked.connect(self.open_log_window)
        

        self.spread_checkbox = QCheckBox("L1 Spread")
        self.spread_checkbox.setChecked(True)
        self.spread_checkbox.setStyleSheet("color: white; font-weight: bold;")
        self.spread_checkbox.stateChanged.connect(self.toggle_spread)

        # Top-Bar Toggles for Indicators
        self.vwap_checkbox = QCheckBox("VWAP")
        self.vwap_checkbox.setChecked(False)
        self.vwap_checkbox.setStyleSheet("color: #FF00FF; font-weight: bold;") # Magenta
        self.vwap_checkbox.stateChanged.connect(self.toggle_indicators)

        self.microprice_checkbox = QCheckBox("Microprice")
        self.microprice_checkbox.setChecked(False)
        self.microprice_checkbox.setStyleSheet("color: #FF8C00; font-weight: bold;") # Dark Orange
        self.microprice_checkbox.stateChanged.connect(self.toggle_indicators)

        self.outermid_checkbox = QCheckBox("Outer Mid")
        self.outermid_checkbox.setChecked(False)
        self.outermid_checkbox.setStyleSheet("color: #FFD700; font-weight: bold;") # Gold
        self.outermid_checkbox.stateChanged.connect(self.toggle_indicators)
        
        self.vwworst_checkbox = QCheckBox("VW Worst Offers")
        self.vwworst_checkbox.setChecked(False)
        self.vwworst_checkbox.setStyleSheet("color: #00FFFF; font-weight: bold;") # Cyan
        self.vwworst_checkbox.stateChanged.connect(self.toggle_indicators)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        controls_layout.addWidget(QLabel("Round:"))
        controls_layout.addWidget(self.round_combo)
        controls_layout.addWidget(QLabel("Product:"))
        controls_layout.addWidget(self.product_combo)
        controls_layout.addWidget(QLabel("Day:"))
        controls_layout.addWidget(self.day_combo)
        controls_layout.addWidget(self.spread_checkbox)
        controls_layout.addWidget(self.btn_open_log) 
        
        controls_layout.addWidget(self.vwap_checkbox)
        controls_layout.addWidget(self.microprice_checkbox)
        controls_layout.addWidget(self.outermid_checkbox)
        controls_layout.addWidget(self.vwworst_checkbox)
        
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)
        layout.addLayout(controls_layout)

        pg.setConfigOptions(antialias=False)
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.p1 = self.graph_widget.addPlot(row=0, col=0, title="Market Microstructure")
        self.p2 = self.graph_widget.addPlot(row=1, col=0, title="Order Book Liquidity & OBI")
        
        self.p2.setXLink(self.p1)
        
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        
        # Restored the original graph legends
        self.p1.addLegend()
        self.p2.addLegend()

        # ==========================================
        # CROSSHAIR & DYNAMIC LABEL INITIALIZATION
        # ==========================================
        
        crosshair_pen = pg.mkPen(color=(180, 180, 180, 150), width=1.5, style=Qt.PenStyle.DashLine)
        
        self.vLine1 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.vLine2 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.hLine1 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.hLine2 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)

        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        
        self.label_p1 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        self.label_p2 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        
        self.hide_crosshairs()
        self.graph_widget.scene().sigMouseMoved.connect(self.mouse_moved)

        if self.rounds_map:
            self.load_round_data(self.round_combo.currentText())
        else:
            self.status_label.setText("Status: No CSV files found in directory.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def hide_crosshairs(self):
        self.vLine1.hide()
        self.vLine2.hide()
        self.hLine1.hide()
        self.hLine2.hide()
        self.label_p1.hide()
        self.label_p2.hide()

    def mouse_moved(self, evt):
        pos = evt
        if not self.has_plotted_data: return

        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = self.p1.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            
            self.vLine1.setPos(x)
            self.vLine2.setPos(x)
            self.hLine1.setPos(y)
            
            self.vLine1.show()
            self.vLine2.show()
            self.hLine1.show()
            self.hLine2.hide()
            
            html_str = f"""
            <div style='text-align: left;'>
                <span style='color: #FFFFFF; font-size: 11pt;'>Time: </span><b style='color: #00BFFF; font-size: 11pt;'>{int(x)}</b><br>
                <span style='color: #FFFFFF; font-size: 11pt;'>Price: </span><b style='color: #00FF00; font-size: 11pt;'>{y:.2f}</b>
            </div>
            """
            self.label_p1.setHtml(html_str)
            self.label_p1.setPos(x, y)
            self.label_p1.show()
            self.label_p2.hide()
            
        elif self.p2.sceneBoundingRect().contains(pos):
            mousePoint = self.p2.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            
            self.vLine1.setPos(x)
            self.vLine2.setPos(x)
            self.hLine2.setPos(y)
            
            self.vLine1.show()
            self.vLine2.show()
            self.hLine2.show()
            self.hLine1.hide()
            
            html_str = f"""
            <div style='text-align: left;'>
                <span style='color: #FFFFFF; font-size: 11pt;'>Time: </span><b style='color: #00BFFF; font-size: 11pt;'>{int(x)}</b><br>
                <span style='color: #FFFFFF; font-size: 11pt;'>Level: </span><b style='color: #FFD700; font-size: 11pt;'>{y:.2f}</b>
            </div>
            """
            self.label_p2.setHtml(html_str)
            self.label_p2.setPos(x, y)
            self.label_p2.show()
            self.label_p1.hide()
        else:
            self.hide_crosshairs()

    def open_log_window(self):
        log_path, _ = QFileDialog.getOpenFileName(self, "Open Log File", ".", "Log Files (*.log);;CSV Files (*.csv);;All Files (*)")
        if log_path:
            if not hasattr(self, 'log_windows'):
                self.log_windows = []
            
            # Clean up closed windows from the list to accurately track active ones
            self.log_windows = [w for w in self.log_windows if w.isVisible()]
            
            try:
                log_win = LogVisualizer(log_path)
                
                # ENHANCEMENT 1: Free up memory when you close a specific log window
                log_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                
                # ENHANCEMENT 2: Inject the file name into the window title for side-by-side clarity
                file_name = os.path.basename(log_path)
                log_win.setWindowTitle(f"Prosperity Log Viewer - {file_name}")
                
                # ENHANCEMENT 3: Cascade the window geometry so they don't perfectly overlap
                offset = len(self.log_windows) * 30
                log_win.setGeometry(50 + offset, 50 + offset, 1600, 920)
                
                log_win.show()
                self.log_windows.append(log_win)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load log:\n{e}")
    def load_round_data(self, round_str):
        if round_str == "None" or round_str not in self.rounds_map: return
        
        self.status_label.setText(f"Status: Loading Round {round_str} files into memory...")
        self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        QApplication.processEvents()

        files = self.rounds_map[round_str]
        
        p_dfs = []
        for f in files['prices']:
            filepath = os.path.join(self.data_dir, f)
            df = pl.read_csv(filepath, separator=";", null_values=["", "NaN"])
            
            # Force target columns to float to prevent Int/Float mis-assignment or mixed type errors
            cast_exprs = []
            for col in df.columns:
                if col not in ("timestamp", "day", "product", "symbol"):
                    cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
            if cast_exprs:
                df = df.with_columns(cast_exprs)
                
            df = df.with_columns((pl.col("timestamp") % 1000000).alias("timestamp"))
            p_dfs.append(df)
        self.prices_df = pl.concat(p_dfs, how="diagonal_relaxed") if p_dfs else None

        t_dfs = []
        for f in files['trades']:
            filepath = os.path.join(self.data_dir, f)
            # FIX: Use re.search instead of re.match to ignore the "dataviz/" prefix
            day_match = re.search(r"trades_round_\d+_day_(-?\d+)\.csv", f)
            if day_match:
                day_val = int(day_match.group(1))
                df = pl.read_csv(filepath, separator=";", null_values=["", "NaN"])
                
                # Force target columns to float
                cast_exprs = []
                for col in df.columns:
                    if col not in ("timestamp", "day", "product", "symbol", "buyer", "seller", "currency"):
                        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
                if cast_exprs:
                    df = df.with_columns(cast_exprs)
                    
                df = df.with_columns([
                    pl.lit(day_val).alias("day"),
                    (pl.col("timestamp") % 1000000).alias("timestamp")
                ])
                t_dfs.append(df)
        self.trades_df = pl.concat(t_dfs, how="diagonal_relaxed") if t_dfs else None

        if self.prices_df is not None and 'profit_and_loss' in self.prices_df.columns:
            unique_days = self.prices_df['day'].unique().sort().to_list()
            products = self.prices_df['product'].unique().to_list()
            pnl_series = np.array(self.prices_df['profit_and_loss'].to_numpy())
            for prod_item in products:
                cum_offset = 0
                for d in unique_days:
                    mask = (self.prices_df['product'] == prod_item).to_numpy() & (self.prices_df['day'] == d).to_numpy()
                    idx = np.where(mask)[0]
                    if len(idx) == 0: continue
                    pnl_series[idx] += cum_offset
                    cum_offset = pnl_series[idx[-1]]
            self.prices_df = self.prices_df.with_columns(pl.Series(name="profit_and_loss", values=pnl_series))

        self.product_combo.blockSignals(True)
        self.day_combo.blockSignals(True)
        
        self.product_combo.clear()
        self.day_combo.clear()

        if self.prices_df is not None:
            products = self.prices_df['product'].unique().sort().to_list()
            self.product_combo.addItems(products)
            
            days = ["All"] + [str(d) for d in self.prices_df['day'].unique().sort().to_list()]
            self.day_combo.addItems(days)

        self.product_combo.blockSignals(False)
        self.day_combo.blockSignals(False)

        self.start_processing()

    def toggle_spread(self):
        is_checked = self.spread_checkbox.isChecked()
        if self.spread_fill_ask: self.spread_fill_ask.setVisible(is_checked)
        if self.spread_fill_bid: self.spread_fill_bid.setVisible(is_checked)
        if self.curve_ask: self.curve_ask.setVisible(is_checked)
        if self.curve_bid: self.curve_bid.setVisible(is_checked)

    def toggle_indicators(self):
        if self.curve_vwap: self.curve_vwap.setVisible(self.vwap_checkbox.isChecked())
        if self.curve_microprice: self.curve_microprice.setVisible(self.microprice_checkbox.isChecked())
        if hasattr(self, 'curve_outermid') and self.curve_outermid: self.curve_outermid.setVisible(self.outermid_checkbox.isChecked())
        if hasattr(self, 'curve_vwworst') and self.curve_vwworst: self.curve_vwworst.setVisible(self.vwworst_checkbox.isChecked())

    def start_processing(self):
        if self.prices_df is None or self.product_combo.count() == 0: return

        self.status_label.setText("Status: Crunching data via Polars...")
        self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        self.round_combo.setEnabled(False)
        self.product_combo.setEnabled(False)
        self.day_combo.setEnabled(False)
        
        self.has_plotted_data = False
        self.hide_crosshairs()

        product = self.product_combo.currentText()
        day_text = self.day_combo.currentText()
        day_val = "All" if day_text == "All" else int(day_text)

        self.worker = DataProcessor(self.prices_df, self.trades_df, product, day_val)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.start()

    def on_data_ready(self, p_data, t_data):
        self.p1.clear()
        self.p2.clear()
        
        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        
        self.hide_crosshairs()

        if len(p_data['time']) == 0:
            self.reset_ui("Status: No Data")
            return

        # Name included here, so Mid Price will appear in the graph legend
        self.p1.plot(p_data['time'], p_data['mid'], pen=pg.mkPen('#00BFFF', width=2), 
                     name="Mid Price", autoDownsample=True)

        self.curve_ask = pg.PlotCurveItem(p_data['time'], p_data['ask'], pen=pg.mkPen('#DC143C', width=1)) 
        self.curve_bid = pg.PlotCurveItem(p_data['time'], p_data['bid'], pen=pg.mkPen('#00FF00', width=1)) 
        
        # CRITICAL FIX: No 'name' argument passed here. These will NOT appear in the graph legend.
        self.curve_vwap = pg.PlotCurveItem(p_data['time'], p_data['vwap'], pen=pg.mkPen('#FF00FF', width=2, style=Qt.PenStyle.DashLine))
        self.curve_microprice = pg.PlotCurveItem(p_data['time'], p_data['microprice'], pen=pg.mkPen('#FF8C00', width=2, style=Qt.PenStyle.DotLine))
        self.curve_outermid = pg.PlotCurveItem(p_data['time'], p_data['outer_mid'], pen=pg.mkPen('#FFD700', width=1.5, style=Qt.PenStyle.DashDotLine))
        self.curve_vwworst = pg.PlotCurveItem(p_data['time'], p_data['vw_worst'], pen=pg.mkPen('#00FFFF', width=1.5, style=Qt.PenStyle.DashDotLine))
        
        curve_mid_anchor = pg.PlotCurveItem(p_data['time'], p_data['mid'])

        self.spread_fill_ask = pg.FillBetweenItem(curve_mid_anchor, self.curve_ask, brush=(220, 20, 60, 50)) 
        self.spread_fill_bid = pg.FillBetweenItem(self.curve_bid, curve_mid_anchor, brush=(0, 255, 0, 50))

        self.p1.addItem(self.curve_ask)
        self.p1.addItem(self.curve_bid)
        self.p1.addItem(self.spread_fill_ask)
        self.p1.addItem(self.spread_fill_bid)
        self.p1.addItem(self.curve_vwap)
        self.p1.addItem(self.curve_microprice)
        self.p1.addItem(self.curve_outermid)
        self.p1.addItem(self.curve_vwworst)
        
        self.toggle_spread()
        self.toggle_indicators()

        if len(t_data['time']) > 0 and len(t_data['quantity']) > 0:
            q = np.abs(t_data['quantity']) 
            
            mask_blue = (q >= 1) & (q <= 3)
            mask_yellow = (q == 4)
            mask_red = (q >= 5)

            if np.any(mask_blue):
                scatter = pg.ScatterPlotItem(
                    x=t_data['time'][mask_blue], y=t_data['price'][mask_blue], 
                    pen=pg.mkPen('#00BFFF', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 1-3", pxMode=True
                )
                self.p1.addItem(scatter)

            if np.any(mask_yellow):
                scatter = pg.ScatterPlotItem(
                    x=t_data['time'][mask_yellow], y=t_data['price'][mask_yellow], 
                    pen=pg.mkPen('#FFFF00', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 4", pxMode=True
                )
                self.p1.addItem(scatter)

            if np.any(mask_red):
                scatter = pg.ScatterPlotItem(
                    x=t_data['time'][mask_red], y=t_data['price'][mask_red], 
                    pen=pg.mkPen('#FF0000', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 5+", pxMode=True
                )
                self.p1.addItem(scatter)

        self.p2.plot(p_data['time'], p_data['bid_vol'], fillLevel=0, 
                     brush=(0, 255, 0, 150), pen='#00FF00', name='Bid Vol', 
                     autoDownsample=True)
                     
        self.p2.plot(p_data['time'], -p_data['ask_vol'], fillLevel=0, 
                     brush=(220, 20, 60, 150), pen='#DC143C', name='Ask Vol', 
                     autoDownsample=True)

        max_vol_scalar = max(np.max(p_data['bid_vol']), np.max(p_data['ask_vol']))
        if max_vol_scalar == 0: max_vol_scalar = 1 
        
        # Scale both raw and EMA. We purposefully 10x the EMA because the smoothing 
        # naturally compresses a zero-centered oscillating signal closely to the 0-line.
        scaled_obi = p_data['obi'] * max_vol_scalar
        scaled_obi_ema = p_data['obi_ema'] * max_vol_scalar * 10

        # Plot raw OBI as an independent curve item (faint)
        self.obi_raw_curve = pg.PlotCurveItem(
            p_data['time'], scaled_obi, 
            pen=pg.mkPen((255, 215, 0, 80), width=1), 
            name="Raw OBI", autoDownsample=True
        )
        self.p2.addItem(self.obi_raw_curve)

        # Plot bold, smoothed OBI EMA as a separate curve item
        self.obi_curve = pg.PlotCurveItem(
            p_data['time'], scaled_obi_ema, 
            pen=pg.mkPen('#FF8C00', width=2.5), 
            name="OBI EMA (Trend)", autoDownsample=True
        )
        self.p2.addItem(self.obi_curve)

        self.p1.autoRange()
        self.p2.autoRange()
        
        self.has_plotted_data = True
        self.reset_ui("Status: Ready")

    def reset_ui(self, status):
        self.status_label.setText(status)
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.round_combo.setEnabled(True)
        self.product_combo.setEnabled(True)
        self.day_combo.setEnabled(True)



def main():
    app = QApplication(sys.argv)
    
    app.setStyleSheet("""
        QMainWindow { background-color: #121212; }
        QWidget { color: #E0E0E0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
        QComboBox { background-color: #1E1E1E; border: 1px solid #444; padding: 4px; border-radius: 4px; }
        QComboBox::drop-down { border: 0px; }
        QComboBox QAbstractItemView { background-color: #1E1E1E; color: #E0E0E0; selection-background-color: #3A3A3A; }
        QLabel { font-weight: bold; }
        QCheckBox { spacing: 5px; }
        QCheckBox::indicator { width: 15px; height: 15px; }
    """)

    DATA_DIR = r"F:\PROSPERITY4\Tutorial Round\TUTORIAL_ROUND_1"
    
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "."

    print(f"Scanning directory: {os.path.abspath(DATA_DIR)}")
    ex = ProsperityVisualizer(data_dir=DATA_DIR)
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
