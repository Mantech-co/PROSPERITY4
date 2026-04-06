import sys, os, json, re
from io import StringIO
import numpy as np
import polars as pl
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit
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

# --- Data Engine ---
def build_ob_heatmap(p_df: pl.DataFrame, product: str, day):
    flt = p_df.filter(pl.col('product') == product)
    if day != 'All': flt = flt.filter(pl.col('day') == int(day))
    flt = flt.sort(['day', 'timestamp'])
    
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
    price_levels = np.unique(np.concatenate(all_p))
    max_vol = max(np.max(np.concatenate(all_v)), 1.0)
    
    w, h = len(times), len(price_levels)
    raw_vol = np.zeros((h, w), dtype=float)
    red, blue = np.zeros(h * w, np.uint8), np.zeros(h * w, np.uint8)

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
            ints = np.clip(va[mask] / max_vol * 255, 0, 255).astype(np.uint8)
            np.maximum.at(red if side == 'ask' else blue, flat_idxs, ints)
            raw_vol.flat[flat_idxs] += va[mask]

    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0], img[..., 2] = red.reshape(h, w), blue.reshape(h, w)
    return {'img': img, 'times': times, 'levels': price_levels, 'raw_vol': raw_vol}

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
        
        # Market View
        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG)
        self.tabs.addTab(self.gw_m, "Market View")
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

        # Other Tabs
        self.gw_p = pg.GraphicsLayoutWidget(); self.gw_p.setBackground(BG)
        self.tabs.addTab(self.gw_p, "PnL")
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
        self.cb_prod.clear(); self.cb_prod.addItems(df['product'].unique().sort().to_list())
        self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False)
        self._build_custom_plots()
        self._build_position_plot()
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
            is_seller = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if not is_buyer and not is_seller:
                continue
            sym = tr.get('symbol', '')
            qty = tr.get('quantity', 0)
            ts = tr.get('timestamp', 0)
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
        self.current_df = self.current_df.sort('timestamp')
        
        t, mid = self.current_df['timestamp'].to_numpy(), self.current_df['mid_price'].to_numpy()
        self.curve_mid.setData(t, mid)
        self.curve_pnl.setData(t, self.current_df['profit_and_loss'].to_numpy() if 'profit_and_loss' in self.current_df.columns else np.zeros(len(t)))

        mb_t, mb_p, ms_t, ms_p, b_t, b_p = [], [], [], [], [], []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            ts, pr = tr['timestamp'], tr['price']
            if tr.get('buyer') == 'SUBMISSION': mb_t.append(ts); mb_p.append(pr)
            elif tr.get('seller') == 'SUBMISSION': ms_t.append(ts); ms_p.append(pr)
            else: b_t.append(ts); b_p.append(pr)
        self.sc_buy.setData(x=mb_t, y=mb_p); self.sc_sell.setData(x=ms_t, y=ms_p); self.sc_bot.setData(x=b_t, y=b_p)

        self.ob_res = build_ob_heatmap(self.data['prices_df'], prod, day)
        if self.ob_res:
            self.img_item.setImage(self.ob_res['img'], autoLevels=False)
            self.img_item.setRect(QRectF(t[0], self.ob_res['levels'][0], t[-1]-t[0], self.ob_res['levels'][-1]-self.ob_res['levels'][0]))
            self.img_item.setVisible(self.img_item.isVisible()) # Preserving current legend toggle state
        else: self.img_item.setVisible(False)

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