import sys, os, json, re
import numpy as np
import polars as pl
import pyqtgraph as pg
from io import StringIO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit,
    QScrollArea, QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt, QEvent

# Add project root and script dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from log_visualizer import (
    LogVisualizer, CheckableComboBox, APP_STYLE, BG, PANEL_BG, BORDER, TEXT, DIM, 
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE, 
    ACCENT_ORANGE, CUSTOM_COLORS, BUILTIN_KEYS, HeatmapLegend,
    InteractiveLegendItem, build_ob_heatmap, build_order_placement_heatmap, get_rect,
    BUY_VOLUME_COLORS, SELL_VOLUME_COLORS, TRADE_VOLUME_COLORS, ORDER_BUY_COLORS, ORDER_SELL_COLORS,
    _is_timestamps_continuous
)

class TraderVisualizer(LogVisualizer):
    def __init__(self, log_path=None):
        QMainWindow.__init__(self)
        self.setWindowTitle('Trader Performance Visualizer')
        self.setGeometry(50, 50, 1600, 920)
        
        self.data, self.current_df, self.ob_res = None, None, None
        self.custom_curves = {}
        self._bid_p_cols = []; self._ask_p_cols = []
        self._custom_ts_arrays = {}
        self.data_settings = {}
        self.trade_tag_map = {}
        self.markup_lines = []
        self.markup_enabled = False
        self.tooltip_enabled = False
        self.tooltip_frozen = False
        self._freeze_lines = []
        self._plot_t = None
        self.sandbox_msgs = {}
        self._current_log_path = None
        self._gen_pane_minimized = False
        
        use_gl = sys.platform != 'linux' 
        pg.setConfigOptions(useOpenGL=use_gl, imageAxisOrder='row-major')
        
        self._build_ui()
        
        if log_path:
            self._load_file(log_path)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)
        
        controls = QHBoxLayout(); controls.setContentsMargins(12, 12, 12, 12); controls.setSpacing(12)
        btn_import = QPushButton("📊 Import Data"); btn_import.clicked.connect(self._import_data); controls.addWidget(btn_import)
        
        controls.addWidget(QLabel("Day:")); self.cb_day = QComboBox(); controls.addWidget(self.cb_day)
        controls.addWidget(QLabel("Product:")); self.cb_prod = QComboBox(); controls.addWidget(self.cb_prod)
        
        controls.addWidget(QLabel("Trader:")); self.cb_trader = QComboBox(); self.cb_trader.setMinimumWidth(150); controls.addWidget(self.cb_trader)
        controls.addWidget(QLabel("Side Filter:")); self.cb_side = QComboBox(); self.cb_side.addItems(["Both", "Buy", "Sell"]); controls.addWidget(self.cb_side)
        
        self.cb_day.currentTextChanged.connect(self._process_selection)
        self.cb_prod.currentTextChanged.connect(self._process_selection)
        self.cb_trader.currentTextChanged.connect(self._process_selection)
        self.cb_side.currentTextChanged.connect(self._process_selection)
        
        controls.addStretch()
        
        self.lbl_markup = QLabel("MARKUP: OFF"); self.lbl_markup.setStyleSheet(f"color: {DIM}; font-weight: bold;"); controls.addWidget(self.lbl_markup)
        self.lbl_tooltip = QLabel("TOOLTIP: OFF"); self.lbl_tooltip.setStyleSheet(f"color: {DIM}; font-weight: bold;"); controls.addWidget(self.lbl_tooltip)
        
        main_layout.addLayout(controls)
        
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self._build_profit_tab()
        self._build_position_tab()
        self._build_market_view_tab()
        
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("A"), self).activated.connect(self._autoscale_all)
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_markup)
        QShortcut(QKeySequence("T"), self).activated.connect(self._toggle_tooltip)
        QShortcut(QKeySequence("Shift+T"), self).activated.connect(self._freeze_tooltip)

    def _build_profit_tab(self):
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(12)
        metrics_frame = QFrame(); metrics_frame.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;"); metrics_layout = QHBoxLayout(metrics_frame)
        self.lbl_pnl = QLabel("Total PnL: --"); self.lbl_pnl.setStyleSheet(f"color: {ACCENT_GREEN}; font-weight: bold; font-size: 11pt;"); metrics_layout.addWidget(self.lbl_pnl)
        self.lbl_volume = QLabel("Total Volume: --"); metrics_layout.addWidget(self.lbl_volume)
        self.lbl_trades = QLabel("Trade Count: --"); metrics_layout.addWidget(self.lbl_trades)
        metrics_layout.addStretch(); layout.addWidget(metrics_frame)
        
        self.gw_pnl = pg.GraphicsLayoutWidget(); self.gw_pnl.setBackground(BG)
        self.p_pnl = self.gw_pnl.addPlot(title="Portfolio Cumulative PnL"); self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_pnl = self.p_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2))
        layout.addWidget(self.gw_pnl)
        
        self.trader_stats_table = QTableWidget(); self.trader_stats_table.setColumnCount(4)
        self.trader_stats_table.setHorizontalHeaderLabels(["Product", "Position", "Realized PnL", "Total PnL"])
        self.trader_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trader_stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.trader_stats_table)
        self.tabs.addTab(container, "Profit")

    def _build_position_tab(self):
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 0, 0, 0)
        self.gw_pos = pg.GraphicsLayoutWidget(); self.gw_pos.setBackground(BG)
        self.p_pos = self.gw_pos.addPlot(title="Trader Position per Product"); self.p_pos.showGrid(x=True, y=True, alpha=0.3); self.p_pos.addLegend()
        self.pos_curves = {}
        layout.addWidget(self.gw_pos)
        self.tabs.addTab(container, "Position")

    def _build_market_view_tab(self):
        market_container = QWidget(); market_layout = QVBoxLayout(market_container); market_layout.setContentsMargins(0, 0, 0, 0); market_layout.setSpacing(0)
        self.hm_legend = HeatmapLegend(); market_layout.addWidget(self.hm_legend)
        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG); market_layout.addWidget(self.gw_m)
        
        self.info_panel = QFrame(self.gw_m)
        self.info_panel.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")
        _ip_scroll = QScrollArea(self.info_panel); _ip_scroll.setWidgetResizable(True); _ip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._info_label = QLabel(); self._info_label.setTextFormat(Qt.TextFormat.RichText); self._info_label.setStyleSheet(f"color: {TEXT}; font-family: 'JetBrains Mono'; font-size: 8pt; padding: 6px;")
        _ip_scroll.setWidget(self._info_label)
        _ip_layout = QVBoxLayout(self.info_panel); _ip_layout.addWidget(_ip_scroll); self.info_panel.setFixedWidth(265); self.info_panel.setVisible(False)
        
        self.p_m = self.gw_m.addPlot(); self.p_m.showGrid(x=True, y=True, alpha=0.3)
        self.img_item = pg.ImageItem(); self.img_item.setZValue(0); self.p_m.addItem(self.img_item)
        self.curve_mid = self.p_m.plot(pen=pg.mkPen(ACCENT_CYAN, width=2), name="Mid Price")
        self.sc_buy = pg.ScatterPlotItem(symbol='t1', size=10, brush=ACCENT_GREEN, name="Trader Buy"); self.sc_buy.setZValue(3)
        self.sc_sell = pg.ScatterPlotItem(symbol='t', size=10, brush=ACCENT_ORANGE, name="Trader Sell"); self.sc_sell.setZValue(3)
        self.p_m.addItem(self.sc_buy); self.p_m.addItem(self.sc_sell)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.p_m.addItem(self.v_line); self.p_m.addItem(self.h_line)
        self.p_m.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.gw_m.installEventFilter(self)
        self.tabs.addTab(market_container, "Market View")

    def _import_data(self):
        data_dir = os.path.join(_PROJECT_ROOT, "data")
        if not os.path.exists(data_dir): return
        p_dfs, t_dicts = [], []
        for f in os.listdir(data_dir):
            if not f.endswith('.csv'): continue
            filepath = os.path.join(data_dir, f)
            if f.startswith("prices_"):
                df = pl.read_csv(filepath, separator=";", null_values=['', 'nan'])
                df = df.rename({c: c.strip() for c in df.columns})
                p_dfs.append(df)
            elif f.startswith("trades_"):
                day_match = re.search(r"day_(-?\d+)", f)
                df = pl.read_csv(filepath, separator=";", null_values=['', 'nan'])
                df = df.rename({c: c.strip() for c in df.columns})
                if day_match: df = df.with_columns(pl.lit(int(day_match.group(1))).alias("day"))
                t_dicts.extend(df.to_dicts())
        if not p_dfs: return
        df = pl.concat(p_dfs, how='diagonal_relaxed')
        if 'day' in df.columns: df = df.with_columns(pl.col('day').cast(pl.Int32, strict=False))
        if 'timestamp' in df.columns: df = df.with_columns(pl.col('timestamp').cast(pl.Int32, strict=False))
        df = df.drop_nulls(subset=['day', 'timestamp']).unique(subset=['day', 'timestamp', 'product'], keep='last').sort(['day', 'timestamp'])
        
        traders = {tr.get('buyer') for tr in t_dicts} | {tr.get('seller') for tr in t_dicts}
        sorted_traders = sorted([str(t) for t in traders if t is not None and str(t).strip() != ""])
        
        self.data = {'prices_df': df, 'trades': t_dicts, 'custom': {}, '_continuous_ts': _is_timestamps_continuous(df), '_min_day': df['day'].min() if 'day' in df.columns else 0}
        self.cb_day.blockSignals(True); self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()]); self.cb_day.blockSignals(False)
        self.cb_prod.blockSignals(True); self.cb_prod.clear(); self.cb_prod.addItems(df['product'].unique().sort().to_list()); self.cb_prod.blockSignals(False)
        self.cb_trader.blockSignals(True); self.cb_trader.clear(); self.cb_trader.addItems(sorted_traders); self.cb_trader.blockSignals(False)
        self.setWindowTitle("Trader Performance Visualizer [CSV Data]")
        self._process_selection()

    def _process_selection(self):
        if not self.data or not self.cb_prod.currentText() or not self.cb_trader.currentText(): return
        prod, day, trader, side_filter = self.cb_prod.currentText(), self.cb_day.currentText(), self.cb_trader.currentText(), self.cb_side.currentText()
        self.current_df = self.data['prices_df'].filter(pl.col('product') == prod)
        try:
            if day != 'All': self.current_df = self.current_df.filter(pl.col('day') == int(day))
        except: pass
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t = self.current_df['timestamp'].to_numpy()
        if day == 'All' and 'day' in self.current_df.columns and not cont_ts: t = t + (self.current_df['day'].to_numpy() - min_day) * 1000000
        self._plot_t = t
        mid = self.current_df['mid_price'].to_numpy(); self.curve_mid.setData(t, mid)
        self._update_trader_stats(trader, day, side_filter)
        self._update_position_plots(trader, day, side_filter)
        mb_t, mb_p, ms_t, ms_p = [], [], [], []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and str(tr.get('day', '')) != day: continue
            is_b, is_s = tr.get('buyer') == trader, tr.get('seller') == trader
            if not is_b and not is_s: continue
            if side_filter == "Buy" and not is_b: continue
            if side_filter == "Sell" and not is_s: continue
            ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr and not cont_ts: ts += (tr['day'] - min_day) * 1000000
            if is_b: mb_t.append(ts); mb_p.append(tr['price'])
            else: ms_t.append(ts); ms_p.append(tr['price'])
        self.sc_buy.setData(x=mb_t, y=mb_p); self.sc_sell.setData(x=ms_t, y=ms_p)
        self.ob_res = build_ob_heatmap(self.data['prices_df'], prod, day, cont_ts, BUY_VOLUME_COLORS, SELL_VOLUME_COLORS)
        if self.ob_res:
            self.img_item.setImage(self.ob_res['img'], autoLevels=False)
            self.img_item.setRect(get_rect(t, self.ob_res['levels'])); self.img_item.setVisible(True)
            self.hm_legend.update_ranges(self.ob_res['max_vol'], [0, 100])
        else: self.img_item.setVisible(False)
        self.p_m.autoRange(); self.p_pnl.autoRange(); self.p_pos.autoRange()

    def _update_trader_stats(self, trader, day, side_filter):
        trades, prices_df = self.data['trades'], self.data['prices_df']
        all_prods = sorted(prices_df['product'].unique().to_list())
        prod_stats = {p: {'pos': 0, 'cash': 0, 'last_mid': 0} for p in all_prods}
        total_volume, trade_count = 0, 0
        for tr in trades:
            if day != 'All' and str(tr.get('day', '')) != day: continue
            is_b, is_s = tr.get('buyer') == trader, tr.get('seller') == trader
            if not is_b and not is_s: continue
            if side_filter == "Buy" and not is_b: continue
            if side_filter == "Sell" and not is_s: continue
            p = tr.get('symbol')
            if p not in prod_stats: continue
            qty, pr = tr.get('quantity', 0), tr.get('price', 0)
            total_volume += qty; trade_count += 1
            if is_b: prod_stats[p]['pos'] += qty; prod_stats[p]['cash'] -= qty * pr
            else: prod_stats[p]['pos'] -= qty; prod_stats[p]['cash'] += qty * pr
        for p in all_prods:
            last_row = prices_df.filter(pl.col('product') == p).tail(1)
            if not last_row.is_empty(): prod_stats[p]['last_mid'] = last_row['mid_price'][0]
        self.trader_stats_table.setRowCount(len(all_prods))
        portfolio_pnl = 0
        for i, p in enumerate(all_prods):
            stats = prod_stats[p]; val_pnl = stats['cash'] + stats['pos'] * stats['last_mid']
            portfolio_pnl += val_pnl
            self.trader_stats_table.setItem(i, 0, QTableWidgetItem(p))
            self.trader_stats_table.setItem(i, 1, QTableWidgetItem(str(stats['pos'])))
            self.trader_stats_table.setItem(i, 2, QTableWidgetItem(f"{stats['cash']:,.2f}"))
            it_val = QTableWidgetItem(f"{val_pnl:,.2f}"); it_val.setForeground(pg.mkColor(ACCENT_GREEN if val_pnl >= 0 else ACCENT_RED))
            self.trader_stats_table.setItem(i, 3, it_val)
        self.lbl_pnl.setText(f"Total PnL: {portfolio_pnl:,.2f}"); self.lbl_pnl.setStyleSheet(f"color: {ACCENT_GREEN if portfolio_pnl >= 0 else ACCENT_RED}; font-weight: bold; font-size: 11pt;")
        self.lbl_volume.setText(f"Total Volume: {total_volume:,}"); self.lbl_trades.setText(f"Trade Count: {trade_count}")
        pnl_t, pnl_v = self._compute_portfolio_pnl(trader, day, side_filter); self.curve_pnl.setData(pnl_t, pnl_v)

    def _compute_portfolio_pnl(self, trader, day, side_filter):
        prices_df = self.data['prices_df']
        if day != 'All': prices_df = prices_df.filter(pl.col('day') == int(day))
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t_col = '_cts' if (day=='All' and not cont_ts) else 'timestamp'
        if t_col == '_cts': prices_df = prices_df.with_columns((pl.col('timestamp') + (pl.col('day') - min_day) * 1000000).alias(t_col))
        all_ts = sorted(prices_df[t_col].unique().to_list())
        if not all_ts: return np.array([]), np.array([])
        portfolio_pnl = np.zeros(len(all_ts))
        for p in prices_df['product'].unique().to_list():
            p_df = prices_df.filter(pl.col('product') == p).sort(t_col); p_ts, p_mid = p_df[t_col].to_list(), p_df['mid_price'].to_numpy()
            trade_events = []
            for tr in self.data['trades']:
                if tr.get('symbol') != p: continue
                if day != 'All' and str(tr.get('day', '')) != day: continue
                is_b, is_s = tr.get('buyer') == trader, tr.get('seller') == trader
                if not is_b and not is_s: continue
                if side_filter == "Buy" and not is_b: continue
                if side_filter == "Sell" and not is_s: continue
                tr_ts = tr.get('timestamp', 0)
                if day == 'All' and 'day' in tr and not cont_ts: tr_ts += (tr['day'] - min_day) * 1000000
                trade_events.append((tr_ts, int(tr.get('quantity', 0)) * (1 if is_b else -1), float(tr.get('price', 0))))
            trade_events.sort(); cash_at, pos_at, cc, cp = {}, {}, 0.0, 0
            for tr_ts, qs, pr in trade_events: cc -= qs * pr; cp += qs; cash_at[tr_ts] = cc; pos_at[tr_ts] = cp
            lc, lp, p_mid_idx = 0.0, 0, 0
            for i, ts in enumerate(all_ts):
                if ts in cash_at: lc, lp = cash_at[ts], pos_at[ts]
                while p_mid_idx < len(p_ts) and p_ts[p_mid_idx] <= ts: p_mid_idx += 1
                cur_mid = p_mid[p_mid_idx-1] if p_mid_idx > 0 else 0
                portfolio_pnl[i] += lc + lp * cur_mid
        return np.array(all_ts), portfolio_pnl

    def _update_position_plots(self, trader, day, side_filter):
        for c in self.pos_curves.values(): self.p_pos.removeItem(c)
        self.pos_curves.clear(); trades = self.data['trades']
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        pos_by_sym = {}
        for tr in trades:
            if day != 'All' and str(tr.get('day', '')) != day: continue
            is_b, is_s = tr.get('buyer') == trader, tr.get('seller') == trader
            if not is_b and not is_s: continue
            if side_filter == "Buy" and not is_b: continue
            if side_filter == "Sell" and not is_s: continue
            sym, qty, ts = tr.get('symbol',''), tr.get('quantity',0), tr.get('timestamp',0)
            pts = ts + (tr.get('day', min_day) - min_day)*1000000 if 'day' in tr and not cont_ts else ts
            pos_by_sym.setdefault(sym, []).append((tr.get('day', ts//1000000), pts, qty if is_b else -qty))
        for i, sym in enumerate(sorted(pos_by_sym.keys())):
            events = sorted(pos_by_sym[sym], key=lambda x: x[1]); ts_list, pos_list, cum, last_day = [], [], 0, events[0][0]
            for d_val, pts, delta in events:
                if d_val != last_day: cum = 0; last_day = d_val
                cum += delta; ts_list.append(pts); pos_list.append(cum)
            self.pos_curves[sym] = self.p_pos.plot(ts_list, pos_list, pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2), name=sym, stepMode='right')

    def _on_mouse_moved(self, pos):
        if not self.p_m.sceneBoundingRect().contains(pos) or self.current_df is None: return
        mouse_point = self.p_m.vb.mapSceneToView(pos); x, y = mouse_point.x(), mouse_point.y(); self.v_line.setPos(x); self.h_line.setPos(y)
        if not self.tooltip_enabled or self.tooltip_frozen: return
        idx = np.searchsorted(self._plot_t, x); idx = int(np.clip(idx, 0, len(self._plot_t) - 1))
        row = self.current_df.row(idx, named=True); raw_ts = int(row['timestamp']); lines = []
        def h(color, text): return f'<span style="color:{color}">{text}</span>'
        lines.append(h(ACCENT_CYAN, f'<b>TS: {raw_ts}</b>')); lines.append(f'Mid Price: {row.get("mid_price", 0):,.2f}')
        trader = self.cb_trader.currentText(); prod = self.cb_prod.currentText(); day = self.cb_day.currentText(); trades_here = []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and str(tr.get('day', '')) != day: continue
            if tr.get('timestamp') == raw_ts:
                is_b, is_s = tr.get('buyer') == trader, tr.get('seller') == trader
                if is_b or is_s: trades_here.append(tr)
        if trades_here:
            lines.append(h(ACCENT_GOLD, '<b>TRADER TRADES:</b>'))
            for tr in trades_here:
                side = 'BUY' if tr.get('buyer') == trader else 'SELL'; other = tr.get('seller') if side == 'BUY' else tr.get('buyer')
                lines.append(f' {side} {tr["quantity"]} @ {tr["price"]} vs {other}')
        self._info_label.setText('<br>'.join(lines)); self.info_panel.setVisible(True)

    def _autoscale_all(self):
        for p in [self.p_m, self.p_pnl, self.p_pos]:
            if p: p.autoRange()

    def _toggle_tooltip(self):
        self.tooltip_enabled = not self.tooltip_enabled
        self.lbl_tooltip.setText(f"TOOLTIP: {'ON' if self.tooltip_enabled else 'OFF'}")
        self.lbl_tooltip.setStyleSheet(f"color: {ACCENT_CYAN if self.tooltip_enabled else DIM}; font-weight: bold;")
        self.info_panel.setVisible(self.tooltip_enabled)

    def _freeze_tooltip(self):
        if not self.tooltip_enabled: return
        self.tooltip_frozen = not self.tooltip_frozen
        self.lbl_tooltip.setText(f"TOOLTIP: {'FROZEN' if self.tooltip_frozen else 'ON'}")
        self.lbl_tooltip.setStyleSheet(f"color: {ACCENT_GOLD if self.tooltip_frozen else ACCENT_CYAN}; font-weight: bold;")

    def _toggle_markup(self):
        self.markup_enabled = not self.markup_enabled
        self.lbl_markup.setText(f"MARKUP: {'ON' if self.markup_enabled else 'OFF'}")
        self.lbl_markup.setStyleSheet(f"color: {ACCENT_GOLD if self.markup_enabled else DIM}; font-weight: bold;")

    def _restore_window_state(self): pass
    def _save_window_state(self): pass

    def eventFilter(self, obj, event):
        if obj is self.gw_m and event.type() == QEvent.Type.Resize: self.info_panel.move(self.gw_m.width() - 280, 10)
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(APP_STYLE)
    win = TraderVisualizer(); win.show(); sys.exit(app.exec())
