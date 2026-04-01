"""
log_visualizer.py — Prosperity Log Visualizer
==============================================
Parses the 3-section .log JSON file produced by the Prosperity sandbox and
renders a PyQt6 / pyqtgraph dashboard with:
  • Order-book heatmap (bid=blue, ask=red) per product / day
  • Mid-price + best bid/ask spread overlay
  • Trades by other bots  (white ×)
  • Your own trades  (buyer or seller == "SUBMISSION")  (gold ★)
  • PnL vs timestamp  (from activitiesLog profit_and_loss column)
  • Valuation = PnL + holdings × mid_price  (dotted cyan line)
  • Any custom key→value data emitted by logger.py  (auto-plotted per key)
  • All panels share the same X-axis and support x/y/z zoom modes

Usage:
    python log_visualizer.py path/to/39187.log
    python log_visualizer.py          # opens file-picker dialog
"""

import sys, os, json, re, ast
from copy import deepcopy
from io import StringIO

import numpy as np
import polars as pl
import pyqtgraph as pg

os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QCheckBox, QPushButton, QFileDialog, QScrollArea,
    QSplitter, QTabWidget, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QShortcut, QKeySequence, QFont


# ─────────────────────────── colour palette (dark theme) ────────────────────
BG          = '#0d0f14'
PANEL_BG    = '#12151c'
BORDER      = '#1e2330'
TEXT        = '#c8d0e0'
DIM         = '#4a5068'
ACCENT_CYAN = '#00d4ff'
ACCENT_GREEN= '#39ff6e'
ACCENT_RED  = '#ff3d5a'
ACCENT_GOLD = '#ffd700'
ACCENT_PURPLE='#b06dff'
ACCENT_WHITE= '#ffffff'

APP_STYLE = f"""
QMainWindow, QWidget {{ background-color: {BG}; color: {TEXT};
    font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 9pt; }}
QComboBox {{ background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 3px 8px; border-radius: 3px; color: {TEXT}; min-width: 80px; }}
QComboBox::drop-down {{ border: 0; }}
QComboBox QAbstractItemView {{ background: {PANEL_BG}; color: {TEXT};
    selection-background-color: #22273a; }}
QLabel {{ color: {TEXT}; }}
QCheckBox {{ spacing: 5px; color: {TEXT}; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {BORDER};
    background: {PANEL_BG}; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: {ACCENT_CYAN}; }}
QPushButton {{ background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 4px 12px; border-radius: 3px; color: {TEXT}; }}
QPushButton:hover {{ border-color: {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}
QPushButton:pressed {{ background: #1a1f2e; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; }}
QTabBar::tab {{ background: {PANEL_BG}; color: {DIM};
    padding: 5px 16px; border: 1px solid {BORDER}; border-bottom: none; }}
QTabBar::tab:selected {{ color: {ACCENT_CYAN}; border-bottom: 2px solid {ACCENT_CYAN}; }}
QScrollBar:vertical {{ background: {BG}; width: 6px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
"""

CUSTOM_COLORS = [
    '#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f',
    '#b06dff', '#ff9f43', '#00d2d3', '#ff6348', '#7bed9f',
]


# ────────────────────────────── log parser ───────────────────────────────────

def parse_log(path: str):
    """
    Returns dict with keys:
      prices_df   – polars DataFrame (CSV section)
      logs        – list of {timestamp, sandboxLog, lambdaLog}
      trades      – list of trade dicts
      custom      – dict {series_name: [(timestamp, value), ...]}
    """
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    # ── section 1: order-book CSV ────────────────────────────────────────────
    activities_csv = raw.get('activitiesLog', '')
    # The CSV may embed \n as literal escape or real newlines
    activities_csv = activities_csv.replace('\\n', '\n')
    prices_df = None
    if activities_csv.strip():
        try:
            prices_df = pl.read_csv(StringIO(activities_csv), separator=';',
                                    null_values=['', 'nan'])
            # Normalise column names (strip whitespace)
            prices_df = prices_df.rename({c: c.strip() for c in prices_df.columns})
        except Exception as e:
            print(f"[warn] CSV parse failed: {e}")

    # ── section 2: lambda / sandbox logs ────────────────────────────────────
    logs_raw = raw.get('logs', [])

    # ── section 3: trade history ─────────────────────────────────────────────
    trades = raw.get('tradeHistory', [])

    # ── custom series from lambdaLog ─────────────────────────────────────────
    custom = {}   # name → [(ts, value)]
    for entry in logs_raw:
        ts = entry.get('timestamp', 0)
        lambda_log = entry.get('lambdaLog', '') or ''
        for line in lambda_log.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Try to decode LOGVIZ: prefix written by logger.py
            if line.startswith('LOGVIZ:'):
                try:
                    payload = json.loads(line[7:])
                    for k, v in payload.items():
                        custom.setdefault(k, []).append((ts, float(v)))
                except Exception:
                    pass

    return dict(prices_df=prices_df, logs=logs_raw, trades=trades, custom=custom)


# ─────────────────────── order-book heatmap builder ─────────────────────────

def build_ob_heatmap(p_df: pl.DataFrame, product: str, day):
    """Returns ob_data dict compatible with the image renderer."""
    flt = p_df.filter(pl.col('product') == product)
    if day != 'All':
        flt = flt.filter(pl.col('day') == int(day))
    flt = flt.sort(['day', 'timestamp'])
    flt = flt.with_columns(pl.col('timestamp').alias('plot_time'))

    plot_time = flt['plot_time'].to_numpy()

    bid_price_cols = sorted([c for c in flt.columns if re.match(r'bid_price_\d+', c)],
                             key=lambda x: int(x.rsplit('_', 1)[1]))
    ask_price_cols = sorted([c for c in flt.columns if re.match(r'ask_price_\d+', c)],
                             key=lambda x: int(x.rsplit('_', 1)[1]))

    def get_levels(pcols, prefix):
        pairs = []
        for c in pcols:
            lvl = c.rsplit('_', 1)[1]
            vc = f'{prefix}_volume_{lvl}'
            if vc in flt.columns:
                pairs.append((c, vc))
        return pairs

    bid_levels = get_levels(bid_price_cols, 'bid')
    ask_levels = get_levels(ask_price_cols, 'ask')

    all_p, all_v = [], []
    for pc, vc in bid_levels + ask_levels:
        pa = flt[pc].to_numpy(); va = flt[vc].to_numpy()
        valid = np.isfinite(pa) & np.isfinite(va) & (pa > 0) & (va > 0)
        if np.any(valid):
            all_p.append(pa[valid]); all_v.append(va[valid])

    if not all_p:
        return {'time': plot_time, 'price_levels': np.array([])}

    price_levels = np.unique(np.concatenate(all_p))
    max_vol = max(np.max(np.concatenate(all_v)), 1.0)
    n = len(plot_time)
    x_idx = np.arange(n, dtype=np.int32)

    def build_side(pairs):
        xs, ys, ins, vs = [], [], [], []
        for pc, vc in pairs:
            pa = flt[pc].to_numpy(); va = flt[vc].to_numpy()
            valid = np.isfinite(pa) & np.isfinite(va) & (pa > 0) & (va > 0)
            if not np.any(valid): continue
            xs.append(x_idx[valid])
            ys.append(np.searchsorted(price_levels, pa[valid]).astype(np.int32))
            ins.append(np.clip(va[valid] / max_vol * 255, 0, 255).astype(np.uint8))
            vs.append(va[valid])
        if not xs:
            return [np.array([], dtype=t) for t in [np.int32, np.int32, np.uint8, float]]
        return [np.concatenate(a) for a in [xs, ys, ins, vs]]

    bx, by, bi, bv = build_side(bid_levels)
    ax, ay, ai, av = build_side(ask_levels)

    return dict(time=plot_time, price_levels=price_levels,
                bid_x=bx, bid_y=by, bid_i=bi, bid_v=bv,
                ask_x=ax, ask_y=ay, ask_i=ai, ask_v=av)


def render_ob_image(ob_data):
    times = ob_data.get('time', np.array([]))
    price_levels = ob_data.get('price_levels', np.array([]))
    if len(times) == 0 or len(price_levels) == 0:
        return None
    w, h = len(times), len(price_levels)
    red   = np.zeros(h * w, np.uint8)
    blue  = np.zeros(h * w, np.uint8)
    ax, ay, ai = ob_data.get('ask_x', []), ob_data.get('ask_y', []), ob_data.get('ask_i', [])
    if len(ax): np.maximum.at(red,  ay.astype(np.int64) * w + ax.astype(np.int64), ai)
    bx, by, bi = ob_data.get('bid_x', []), ob_data.get('bid_y', []), ob_data.get('bid_i', [])
    if len(bx): np.maximum.at(blue, by.astype(np.int64) * w + bx.astype(np.int64), bi)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = red.reshape(h, w)
    img[..., 2] = blue.reshape(h, w)
    return img


# ────────────────────── zoom-aware viewbox (from original) ──────────────────

class ModeViewBox(pg.ViewBox):
    def __init__(self, mode_getter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_getter = mode_getter

    def wheelEvent(self, ev, axis=None):
        mode = self._mode_getter()
        if mode == 'x':   super().wheelEvent(ev, axis=0); return
        if mode == 'y':   super().wheelEvent(ev, axis=1); return
        if mode == 'z':
            try:    delta = ev.delta()
            except: delta = ev.angleDelta().y()
            x0, x1 = self.viewRange()[0]
            self.translateBy(x=-np.sign(delta) * (x1 - x0) * 0.08, y=0)
            ev.accept(); return
        super().wheelEvent(ev, axis=None)


# ──────────────────────────── background worker ──────────────────────────────

class Worker(QThread):
    done = pyqtSignal(object)  # emits result dict

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        result = self._fn(*self._args, **self._kwargs)
        self.done.emit(result)


# ─────────────────────────── main window ────────────────────────────────────

class LogVisualizer(QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.setWindowTitle('Prosperity Log Visualizer')
        self.setGeometry(80, 80, 1600, 960)

        self._data       = None   # parsed log dict
        self._zoom_mode  = 'xy'
        self._ob_data    = None
        self._p_flt      = None   # current price df slice

        # pyqtgraph dark setup
        pg.setConfigOptions(antialias=False, useOpenGL=True, imageAxisOrder='row-major')
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        fmt.setSwapInterval(0); fmt.setSamples(0)
        QSurfaceFormat.setDefaultFormat(fmt)

        self._build_ui()
        self._bind_shortcuts()

        if log_path:
            self._load_file(log_path)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        vbox = QVBoxLayout(root); vbox.setSpacing(4); vbox.setContentsMargins(6, 6, 6, 6)

        # ── top control bar ──────────────────────────────────────────────────
        cbar = QHBoxLayout(); cbar.setSpacing(8)

        self._btn_open = QPushButton('📂 Open Log')
        self._btn_open.clicked.connect(self._open_dialog)
        cbar.addWidget(self._btn_open)

        def lbl(t): w = QLabel(t); w.setStyleSheet(f'color:{DIM};'); cbar.addWidget(w); return w
        def combo(items=None):
            w = QComboBox()
            if items: w.addItems(items)
            cbar.addWidget(w); return w

        lbl('Product:');  self._cb_product = combo()
        lbl('Day:');      self._cb_day     = combo()

        self._cb_product.currentTextChanged.connect(self._on_selection_change)
        self._cb_day.currentTextChanged.connect(self._on_selection_change)

        cbar.addSpacing(16)
        self._chk_spread = QCheckBox('Spread'); self._chk_spread.setChecked(True)
        self._chk_spread.stateChanged.connect(self._toggle_spread)
        cbar.addWidget(self._chk_spread)

        self._chk_my_trades = QCheckBox('My Trades'); self._chk_my_trades.setChecked(True)
        self._chk_my_trades.stateChanged.connect(self._refresh_trades)
        cbar.addWidget(self._chk_my_trades)

        self._chk_bot_trades = QCheckBox('Bot Trades'); self._chk_bot_trades.setChecked(True)
        self._chk_bot_trades.stateChanged.connect(self._refresh_trades)
        cbar.addWidget(self._chk_bot_trades)

        # self._chk_valuation = QCheckBox('Valuation'); self._chk_valuation.setChecked(True)
        # self._chk_valuation.stateChanged.connect(self._refresh_pnl)
        # cbar.addWidget(self._chk_valuation)

        self._chk_ob_mid = QCheckBox('OB Mid'); self._chk_ob_mid.setChecked(True)
        self._chk_ob_mid.stateChanged.connect(self._toggle_ob_mid)
        cbar.addWidget(self._chk_ob_mid)

        cbar.addStretch()
        self._zoom_lbl = QLabel('XY zoom | X Y Z keys')
        self._zoom_lbl.setStyleSheet(f'color:{DIM}; font-size:8pt;')
        cbar.addWidget(self._zoom_lbl)

        self._status = QLabel('No file loaded')
        self._status.setStyleSheet(f'color:{DIM};')
        cbar.addWidget(self._status)

        vbox.addLayout(cbar)

        # ── tab widget ───────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        vbox.addWidget(self._tabs)

        # Tab 1 – market view
        self._gw_market = pg.GraphicsLayoutWidget()
        self._gw_market.setBackground(BG)
        self._tabs.addTab(self._gw_market, 'Market / OB')

        # Tab 2 – PnL / valuation
        self._gw_pnl = pg.GraphicsLayoutWidget()
        self._gw_pnl.setBackground(BG)
        self._tabs.addTab(self._gw_pnl, 'PnL')

        # Tab 3 – custom series (built dynamically)
        self._gw_custom = pg.GraphicsLayoutWidget()
        self._gw_custom.setBackground(BG)
        self._tabs.addTab(self._gw_custom, 'Custom Plots')

        self._init_market_plots()
        self._init_pnl_plots()
        # custom plots are built on data load

    def _styled_plot(self, gw, row, col, title='', share_x=None):
        vb = ModeViewBox(lambda: self._zoom_mode)
        p = gw.addPlot(row=row, col=col, title=title, viewBox=vb)
        if share_x: p.setXLink(share_x)
        p.setClipToView(True); p.setDownsampling(mode='peak')
        p.showGrid(x=True, y=True, alpha=0.18)
        for axis in ('left', 'bottom'):
            p.getAxis(axis).setTextPen(pg.mkPen(TEXT))
            p.getAxis(axis).setPen(pg.mkPen(BORDER))
        p.getAxis('bottom').setStyle(tickFont=QFont('Consolas', 7))
        p.addLegend(offset=(10, 10))
        return p

    def _init_market_plots(self):
        gw = self._gw_market

        # p_price: mid + spread + all trades
        self._pp = self._styled_plot(gw, 0, 0, 'Price — Mid / Spread / Trades')
        self._curve_mid  = self._pp.plot([], [], pen=pg.mkPen(ACCENT_CYAN,  width=2), name='Mid')
        self._curve_ask  = pg.PlotCurveItem([], [], pen=pg.mkPen(ACCENT_RED,   width=1))
        self._curve_bid  = pg.PlotCurveItem([], [], pen=pg.mkPen(ACCENT_GREEN, width=1))
        self._curve_anch = pg.PlotCurveItem([], [])
        self._fill_ask   = pg.FillBetweenItem(self._curve_anch, self._curve_ask,
                                               brush=(255, 61, 90, 40))
        self._fill_bid   = pg.FillBetweenItem(self._curve_bid, self._curve_anch,
                                               brush=(57, 255, 110, 40))
        for item in [self._curve_ask, self._curve_bid, self._curve_anch,
                     self._fill_ask, self._fill_bid]:
            self._pp.addItem(item)

        # bot trades (white ×)
        self._sc_bot = pg.ScatterPlotItem(x=[], y=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(ACCENT_WHITE),
            size=7, symbol='x', pxMode=True, name='Bot Trades')
        self._pp.addItem(self._sc_bot)

        # my trades – buy (▲ green) and sell (▼ red)
        self._sc_my_buy = pg.ScatterPlotItem(x=[], y=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(ACCENT_GREEN),
            size=9, symbol='t1', pxMode=True, name='My Buy')
        self._sc_my_sell = pg.ScatterPlotItem(x=[], y=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(ACCENT_RED),
            size=9, symbol='t', pxMode=True, name='My Sell')
        self._pp.addItem(self._sc_my_buy)
        self._pp.addItem(self._sc_my_sell)

        # p_ob: order-book heatmap
        self._po = self._styled_plot(gw, 1, 0, 'Order Book Heatmap  (red=ask  blue=bid)',
                                     share_x=self._pp)
        self._ob_img  = pg.ImageItem(); self._ob_img.setOpts(axisOrder='row-major')
        self._po.addItem(self._ob_img)
        self._ob_mid_curve = pg.PlotCurveItem([], [], pen=pg.mkPen(ACCENT_CYAN, width=2))
        self._po.addItem(self._ob_mid_curve)
        self._po.setLabel('left', 'Price'); self._po.setLabel('bottom', 'Timestamp')

        gw.ci.layout.setRowStretchFactor(0, 2)
        gw.ci.layout.setRowStretchFactor(1, 3)

    def _init_pnl_plots(self):
        gw = self._gw_pnl
        self._pnl_plot = self._styled_plot(gw, 0, 0, 'PnL & Valuation')
        self._curve_pnl = self._pnl_plot.plot([], [], pen=pg.mkPen(ACCENT_GREEN, width=2),
                                               name='PnL')
        # self._curve_val = self._pnl_plot.plot([], [],
        #     pen=pg.mkPen(ACCENT_CYAN, width=2, style=Qt.PenStyle.DotLine),
        #     name='Valuation')

    # ── data loading ─────────────────────────────────────────────────────────

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Log File', '',
                                               'Log files (*.log *.json);;All (*)')
        if path:
            self._load_file(path)

    def _load_file(self, path):
        self._set_status(f'Loading {os.path.basename(path)}…', ACCENT_GOLD)
        self._worker = Worker(parse_log, path)
        self._worker.done.connect(self._on_data_loaded)
        self._worker.start()

    def _on_data_loaded(self, data):
        self._data = data
        df = data['prices_df']

        self._cb_product.blockSignals(True)
        self._cb_day.blockSignals(True)
        self._cb_product.clear(); self._cb_day.clear()

        if df is not None:
            products = df['product'].unique().sort().to_list()
            self._cb_product.addItems(products)
            days = ['All'] + [str(d) for d in df['day'].unique().sort().to_list()]
            self._cb_day.addItems(days)

        self._cb_product.blockSignals(False)
        self._cb_day.blockSignals(False)

        self._build_custom_plots()
        self._on_selection_change()
        self._set_status('Ready', ACCENT_GREEN)

    # ── selection change → reprocess ─────────────────────────────────────────

    def _on_selection_change(self):
        if self._data is None or self._data['prices_df'] is None:
            return
        product = self._cb_product.currentText()
        day     = self._cb_day.currentText()
        if not product:
            return
        self._set_status('Processing…', ACCENT_GOLD)
        self._worker2 = Worker(self._process, product, day)
        self._worker2.done.connect(self._on_processed)
        self._worker2.start()

    def _process(self, product, day):
        df = self._data['prices_df']
        flt = df.filter(pl.col('product') == product)
        if day != 'All':
            flt = flt.filter(pl.col('day') == int(day))
        flt = flt.sort(['day', 'timestamp'])
        flt = flt.with_columns(pl.col('timestamp').alias('plot_time'))

        time  = flt['plot_time'].to_numpy()
        mid   = flt['mid_price'].to_numpy() if 'mid_price' in flt.columns else np.zeros(len(time))
        bid1  = flt['bid_price_1'].to_numpy() if 'bid_price_1' in flt.columns else mid.copy()
        ask1  = flt['ask_price_1'].to_numpy() if 'ask_price_1' in flt.columns else mid.copy()
        pnl   = flt['profit_and_loss'].to_numpy() if 'profit_and_loss' in flt.columns else np.zeros(len(time))

        # ── valuation: PnL + holdings * mid ──────────────────────────────────
        trades = self._data['trades']
        # compute holdings over time for this product
        holdings = self._compute_holdings(trades, product, time)
        # valuation = pnl + holdings * mid

        # ── trade scatter ────────────────────────────────────────────────────
        my_buy_t, my_buy_p   = [], []
        my_sell_t, my_sell_p = [], []
        bot_t, bot_p         = [], []

        for t in trades:
            if t.get('symbol') != product: continue
            ts = t['timestamp']; pr = t['price']
            buyer  = (t.get('buyer')  or '').upper()
            seller = (t.get('seller') or '').upper()
            is_mine = (buyer == 'SUBMISSION' or seller == 'SUBMISSION')
            if is_mine:
                if buyer == 'SUBMISSION':
                    my_buy_t.append(ts); my_buy_p.append(pr)
                else:
                    my_sell_t.append(ts); my_sell_p.append(pr)
            else:
                bot_t.append(ts); bot_p.append(pr)

        ob = build_ob_heatmap(df, product, day)

        return dict(time=time, mid=mid, bid=bid1, ask=ask1, pnl=pnl,
                    my_buy=(np.array(my_buy_t), np.array(my_buy_p)),
                    my_sell=(np.array(my_sell_t), np.array(my_sell_p)),
                    bot=(np.array(bot_t), np.array(bot_p)),
                    ob=ob)

    def _compute_holdings(self, trades, product, time_axis):
        """Simple running position for SUBMISSION on given product."""
        events = []
        for t in trades:
            if t.get('symbol') != product: continue
            buyer  = (t.get('buyer')  or '').upper()
            seller = (t.get('seller') or '').upper()
            qty    = t.get('quantity', 0)
            ts     = t['timestamp']
            if buyer  == 'SUBMISSION': events.append((ts, +qty))
            if seller == 'SUBMISSION': events.append((ts, -qty))
        events.sort(key=lambda x: x[0])
        holdings = np.zeros(len(time_axis), dtype=float)
        pos = 0.0
        ei = 0
        for i, t in enumerate(time_axis):
            while ei < len(events) and events[ei][0] <= t:
                pos += events[ei][1]; ei += 1
            holdings[i] = pos
        return holdings

    def _on_processed(self, result):
        t    = result['time']
        mid  = result['mid']
        bid  = result['bid']
        ask  = result['ask']

        self._curve_mid.setData(t, mid)
        self._curve_anch.setData(t, mid)
        self._curve_bid.setData(t, bid)
        self._curve_ask.setData(t, ask)
        self._toggle_spread()

        # trades
        bx, by = result['bot']
        self._sc_bot.setData(x=bx, y=by)
        mx, my = result['my_buy']
        self._sc_my_buy.setData(x=mx, y=my)
        sx, sy = result['my_sell']
        self._sc_my_sell.setData(x=sx, y=sy)
        self._refresh_trades()

        # OB heatmap
        ob = result['ob']
        img = render_ob_image(ob)
        if img is not None:
            times   = ob['time']
            plevels = ob['price_levels']
            self._ob_img.setImage(img, autoLevels=False)
            xs = np.diff(times.astype(float))
            xs = xs[xs > 0]
            xstep = float(np.median(xs)) if len(xs) else 1.0
            ys = np.diff(plevels.astype(float))
            ys = ys[ys > 0]
            ystep = float(np.median(ys)) if len(ys) else 1.0
            self._ob_img.setRect(QRectF(
                float(times[0]) - 0.5 * xstep,
                float(plevels[0]) - 0.5 * ystep,
                len(times) * xstep,
                len(plevels) * ystep
            ))
            self._ob_mid_curve.setData(t, mid)
            self._toggle_ob_mid()
        else:
            self._ob_img.clear()
        self._ob_data = ob

        # PnL / valuation
        self._curve_pnl.setData(t, result['pnl'])
        # self._curve_val.setData(t, result['valuation'])
        self._refresh_pnl()

        self._pp.autoRange(); self._po.autoRange(); self._pnl_plot.autoRange()
        self._set_status('Ready', ACCENT_GREEN)

    # ── custom plots ─────────────────────────────────────────────────────────

    def _build_custom_plots(self):
        custom = self._data.get('custom', {})
        gw = self._gw_custom
        gw.clear()
        if not custom:
            lbl = pg.LabelItem('No custom LOGVIZ data found in lambda logs.', color=DIM)
            gw.addItem(lbl, 0, 0)
            return

        anchor = None
        for i, (name, points) in enumerate(custom.items()):
            color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
            p = self._styled_plot(gw, i, 0, name, share_x=anchor)
            if anchor is None: anchor = p
            ts = np.array([x[0] for x in points])
            vs = np.array([x[1] for x in points])
            p.plot(ts, vs, pen=pg.mkPen(color, width=2), name=name)

    # ── toggle helpers ───────────────────────────────────────────────────────

    def _toggle_spread(self):
        on = self._chk_spread.isChecked()
        for item in [self._fill_ask, self._fill_bid, self._curve_ask, self._curve_bid]:
            item.setVisible(on)

    def _refresh_trades(self):
        self._sc_bot.setVisible(self._chk_bot_trades.isChecked())
        self._sc_my_buy.setVisible(self._chk_my_trades.isChecked())
        self._sc_my_sell.setVisible(self._chk_my_trades.isChecked())

    def _refresh_pnl(self):
        # self._curve_val.setVisible(self._chk_valuation.isChecked())
        pass

    def _toggle_ob_mid(self):
        self._ob_mid_curve.setVisible(self._chk_ob_mid.isChecked())

    # ── zoom ─────────────────────────────────────────────────────────────────

    def _set_zoom(self, mode):
        self._zoom_mode = mode
        labels = {'xy': 'XY zoom', 'x': 'X-only zoom', 'y': 'Y-only zoom', 'z': 'X-scroll'}
        self._zoom_lbl.setText(f'{labels[mode]}  |  X Y Z keys')

    def _bind_shortcuts(self):
        def toggle(m):
            self._set_zoom('xy' if self._zoom_mode == m else m)
        QShortcut(QKeySequence('X'), self).activated.connect(lambda: toggle('x'))
        QShortcut(QKeySequence('Y'), self).activated.connect(lambda: toggle('y'))
        QShortcut(QKeySequence('Z'), self).activated.connect(lambda: toggle('z'))

    # ── status ───────────────────────────────────────────────────────────────

    def _set_status(self, msg, color=DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f'color:{color}; font-weight:bold;')


# ─────────────────────────────── entry point ────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    path = sys.argv[1] if len(sys.argv) > 1 else None
    win  = LogVisualizer(log_path=path)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
