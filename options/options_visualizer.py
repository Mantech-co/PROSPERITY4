import sys
import numpy as np
from scipy.stats import norm

import re

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QComboBox, QFrame, QDoubleSpinBox, QSpinBox, QSplitter,
    QLineEdit
)
from PyQt6.QtCore import Qt, QRectF, QEvent
from PyQt6.QtGui import QShortcut, QKeySequence, QColor, QBrush
import pyqtgraph as pg

BG, PANEL_BG, BORDER, TEXT, DIM = '#0d0f14', '#12151c', '#1e2330', '#c8d0e0', '#4a5068'
ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE, ACCENT_ORANGE = \
    '#00d4ff', '#39ff6e', '#ff3d5a', '#ffd700', '#ffffff', '#b06dff', '#ff9f43'

# Diverging colormap: red (loss) → dark BG (zero) → green (profit)
# Stored as float32 [0-255] for direct numpy indexing, bypassing pg LUT pipeline.
def _build_lut(n=512):
    stops = np.array([0.0, 0.40, 0.50, 0.60, 1.0])
    cols  = np.array([
        [180, 30,  30,  255],
        [55,  0,   0,   230],
        [13,  15,  20,  200],
        [0,   55,  0,   230],
        [30,  180, 30,  255],
    ], dtype=np.float32)
    t = np.linspace(0.0, 1.0, n)
    lut = np.zeros((n, 4), dtype=np.float32)
    for ch in range(4):
        lut[:, ch] = np.interp(t, stops, cols[:, ch])
    return lut

PNL_LUT = _build_lut()   # (512, 4) float32 in [0, 255]


def apply_colormap(grid, vmax):
    """Map float32 P&L grid → uint8 RGBA via diverging LUT."""
    norm = np.clip((grid / vmax + 1.0) * 0.5, 0.0, 1.0)
    idx  = (norm * (len(PNL_LUT) - 1)).astype(np.int32)
    return PNL_LUT[idx].astype(np.uint8)

APP_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 9pt;
}}
QComboBox {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 2px 8px; border-radius: 3px; color: {TEXT}; min-width: 80px;
}}
QPushButton {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 3px 12px; border-radius: 3px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT_CYAN}; }}
#DataStrip {{
    background-color: {PANEL_BG}; border-top: 1px solid {BORDER};
    color: {ACCENT_CYAN}; padding: 4px 15px; font-size: 9pt;
}}
QTableWidget {{
    background-color: {BG}; color: {TEXT}; gridline-color: {BORDER};
    border: none; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 8.5pt;
}}
QTableWidget::item {{ padding: 2px 6px; border-bottom: 1px solid {BORDER}; }}
QTableWidget::item:selected {{ background-color: {BORDER}; }}
QHeaderView::section {{
    background-color: {PANEL_BG}; color: {ACCENT_CYAN};
    border: 1px solid {BORDER}; padding: 4px 8px; font-weight: bold; font-size: 8.5pt;
}}
QDoubleSpinBox, QSpinBox {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 3px 8px; border-radius: 3px; color: {TEXT};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{ background: {BORDER}; border: none; width: 14px; }}
QSplitter::handle {{ background: {BORDER}; width: 2px; }}
"""


def bs_price(S, K, r, sigma, T, opt_type):
    if T <= 0:
        return max(S - K, 0.0) if opt_type == 'call' else max(K - S, 0.0)
    if sigma <= 0 or S <= 0:
        df = np.exp(-r * T)
        return max(S - K * df, 0.0) if opt_type == 'call' else max(K * df - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price_vec(S_arr, K, r, sigma, T, opt_type):
    if T <= 0:
        return np.maximum(S_arr - K, 0.0) if opt_type == 'call' else np.maximum(K - S_arr, 0.0)
    safe_S = np.where(S_arr > 0, S_arr, 1e-10)
    if sigma <= 0:
        df = np.exp(-r * T)
        return np.maximum(safe_S - K * df, 0.0) if opt_type == 'call' else np.maximum(K * df - safe_S, 0.0)
    d1 = (np.log(safe_S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'call':
        return S_arr * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S_arr * norm.cdf(-d1)


def expiry_payoff_vec(S_arr, K, opt_type):
    return np.maximum(S_arr - K, 0.0) if opt_type == 'call' else np.maximum(K - S_arr, 0.0)


def _section(txt):
    l = QLabel(txt)
    l.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 8.5pt;")
    return l


def _sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


class OptionsVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Options Basket Visualizer')
        self.setGeometry(50, 50, 1440, 900)
        self.basket = []
        pg.setConfigOptions(useOpenGL=True, imageAxisOrder='row-major')
        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── Left panel ───────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(330)
        left.setStyleSheet(f"background-color: {PANEL_BG}; border-right: 1px solid {BORDER};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(10)

        lv.addWidget(_section("GLOBAL PARAMS"))
        gp = QWidget()
        grid = QGridLayout(gp)
        grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(6)

        self.spin_S     = QDoubleSpinBox(); self.spin_S.setRange(0.01, 1e6);     self.spin_S.setValue(100);  self.spin_S.setDecimals(2)
        self.spin_r     = QDoubleSpinBox(); self.spin_r.setRange(0, 1);          self.spin_r.setValue(0.05); self.spin_r.setDecimals(4); self.spin_r.setSingleStep(0.001)
        self.spin_sigma = QDoubleSpinBox(); self.spin_sigma.setRange(0.001, 10); self.spin_sigma.setValue(0.20); self.spin_sigma.setDecimals(4); self.spin_sigma.setSingleStep(0.01)
        self.spin_T     = QDoubleSpinBox(); self.spin_T.setRange(0.001, 30);     self.spin_T.setValue(0.25); self.spin_T.setDecimals(4); self.spin_T.setSingleStep(0.01)

        for row, (lbl, w) in enumerate([("Spot (S):", self.spin_S), ("Rate (r):", self.spin_r),
                                         ("Vol (σ):", self.spin_sigma), ("Expiry T (yr):", self.spin_T)]):
            grid.addWidget(QLabel(lbl), row, 0); grid.addWidget(w, row, 1)
        lv.addWidget(gp)
        lv.addWidget(_sep())

        lv.addWidget(_section("ADD OPTION"))
        af = QWidget()
        aform = QGridLayout(af)
        aform.setContentsMargins(0, 0, 0, 0); aform.setSpacing(6)

        self.cb_dir   = QComboBox(); self.cb_dir.addItems(["BUY", "SELL"])
        self.cb_dir.currentTextChanged.connect(
            lambda t: self.cb_dir.setStyleSheet(
                f"color: {ACCENT_GREEN};" if t == "BUY" else f"color: {ACCENT_RED};"
            )
        )
        self.cb_dir.setStyleSheet(f"color: {ACCENT_GREEN};")
        self.cb_type  = QComboBox(); self.cb_type.addItems(["CALL", "PUT"])
        self.spin_K   = QDoubleSpinBox(); self.spin_K.setRange(0.01, 1e6); self.spin_K.setValue(100); self.spin_K.setDecimals(2)
        self.spin_qty = QSpinBox(); self.spin_qty.setRange(1, 10000); self.spin_qty.setValue(1)
        self.spin_iv  = QDoubleSpinBox(); self.spin_iv.setRange(0.001, 10); self.spin_iv.setValue(0.20); self.spin_iv.setDecimals(4); self.spin_iv.setSingleStep(0.01)

        for row, (lbl, w) in enumerate([("Dir:", self.cb_dir), ("Type:", self.cb_type), ("Strike (K):", self.spin_K),
                                         ("Qty:", self.spin_qty), ("IV (σ override):", self.spin_iv)]):
            aform.addWidget(QLabel(lbl), row, 0); aform.addWidget(w, row, 1)
        lv.addWidget(af)

        btn_add = QPushButton("+ Add to Basket")
        btn_add.setStyleSheet(f"QPushButton {{ border-color: {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}")
        btn_add.clicked.connect(self._add_option)
        lv.addWidget(btn_add)
        lv.addWidget(_sep())

        lv.addWidget(_section("BASKET"))
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels(["Dir", "Type", "K", "Qty", "IV", "Price"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setMaximumHeight(220)
        lv.addWidget(self.tbl)

        btn_row = QHBoxLayout()
        btn_rm = QPushButton("Remove Selected"); btn_rm.clicked.connect(self._remove_option)
        btn_cl = QPushButton("Clear All");       btn_cl.clicked.connect(self._clear_basket)
        btn_row.addWidget(btn_rm); btn_row.addWidget(btn_cl)
        lv.addLayout(btn_row)
        lv.addWidget(_sep())

        btn_upd = QPushButton("Update Plot  [Enter]")
        btn_upd.setStyleSheet(f"QPushButton {{ border-color: {ACCENT_GREEN}; color: {ACCENT_GREEN}; font-weight: bold; }}")
        btn_upd.clicked.connect(self._update_plot)
        lv.addWidget(btn_upd)
        lv.addStretch()

        splitter.addWidget(left)

        # ── Right panel ──────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(0)

        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground(BG)
        rv.addWidget(self.gw)

        self.data_strip = QLabel()
        self.data_strip.setObjectName("DataStrip")
        self.data_strip.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        rv.addWidget(self.data_strip)

        self.cmd_bar = QLineEdit()
        self.cmd_bar.setPlaceholderText(":  {B/S}[qty]{C/P}{strike}  —  e.g.  BC100  S3P95.5  B2C110")
        self.cmd_bar.setVisible(False)
        self.cmd_bar.setStyleSheet(f"""
            QLineEdit {{
                background: {PANEL_BG}; border: none; border-top: 2px solid {ACCENT_CYAN};
                color: {ACCENT_CYAN}; padding: 5px 15px; font-size: 9pt;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
            }}
        """)
        self.cmd_bar.installEventFilter(self)
        rv.addWidget(self.cmd_bar)

        # ── Top plot: line chart ─────────────────────────────────────────────
        self.plot = self.gw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'P&L')
        self.plot.hideAxis('bottom')
        self.plot.addLegend(offset=(10, 10))

        self.zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.zero_line, ignoreBounds=True)

        self.spot_line = pg.InfiniteLine(
            pos=100, angle=90,
            pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine),
            label='S={value:.2f}', labelOpts={'color': ACCENT_GOLD, 'position': 0.95}
        )
        self.plot.addItem(self.spot_line, ignoreBounds=True)

        self.curve_expiry = self.plot.plot(pen=pg.mkPen(ACCENT_CYAN, width=2.5), name="Expiry P&L")
        self.curve_bs     = self.plot.plot(pen=pg.mkPen(ACCENT_GREEN, width=2),   name="BS Value P&L (now)")

        # ── Bottom plot: S×T contour heatmap ─────────────────────────────────
        self.plot_c = self.gw.addPlot(row=1, col=0)
        self.plot_c.setXLink(self.plot)
        self.plot_c.showGrid(x=True, y=True, alpha=0.2)
        self.plot_c.setLabel('bottom', 'Underlying Price (S)')
        self.plot_c.setLabel('left', 'Time Remaining (yr)')

        self.img_contour = pg.ImageItem()
        self.plot_c.addItem(self.img_contour)

        # Zero-contour line (horizontal at T=current → not meaningful as a line;
        # spot line tracks current S on the contour pane)
        self.spot_line_c = pg.InfiniteLine(
            pos=100, angle=90,
            pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_c.addItem(self.spot_line_c, ignoreBounds=True)

        # Horizontal line at current T (top of contour = "now")
        self.t_line_c = pg.InfiniteLine(
            pos=0.25, angle=0,
            pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine),
            label='T={value:.4f}', labelOpts={'color': ACCENT_GOLD, 'position': 0.02}
        )
        self.plot_c.addItem(self.t_line_c, ignoreBounds=True)

        # Colorbar legend (manual gradient bar)
        self._build_colorbar()

        self.gw.ci.layout.setRowStretchFactor(0, 2)
        self.gw.ci.layout.setRowStretchFactor(1, 3)

        splitter.addWidget(right)
        splitter.setSizes([330, 1110])

        for w in [self.spin_S, self.spin_r, self.spin_sigma, self.spin_T]:
            w.valueChanged.connect(self._update_plot)
        self.spin_sigma.valueChanged.connect(lambda v: self.spin_iv.setValue(v))
        self.return_sc = QShortcut(QKeySequence("Return"), self)
        self.return_sc.activated.connect(self._update_plot)
        QShortcut(QKeySequence(":"), self).activated.connect(self._open_cmd)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._close_cmd)
        self.cmd_bar.returnPressed.connect(self._exec_cmd)

    def _build_colorbar(self):
        cb_plot = self.gw.addPlot(row=1, col=1)
        cb_plot.setFixedWidth(60)
        cb_plot.hideAxis('bottom')
        cb_plot.setLabel('right', 'P&L')
        cb_plot.setMouseEnabled(x=False, y=False)
        cb_plot.setMenuEnabled(False)

        # Build colorbar image via same LUT (1-pixel wide, 512-pixel tall)
        n = 512
        bar = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(n, 1)
        cb_rgba = apply_colormap(bar, 1.0)   # (512, 1, 4)
        self._cb_img = pg.ImageItem(cb_rgba)
        cb_plot.addItem(self._cb_img)
        cb_plot.setXRange(0, 1, padding=0)
        cb_plot.setYRange(0, n, padding=0)
        self._cb_plot = cb_plot

        self._cb_min_label  = pg.TextItem("", anchor=(0, 0),   color=ACCENT_RED)
        self._cb_zero_label = pg.TextItem("0", anchor=(0, 0.5), color=DIM)
        self._cb_max_label  = pg.TextItem("", anchor=(0, 1),   color=ACCENT_GREEN)
        for item in [self._cb_min_label, self._cb_zero_label, self._cb_max_label]:
            cb_plot.addItem(item)
        self._cb_min_label.setPos(0.05, 5)
        self._cb_zero_label.setPos(0.05, n // 2)
        self._cb_max_label.setPos(0.05, n - 5)

    # ── Basket management ────────────────────────────────────────────────────

    def _add_option(self):
        sign = 1 if self.cb_dir.currentText() == "BUY" else -1
        self.basket.append({
            'type': self.cb_type.currentText().lower(),
            'K':    self.spin_K.value(),
            'qty':  sign * self.spin_qty.value(),
            'iv':   self.spin_iv.value(),
        })
        self._refresh_table()
        self._update_plot()

    def _remove_option(self):
        rows = sorted({i.row() for i in self.tbl.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.basket):
                self.basket.pop(r)
        self._refresh_table()
        self._update_plot()

    def _clear_basket(self):
        self.basket.clear()
        self._refresh_table()
        self._update_plot()

    def _refresh_table(self):
        S = self.spin_S.value()
        r = self.spin_r.value()
        T = self.spin_T.value()
        self.tbl.setRowCount(len(self.basket))
        for i, opt in enumerate(self.basket):
            color = ACCENT_CYAN if opt['type'] == 'call' else ACCENT_ORANGE
            dir_str = "BUY" if opt['qty'] >= 0 else "SELL"
            dir_color = ACCENT_GREEN if opt['qty'] >= 0 else ACCENT_RED
            price = bs_price(S, opt['K'], r, opt['iv'], T, opt['type'])
            for col, val in enumerate([dir_str, opt['type'].upper(), f"{opt['K']:.2f}", str(abs(opt['qty'])), f"{opt['iv']:.4f}", f"{price:.4f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QBrush(QColor(dir_color if col == 0 else color)))
                self.tbl.setItem(i, col, item)

    # ── Plot ─────────────────────────────────────────────────────────────────

    def _update_plot(self):
        S = self.spin_S.value()
        r = self.spin_r.value()
        T = self.spin_T.value()

        self.spot_line.setValue(S)
        self.spot_line_c.setValue(S)
        self.t_line_c.setValue(T)

        if not self.basket:
            self.curve_expiry.setData([], [])
            self.curve_bs.setData([], [])
            self.img_contour.setImage(np.zeros((1, 1, 4), dtype=np.uint8))
            self.data_strip.setText("  No options in basket")
            return

        strikes = [o['K'] for o in self.basket]
        S_lo = max(0.01, min(strikes + [S]) * 0.5)
        S_hi = max(strikes + [S]) * 1.5
        S_arr = np.linspace(S_lo, S_hi, 400)

        # Cost basis at current (S, T)
        cost_basis = sum(o['qty'] * bs_price(S, o['K'], r, o['iv'], T, o['type']) for o in self.basket)

        # ── Line chart ───────────────────────────────────────────────────────
        expiry_total = np.zeros(len(S_arr))
        bs_total     = np.zeros(len(S_arr))
        for opt in self.basket:
            q, K, iv, otype = opt['qty'], opt['K'], opt['iv'], opt['type']
            expiry_total += q * expiry_payoff_vec(S_arr, K, otype)
            bs_total     += q * bs_price_vec(S_arr, K, r, iv, T, otype)

        self.curve_expiry.setData(S_arr, expiry_total - cost_basis)
        self.curve_bs.setData(S_arr, bs_total - cost_basis)

        # ── Contour heatmap: grid[i_T, i_S] = BS P&L at (S_arr[i_S], T_arr[i_T]) ──
        # T_arr[0]  = near expiry (bottom of Y axis)
        # T_arr[-1] = T_current  (top of Y axis)
        n_T   = 150
        T_arr = np.linspace(1e-6, T, n_T)
        grid  = np.empty((n_T, len(S_arr)), dtype=np.float32)
        for i, t in enumerate(T_arr):
            row_val = np.zeros(len(S_arr))
            for opt in self.basket:
                row_val += opt['qty'] * bs_price_vec(S_arr, opt['K'], r, opt['iv'], t, opt['type'])
            grid[i, :] = row_val - cost_basis

        vmax = max(float(np.abs(grid).max()), 1e-10)
        rgba = apply_colormap(grid, vmax)   # (n_T, n_S, 4) uint8
        self.img_contour.setImage(rgba)
        self.img_contour.setRect(QRectF(float(S_lo), float(T_arr[0]),
                                        float(S_hi - S_lo), float(T_arr[-1] - T_arr[0])))

        self._cb_min_label.setText(f"{-vmax:.3g}")
        self._cb_max_label.setText(f"+{vmax:.3g}")

        # ── Data strip ───────────────────────────────────────────────────────
        expiry_pnl = expiry_total - cost_basis
        signs = np.sign(expiry_pnl)
        be_prices = []
        for i in np.where(np.diff(signs) != 0)[0]:
            d = expiry_pnl[i+1] - expiry_pnl[i]
            if d: be_prices.append(S_arr[i] - expiry_pnl[i] / d * (S_arr[i+1] - S_arr[i]))
        be_str = ", ".join(f"{p:.2f}" for p in be_prices) or "N/A"

        bs_at_S    = float(np.interp(S, S_arr, bs_total))
        cost_label = "Debit" if cost_basis > 0 else "Credit"
        self.data_strip.setText(
            f"  {cost_label}: {abs(cost_basis):.4f}"
            f"  |  BS Value @S: {bs_at_S:.4f}"
            f"  |  BS P&L @S: {bs_at_S - cost_basis:+.4f}"
            f"  |  Max Profit: {np.max(expiry_pnl):.4f}"
            f"  |  Max Loss: {np.min(expiry_pnl):.4f}"
            f"  |  Breakevens: {be_str}"
            f"  |  Legs: {len(self.basket)}"
        )
        self._refresh_table()


    # ── Command bar ──────────────────────────────────────────────────────────

    def _open_cmd(self):
        self.return_sc.setEnabled(False)
        self.cmd_bar.setVisible(True)
        self.cmd_bar.setFocus()
        self.cmd_bar.setText(":")

    def _close_cmd(self):
        self.return_sc.setEnabled(True)
        self.cmd_bar.setVisible(False)
        self.cmd_bar.clear()

    def eventFilter(self, obj, event):
        if obj is self.cmd_bar and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._close_cmd()
                return True
        return super().eventFilter(obj, event)

    def _exec_cmd(self):
        raw = self.cmd_bar.text().lstrip(":").strip()
        self.cmd_bar.setText(":")
        self.cmd_bar.setFocus()
        if not raw:
            return
        m = re.fullmatch(r'([BSbs])(\d+)?([CPcp])([\d.]+)', raw)
        if not m:
            self._restore_cmd_style(error=f"bad: '{raw}'")
            return
        direction, qty_str, opt_type, strike_str = m.groups()
        sign = 1 if direction.upper() == 'B' else -1
        qty  = int(qty_str) if qty_str else 1
        self.basket.append({
            'type': 'call' if opt_type.upper() == 'C' else 'put',
            'K':    float(strike_str),
            'qty':  sign * qty,
            'iv':   self.spin_iv.value(),
        })
        self._refresh_table()
        self._update_plot()

    def _restore_cmd_style(self, error=None):
        if error:
            self.cmd_bar.setStyleSheet(f"""
                QLineEdit {{
                    background: {PANEL_BG}; border: none; border-top: 2px solid {ACCENT_RED};
                    color: {ACCENT_RED}; padding: 5px 15px; font-size: 9pt;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                }}
            """)
            self.cmd_bar.setPlaceholderText(f"  {error}  — e.g. BC100  S3P95.5  B2C110")
            self.cmd_bar.textChanged.connect(lambda: self._restore_cmd_style())
        else:
            self.cmd_bar.setStyleSheet(f"""
                QLineEdit {{
                    background: {PANEL_BG}; border: none; border-top: 2px solid {ACCENT_CYAN};
                    color: {ACCENT_CYAN}; padding: 5px 15px; font-size: 9pt;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                }}
            """)
            self.cmd_bar.setPlaceholderText(":  {B/S}[qty]{C/P}{strike}  —  e.g.  BC100  S3P95.5  B2C110")
            try:
                self.cmd_bar.textChanged.disconnect()
            except TypeError:
                pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = OptionsVisualizer()
    win.show()
    sys.exit(app.exec())
