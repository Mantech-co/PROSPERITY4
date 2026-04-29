#!/usr/bin/env python3
"""
midviz.py — Mid-price technical analysis dashboard for round 5 data.
Usage: python midviz.py
"""

import sys, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyqtgraph as pg

pg.setConfigOptions(antialias=True)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QLineEdit, QCheckBox, QScrollArea,
    QFrame, QSpinBox, QDoubleSpinBox, QComboBox, QToolButton,
    QSizePolicy, QGroupBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QShortcut, QKeySequence

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "round5"

BG        = '#0d0f14'
PANEL_BG  = '#12151c'
BORDER    = '#1e2330'
TEXT      = '#c8d0e0'
DIM       = '#4a5068'
CYAN      = '#00d4ff'
GREEN     = '#39ff6e'
RED       = '#ff3d5a'
GOLD      = '#ffd700'
PURPLE    = '#b06dff'
ORANGE    = '#ff9f43'

OVERLAY_PALETTE = [
    '#00d4ff', '#39ff6e', '#ffd700', '#ff6b6b', '#b06dff',
    '#ff9f43', '#06d6a0', '#118ab2', '#ef476f', '#ffd166',
    '#a8dadc', '#457b9d', '#e63946', '#2a9d8f', '#e9c46a',
]

APP_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 9pt;
}}
QSplitter::handle {{ background: {BORDER}; width: 2px; height: 2px; }}
QPushButton {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 3px 10px; border-radius: 3px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {CYAN}; color: {CYAN}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QToolButton {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 2px 6px; border-radius: 3px; color: {TEXT};
}}
QToolButton:hover {{ border-color: {CYAN}; }}
QLineEdit {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 3px 8px; border-radius: 3px; color: {TEXT};
}}
QLineEdit:focus {{ border-color: {CYAN}; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER}; border-radius: 2px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QSpinBox, QDoubleSpinBox {{
    background: {PANEL_BG}; border: 1px solid {BORDER};
    padding: 2px 4px; border-radius: 3px; color: {TEXT};
}}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 4px;
    margin-top: 8px; padding-top: 4px;
    font-size: 8pt; color: {DIM};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QScrollBar:vertical {{
    background: {PANEL_BG}; width: 6px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#dim {{ color: {DIM}; font-size: 8pt; }}
"""

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_pivot(days: list[int]) -> pd.DataFrame:
    frames = []
    for d in days:
        f = DATA_DIR / f"prices_round_5_day_{d}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, sep=';'))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[['day', 'timestamp', 'product', 'mid_price']].copy()
    pivot = df.pivot_table(index=['day', 'timestamp'], columns='product', values='mid_price')
    pivot = pivot.sort_index().ffill()
    return pivot

def all_products() -> list[str]:
    frames = []
    for f in DATA_DIR.glob("prices_round_5_day_*.csv"):
        df = pd.read_csv(f, sep=';', usecols=['product'])
        frames.append(df)
    if not frames:
        return []
    return sorted(pd.concat(frames)['product'].unique().tolist())

def group_products(products: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in products:
        parts = p.rsplit('_', 1)
        prefix = parts[0] if len(parts) == 2 and not parts[1].isdigit() else p
        # Detect known prefixes
        for sep in ['_']:
            tokens = p.split(sep)
            if len(tokens) >= 2:
                # Use all but last token as group key
                grp = sep.join(tokens[:-1])
                break
        else:
            grp = p
        groups.setdefault(grp, []).append(p)
    # If group has only 1 member, keep ungrouped
    result: dict[str, list[str]] = {}
    for grp, members in groups.items():
        if len(members) == 1:
            result.setdefault('OTHER', []).extend(members)
        else:
            result[grp] = members
    return result

# ── Indicator Math ────────────────────────────────────────────────────────────
def sma(y: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if period < 1:
        return out
    k = np.ones(period) / period
    conv = np.convolve(y, k, mode='valid')
    out[period - 1:] = conv
    return out

def ema(y: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if period < 1 or len(y) == 0:
        return out
    alpha = 2.0 / (period + 1)
    val = float(y[0])
    out[0] = val
    for i in range(1, len(y)):
        val = alpha * float(y[i]) + (1 - alpha) * val
        out[i] = val
    out[:period - 1] = np.nan
    return out

def bollinger(y: np.ndarray, period: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(y, period)
    std = np.full_like(y, np.nan, dtype=float)
    for i in range(period - 1, len(y)):
        std[i] = float(np.std(y[i - period + 1:i + 1]))
    upper = mid + sigma * std
    lower = mid - sigma * std
    return upper, mid, lower

def rolling_vol(y: np.ndarray, period: int) -> np.ndarray:
    """Rolling historical volatility: std of log-returns over `period` bars."""
    out = np.full(len(y), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where(y[:-1] > 0, np.log(y[1:] / y[:-1]), np.nan)
    for i in range(period - 1, len(lr)):
        window = lr[i - period + 1:i + 1]
        if np.isfinite(window).sum() >= 2:
            out[i + 1] = float(np.nanstd(window))
    return out

# ── Expression eval (same as midplot.py) ─────────────────────────────────────
def eval_expr(expr: str, pivot: pd.DataFrame) -> np.ndarray | None:
    if pivot is None or pivot.empty:
        return None
    products = sorted(pivot.columns.tolist(), key=len, reverse=True)
    ns: dict = {p: pivot[p].values.astype(float) for p in products if p in pivot.columns}
    mapping: dict = {}
    safe_expr = expr
    for p in products:
        safe = '__p_' + re.sub(r'\W', '_', p)
        safe_expr = safe_expr.replace(p, safe)
        mapping[safe] = ns[p]
    mapping['np'] = np
    try:
        result = eval(safe_expr, {'__builtins__': {}}, mapping)
        return np.asarray(result, dtype=float)
    except Exception as e:
        return None

# ── Overlay entry ─────────────────────────────────────────────────────────────
class Overlay:
    _next_id = 0

    def __init__(self, label: str, color: str):
        self.id = Overlay._next_id
        Overlay._next_id += 1
        self.label = label
        self.color = color
        self.visible = True
        self.items: list[pg.PlotDataItem] = []
        self.plot = None  # set to the PlotWidget that owns the items

# ── Overlay row widget ────────────────────────────────────────────────────────
class OverlayRow(QWidget):
    def __init__(self, overlay: Overlay, on_toggle, on_remove, parent=None):
        super().__init__(parent)
        self.overlay = overlay
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f'background:{overlay.color}; border-radius:2px;')
        layout.addWidget(swatch)

        self.vis_btn = QToolButton()
        self.vis_btn.setText('●')
        self.vis_btn.setFixedSize(18, 18)
        self.vis_btn.setStyleSheet(f'color:{overlay.color}; border:none; background:transparent;')
        self.vis_btn.clicked.connect(lambda: on_toggle(overlay))
        layout.addWidget(self.vis_btn)

        lbl = QLabel(overlay.label)
        lbl.setStyleSheet(f'color:{TEXT}; font-size:8pt;')
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(lbl)

        rm = QToolButton()
        rm.setText('✕')
        rm.setFixedSize(16, 16)
        rm.setStyleSheet(f'color:{DIM}; border:none; background:transparent;')
        rm.clicked.connect(lambda: on_remove(overlay))
        layout.addWidget(rm)

    def set_visible(self, v: bool):
        self.vis_btn.setStyleSheet(
            f'color:{"#fff" if v else DIM}; border:none; background:transparent;'
        )

# ── Main Window ───────────────────────────────────────────────────────────────
class MidViz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MidViz — Round 5')
        self.resize(1600, 900)
        self.setStyleSheet(APP_STYLE)

        self._products = all_products()
        self._groups = group_products(self._products)
        self._pivot: pd.DataFrame | None = None
        self._overlays: list[Overlay] = []
        self._overlay_rows: dict[int, OverlayRow] = {}
        self._color_idx = 0
        self._day_checks: dict[int, QCheckBox] = {}
        self._x: np.ndarray | None = None
        self._day_boundaries: list[int] = []

        self._build_ui()
        self._reload_data()

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([200, 1150, 250])

        QShortcut(QKeySequence('Ctrl+L'), self).activated.connect(self._clear_all)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(160)
        panel.setMaximumWidth(280)
        panel.setStyleSheet(f'background:{PANEL_BG}; border-right:1px solid {BORDER};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel('  PRODUCTS')
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(f'background:{PANEL_BG}; color:{CYAN}; font-size:8pt; font-weight:bold; border-bottom:1px solid {BORDER};')
        layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet('border:none;')

        container = QWidget()
        self._product_layout = QVBoxLayout(container)
        self._product_layout.setContentsMargins(4, 4, 4, 4)
        self._product_layout.setSpacing(2)
        self._product_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for grp, members in sorted(self._groups.items()):
            self._add_group_widget(grp, members)

        scroll.setWidget(container)
        layout.addWidget(scroll)
        return panel

    def _add_group_widget(self, grp: str, members: list[str]):
        # Group header (collapsible)
        grp_widget = QWidget()
        grp_layout = QVBoxLayout(grp_widget)
        grp_layout.setContentsMargins(0, 0, 0, 2)
        grp_layout.setSpacing(1)

        header = QToolButton()
        short = grp.replace('_', ' ')
        header.setText(f'▾ {short}')
        header.setCheckable(True)
        header.setChecked(True)
        header.setStyleSheet(f'''
            QToolButton {{
                text-align:left; background:{PANEL_BG}; border:none;
                color:{DIM}; font-size:7.5pt; padding:2px 4px;
            }}
            QToolButton:hover {{ color:{TEXT}; }}
        ''')

        member_widget = QWidget()
        member_layout = QVBoxLayout(member_widget)
        member_layout.setContentsMargins(8, 0, 0, 0)
        member_layout.setSpacing(1)

        for p in sorted(members):
            short_name = p[len(grp) + 1:] if p.startswith(grp + '_') else p
            btn = QPushButton(short_name)
            btn.setToolTip(p)
            btn.setFixedHeight(20)
            btn.setStyleSheet(f'''
                QPushButton {{
                    text-align:left; background:transparent; border:none;
                    color:{TEXT}; font-size:8pt; padding:0 4px;
                }}
                QPushButton:hover {{ color:{CYAN}; background:{BORDER}; border-radius:2px; }}
            ''')
            btn.clicked.connect(lambda checked, prod=p: self._add_product_overlay(prod))
            member_layout.addWidget(btn)

        header.toggled.connect(member_widget.setVisible)
        grp_layout.addWidget(header)
        grp_layout.addWidget(member_widget)
        self._product_layout.addWidget(grp_widget)

    def _build_center(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(44)
        toolbar.setStyleSheet(f'background:{PANEL_BG}; border-bottom:1px solid {BORDER};')
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 6, 10, 6)
        tb_layout.setSpacing(8)

        # Day selectors
        day_lbl = QLabel('DAYS:')
        day_lbl.setStyleSheet(f'color:{DIM}; font-size:8pt;')
        tb_layout.addWidget(day_lbl)
        for d in [2, 3, 4]:
            cb = QCheckBox(str(d))
            cb.setChecked(True)
            cb.stateChanged.connect(self._reload_data)
            self._day_checks[d] = cb
            tb_layout.addWidget(cb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f'color:{BORDER};')
        tb_layout.addWidget(sep)

        # Expression bar
        expr_lbl = QLabel('EXPR:')
        expr_lbl.setStyleSheet(f'color:{DIM}; font-size:8pt;')
        tb_layout.addWidget(expr_lbl)
        self._expr_input = QLineEdit()
        self._expr_input.setPlaceholderText('e.g. PEBBLES_L - PEBBLES_S  or  PEBBLES_L * f')
        self._expr_input.setMinimumWidth(350)
        self._expr_input.returnPressed.connect(self._plot_expr)
        tb_layout.addWidget(self._expr_input)

        plot_btn = QPushButton('PLOT')
        plot_btn.setFixedWidth(60)
        plot_btn.clicked.connect(self._plot_expr)
        tb_layout.addWidget(plot_btn)

        tb_layout.addStretch()

        # Clear all
        clear_btn = QPushButton('CLEAR  Ctrl+L')
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(self._clear_all)
        tb_layout.addWidget(clear_btn)

        layout.addWidget(toolbar)

        # Indicator toolbar
        ind_bar = QWidget()
        ind_bar.setFixedHeight(36)
        ind_bar.setStyleSheet(f'background:{BG}; border-bottom:1px solid {BORDER};')
        ind_layout = QHBoxLayout(ind_bar)
        ind_layout.setContentsMargins(10, 4, 10, 4)
        ind_layout.setSpacing(6)

        # SMA
        ind_layout.addWidget(self._ind_label('SMA'))
        self._sma_period = self._spin(20, 1, 500)
        ind_layout.addWidget(self._sma_period)
        sma_btn = QPushButton('ADD')
        sma_btn.setFixedWidth(44)
        sma_btn.clicked.connect(self._add_sma)
        ind_layout.addWidget(sma_btn)

        ind_layout.addWidget(self._vsep())

        # EMA
        ind_layout.addWidget(self._ind_label('EMA'))
        self._ema_period = self._spin(20, 1, 500)
        ind_layout.addWidget(self._ema_period)
        ema_btn = QPushButton('ADD')
        ema_btn.setFixedWidth(44)
        ema_btn.clicked.connect(self._add_ema)
        ind_layout.addWidget(ema_btn)

        ind_layout.addWidget(self._vsep())

        # Bollinger
        ind_layout.addWidget(self._ind_label('BB'))
        self._bb_period = self._spin(20, 2, 500)
        ind_layout.addWidget(self._bb_period)
        self._bb_sigma = QDoubleSpinBox()
        self._bb_sigma.setRange(0.1, 5.0)
        self._bb_sigma.setSingleStep(0.1)
        self._bb_sigma.setValue(2.0)
        self._bb_sigma.setDecimals(1)
        self._bb_sigma.setFixedWidth(52)
        ind_layout.addWidget(self._bb_sigma)
        bb_btn = QPushButton('ADD')
        bb_btn.setFixedWidth(44)
        bb_btn.clicked.connect(self._add_bb)
        ind_layout.addWidget(bb_btn)

        ind_layout.addWidget(self._vsep())

        # Apply to
        ind_layout.addWidget(self._ind_label('ON:'))
        self._ind_target = QComboBox()
        self._ind_target.setFixedWidth(160)
        self._ind_target.setStyleSheet(f'background:{PANEL_BG}; color:{TEXT}; border:1px solid {BORDER}; padding:2px 4px;')
        ind_layout.addWidget(self._ind_target)

        # VOL
        ind_layout.addWidget(self._vsep())
        ind_layout.addWidget(self._ind_label('VOL'))
        self._vol_period = self._spin(20, 2, 500)
        ind_layout.addWidget(self._vol_period)
        vol_btn = QPushButton('ADD')
        vol_btn.setFixedWidth(44)
        vol_btn.clicked.connect(self._add_vol)
        ind_layout.addWidget(vol_btn)

        ind_layout.addStretch()

        layout.addWidget(ind_bar)

        # Vertical splitter: main plot on top, vol panel below
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.setHandleWidth(3)

        # Main plot
        self._plot = pg.PlotWidget()
        self._plot.setBackground(BG)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.getAxis('left').setTextPen(TEXT)
        self._plot.getAxis('bottom').setTextPen(TEXT)
        self._plot.getAxis('left').setPen(BORDER)
        self._plot.getAxis('bottom').setPen(BORDER)
        self._plot.getAxis('bottom').setStyle(showValues=False)

        # Crosshair — ignoreBounds so they never affect autoscale
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, width=1, style=Qt.PenStyle.DotLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, width=1, style=Qt.PenStyle.DotLine))
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._plot.addItem(self._vline, ignoreBounds=True)
        self._plot.addItem(self._hline, ignoreBounds=True)
        self._coord_label = pg.TextItem('', anchor=(0, 1), color=DIM)
        self._coord_label.setFont(QFont('Consolas', 8))
        self._plot.addItem(self._coord_label, ignoreBounds=True)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_move)

        plot_splitter.addWidget(self._plot)

        # Vol panel
        self._vol_panel = QWidget()
        self._vol_panel.setStyleSheet(f'background:{BG};')
        vol_panel_layout = QVBoxLayout(self._vol_panel)
        vol_panel_layout.setContentsMargins(0, 0, 0, 0)
        vol_panel_layout.setSpacing(0)

        vol_header = QWidget()
        vol_header.setFixedHeight(22)
        vol_header.setStyleSheet(f'background:{PANEL_BG}; border-top:1px solid {BORDER}; border-bottom:1px solid {BORDER};')
        vol_hdr_layout = QHBoxLayout(vol_header)
        vol_hdr_layout.setContentsMargins(8, 0, 8, 0)
        vol_hdr_layout.setSpacing(6)
        vol_title = QLabel('VOLATILITY  (rolling σ of log-returns)')
        vol_title.setStyleSheet(f'color:{GOLD}; font-size:7.5pt; font-weight:bold;')
        vol_hdr_layout.addWidget(vol_title)
        vol_hdr_layout.addStretch()
        vol_close = QToolButton()
        vol_close.setText('✕')
        vol_close.setFixedSize(16, 16)
        vol_close.setStyleSheet(f'color:{DIM}; border:none; background:transparent;')
        vol_close.clicked.connect(lambda: self._vol_panel.setVisible(False))
        vol_hdr_layout.addWidget(vol_close)
        vol_panel_layout.addWidget(vol_header)

        self._vol_plot = pg.PlotWidget()
        self._vol_plot.setBackground(BG)
        self._vol_plot.showGrid(x=True, y=True, alpha=0.15)
        self._vol_plot.getAxis('left').setTextPen(TEXT)
        self._vol_plot.getAxis('bottom').setTextPen(TEXT)
        self._vol_plot.getAxis('left').setPen(BORDER)
        self._vol_plot.getAxis('bottom').setPen(BORDER)
        self._vol_plot.setLabel('bottom', 'timestep', color=DIM)
        self._vol_plot.setXLink(self._plot)
        self._vol_plot.scene().sigMouseMoved.connect(self._on_mouse_move_vol)
        vol_panel_layout.addWidget(self._vol_plot)

        # Vol crosshair
        self._vol_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, width=1, style=Qt.PenStyle.DotLine))
        self._vol_vline.setVisible(False)
        self._vol_plot.addItem(self._vol_vline, ignoreBounds=True)

        self._vol_panel.setVisible(False)
        plot_splitter.addWidget(self._vol_panel)
        plot_splitter.setSizes([700, 200])
        plot_splitter.setStretchFactor(0, 3)
        plot_splitter.setStretchFactor(1, 1)

        layout.addWidget(plot_splitter)

        # Status bar
        self._status = QLabel('  ready')
        self._status.setFixedHeight(22)
        self._status.setStyleSheet(f'background:{PANEL_BG}; color:{DIM}; font-size:8pt; border-top:1px solid {BORDER}; padding-left:8px;')
        layout.addWidget(self._status)

        return w

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(320)
        panel.setStyleSheet(f'background:{PANEL_BG}; border-left:1px solid {BORDER};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel('  OVERLAYS')
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(f'background:{PANEL_BG}; color:{CYAN}; font-size:8pt; font-weight:bold; border-bottom:1px solid {BORDER};')
        layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet('border:none;')

        self._overlay_container = QWidget()
        self._overlay_list_layout = QVBoxLayout(self._overlay_container)
        self._overlay_list_layout.setContentsMargins(4, 4, 4, 4)
        self._overlay_list_layout.setSpacing(2)
        self._overlay_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self._overlay_container)
        layout.addWidget(scroll)

        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _ind_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f'color:{CYAN}; font-size:8pt; font-weight:bold;')
        return lbl

    def _spin(self, val: int, mn: int, mx: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(mn, mx)
        s.setValue(val)
        s.setFixedWidth(52)
        return s

    def _vsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(20)
        sep.setStyleSheet(f'color:{BORDER};')
        return sep

    def _next_color(self) -> str:
        c = OVERLAY_PALETTE[self._color_idx % len(OVERLAY_PALETTE)]
        self._color_idx += 1
        return c

    # ── Data ─────────────────────────────────────────────────────────────────
    def _reload_data(self):
        days = [d for d, cb in self._day_checks.items() if cb.isChecked()]
        if not days:
            self._pivot = None
            self._x = None
            self._day_boundaries = []
            return
        self._pivot = load_pivot(days)
        if self._pivot.empty:
            self._pivot = None
            self._x = None
            return
        self._x = np.arange(len(self._pivot))
        # day boundaries
        day_col = [d for d, _ in self._pivot.index]
        self._day_boundaries = [0] + [i for i in range(1, len(day_col)) if day_col[i] != day_col[i - 1]]
        self._refresh_all_overlays()
        self._status.setText(f'  loaded days {days}  —  {len(self._pivot)} timesteps  —  {len(self._products)} products')

    def _draw_boundaries(self):
        # Remove old boundary lines
        for item in list(self._plot.items()):
            if getattr(item, '_is_day_boundary', False):
                self._plot.removeItem(item)
        for b in self._day_boundaries[1:]:
            line = pg.InfiniteLine(pos=b, angle=90, movable=False,
                                   pen=pg.mkPen(BORDER, width=1, style=Qt.PenStyle.DashLine))
            line._is_day_boundary = True
            self._plot.addItem(line, ignoreBounds=True)

    # ── Overlays ──────────────────────────────────────────────────────────────
    def _add_overlay(self, label: str, y: np.ndarray, color: str | None = None, dash: bool = False) -> Overlay:
        if color is None:
            color = self._next_color()
        ov = Overlay(label, color)

        pen = pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
        x = self._x if self._x is not None else np.arange(len(y))
        curve = self._plot.plot(x, y, pen=pen, name=label)
        ov.items.append(curve)
        ov.plot = self._plot
        self._overlays.append(ov)

        # Row in right panel
        row = OverlayRow(ov, self._toggle_overlay, self._remove_overlay)
        self._overlay_list_layout.addWidget(row)
        self._overlay_rows[ov.id] = row

        self._update_ind_target()
        return ov

    def _add_bb_overlay(self, label: str, upper: np.ndarray, mid: np.ndarray, lower: np.ndarray, color: str) -> Overlay:
        ov = Overlay(label, color)
        x = self._x if self._x is not None else np.arange(len(mid))

        pen_mid = pg.mkPen(color, width=1.5)
        pen_band = pg.mkPen(color, width=0.8, style=Qt.PenStyle.DashLine)

        c_mid = self._plot.plot(x, mid, pen=pen_mid, name=label + ' mid')
        c_up  = self._plot.plot(x, upper, pen=pen_band)
        c_lo  = self._plot.plot(x, lower, pen=pen_band)

        # Fill between
        fill = pg.FillBetweenItem(c_up, c_lo, brush=pg.mkBrush(QColor(color).darker(300)))
        fill.setZValue(-5)
        self._plot.addItem(fill)

        ov.items = [c_mid, c_up, c_lo, fill]
        ov.plot = self._plot
        self._overlays.append(ov)

        row = OverlayRow(ov, self._toggle_overlay, self._remove_overlay)
        self._overlay_list_layout.addWidget(row)
        self._overlay_rows[ov.id] = row

        self._update_ind_target()
        return ov

    def _toggle_overlay(self, ov: Overlay):
        ov.visible = not ov.visible
        for item in ov.items:
            item.setVisible(ov.visible)
        row = self._overlay_rows.get(ov.id)
        if row:
            row.set_visible(ov.visible)

    def _remove_overlay(self, ov: Overlay):
        target = ov.plot or self._plot
        for item in ov.items:
            target.removeItem(item)
        self._overlays.remove(ov)
        row = self._overlay_rows.pop(ov.id, None)
        if row:
            row.setParent(None)
            row.deleteLater()
        self._update_ind_target()

    def _clear_all(self):
        for ov in list(self._overlays):
            target = ov.plot or self._plot
            for item in ov.items:
                target.removeItem(item)
            row = self._overlay_rows.pop(ov.id, None)
            if row:
                row.setParent(None)
                row.deleteLater()
        self._overlays.clear()
        self._draw_boundaries()
        self._update_ind_target()

    def _refresh_all_overlays(self):
        to_remove = list(self._overlays)
        for ov in to_remove:
            target = ov.plot or self._plot
            for item in ov.items:
                target.removeItem(item)
            row = self._overlay_rows.pop(ov.id, None)
            if row:
                row.setParent(None)
                row.deleteLater()
        self._overlays.clear()
        self._draw_boundaries()
        self._update_ind_target()

    def _update_ind_target(self):
        self._ind_target.clear()
        for ov in self._overlays:
            if not ov.label.startswith(('SMA', 'EMA', 'BB')):
                self._ind_target.addItem(ov.label, userData=ov)

    def _get_target_y(self) -> np.ndarray | None:
        ov: Overlay | None = self._ind_target.currentData()
        if ov is None or not ov.items:
            return None
        item = ov.items[0]
        _, y = item.getData()
        return y

    # ── Product + Expr plotting ───────────────────────────────────────────────
    def _add_product_overlay(self, product: str):
        if self._pivot is None or product not in self._pivot.columns:
            self._status.setText(f'  {product} not in loaded data')
            return
        y = self._pivot[product].values.astype(float)
        self._add_overlay(product, y)

    def _plot_expr(self):
        expr = self._expr_input.text().strip()
        if not expr:
            return
        if self._pivot is None:
            self._status.setText('  no data loaded')
            return
        y = eval_expr(expr, self._pivot)
        if y is None:
            self._status.setText(f'  expr error: {expr}')
            return
        self._add_overlay(expr, y)
        self._status.setText(f'  plotted: {expr}')

    # ── Indicators ───────────────────────────────────────────────────────────
    def _add_sma(self):
        y = self._get_target_y()
        if y is None:
            self._status.setText('  no base overlay selected')
            return
        p = self._sma_period.value()
        result = sma(y, p)
        base_label = self._ind_target.currentText()
        self._add_overlay(f'SMA({p}) {base_label}', result, dash=False)

    def _add_ema(self):
        y = self._get_target_y()
        if y is None:
            self._status.setText('  no base overlay selected')
            return
        p = self._ema_period.value()
        result = ema(y, p)
        base_label = self._ind_target.currentText()
        self._add_overlay(f'EMA({p}) {base_label}', result, dash=False)

    def _add_bb(self):
        y = self._get_target_y()
        if y is None:
            self._status.setText('  no base overlay selected')
            return
        p = self._bb_period.value()
        sig = self._bb_sigma.value()
        upper, mid, lower = bollinger(y, p, sig)
        base_label = self._ind_target.currentText()
        color = self._next_color()
        self._add_bb_overlay(f'BB({p},{sig}) {base_label}', upper, mid, lower, color)

    def _add_vol(self):
        y = self._get_target_y()
        if y is None:
            self._status.setText('  no base overlay selected')
            return
        p = self._vol_period.value()
        base_label = self._ind_target.currentText()
        vol = rolling_vol(y, p)
        color = self._next_color()
        x = self._x if self._x is not None else np.arange(len(vol))
        pen = pg.mkPen(color, width=1.5)
        curve = self._vol_plot.plot(x, vol, pen=pen, name=f'VOL({p}) {base_label}')
        # Track in overlay list so clear-all removes it
        ov = Overlay(f'VOL({p}) {base_label}', color)
        ov.items.append(curve)
        ov.plot = self._vol_plot
        self._overlays.append(ov)
        row = OverlayRow(ov, self._toggle_overlay, self._remove_overlay)
        self._overlay_list_layout.addWidget(row)
        self._overlay_rows[ov.id] = row
        self._update_ind_target()
        # Show vol panel if hidden
        if not self._vol_panel.isVisible():
            self._vol_panel.setVisible(True)

    # ── Crosshair ─────────────────────────────────────────────────────────────
    def _on_mouse_move(self, pos):
        in_plot = self._plot.sceneBoundingRect().contains(pos)
        self._vline.setVisible(in_plot)
        self._hline.setVisible(in_plot)
        if not in_plot:
            self._coord_label.setText('')
            return
        pt = self._plot.plotItem.vb.mapSceneToView(pos)
        self._vline.setPos(pt.x())
        self._hline.setPos(pt.y())
        self._coord_label.setPos(pt.x(), pt.y())
        self._coord_label.setText(f'  t={int(pt.x())}  y={pt.y():.4f}')
        # Sync vol crosshair
        self._vol_vline.setPos(pt.x())

    def _on_mouse_move_vol(self, pos):
        in_plot = self._vol_plot.sceneBoundingRect().contains(pos)
        self._vol_vline.setVisible(in_plot)
        if not in_plot:
            return
        pt = self._vol_plot.plotItem.vb.mapSceneToView(pos)
        self._vol_vline.setPos(pt.x())
        # Sync main crosshair x position
        self._vline.setPos(pt.x())


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MidViz()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
