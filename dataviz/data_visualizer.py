import sys
import os
import re
from copy import deepcopy
import polars as pl
import numpy as np
import pyqtgraph as pg

# Lock backend to PyQt6
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QCheckBox, QPushButton)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QShortcut, QKeySequence, QIcon


class ModeZoomViewBox(pg.ViewBox):
    def __init__(self, zoom_mode_getter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_mode_getter = zoom_mode_getter

    def wheelEvent(self, ev, axis=None):
        mode = self.zoom_mode_getter()
        if mode == 'x':
            super().wheelEvent(ev, axis=0)
            return
        if mode == 'y':
            super().wheelEvent(ev, axis=1)
            return
        if mode == 'z':
            try:
                delta = ev.delta()
            except Exception:
                delta = ev.angleDelta().y()

            x_min, x_max = self.viewRange()[0]
            span_x = max(x_max - x_min, 1e-9)
            shift_x = -np.sign(delta) * span_x * 0.08
            self.translateBy(x=shift_x, y=0)
            ev.accept()
            return
        super().wheelEvent(ev, axis=None)

class DataProcessor(QThread):
    data_ready = pyqtSignal(object, object, object)

    DEFAULT_COLUMN_MAP = {
        'prices': {
            'product': 'product',
            'day': 'day',
            'timestamp': 'timestamp',
            'mid_price': 'mid_price',
            'bid_price_1': 'bid_price_1',
            'ask_price_1': 'ask_price_1',
            'bid_volume_1': 'bid_volume_1',
            'ask_volume_1': 'ask_volume_1',
            'bid_price_prefix': 'bid_price_',
            'ask_price_prefix': 'ask_price_',
            'bid_volume_prefix': 'bid_volume_',
            'ask_volume_prefix': 'ask_volume_'
        },
        'trades': {
            'symbol': 'symbol',
            'day': 'day',
            'timestamp': 'timestamp',
            'price': 'price'
        }
    }

    def __init__(self, prices_df, trades_df, product, day_val, column_map=None):
        super().__init__()
        self.prices_df = prices_df
        self.trades_df = trades_df
        self.product = product
        self.day_val = day_val
        self.column_map = deepcopy(self.DEFAULT_COLUMN_MAP)
        if column_map:
            for section, values in column_map.items():
                if section in self.column_map and isinstance(values, dict):
                    self.column_map[section].update(values)

    def pc(self, key):
        return self.column_map['prices'][key]

    def tc(self, key):
        return self.column_map['trades'][key]

    def run(self):
        if self.prices_df is None or len(self.prices_df) == 0:
            self.data_ready.emit({'time': []}, {'time': []}, {'time': [], 'price_levels': []})
            return

        p_filtered = self.prices_df.filter(pl.col(self.pc('product')) == self.product)
        
        if self.trades_df is not None and self.tc('day') in self.trades_df.columns and len(self.trades_df) > 0:
            t_filtered = self.trades_df.filter(pl.col(self.tc('symbol')) == self.product)
        else:
            t_filtered = pl.DataFrame() # Empty fallback

        if self.day_val == "All":
            p_filtered = p_filtered.sort([self.pc('day'), self.pc('timestamp')])
            if len(t_filtered) > 0:
                t_filtered = t_filtered.sort([self.tc('day'), self.tc('timestamp')])

            DAY_LENGTH = 1_000_000
            min_day = p_filtered[self.pc('day')].min()

            p_filtered = p_filtered.with_columns(
                (pl.col(self.pc('timestamp')) + (pl.col(self.pc('day')) - min_day) * DAY_LENGTH).alias("plot_time")
            )
            
            if len(t_filtered) > 0:
                t_filtered = t_filtered.with_columns(
                    (pl.col(self.tc('timestamp')) + (pl.col(self.tc('day')) - min_day) * DAY_LENGTH).alias("plot_time")
                )
        else:
            p_filtered = p_filtered.filter(pl.col(self.pc('day')) == self.day_val).sort(self.pc('timestamp'))
            p_filtered = p_filtered.with_columns(pl.col(self.pc('timestamp')).alias("plot_time"))
            
            if len(t_filtered) > 0:
                t_filtered = t_filtered.filter(pl.col(self.tc('day')) == self.day_val).sort(self.tc('timestamp'))
                t_filtered = t_filtered.with_columns(pl.col(self.tc('timestamp')).alias("plot_time"))

        bid_vol = p_filtered[self.pc('bid_volume_1')].to_numpy()
        ask_vol = p_filtered[self.pc('ask_volume_1')].to_numpy()

        total_vol = bid_vol + ask_vol
        obi = np.divide(bid_vol - ask_vol, total_vol, out=np.zeros_like(bid_vol, dtype=float), where=total_vol!=0)

        p_data = {
            'time': p_filtered['plot_time'].to_numpy(),
            'mid': p_filtered[self.pc('mid_price')].to_numpy(),
            'bid': p_filtered[self.pc('bid_price_1')].to_numpy(),
            'ask': p_filtered[self.pc('ask_price_1')].to_numpy(),
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'obi': obi
        }

        ob_data = self.build_order_book_heatmap_data(p_filtered, p_data['time'])
        
        t_data = {
            'time': t_filtered['plot_time'].to_numpy() if len(t_filtered) > 0 else np.array([]),
            'price': t_filtered[self.tc('price')].to_numpy() if len(t_filtered) > 0 else np.array([])
        }

        self.data_ready.emit(p_data, t_data, ob_data)

    def build_order_book_heatmap_data(self, p_filtered, plot_time):
        if len(p_filtered) == 0:
            return {'time': np.array([]), 'price_levels': np.array([])}

        bid_price_prefix = self.pc('bid_price_prefix')
        ask_price_prefix = self.pc('ask_price_prefix')
        bid_volume_prefix = self.pc('bid_volume_prefix')
        ask_volume_prefix = self.pc('ask_volume_prefix')

        bid_price_cols = sorted(
            [c for c in p_filtered.columns if c.startswith(bid_price_prefix)],
            key=lambda x: int(x.rsplit("_", 1)[1])
        )
        ask_price_cols = sorted(
            [c for c in p_filtered.columns if c.startswith(ask_price_prefix)],
            key=lambda x: int(x.rsplit("_", 1)[1])
        )

        bid_levels = []
        for c in bid_price_cols:
            lvl = c.rsplit("_", 1)[1]
            vcol = f"{bid_volume_prefix}{lvl}"
            if vcol in p_filtered.columns:
                bid_levels.append((c, vcol))

        ask_levels = []
        for c in ask_price_cols:
            lvl = c.rsplit("_", 1)[1]
            vcol = f"{ask_volume_prefix}{lvl}"
            if vcol in p_filtered.columns:
                ask_levels.append((c, vcol))

        if not bid_levels and not ask_levels:
            return {'time': np.array([]), 'price_levels': np.array([])}

        n = len(plot_time)
        x_idx_all = np.arange(n, dtype=np.int32)

        all_price_arrays = []
        all_vol_arrays = []

        for pcol, vcol in bid_levels + ask_levels:
            p_arr = p_filtered[pcol].to_numpy()
            v_arr = p_filtered[vcol].to_numpy()
            valid = np.isfinite(p_arr) & np.isfinite(v_arr) & (p_arr > 0) & (v_arr > 0)
            if np.any(valid):
                all_price_arrays.append(p_arr[valid])
                all_vol_arrays.append(v_arr[valid])

        if not all_price_arrays:
            return {'time': np.array([]), 'price_levels': np.array([])}

        price_levels = np.unique(np.concatenate(all_price_arrays))
        max_vol = np.max(np.concatenate(all_vol_arrays))
        if max_vol <= 0:
            max_vol = 1.0

        def build_side(level_pairs):
            x_parts, y_parts, i_parts, v_parts = [], [], [], []
            for pcol, vcol in level_pairs:
                p_arr = p_filtered[pcol].to_numpy()
                v_arr = p_filtered[vcol].to_numpy()
                valid = np.isfinite(p_arr) & np.isfinite(v_arr) & (p_arr > 0) & (v_arr > 0)
                if not np.any(valid):
                    continue

                x_sel = x_idx_all[valid]
                p_sel = p_arr[valid]
                v_sel = v_arr[valid]

                y_sel = np.searchsorted(price_levels, p_sel).astype(np.int32)
                intensity = np.clip((v_sel / max_vol) * 255.0, 0, 255).astype(np.uint8)

                x_parts.append(x_sel)
                y_parts.append(y_sel)
                i_parts.append(intensity)
                v_parts.append(v_sel)

            if not x_parts:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.uint8), np.array([], dtype=float)

            return (
                np.concatenate(x_parts),
                np.concatenate(y_parts),
                np.concatenate(i_parts),
                np.concatenate(v_parts)
            )

        bid_x, bid_y, bid_i, bid_v = build_side(bid_levels)
        ask_x, ask_y, ask_i, ask_v = build_side(ask_levels)

        return {
            'time': plot_time,
            'price_levels': price_levels,
            'bid_x': bid_x,
            'bid_y': bid_y,
            'bid_i': bid_i,
            'bid_v': bid_v,
            'ask_x': ask_x,
            'ask_y': ask_y,
            'ask_i': ask_i,
            'ask_v': ask_v
        }


class ProsperityVisualizer(QMainWindow):
    def __init__(self, data_dir="."):
        super().__init__()
        self.data_dir = data_dir
        self.rounds_map = self.scan_directory()
        
        self.prices_df = None
        self.trades_df = None
        
        self.spread_fill_ask = None 
        self.spread_fill_bid = None 
        self.curve_ask = None 
        self.curve_bid = None 
        self.obi_curve = None
        self.curve_mid_anchor = None
        self.mid_curve = None
        self.trade_scatter = None
        self.bid_vol_curve = None
        self.ask_vol_curve = None
        self.gpu_enabled = False
        self.orderbook_image_item = None
        self.orderbook_mid_overlay = None
        self.latest_p_data = None
        self.latest_t_data = None
        self.latest_ob_data = None
        self.zoom_mode = 'xy'
        self.zoom_mode_label = None
        self.shortcut_x = None
        self.shortcut_y = None
        self.shortcut_z = None
        self.column_map = deepcopy(DataProcessor.DEFAULT_COLUMN_MAP)
        self.overlay_lines = {'top': {}, 'bottom': {}, 'orderbook': {}}
        self.plot_panels = {}
        
        self.init_ui()

    def scan_directory(self):
        """Scans the current directory for Prosperity format CSVs and groups them by round."""
        rounds = {}
        # Regex to capture Round number and Day number
        price_pattern = re.compile(r"prices_round_(\d+)_day_(-?\d+)\.csv")
        trade_pattern = re.compile(r"trades_round_(\d+)_day_(-?\d+)\.csv")

        for f in os.listdir(self.data_dir):
            if not f.endswith('.csv'): continue
            
            p_match = price_pattern.match(f)
            if p_match:
                rnd = p_match.group(1)
                rounds.setdefault(rnd, {'prices': [], 'trades': []})['prices'].append(f)
                continue
                
            t_match = trade_pattern.match(f)
            if t_match:
                rnd = t_match.group(1)
                rounds.setdefault(rnd, {'prices': [], 'trades': []})['trades'].append(f)
                
        return rounds

    def init_ui(self):
        self.setWindowTitle("Prosperity 4 - Smart Alpha Scanner")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosmic_latte.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(100, 100, 1400, 900)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        controls_layout = QHBoxLayout()
        
        # --- NEW: Round Selector ---
        self.round_combo = QComboBox()
        available_rounds = sorted(list(self.rounds_map.keys()), key=int) if self.rounds_map else ["None"]
        self.round_combo.addItems(available_rounds)
        self.round_combo.currentTextChanged.connect(self.load_round_data)

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.start_processing)
        
        self.day_combo = QComboBox()
        self.day_combo.currentTextChanged.connect(self.start_processing)

        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "View 1: Price + Executions + L1 Spread",
            "View 2: Liquidity + OBI",
            "View 3: Order Book Heatmap"
        ])
        self.view_combo.currentTextChanged.connect(self.apply_view_mode)

        self.spread_checkbox = QCheckBox("Show L1 Spread")
        self.spread_checkbox.setChecked(True)
        self.spread_checkbox.setStyleSheet("color: white; font-weight: bold;")
        self.spread_checkbox.stateChanged.connect(self.toggle_spread)

        self.overlay_mid_checkbox = QCheckBox("Midprice")
        self.overlay_mid_checkbox.setChecked(False)
        self.overlay_mid_checkbox.setStyleSheet("color: white; font-weight: bold;")
        self.overlay_mid_checkbox.stateChanged.connect(self.toggle_orderbook_mid_overlay)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_view)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        controls_layout.addWidget(QLabel("Round:"))
        controls_layout.addWidget(self.round_combo)
        controls_layout.addWidget(QLabel("Product:"))
        controls_layout.addWidget(self.product_combo)
        controls_layout.addWidget(QLabel("Day:"))
        controls_layout.addWidget(self.day_combo)
        controls_layout.addWidget(QLabel("View:"))
        controls_layout.addWidget(self.view_combo)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.spread_checkbox) 
        controls_layout.addWidget(self.overlay_mid_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)
        layout.addLayout(controls_layout)

        # Request an OpenGL context and ask pyqtgraph to use it for rendering.
        gl_format = QSurfaceFormat()
        gl_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        gl_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        gl_format.setSwapInterval(0)
        gl_format.setSamples(0)
        QSurfaceFormat.setDefaultFormat(gl_format)

        pg.setConfigOptions(antialias=False, useOpenGL=True, imageAxisOrder='row-major')
        self.graph_widget = pg.GraphicsLayoutWidget()
        try:
            self.graph_widget.setViewport(QOpenGLWidget())
            self.gpu_enabled = True
            self.status_label.setText("Status: OpenGL renderer enabled")
            self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        except Exception:
            # Fall back to software rendering if OpenGL viewport setup fails.
            pg.setConfigOptions(useOpenGL=False)
            self.gpu_enabled = False
            self.status_label.setText("Status: OpenGL unavailable, using software renderer")
            self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        self.graph_widget.setBackground('#121212')
        layout.addWidget(self.graph_widget)

        footer_layout = QHBoxLayout()
        
        self.tooltip_checkbox = QCheckBox("Enable Hover Tooltip")
        self.tooltip_checkbox.setChecked(False)
        self.tooltip_checkbox.setStyleSheet("color: white; font-weight: bold;")
        footer_layout.addWidget(self.tooltip_checkbox)
        
        footer_layout.addStretch()
        self.zoom_mode_label = QLabel("")
        self.zoom_mode_label.setStyleSheet("color: #777777; font-size: 9pt;")
        footer_layout.addWidget(self.zoom_mode_label)
        layout.addLayout(footer_layout)

        self.shortcut_x = QShortcut(QKeySequence("X"), self)
        self.shortcut_x.activated.connect(lambda: self.set_zoom_mode('xy' if self.zoom_mode == 'x' else 'x'))
        self.shortcut_y = QShortcut(QKeySequence("Y"), self)
        self.shortcut_y.activated.connect(lambda: self.set_zoom_mode('xy' if self.zoom_mode == 'y' else 'y'))
        self.shortcut_z = QShortcut(QKeySequence("Z"), self)
        self.shortcut_z.activated.connect(lambda: self.set_zoom_mode('xy' if self.zoom_mode == 'z' else 'z'))
        self.shortcut_t = QShortcut(QKeySequence("T"), self)
        self.shortcut_t.activated.connect(self.tooltip_checkbox.toggle)

        self.p1 = self.graph_widget.addPlot(
            row=0,
            col=0,
            title="Market Microstructure (Dual-Tone Spread)",
            viewBox=ModeZoomViewBox(self.get_zoom_mode)
        )
        self.p2 = self.graph_widget.addPlot(
            row=1,
            col=0,
            title="Order Book Liquidity & OBI",
            viewBox=ModeZoomViewBox(self.get_zoom_mode)
        )
        self.p3 = self.graph_widget.addPlot(
            row=2,
            col=0,
            title="Order Book Heatmap (Red=Sell, Blue=Buy)",
            viewBox=ModeZoomViewBox(self.get_zoom_mode)
        )
        self.plot_panels = {
            'top': self.p1,
            'bottom': self.p2,
            'orderbook': self.p3
        }
        
        self.p2.setXLink(self.p1)
        self.p3.setXLink(self.p1)
        self.p1.setClipToView(True)
        self.p2.setClipToView(True)
        self.p3.setClipToView(True)
        self.p1.setDownsampling(mode='peak')
        self.p2.setDownsampling(mode='peak')
        self.p3.setDownsampling(mode='peak')
        self.p1.getAxis('left').setTextPen(pg.mkPen('#E0E0E0'))
        self.p1.getAxis('bottom').setTextPen(pg.mkPen('#E0E0E0'))
        self.p2.getAxis('left').setTextPen(pg.mkPen('#E0E0E0'))
        self.p2.getAxis('bottom').setTextPen(pg.mkPen('#E0E0E0'))
        self.p3.getAxis('left').setTextPen(pg.mkPen('#E0E0E0'))
        self.p3.getAxis('bottom').setTextPen(pg.mkPen('#E0E0E0'))
        self.p1.getAxis('left').setPen(pg.mkPen('#E0E0E0'))
        self.p1.getAxis('bottom').setPen(pg.mkPen('#E0E0E0'))
        self.p2.getAxis('left').setPen(pg.mkPen('#E0E0E0'))
        self.p2.getAxis('bottom').setPen(pg.mkPen('#E0E0E0'))
        self.p3.getAxis('left').setPen(pg.mkPen('#E0E0E0'))
        self.p3.getAxis('bottom').setPen(pg.mkPen('#E0E0E0'))
        
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p3.showGrid(x=True, y=True, alpha=0.2)
        self.p3.getViewBox().invertY(False)
        
        self.p1.addLegend()
        self.p2.addLegend()
        self.p3.setLabel('left', 'Price')
        self.p3.setLabel('bottom', 'Timestamp')

        self.init_plot_items()
        self.set_zoom_mode('xy')
        self.apply_view_mode(self.view_combo.currentText())

        # Trigger initial data load
        if self.rounds_map:
            self.load_round_data(self.round_combo.currentText())
        else:
            self.status_label.setText("Status: No CSV files found in directory.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def init_plot_items(self):
        self.mid_curve = self.p1.plot([], [], pen=pg.mkPen('#00BFFF', width=2), name="Mid Price", autoDownsample=True)

        self.curve_ask = pg.PlotCurveItem([], [], pen=pg.mkPen('#FF2D55', width=2), autoDownsample=True)
        self.curve_bid = pg.PlotCurveItem([], [], pen=pg.mkPen('#39FF14', width=2), autoDownsample=True)
        self.curve_mid_anchor = pg.PlotCurveItem([], [])

        # OpenGL can render translucent overlays darker; reduce opacity impact in GPU mode.
        ask_fill_alpha = 35 if self.gpu_enabled else 50
        bid_fill_alpha = 35 if self.gpu_enabled else 50
        self.spread_fill_ask = pg.FillBetweenItem(self.curve_mid_anchor, self.curve_ask, brush=(255, 45, 85, ask_fill_alpha))
        self.spread_fill_bid = pg.FillBetweenItem(self.curve_bid, self.curve_mid_anchor, brush=(57, 255, 20, bid_fill_alpha))

        self.p1.addItem(self.curve_ask)
        self.p1.addItem(self.curve_bid)
        self.p1.addItem(self.spread_fill_ask)
        self.p1.addItem(self.spread_fill_bid)

        self.trade_scatter = pg.ScatterPlotItem(
            x=[], y=[],
            pen=pg.mkPen(None), brush=pg.mkBrush('#FF1493'),
            size=6, symbol='x', pxMode=True, name="Executions"
        )
        self.p1.addItem(self.trade_scatter)

        if self.gpu_enabled:
            self.bid_vol_curve = self.p2.plot([], [], pen=pg.mkPen('#39FF14', width=2),
                                              name='Bid Vol', autoDownsample=True)
            self.ask_vol_curve = self.p2.plot([], [], pen=pg.mkPen('#FF2D55', width=2),
                                              name='Ask Vol', autoDownsample=True)
        else:
            self.bid_vol_curve = self.p2.plot([], [], fillLevel=0, brush=(57, 255, 20, 150), pen='#39FF14',
                                              name='Bid Vol', autoDownsample=True)
            self.ask_vol_curve = self.p2.plot([], [], fillLevel=0, brush=(255, 45, 85, 150), pen='#FF2D55',
                                              name='Ask Vol', autoDownsample=True)

        self.obi_curve = pg.PlotCurveItem([], [], pen=pg.mkPen('#FFD700', width=2), name="OBI (Scaled)", autoDownsample=True)
        self.p2.addItem(self.obi_curve)

        self.orderbook_image_item = pg.ImageItem()
        self.orderbook_image_item.setOpts(axisOrder='row-major')
        self.p3.addItem(self.orderbook_image_item)
        self.orderbook_mid_overlay = pg.PlotCurveItem([], [], pen=pg.mkPen('#00BFFF', width=2), autoDownsample=True)
        self.p3.addItem(self.orderbook_mid_overlay)

        # Tooltip for orderbook heatmap
        self.orderbook_tooltip = pg.TextItem("", anchor=(0, 1), color='w', fill=(0, 0, 0, 200))
        self.p3.addItem(self.orderbook_tooltip)
        self.orderbook_tooltip.setVisible(False)
        self.p3.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def on_mouse_moved(self, evt):
        if not hasattr(self, 'tooltip_checkbox') or not self.tooltip_checkbox.isChecked():
            if hasattr(self, 'orderbook_tooltip'): self.orderbook_tooltip.setVisible(False)
            return

        if getattr(self, 'latest_ob_data', None) is None:
            return
            
        pos = evt
        if self.p3.sceneBoundingRect().contains(pos):
            mouse_point = self.p3.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()
            
            times = self.latest_ob_data.get('time', [])
            price_levels = self.latest_ob_data.get('price_levels', [])
            
            if len(times) == 0 or len(price_levels) == 0:
                if hasattr(self, 'orderbook_tooltip'): self.orderbook_tooltip.setVisible(False)
                return
                
            t_idx = np.searchsorted(times, x_val)
            if t_idx >= len(times): t_idx = len(times) - 1
            if t_idx > 0 and abs(times[t_idx-1] - x_val) < abs(times[t_idx] - x_val):
                t_idx = t_idx - 1
                
            p_idx = np.searchsorted(price_levels, y_val)
            if p_idx >= len(price_levels): p_idx = len(price_levels) - 1
            if p_idx > 0 and abs(price_levels[p_idx-1] - y_val) < abs(price_levels[p_idx] - y_val):
                p_idx = p_idx - 1
                
            exact_time = times[t_idx]
            exact_price = price_levels[p_idx]
            
            ob = self.latest_ob_data
            
            bid_mask = (ob.get('bid_x', np.array([])) == t_idx) & (ob.get('bid_y', np.array([])) == p_idx)
            ask_mask = (ob.get('ask_x', np.array([])) == t_idx) & (ob.get('ask_y', np.array([])) == p_idx)
            
            bid_v = ob.get('bid_v', np.array([]))[bid_mask]
            ask_v = ob.get('ask_v', np.array([]))[ask_mask]
            
            details = f"Time: {exact_time}\nPrice: {exact_price}"
            if len(bid_v) > 0:
                details += f"\nBid Vol: {bid_v[0]}"
            if len(ask_v) > 0:
                details += f"\nAsk Vol: {ask_v[0]}"
                
            self.orderbook_tooltip.setText(details)
            self.orderbook_tooltip.setPos(x_val, y_val)
            self.orderbook_tooltip.setVisible(True)
        else:
            if hasattr(self, 'orderbook_tooltip'): self.orderbook_tooltip.setVisible(False)

    def decimate_points(self, x, y, max_points=5000):
        n = len(x)
        if n <= max_points:
            return x, y
        step = max(1, n // max_points)
        return x[::step], y[::step]

    def get_zoom_mode(self):
        return self.zoom_mode

    def set_zoom_mode(self, mode):
        if mode not in ('xy', 'x', 'y', 'z'):
            return
        self.zoom_mode = mode
        if self.zoom_mode_label is not None:
            if mode == 'xy':
                self.zoom_mode_label.setText("Mode: XY zoom | X=x-zoom Y=y-zoom Z=x-scroll")
            elif mode == 'x':
                self.zoom_mode_label.setText("Mode: X-only zoom | press X again for normal")
            elif mode == 'y':
                self.zoom_mode_label.setText("Mode: Y-only zoom | press Y again for normal")
            else:
                self.zoom_mode_label.setText("Mode: X-scroll (wheel pans time) | press Z again for normal")

    def set_column_map(self, column_map):
        """Update input schema mapping for price/trade columns."""
        for section, values in column_map.items():
            if section in self.column_map and isinstance(values, dict):
                self.column_map[section].update(values)

    def add_overlay_line(self, overlay_id, panel='top', name=None, pen=None, width=2, visible=True):
        """Register a reusable overlay curve on any panel: top, bottom, or orderbook."""
        if panel not in self.plot_panels:
            raise ValueError(f"Unknown panel '{panel}'. Use: top, bottom, orderbook")
        if overlay_id in self.overlay_lines[panel]:
            return self.overlay_lines[panel][overlay_id]

        curve_name = name if name is not None else overlay_id
        curve_pen = pen if pen is not None else '#FFFFFF'
        curve = pg.PlotCurveItem([], [], pen=pg.mkPen(curve_pen, width=width), name=curve_name, autoDownsample=True)
        self.plot_panels[panel].addItem(curve)
        curve.setVisible(visible)
        self.overlay_lines[panel][overlay_id] = curve
        return curve

    def update_overlay_line(self, overlay_id, x_data, y_data, panel='top'):
        if panel not in self.overlay_lines or overlay_id not in self.overlay_lines[panel]:
            raise KeyError(f"Overlay '{overlay_id}' not found on panel '{panel}'.")
        self.overlay_lines[panel][overlay_id].setData(x_data, y_data)

    def clear_overlay_line(self, overlay_id, panel='top'):
        if panel in self.overlay_lines and overlay_id in self.overlay_lines[panel]:
            self.overlay_lines[panel][overlay_id].setData([], [])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_T:
            if hasattr(self, 'tooltip_checkbox'):
                self.tooltip_checkbox.toggle()
            event.accept()
            return
        if event.key() == Qt.Key.Key_X:
            self.set_zoom_mode('xy' if self.zoom_mode == 'x' else 'x')
            event.accept()
            return
        if event.key() == Qt.Key.Key_Y:
            self.set_zoom_mode('xy' if self.zoom_mode == 'y' else 'y')
            event.accept()
            return
        if event.key() == Qt.Key.Key_Z:
            self.set_zoom_mode('xy' if self.zoom_mode == 'z' else 'z')
            event.accept()
            return
        super().keyPressEvent(event)

    def apply_view_mode(self, view_text):
        show_top = view_text.startswith("View 1")
        show_bottom = view_text.startswith("View 2")
        show_orderbook = view_text.startswith("View 3")

        self.p1.setVisible(show_top)
        self.p2.setVisible(show_bottom)
        self.p3.setVisible(show_orderbook)

        self.spread_checkbox.setVisible(show_top)
        self.overlay_mid_checkbox.setVisible(show_orderbook)

        if show_orderbook:
            self.toggle_orderbook_mid_overlay()

        # Force a repaint using the latest loaded data so view switches update immediately.
        if self.latest_p_data is not None:
            self.render_plots(auto_range=True)

    def refresh_view(self):
        if self.prices_df is None or self.product_combo.count() == 0:
            return
        self.start_processing()

    def toggle_orderbook_mid_overlay(self):
        if self.orderbook_mid_overlay is None:
            return
        should_show = self.overlay_mid_checkbox.isChecked() and self.p3.isVisible()
        self.orderbook_mid_overlay.setVisible(should_show)

    def build_orderbook_image(self, ob_data):
        times = ob_data.get('time', np.array([]))
        price_levels = ob_data.get('price_levels', np.array([]))

        if len(times) == 0 or len(price_levels) == 0:
            return None

        w = len(times)
        h = len(price_levels)

        red = np.zeros(h * w, dtype=np.uint8)
        blue = np.zeros(h * w, dtype=np.uint8)

        ask_x = ob_data.get('ask_x', np.array([], dtype=np.int32))
        ask_y = ob_data.get('ask_y', np.array([], dtype=np.int32))
        ask_i = ob_data.get('ask_i', np.array([], dtype=np.uint8))
        if len(ask_x) > 0:
            ask_idx = ask_y.astype(np.int64) * w + ask_x.astype(np.int64)
            np.maximum.at(red, ask_idx, ask_i)

        bid_x = ob_data.get('bid_x', np.array([], dtype=np.int32))
        bid_y = ob_data.get('bid_y', np.array([], dtype=np.int32))
        bid_i = ob_data.get('bid_i', np.array([], dtype=np.uint8))
        if len(bid_x) > 0:
            bid_idx = bid_y.astype(np.int64) * w + bid_x.astype(np.int64)
            np.maximum.at(blue, bid_idx, bid_i)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = red.reshape(h, w)
        img[..., 2] = blue.reshape(h, w)
        return img

    def load_round_data(self, round_str):
        """Loads all CSVs associated with the selected round dynamically."""
        if round_str == "None" or round_str not in self.rounds_map: return
        
        self.status_label.setText(f"Status: Loading Round {round_str} files into memory...")
        self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        QApplication.processEvents() # Force UI to update text immediately

        files = self.rounds_map[round_str]
        
        # Load Prices
        p_dfs = []
        for f in files['prices']:
            filepath = os.path.join(self.data_dir, f)
            p_dfs.append(pl.read_csv(filepath, separator=";"))
        self.prices_df = pl.concat(p_dfs) if p_dfs else None

        # Load Trades and safely inject the "day" column
        t_dfs = []
        for f in files['trades']:
            filepath = os.path.join(self.data_dir, f)
            # Extract day from filename since it's not in the trade CSV
            day_match = re.match(r"trades_round_\d+_day_(-?\d+)\.csv", f)
            if day_match:
                day_val = int(day_match.group(1))
                df = pl.read_csv(filepath, separator=";")
                df = df.with_columns(pl.lit(day_val).alias("day"))
                t_dfs.append(df)
        self.trades_df = pl.concat(t_dfs) if t_dfs else None

        # Update dependent dropdowns (block signals to prevent premature plotting)
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

    def start_processing(self):
        if self.prices_df is None or self.product_combo.count() == 0: return

        self.status_label.setText("Status: Crunching data via Polars...")
        self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        self.round_combo.setEnabled(False)
        self.product_combo.setEnabled(False)
        self.day_combo.setEnabled(False)
        self.view_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)

        product = self.product_combo.currentText()
        day_text = self.day_combo.currentText()
        day_val = "All" if day_text == "All" else int(day_text)

        self.worker = DataProcessor(self.prices_df, self.trades_df, product, day_val, column_map=self.column_map)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.start()

    def on_data_ready(self, p_data, t_data, ob_data):
        self.latest_p_data = p_data
        self.latest_t_data = t_data
        self.latest_ob_data = ob_data

        self.render_plots(auto_range=True)

    def render_plots(self, auto_range=False):
        p_data = self.latest_p_data
        t_data = self.latest_t_data
        ob_data = self.latest_ob_data

        if p_data is None or t_data is None or ob_data is None:
            return

        if len(p_data['time']) == 0:
            self.mid_curve.setData([], [])
            self.curve_ask.setData([], [])
            self.curve_bid.setData([], [])
            self.curve_mid_anchor.setData([], [])
            self.trade_scatter.setData(x=[], y=[])
            self.bid_vol_curve.setData([], [])
            self.ask_vol_curve.setData([], [])
            self.obi_curve.setData([], [])
            self.orderbook_image_item.clear()
            self.orderbook_mid_overlay.setData([], [])
            self.reset_ui("Status: No Data")
            return

        self.mid_curve.setData(p_data['time'], p_data['mid'])
        self.curve_ask.setData(p_data['time'], p_data['ask'])
        self.curve_bid.setData(p_data['time'], p_data['bid'])
        self.curve_mid_anchor.setData(p_data['time'], p_data['mid'])
        
        self.toggle_spread()

        if len(t_data['time']) > 0:
            t_x, t_y = self.decimate_points(t_data['time'], t_data['price'])
            self.trade_scatter.setData(x=t_x, y=t_y)
        else:
            self.trade_scatter.setData(x=[], y=[])

        self.bid_vol_curve.setData(p_data['time'], p_data['bid_vol'])
        self.ask_vol_curve.setData(p_data['time'], -p_data['ask_vol'])

        max_vol_scalar = max(np.max(p_data['bid_vol']), np.max(p_data['ask_vol']))
        if max_vol_scalar == 0: max_vol_scalar = 1 
        scaled_obi = p_data['obi'] * max_vol_scalar

        self.obi_curve.setData(p_data['time'], scaled_obi)

        ob_img = self.build_orderbook_image(ob_data)
        if ob_img is None:
            self.orderbook_image_item.clear()
            self.orderbook_mid_overlay.setData([], [])
        else:
            times = ob_data['time']
            price_levels = ob_data['price_levels']
            self.orderbook_image_item.setImage(ob_img, autoLevels=False)

            x_min = float(times[0])
            x_max = float(times[-1])
            y_min = float(price_levels[0])
            y_max = float(price_levels[-1])

            # Map each image cell to an actual time/price bin size to avoid stretched rectangles.
            x_step = 1.0
            if len(times) > 1:
                x_d = np.diff(times.astype(np.float64))
                x_d = x_d[np.isfinite(x_d) & (x_d > 0)]
                if len(x_d) > 0:
                    x_step = float(np.median(x_d))

            y_step = 1.0
            if len(price_levels) > 1:
                y_d = np.diff(price_levels.astype(np.float64))
                y_d = y_d[np.isfinite(y_d) & (y_d > 0)]
                if len(y_d) > 0:
                    y_step = float(np.median(y_d))

            x0 = x_min - 0.5 * x_step
            y0 = y_min - 0.5 * y_step
            width = max(len(times) * x_step, x_step)
            height = max(len(price_levels) * y_step, y_step)
            self.orderbook_image_item.setRect(QRectF(x0, y0, width, height))

            self.orderbook_mid_overlay.setData(p_data['time'], p_data['mid'])
            self.toggle_orderbook_mid_overlay()

        if auto_range:
            if self.p1.isVisible():
                self.p1.autoRange()
            if self.p2.isVisible():
                self.p2.autoRange()
            if self.p3.isVisible():
                self.p3.autoRange()

        self.reset_ui("Status: Ready")

    def reset_ui(self, status):
        self.status_label.setText(status)
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.round_combo.setEnabled(True)
        self.product_combo.setEnabled(True)
        self.day_combo.setEnabled(True)
        self.view_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosmic_latte.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
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

    # NOTE: Set this to the folder containing your CSV files. 
    # Use "." for the current working directory.
    DATA_DIR = r"F:\PROSPERITY4\Tutorial Round\TUTORIAL_ROUND_1"
    
    # Fallback to current directory if the hardcoded path doesn't exist
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "."

    print(f"Scanning directory: {os.path.abspath(DATA_DIR)}")
    ex = ProsperityVisualizer(data_dir=DATA_DIR)
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()