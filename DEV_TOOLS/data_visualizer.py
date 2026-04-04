import sys
import os
import re
import json
import io
import polars as pl
import numpy as np
import pyqtgraph as pg

# Lock backend to PyQt6
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QCheckBox, QPushButton, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# ==========================================
# ORIGINAL CSV DATA PROCESSOR
# ==========================================
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
        
        obi = np.divide(bid_vol - ask_vol, total_vol, out=np.zeros_like(bid_vol, dtype=float), where=total_vol!=0)
        vwap = np.divide((bid_price * bid_vol) + (ask_price * ask_vol), total_vol, out=np.copy(mid_price), where=total_vol!=0)
        ofi_micro = np.divide((bid_price * ask_vol) + (ask_price * bid_vol), total_vol, out=np.copy(mid_price), where=total_vol!=0)

        p_data = {
            'time': p_filtered['plot_time'].to_numpy(),
            'mid': mid_price,
            'bid': bid_price,
            'ask': ask_price,
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'obi': obi,
            'vwap': vwap,
            'ofi': ofi_micro
        }
        
        t_data = {
            'time': t_filtered['plot_time'].to_numpy() if len(t_filtered) > 0 else np.array([]),
            'price': t_filtered['price'].to_numpy() if len(t_filtered) > 0 else np.array([]),
            'quantity': t_filtered['quantity'].to_numpy() if len(t_filtered) > 0 and 'quantity' in t_filtered.columns else np.array([])
        }

        self.data_ready.emit(p_data, t_data)

# ==========================================
# ROBUST TXT/JSON LOG PROCESSOR
# ==========================================
class LogProcessor(QThread):
    data_ready = pyqtSignal(object, object, float, float)

    def __init__(self, file_path, product=None):
        super().__init__()
        self.file_path = file_path
        self.product = product

    def run(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                clean_json_str = content[start_idx:end_idx]
                data = json.loads(clean_json_str)
            else:
                self.data_ready.emit(None, None, 0, 0)
                return
            
            act_str = data.get("activitiesLog", "")
            if not act_str:
                self.data_ready.emit(None, None, 0, 0)
                return
                
            p_df = pl.read_csv(io.StringIO(act_str), separator=";")
            trade_list = data.get("tradeHistory", [])
            t_df = pl.DataFrame(trade_list) if trade_list else pl.DataFrame()

            products_available = p_df['product'].unique().to_list()
            prod_to_filter = self.product if self.product in products_available else products_available[0]

            p_filtered = p_df.filter(pl.col("product") == prod_to_filter).sort("timestamp")
            
            if len(t_df) > 0 and "symbol" in t_df.columns:
                t_filtered = t_df.filter(pl.col("symbol") == prod_to_filter).sort("timestamp")
            else:
                t_filtered = pl.DataFrame()

            bid_price = p_filtered['bid_price_1'].to_numpy()
            ask_price = p_filtered['ask_price_1'].to_numpy()
            bid_vol = p_filtered['bid_volume_1'].to_numpy()
            ask_vol = p_filtered['ask_volume_1'].to_numpy()
            mid_price = p_filtered['mid_price'].to_numpy()
            p_times = p_filtered['timestamp'].to_numpy()

            p_data = {
                'time': p_times,
                'mid': mid_price,
                'bid': bid_price,
                'ask': ask_price,
                'bid_vol': bid_vol,
                'ask_vol': ask_vol,
            }

            p80, p90 = 0.0, 0.0
            
            if len(t_filtered) > 0:
                q_raw = np.abs(t_filtered['quantity'].to_numpy())
                t_times = t_filtered['timestamp'].to_numpy()
                prices = t_filtered['price'].to_numpy()
                buyers = t_filtered['buyer'].to_numpy()
                sellers = t_filtered['seller'].to_numpy()
                
                if len(q_raw) > 0:
                    p80 = np.percentile(q_raw, 80)
                    p90 = np.percentile(q_raw, 90)

                our_buy = (buyers == "SUBMISSION")
                our_sell = (sellers == "SUBMISSION")
                
                pos_change = np.zeros_like(q_raw, dtype=float)
                pos_change[our_buy] = q_raw[our_buy]
                pos_change[our_sell] = -q_raw[our_sell]

                cash_flow = -pos_change * prices
                cum_pos = np.cumsum(pos_change)
                cum_cash = np.cumsum(cash_flow)

                indices = np.searchsorted(t_times, p_times, side='right') - 1
                current_pos = np.zeros(len(p_times))
                current_cash = np.zeros(len(p_times))
                
                valid = indices >= 0
                current_pos[valid] = cum_pos[indices[valid]]
                current_cash[valid] = cum_cash[indices[valid]]

                pnl = current_cash + current_pos * mid_price

                t_data = {
                    'time': t_times,
                    'price': prices,
                    'quantity': q_raw,
                    'buyer': buyers.tolist(),
                    'seller': sellers.tolist()
                }
            else:
                current_pos = np.zeros(len(p_times))
                pnl = np.zeros(len(p_times))
                t_data = {'time': np.array([]), 'price': np.array([]), 'quantity': np.array([]), 'buyer': [], 'seller': []}

            p_data['pos'] = current_pos
            p_data['pnl'] = pnl

            self.data_ready.emit(p_data, t_data, p80, p90)
            
        except Exception as e:
            print(f"Error parsing Log: {e}")
            self.data_ready.emit(None, None, 0, 0)

# ==========================================
# LOG VISUALIZER WINDOW
# ==========================================
class LogVisualizerWindow(QMainWindow):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.setWindowTitle("Prosperity 4 - Simulation Log Viewer")
        self.setGeometry(150, 150, 1200, 900)
        self.has_plotted_data = False

        self.init_ui()
        self.load_initial_products()

    def init_ui(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        controls_layout = QHBoxLayout()
        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.start_processing)
        
        self.status_label = QLabel(f"Loaded: {os.path.basename(self.file_path)}")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        controls_layout.addWidget(QLabel("Product:"))
        controls_layout.addWidget(self.product_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)
        layout.addLayout(controls_layout)

        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.p1 = self.graph_widget.addPlot(row=0, col=0, title="Execution Log & Tape Inference")
        self.p2 = self.graph_widget.addPlot(row=1, col=0, title="Current Volume Held (Position)")
        self.p3 = self.graph_widget.addPlot(row=2, col=0, title="Cumulative Net PnL")
        
        self.p2.setXLink(self.p1)
        self.p3.setXLink(self.p1)

        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p3.showGrid(x=True, y=True, alpha=0.3)

        self.p1.addLegend()

        crosshair_pen = pg.mkPen(color=(180, 180, 180, 150), width=1.5, style=Qt.PenStyle.DashLine)
        
        self.vLine1 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.hLine1 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.vLine2 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.hLine2 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.vLine3 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.hLine3 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        
        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        self.p3.addItem(self.vLine3, ignoreBounds=True)
        self.p3.addItem(self.hLine3, ignoreBounds=True)

        self.label_p1 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        self.label_p2 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        self.label_p3 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        self.p3.addItem(self.label_p3, ignoreBounds=True)
        
        self.hide_crosshairs()
        self.graph_widget.scene().sigMouseMoved.connect(self.mouse_moved)

    def load_initial_products(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                data = json.loads(content[start_idx:end_idx])
                act_str = data.get("activitiesLog", "")
                if act_str:
                    p_df = pl.read_csv(io.StringIO(act_str), separator=";")
                    products = p_df['product'].unique().sort().to_list()
                    self.product_combo.blockSignals(True)
                    self.product_combo.addItems(products)
                    self.product_combo.blockSignals(False)
                    self.start_processing()
        except Exception as e:
            self.status_label.setText("Failed to read Log File")

    def hide_crosshairs(self):
        self.vLine1.hide()
        self.hLine1.hide()
        self.vLine2.hide()
        self.hLine2.hide()
        self.vLine3.hide()
        self.hLine3.hide()
        self.label_p1.hide()
        self.label_p2.hide()
        self.label_p3.hide()

    def mouse_moved(self, evt):
        pos = evt
        if not self.has_plotted_data: return
        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = self.p1.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine1.setPos(x)
            self.hLine1.setPos(y)
            self.vLine2.setPos(x)
            self.vLine3.setPos(x)
            self.vLine1.show()
            self.hLine1.show()
            self.vLine2.show()
            self.vLine3.show()
            self.hLine2.hide()
            self.hLine3.hide()
            html_str = f"<div style='text-align: left;'><span style='color: white;'>Time: </span><b style='color: #00BFFF;'>{int(x)}</b><br><span style='color: white;'>Price: </span><b style='color: #00FF00;'>{y:.2f}</b></div>"
            self.label_p1.setHtml(html_str)
            self.label_p1.setPos(x, y)
            self.label_p1.show()
            self.label_p2.hide()
            self.label_p3.hide()
        elif self.p2.sceneBoundingRect().contains(pos):
            mousePoint = self.p2.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine1.setPos(x)
            self.vLine2.setPos(x)
            self.hLine2.setPos(y)
            self.vLine3.setPos(x)
            self.vLine1.show()
            self.vLine2.show()
            self.hLine2.show()
            self.vLine3.show()
            self.hLine1.hide()
            self.hLine3.hide()
            html_str = f"<div style='text-align: left;'><span style='color: white;'>Time: </span><b style='color: #00BFFF;'>{int(x)}</b><br><span style='color: white;'>Pos: </span><b style='color: #FFA500;'>{y:.0f}</b></div>"
            self.label_p2.setHtml(html_str)
            self.label_p2.setPos(x, y)
            self.label_p2.show()
            self.label_p1.hide()
            self.label_p3.hide()
        elif self.p3.sceneBoundingRect().contains(pos):
            mousePoint = self.p3.vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            self.vLine1.setPos(x)
            self.vLine2.setPos(x)
            self.vLine3.setPos(x)
            self.hLine3.setPos(y)
            self.vLine1.show()
            self.vLine2.show()
            self.vLine3.show()
            self.hLine3.show()
            self.hLine1.hide()
            self.hLine2.hide()
            html_str = f"<div style='text-align: left;'><span style='color: white;'>Time: </span><b style='color: #00BFFF;'>{int(x)}</b><br><span style='color: white;'>PnL: </span><b style='color: #00FF00;'>{y:.2f}</b></div>"
            self.label_p3.setHtml(html_str)
            self.label_p3.setPos(x, y)
            self.label_p3.show()
            self.label_p1.hide()
            self.label_p2.hide()
        else:
            self.hide_crosshairs()

    def start_processing(self):
        prod = self.product_combo.currentText()
        if not prod: return
        self.status_label.setText("Parsing Log Data...")
        self.worker = LogProcessor(self.file_path, prod)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.start()

    def on_data_ready(self, p_data, t_data, p80, p90):
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        
        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        self.p3.addItem(self.vLine3, ignoreBounds=True)
        self.p3.addItem(self.hLine3, ignoreBounds=True)
        self.hide_crosshairs()

        if not p_data or len(p_data['time']) == 0:
            self.status_label.setText("No data found for product.")
            return

        self.p1.plot(p_data['time'], p_data['mid'], pen=pg.mkPen('#00BFFF', width=2), name="Mid Price")
        curve_ask = pg.PlotCurveItem(p_data['time'], p_data['ask'], pen=pg.mkPen('#DC143C', width=1)) 
        curve_bid = pg.PlotCurveItem(p_data['time'], p_data['bid'], pen=pg.mkPen('#00FF00', width=1)) 
        curve_mid = pg.PlotCurveItem(p_data['time'], p_data['mid'])
        
        self.p1.addItem(curve_ask)
        self.p1.addItem(curve_bid)
        self.p1.addItem(pg.FillBetweenItem(curve_mid, curve_ask, brush=(220, 20, 60, 50)))
        self.p1.addItem(pg.FillBetweenItem(curve_bid, curve_mid, brush=(0, 255, 0, 50)))

        self.p2.plot(p_data['time'], p_data['pos'], pen=pg.mkPen('#FFA500', width=2), name="Position", fillLevel=0, brush=(255, 165, 0, 80))
        self.p3.plot(p_data['time'], p_data['pnl'], pen=pg.mkPen('#00FF00', width=2), name="PnL", fillLevel=0, brush=(0, 255, 0, 80))

        if len(t_data['time']) > 0:
            vol_ranges = [
                {"name": f"Top 10% (Vol >= {p90:.0f})", "mask": t_data['quantity'] >= p90, "color": '#FFD700'},
                {"name": f"Next 10% ({p80:.0f} - {p90:.0f})", "mask": (t_data['quantity'] >= p80) & (t_data['quantity'] < p90), "color": '#00BFFF'},
                {"name": f"Normal (Vol < {p80:.0f})", "mask": t_data['quantity'] < p80, "color": '#888888'}
            ]

            buyer_arr = np.array(t_data['buyer'])
            seller_arr = np.array(t_data['seller'])
            
            mask_our_buy = buyer_arr == "SUBMISSION"
            mask_our_sell = seller_arr == "SUBMISSION"
            mask_bot_only = (buyer_arr == "") & (seller_arr == "")

            trade_types = [
                {"name": "Our Buy", "mask": mask_our_buy, "symbol": 't1'},
                {"name": "Our Sell", "mask": mask_our_sell, "symbol": 't'},
                {"name": "Bot Trade", "mask": mask_bot_only, "symbol": 'o'}
            ]

            for t_type in trade_types:
                for v_rng in vol_ranges:
                    combined_mask = t_type["mask"] & v_rng["mask"]
                    if np.any(combined_mask):
                        scatter = pg.ScatterPlotItem(
                            x=t_data['time'][combined_mask], 
                            y=t_data['price'][combined_mask],
                            pen=pg.mkPen(v_rng["color"], width=1), 
                            brush=pg.mkBrush(v_rng["color"]), 
                            size=12, symbol=t_type["symbol"], 
                            name=f"{t_type['name']} | {v_rng['name']}", 
                            pxMode=True
                        )
                        self.p1.addItem(scatter)

        self.p1.autoRange()
        self.p2.autoRange()
        self.p3.autoRange()

        self.has_plotted_data = True
        self.status_label.setText(f"Status: Ready | P80={p80:.1f}, P90={p90:.1f}")

# ==========================================
# MAIN VISUALIZER (MODIFIED)
# ==========================================
class ProsperityVisualizer(QMainWindow):
    def __init__(self, data_dir="."):
        super().__init__()
        self.data_dir = data_dir
        self.rounds_map = self.scan_directory()
        
        self.prices_df = None
        self.trades_df = None
        self.has_plotted_data = False
        self.log_window = None 
        
        self.spread_fill_ask = None 
        self.spread_fill_bid = None 
        self.curve_ask = None 
        self.curve_bid = None 
        self.obi_curve = None
        self.curve_vwap = None
        self.curve_ofi = None
        
        self.init_ui()

    def scan_directory(self):
        rounds = {}
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

        self.spread_checkbox = QCheckBox("L1 Spread")
        self.spread_checkbox.setChecked(True)
        self.spread_checkbox.setStyleSheet("color: white; font-weight: bold;")
        self.spread_checkbox.stateChanged.connect(self.toggle_spread)

        self.vwap_checkbox = QCheckBox("VWAP")
        self.vwap_checkbox.setChecked(False)
        self.vwap_checkbox.setStyleSheet("color: #FF00FF; font-weight: bold;") 
        self.vwap_checkbox.stateChanged.connect(self.toggle_indicators)

        self.ofi_checkbox = QCheckBox("OFI (Microprice)")
        self.ofi_checkbox.setChecked(False)
        self.ofi_checkbox.setStyleSheet("color: #FF8C00; font-weight: bold;") 
        self.ofi_checkbox.stateChanged.connect(self.toggle_indicators)

        self.log_btn = QPushButton("Open Simulation Log (.txt)")
        self.log_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; border-radius: 4px; padding: 4px 10px;")
        self.log_btn.clicked.connect(self.open_log_window)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        controls_layout.addWidget(QLabel("Round:"))
        controls_layout.addWidget(self.round_combo)
        controls_layout.addWidget(QLabel("Product:"))
        controls_layout.addWidget(self.product_combo)
        controls_layout.addWidget(QLabel("Day:"))
        controls_layout.addWidget(self.day_combo)
        controls_layout.addWidget(self.spread_checkbox) 
        controls_layout.addWidget(self.vwap_checkbox)
        controls_layout.addWidget(self.ofi_checkbox)
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.log_btn)
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
        
        self.p1.addLegend()
        self.p2.addLegend()

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

    def open_log_window(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Submission Log File", 
            self.data_dir, 
            "Log Files (*.txt *.log *.json);;All Files (*)"
        )
        if filepath:
            self.log_window = LogVisualizerWindow(filepath)
            self.log_window.show()

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
            
            html_str = f"<div style='text-align: left;'><span style='color: #FFFFFF; font-size: 11pt;'>Time: </span><b style='color: #00BFFF; font-size: 11pt;'>{int(x)}</b><br><span style='color: #FFFFFF; font-size: 11pt;'>Price: </span><b style='color: #00FF00; font-size: 11pt;'>{y:.2f}</b></div>"
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
            
            html_str = f"<div style='text-align: left;'><span style='color: #FFFFFF; font-size: 11pt;'>Time: </span><b style='color: #00BFFF; font-size: 11pt;'>{int(x)}</b><br><span style='color: #FFFFFF; font-size: 11pt;'>Level: </span><b style='color: #FFD700; font-size: 11pt;'>{y:.2f}</b></div>"
            self.label_p2.setHtml(html_str)
            self.label_p2.setPos(x, y)
            self.label_p2.show()
            self.label_p1.hide()
        else:
            self.hide_crosshairs()

    def load_round_data(self, round_str):
        if round_str == "None" or round_str not in self.rounds_map: return
        
        self.status_label.setText(f"Status: Loading Round {round_str} files...")
        QApplication.processEvents()

        files = self.rounds_map[round_str]
        p_dfs = [pl.read_csv(os.path.join(self.data_dir, f), separator=";") for f in files['prices']]
        self.prices_df = pl.concat(p_dfs) if p_dfs else None

        t_dfs = []
        for f in files['trades']:
            filepath = os.path.join(self.data_dir, f)
            day_match = re.match(r"trades_round_\d+_day_(-?\d+)\.csv", f)
            if day_match:
                df = pl.read_csv(filepath, separator=";").with_columns(pl.lit(int(day_match.group(1))).alias("day"))
                t_dfs.append(df)
        self.trades_df = pl.concat(t_dfs) if t_dfs else None

        self.product_combo.blockSignals(True)
        self.day_combo.blockSignals(True)
        self.product_combo.clear()
        self.day_combo.clear()

        if self.prices_df is not None:
            self.product_combo.addItems(self.prices_df['product'].unique().sort().to_list())
            self.day_combo.addItems(["All"] + [str(d) for d in self.prices_df['day'].unique().sort().to_list()])

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
        if self.curve_ofi: self.curve_ofi.setVisible(self.ofi_checkbox.isChecked())

    def start_processing(self):
        if self.prices_df is None or self.product_combo.count() == 0: return

        self.status_label.setText("Status: Crunching data via Polars...")
        self.has_plotted_data = False
        self.hide_crosshairs()

        day_text = self.day_combo.currentText()
        day_val = "All" if day_text == "All" else int(day_text)

        self.worker = DataProcessor(self.prices_df, self.trades_df, self.product_combo.currentText(), day_val)
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

        self.p1.plot(p_data['time'], p_data['mid'], pen=pg.mkPen('#00BFFF', width=2), name="Mid Price", autoDownsample=True)

        self.curve_ask = pg.PlotCurveItem(p_data['time'], p_data['ask'], pen=pg.mkPen('#DC143C', width=1)) 
        self.curve_bid = pg.PlotCurveItem(p_data['time'], p_data['bid'], pen=pg.mkPen('#00FF00', width=1)) 
        self.curve_vwap = pg.PlotCurveItem(p_data['time'], p_data['vwap'], pen=pg.mkPen('#FF00FF', width=2, style=Qt.PenStyle.DashLine))
        self.curve_ofi = pg.PlotCurveItem(p_data['time'], p_data['ofi'], pen=pg.mkPen('#FF8C00', width=2, style=Qt.PenStyle.DotLine))
        
        curve_mid_anchor = pg.PlotCurveItem(p_data['time'], p_data['mid'])
        self.spread_fill_ask = pg.FillBetweenItem(curve_mid_anchor, self.curve_ask, brush=(220, 20, 60, 50)) 
        self.spread_fill_bid = pg.FillBetweenItem(self.curve_bid, curve_mid_anchor, brush=(0, 255, 0, 50))

        self.p1.addItem(self.curve_ask)
        self.p1.addItem(self.curve_bid)
        self.p1.addItem(self.spread_fill_ask)
        self.p1.addItem(self.spread_fill_bid)
        self.p1.addItem(self.curve_vwap)
        self.p1.addItem(self.curve_ofi)
        
        self.toggle_spread()
        self.toggle_indicators()

        if len(t_data['time']) > 0 and len(t_data['quantity']) > 0:
            q = np.abs(t_data['quantity']) 
            
            mask_blue = (q >= 1) & (q <= 3)
            mask_yellow = (q == 4)
            mask_red = (q >= 5)

            if np.any(mask_blue):
                self.p1.addItem(pg.ScatterPlotItem(x=t_data['time'][mask_blue], y=t_data['price'][mask_blue], pen=pg.mkPen('#00BFFF', width=2), brush=pg.mkBrush(None), size=10, symbol='x', name="Exec Vol 1-3", pxMode=True))
            if np.any(mask_yellow):
                self.p1.addItem(pg.ScatterPlotItem(x=t_data['time'][mask_yellow], y=t_data['price'][mask_yellow], pen=pg.mkPen('#FFFF00', width=2), brush=pg.mkBrush(None), size=10, symbol='x', name="Exec Vol 4", pxMode=True))
            if np.any(mask_red):
                self.p1.addItem(pg.ScatterPlotItem(x=t_data['time'][mask_red], y=t_data['price'][mask_red], pen=pg.mkPen('#FF0000', width=2), brush=pg.mkBrush(None), size=10, symbol='x', name="Exec Vol 5+", pxMode=True))

        self.p2.plot(p_data['time'], p_data['bid_vol'], fillLevel=0, brush=(0, 255, 0, 150), pen='#00FF00', name='Bid Vol', autoDownsample=True)
        self.p2.plot(p_data['time'], -p_data['ask_vol'], fillLevel=0, brush=(220, 20, 60, 150), pen='#DC143C', name='Ask Vol', autoDownsample=True)

        max_vol_scalar = max(np.max(p_data['bid_vol']), np.max(p_data['ask_vol']))
        if max_vol_scalar == 0: max_vol_scalar = 1 
        self.obi_curve = pg.PlotCurveItem(p_data['time'], p_data['obi'] * max_vol_scalar, pen=pg.mkPen('#FFD700', width=2), name="OBI (Scaled)", autoDownsample=True)
        self.p2.addItem(self.obi_curve)

        self.p1.autoRange()
        self.p2.autoRange()
        
        self.has_plotted_data = True
        self.reset_ui("Status: Ready")

    def reset_ui(self, status):
        self.status_label.setText(status)

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
        QPushButton:hover { background-color: #45a049; }
    """)

    DATA_DIR = r"F:\PROSPERITY4\Tutorial Round\TUTORIAL_ROUND_1"
    
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "."

    ex = ProsperityVisualizer(data_dir=DATA_DIR)
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
