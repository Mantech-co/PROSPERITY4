import sys
import os
import re
import polars as pl
import numpy as np
import pyqtgraph as pg

# Lock backend to PyQt6
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

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

        bid_vol = p_filtered['bid_volume_1'].to_numpy()
        ask_vol = p_filtered['ask_volume_1'].to_numpy()

        total_vol = bid_vol + ask_vol
        obi = np.divide(bid_vol - ask_vol, total_vol, out=np.zeros_like(bid_vol, dtype=float), where=total_vol!=0)

        p_data = {
            'time': p_filtered['plot_time'].to_numpy(),
            'mid': p_filtered['mid_price'].to_numpy(),
            'bid': p_filtered['bid_price_1'].to_numpy(),
            'ask': p_filtered['ask_price_1'].to_numpy(),
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'obi': obi
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
        
        self.scatter_blue = None
        self.scatter_yellow = None
        self.scatter_red = None
        
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

        self.exec_blue_cb = QCheckBox("Vol 1-3")
        self.exec_blue_cb.setChecked(True)
        self.exec_blue_cb.setStyleSheet("color: #00BFFF; font-weight: bold;")
        self.exec_blue_cb.stateChanged.connect(self.toggle_executions)

        self.exec_yellow_cb = QCheckBox("Vol 4")
        self.exec_yellow_cb.setChecked(True)
        self.exec_yellow_cb.setStyleSheet("color: #FFFF00; font-weight: bold;")
        self.exec_yellow_cb.stateChanged.connect(self.toggle_executions)

        self.exec_red_cb = QCheckBox("Vol 5+")
        self.exec_red_cb.setChecked(True)
        self.exec_red_cb.setStyleSheet("color: #FF0000; font-weight: bold;")
        self.exec_red_cb.stateChanged.connect(self.toggle_executions)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")

        controls_layout.addWidget(QLabel("Round:"))
        controls_layout.addWidget(self.round_combo)
        controls_layout.addWidget(QLabel("Product:"))
        controls_layout.addWidget(self.product_combo)
        controls_layout.addWidget(QLabel("Day:"))
        controls_layout.addWidget(self.day_combo)
        controls_layout.addWidget(self.spread_checkbox) 
        
        controls_layout.addWidget(self.exec_blue_cb)
        controls_layout.addWidget(self.exec_yellow_cb)
        controls_layout.addWidget(self.exec_red_cb)
        
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)
        layout.addLayout(controls_layout)

        pg.setConfigOptions(antialias=False)
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.p1 = self.graph_widget.addPlot(row=0, col=0, title="Market Microstructure (Dual-Tone Spread)")
        self.p2 = self.graph_widget.addPlot(row=1, col=0, title="Order Book Liquidity & OBI")
        
        self.p2.setXLink(self.p1)
        
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        
        self.p1.addLegend()
        self.p2.addLegend()

        # ==========================================
        # CROSSHAIR & DYNAMIC LABEL INITIALIZATION
        # ==========================================
        
        crosshair_pen = pg.mkPen(color=(180, 180, 180, 150), width=1.5, style=Qt.PenStyle.DashLine)
        
        # Vertical lines (Time)
        self.vLine1 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.vLine2 = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        
        # Horizontal lines (Price / Volume)
        self.hLine1 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.hLine2 = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)

        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        
        # Pop-out Text Labels attached to cursor
        self.label_p1 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        self.label_p2 = pg.TextItem(anchor=(-0.1, 1.1), fill=pg.mkBrush(0, 0, 0, 200), border=pg.mkPen(100, 100, 100))
        
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        
        self.hide_crosshairs()

        # Connect mouse movement
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
        """Triggered when mouse moves over the graph widget."""
        pos = evt
        if not self.has_plotted_data:
            return

        # Check if mouse is inside Plot 1
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
            
            # Formatted HTML pop-out box
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
            
        # Check if mouse is inside Plot 2
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

    def load_round_data(self, round_str):
        if round_str == "None" or round_str not in self.rounds_map: return
        
        self.status_label.setText(f"Status: Loading Round {round_str} files into memory...")
        self.status_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        QApplication.processEvents()

        files = self.rounds_map[round_str]
        
        p_dfs = []
        for f in files['prices']:
            filepath = os.path.join(self.data_dir, f)
            p_dfs.append(pl.read_csv(filepath, separator=";"))
        self.prices_df = pl.concat(p_dfs) if p_dfs else None

        t_dfs = []
        for f in files['trades']:
            filepath = os.path.join(self.data_dir, f)
            day_match = re.match(r"trades_round_\d+_day_(-?\d+)\.csv", f)
            if day_match:
                day_val = int(day_match.group(1))
                df = pl.read_csv(filepath, separator=";")
                df = df.with_columns(pl.lit(day_val).alias("day"))
                t_dfs.append(df)
        self.trades_df = pl.concat(t_dfs) if t_dfs else None

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

    def toggle_executions(self):
        if self.scatter_blue: self.scatter_blue.setVisible(self.exec_blue_cb.isChecked())
        if self.scatter_yellow: self.scatter_yellow.setVisible(self.exec_yellow_cb.isChecked())
        if self.scatter_red: self.scatter_red.setVisible(self.exec_red_cb.isChecked())

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
        
        # Re-add crosshair items since clear() removes everything
        self.p1.addItem(self.vLine1, ignoreBounds=True)
        self.p1.addItem(self.hLine1, ignoreBounds=True)
        self.p1.addItem(self.label_p1, ignoreBounds=True)
        
        self.p2.addItem(self.vLine2, ignoreBounds=True)
        self.p2.addItem(self.hLine2, ignoreBounds=True)
        self.p2.addItem(self.label_p2, ignoreBounds=True)
        
        self.hide_crosshairs()

        self.scatter_blue = None
        self.scatter_yellow = None
        self.scatter_red = None

        if len(p_data['time']) == 0:
            self.reset_ui("Status: No Data")
            return

        self.p1.plot(p_data['time'], p_data['mid'], pen=pg.mkPen('#00BFFF', width=2), 
                     name="Mid Price", autoDownsample=True)

        self.curve_ask = pg.PlotCurveItem(p_data['time'], p_data['ask'], pen=pg.mkPen('#DC143C', width=1)) 
        self.curve_bid = pg.PlotCurveItem(p_data['time'], p_data['bid'], pen=pg.mkPen('#00FF00', width=1)) 
        
        curve_mid_anchor = pg.PlotCurveItem(p_data['time'], p_data['mid'])

        self.spread_fill_ask = pg.FillBetweenItem(curve_mid_anchor, self.curve_ask, brush=(220, 20, 60, 50)) 
        self.spread_fill_bid = pg.FillBetweenItem(self.curve_bid, curve_mid_anchor, brush=(0, 255, 0, 50))

        self.p1.addItem(self.curve_ask)
        self.p1.addItem(self.curve_bid)
        self.p1.addItem(self.spread_fill_ask)
        self.p1.addItem(self.spread_fill_bid)
        
        self.toggle_spread()

        if len(t_data['time']) > 0 and len(t_data['quantity']) > 0:
            q = np.abs(t_data['quantity']) 
            
            mask_blue = (q >= 1) & (q <= 3)
            mask_yellow = (q == 4)
            mask_red = (q >= 5)

            if np.any(mask_blue):
                self.scatter_blue = pg.ScatterPlotItem(
                    x=t_data['time'][mask_blue], y=t_data['price'][mask_blue], 
                    pen=pg.mkPen('#00BFFF', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 1-3", pxMode=True
                )
                self.p1.addItem(self.scatter_blue)

            if np.any(mask_yellow):
                self.scatter_yellow = pg.ScatterPlotItem(
                    x=t_data['time'][mask_yellow], y=t_data['price'][mask_yellow], 
                    pen=pg.mkPen('#FFFF00', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 4", pxMode=True
                )
                self.p1.addItem(self.scatter_yellow)

            if np.any(mask_red):
                self.scatter_red = pg.ScatterPlotItem(
                    x=t_data['time'][mask_red], y=t_data['price'][mask_red], 
                    pen=pg.mkPen('#FF0000', width=2), brush=pg.mkBrush(None), 
                    size=10, symbol='x', name="Exec Vol 5+", pxMode=True
                )
                self.p1.addItem(self.scatter_red)

            self.toggle_executions()

        self.p2.plot(p_data['time'], p_data['bid_vol'], fillLevel=0, 
                     brush=(0, 255, 0, 150), pen='#00FF00', name='Bid Vol', 
                     autoDownsample=True)
                     
        self.p2.plot(p_data['time'], -p_data['ask_vol'], fillLevel=0, 
                     brush=(220, 20, 60, 150), pen='#DC143C', name='Ask Vol', 
                     autoDownsample=True)

        max_vol_scalar = max(np.max(p_data['bid_vol']), np.max(p_data['ask_vol']))
        if max_vol_scalar == 0: max_vol_scalar = 1 
        scaled_obi = p_data['obi'] * max_vol_scalar

        self.obi_curve = pg.PlotCurveItem(
            p_data['time'], scaled_obi, 
            pen=pg.mkPen('#FFD700', width=2), 
            name="OBI (Scaled)", autoDownsample=True
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
