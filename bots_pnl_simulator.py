import sys, os, re, json
import pandas as pd
import numpy as np
import pyqtgraph as pg
from io import StringIO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

# --- Styling & Colors ---
BG = '#0d0f14'
PANEL_BG = '#12151c'
BORDER = '#1e2330'
TEXT = '#c8d0e0'
DIM = '#4a5068'
ACCENT_CYAN = '#00d4ff'

FIXED_BOT_COLORS = {
    'Mark 01': '#ff6b6b',
    'Mark 14': '#ffd166',
    'Mark 22': '#06d6a0',
    'Mark 38': '#118ab2',
    'Mark 49': '#ef476f',
    'Mark 55': '#b06dff',
    'Mark 67': '#ff9f43',
}
ACCENT_GOLD = '#ffd700'
ACCENT_ORANGE = '#ff9f43'

APP_STYLE = f"""
QMainWindow, QWidget {{ 
    background-color: {BG}; 
    color: {TEXT}; 
    font-family: 'JetBrains Mono', 'Consolas', monospace; 
    font-size: 9pt; 
}}
QComboBox, QPushButton {{ 
    background: {PANEL_BG}; 
    border: 1px solid {BORDER}; 
    padding: 4px 10px; 
    border-radius: 3px; 
    color: {TEXT}; 
    min-width: 120px; 
}}
QPushButton:hover {{
    border-color: {ACCENT_CYAN};
}}
QLabel {{
    font-weight: bold;
    color: {ACCENT_CYAN};
}}
QTableWidget {{
    background-color: {BG};
    color: {TEXT};
    gridline-color: {BORDER};
    border: none;
}}
QTableWidget::item {{
    padding: 4px;
    border-bottom: 1px solid {BORDER};
}}
QHeaderView::section {{
    background-color: {PANEL_BG};
    color: {ACCENT_CYAN};
    border: 1px solid {BORDER};
    padding: 6px;
    font-weight: bold;
}}
"""

class BotSimEngine(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bot Simulation Engine: Advanced Analytics")
        self.setGeometry(100, 100, 1400, 900)
        pg.setConfigOptions(useOpenGL=True, antialias=True)
        
        self.prices_df = pd.DataFrame()
        self.legs_df = pd.DataFrame()
        self.bot_colors = {}
        self.markup_lines = []
        self.current_stats = {}
        
        self._build_ui()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("A"), self).activated.connect(self._autoscale_all)
        QShortcut(QKeySequence("C"), self).activated.connect(self._clear_markup)
        QShortcut(QKeySequence("D"), self).activated.connect(self._show_data_pane)
        QShortcut(QKeySequence("X"), self).activated.connect(lambda: self._set_zoom('x'))
        QShortcut(QKeySequence("Y"), self).activated.connect(lambda: self._set_zoom('y'))
        QShortcut(QKeySequence("Z"), self).activated.connect(lambda: self._set_zoom('xy'))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        controls = QHBoxLayout()
        
        btn_import = QPushButton("📂 Import Data")
        btn_import.clicked.connect(self._import_data)
        controls.addWidget(btn_import)
        
        controls.addWidget(QLabel("Day:"))
        self.cb_day = QComboBox()
        self.cb_day.currentTextChanged.connect(self._update_plots)
        controls.addWidget(self.cb_day)
        
        controls.addWidget(QLabel("Product:"))
        self.cb_prod = QComboBox()
        self.cb_prod.currentTextChanged.connect(self._update_plots)
        controls.addWidget(self.cb_prod)
        
        controls.addWidget(QLabel("Bot:"))
        self.cb_bot = QComboBox()
        self.cb_bot.currentTextChanged.connect(self._update_plots)
        controls.addWidget(self.cb_bot)
        
        btn_data = QPushButton("📊 View Stats [D]")
        btn_data.clicked.connect(self._show_data_pane)
        controls.addWidget(btn_data)
        
        controls.addStretch()
        
        self.lbl_info = QLabel("Use [A] AutoScale | [C] Clear | [X/Y/Z] Zoom Modes")
        self.lbl_info.setStyleSheet(f"color: {DIM}; font-weight: normal;")
        controls.addWidget(self.lbl_info)
        
        main_layout.addLayout(controls)
        
        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground(BG)
        main_layout.addWidget(self.gw, stretch=1)
        
        self.p_pnl = self.gw.addPlot(row=0, col=0, title="Net PnL")
        self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.p_pnl.addLegend(offset=(10, 10))
        
        self.p_pos = self.gw.addPlot(row=1, col=0, title="Total Position")
        self.p_pos.showGrid(x=True, y=True, alpha=0.3)
        self.p_pos.setXLink(self.p_pnl)
        
        self.p_pnl.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.p_pos.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def _import_data(self):
        data_dir = QFileDialog.getExistingDirectory(self, "Select Data Directory containing CSVs")
        if not data_dir: return
        
        p_dfs, t_dfs = [], []
        for f in os.listdir(data_dir):
            if not f.endswith('.csv'): continue
            path = os.path.join(data_dir, f)
            
            try:
                if f.startswith("prices_"):
                    df = pd.read_csv(path, sep=';')
                    p_dfs.append(df)
                elif f.startswith("trades_"):
                    df = pd.read_csv(path, sep=';')
                    day_match = re.search(r"day_(-?\d+)", f)
                    if day_match:
                        df['day'] = int(day_match.group(1))
                    t_dfs.append(df)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                
        if p_dfs:
            self.prices_df = pd.concat(p_dfs, ignore_index=True)
            self.prices_df['mid_price'] = self.prices_df['mid_price'].fillna(
                (self.prices_df['bid_price_1'] + self.prices_df['ask_price_1']) / 2
            )
            
        if t_dfs:
            trades_df = pd.concat(t_dfs, ignore_index=True)
            
            buys = trades_df[['day', 'timestamp', 'symbol', 'buyer', 'price', 'quantity']].copy()
            buys.columns = ['day', 'timestamp', 'symbol', 'bot', 'price', 'quantity']
            buys['pos_change'] = buys['quantity']
            buys['cash_change'] = -buys['price'] * buys['quantity']
            
            sells = trades_df[['day', 'timestamp', 'symbol', 'seller', 'price', 'quantity']].copy()
            sells.columns = ['day', 'timestamp', 'symbol', 'bot', 'price', 'quantity']
            sells['pos_change'] = -sells['quantity']
            sells['cash_change'] = sells['price'] * sells['quantity']
            
            legs = pd.concat([buys, sells], ignore_index=True)
            legs = legs[~legs['bot'].astype(str).str.upper().isin(['SUBMISSION', 'NAN', ''])]
            self.legs_df = legs
            
            unique_bots = sorted(self.legs_df['bot'].dropna().unique().tolist())
            self.bot_colors = {}
            for i, bot in enumerate(unique_bots):
                if bot in FIXED_BOT_COLORS:
                    self.bot_colors[bot] = pg.mkColor(FIXED_BOT_COLORS[bot])
                else:
                    self.bot_colors[bot] = pg.intColor(i, hues=max(1, len(unique_bots)), values=1, alpha=255)
                    
        self._populate_dropdowns()
        self._update_plots()

    def _populate_dropdowns(self):
        if self.prices_df.empty: return
        
        self.cb_day.blockSignals(True)
        self.cb_day.clear()
        days = ['All'] + [str(d) for d in sorted(self.prices_df['day'].unique())]
        self.cb_day.addItems(days)
        self.cb_day.blockSignals(False)
        
        self.cb_prod.blockSignals(True)
        self.cb_prod.clear()
        prods = sorted(self.prices_df['product'].unique())
        self.cb_prod.addItems(prods)
        self.cb_prod.blockSignals(False)
        
        self.cb_bot.blockSignals(True)
        self.cb_bot.clear()
        bots = ['All'] + sorted(self.legs_df['bot'].unique().tolist())
        self.cb_bot.addItems(bots)
        self.cb_bot.blockSignals(False)

    def _update_plots(self):
        self.p_pnl.clear()
        self.p_pos.clear()
        self.current_stats.clear()
        
        if self.prices_df.empty or self.legs_df.empty: return
            
        day_sel = self.cb_day.currentText()
        prod_sel = self.cb_prod.currentText()
        bot_sel = self.cb_bot.currentText()
        
        if not prod_sel: return
        
        p_mask = self.prices_df['product'] == prod_sel
        if day_sel != 'All':
            p_mask &= self.prices_df['day'] == int(day_sel)
        p_sub = self.prices_df[p_mask].sort_values(['day', 'timestamp'])
        
        if p_sub.empty: return
        
        min_day = p_sub['day'].min()
        if day_sel == 'All':
            p_sub['cts'] = p_sub['timestamp'] + (p_sub['day'] - min_day) * 1000000
            t_col = 'cts'
        else:
            t_col = 'timestamp'
            
        timeline = p_sub[[t_col, 'mid_price']].copy()
        
        l_mask = self.legs_df['symbol'] == prod_sel
        if day_sel != 'All':
            l_mask &= self.legs_df['day'] == int(day_sel)
        l_sub = self.legs_df[l_mask].copy()
        
        if day_sel == 'All' and not l_sub.empty:
            l_sub['cts'] = l_sub['timestamp'] + (l_sub['day'] - min_day) * 1000000
            l_t_col = 'cts'
        else:
            l_t_col = 'timestamp'

        bots_to_plot = [bot_sel] if bot_sel != 'All' else sorted(l_sub['bot'].unique())
        
        for bot in bots_to_plot:
            bot_legs = l_sub[l_sub['bot'] == bot]
            if bot_legs.empty: continue
            
            agg_legs = bot_legs.groupby(l_t_col)[['pos_change', 'cash_change']].sum().reset_index()
            
            merged = pd.merge(timeline, agg_legs, left_on=t_col, right_on=l_t_col, how='left')
            merged['pos_change'] = merged['pos_change'].fillna(0)
            merged['cash_change'] = merged['cash_change'].fillna(0)
            
            merged['position'] = merged['pos_change'].cumsum()
            merged['cash'] = merged['cash_change'].cumsum()
            merged['pnl'] = merged['cash'] + merged['position'] * merged['mid_price']
            
            t_vals = merged[t_col].to_numpy()
            pnl_vals = merged['pnl'].to_numpy()
            pos_vals = merged['position'].to_numpy()
            
            # Calculate basic stats
            returns = np.diff(pnl_vals)
            sharpe = 0
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))
                
            self.current_stats[bot] = {
                'Net PnL': pnl_vals[-1] if len(pnl_vals) > 0 else 0,
                'Max Abs Pos': np.max(np.abs(pos_vals)) if len(pos_vals) > 0 else 0,
                'Sharpe': sharpe
            }
            
            color = self.bot_colors.get(bot, pg.mkColor(ACCENT_CYAN))
            pen = pg.mkPen(color, width=2)
            
            self.p_pnl.plot(t_vals, pnl_vals, pen=pen, name=bot)
            self.p_pos.plot(t_vals, pos_vals, pen=pen, stepMode='right')

    def _set_zoom(self, mode):
        self.p_pnl.setMouseEnabled(x=(mode in ['x', 'xy']), y=(mode in ['y', 'xy']))
        self.p_pos.setMouseEnabled(x=(mode in ['x', 'xy']), y=(mode in ['y', 'xy']))

    def _autoscale_all(self):
        self.p_pnl.autoRange()
        self.p_pos.autoRange()

    def _on_mouse_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        
        for p in [self.p_pnl, self.p_pos]:
            if p.sceneBoundingRect().contains(event.scenePos()):
                mouse_point = p.vb.mapSceneToView(event.scenePos())
                x, y = mouse_point.x(), mouse_point.y()
                
                vl = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
                hl = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
                txt = pg.TextItem(text=f"({x:.0f}, {y:.1f})", color=ACCENT_GOLD, anchor=(0, 1))
                txt.setPos(x, y)
                
                p.addItem(vl)
                p.addItem(hl)
                p.addItem(txt)
                self.markup_lines.extend([vl, hl, txt])

    def _clear_markup(self):
        for item in self.markup_lines:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.markup_lines = []

    def _show_data_pane(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Bot Statistics")
        dlg.resize(450, 300)
        dlg.setStyleSheet(APP_STYLE)
        
        layout = QVBoxLayout(dlg)
        table = QTableWidget(len(self.current_stats), 4)
        table.setHorizontalHeaderLabels(['Bot', 'Net PnL', 'Max Abs Pos', 'Sharpe'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        for row, (bot, stats) in enumerate(self.current_stats.items()):
            table.setItem(row, 0, QTableWidgetItem(bot))
            table.setItem(row, 1, QTableWidgetItem(f"{stats['Net PnL']:,.2f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{stats['Max Abs Pos']:.0f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{stats['Sharpe']:.3f}"))
            
        layout.addWidget(table)
        dlg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = BotSimEngine()
    win.show()
    sys.exit(app.exec())
