import sys, os, json, re, subprocess
from io import StringIO
import numpy as np
import polars as pl
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QFileDialog, QTabWidget, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit,
    QScrollArea, QCheckBox, QDialog, QRadioButton, QButtonGroup, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, Qt, QEvent
from PyQt6.QtGui import QShortcut, QKeySequence, QFont, QColor, QBrush, QStandardItemModel, QStandardItem

from plotting_utils import build_ob_heatmap, build_order_placement_heatmap, get_rect

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Styling & Colors ---
BG, PANEL_BG, BORDER, TEXT, DIM = '#0d0f14', '#12151c', '#1e2330', '#c8d0e0', '#4a5068'
ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_WHITE, ACCENT_PURPLE, ACCENT_ORANGE = \
    '#00d4ff', '#39ff6e', '#ff3d5a', '#ffd700', '#ffffff', '#b06dff', '#ff9f43'
CUSTOM_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f', '#b06dff', '#ff9f43']

# ── Heatmap Volume Coloring (10-Band Gradient Hues) ─────────────────────────
# Format: [R, G, B]. Buy = Cyan/Teal tones, Sell = Gold/Orange tones.
BUY_VOLUME_COLORS = [
    [0, 100, 100], [0, 130, 130], [0, 160, 160], [0, 190, 190], [0, 220, 220],
    [0, 255, 255], [100, 255, 255], [150, 255, 255], [200, 255, 255], [255, 255, 255]
]
SELL_VOLUME_COLORS = [
    [100, 60, 0], [130, 80, 0], [160, 100, 0], [190, 120, 0], [220, 140, 0],
    [255, 160, 0], [255, 180, 50], [255, 200, 100], [255, 220, 150], [255, 255, 255]
]

# ── Order Placement Heatmap Coloring ─────────────────────────────────────────
ORDER_BUY_COLORS  = [[int(i * 5), int(100 + i * 15.5), int(i * 5)] for i in range(10)]
ORDER_SELL_COLORS = [[int(100 + i * 15.5), int(i * 5), int(100 + i * 15.5)] for i in range(10)]

# Palette for bot trade markers
TRADE_VOLUME_COLORS = [
    [0, 255, 255], [0, 230, 230], [0, 200, 200], [0, 170, 170],
    [255, 215, 0], [255, 180, 0], [255, 140, 0], [255, 100, 0],
    [255, 60, 0], [255, 0, 0]
]

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
QRadioButton {{
    spacing: 8px;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {BORDER};
    border-radius: 11px;
    background: {PANEL_BG};
}}
QRadioButton::indicator:checked {{
    background: {ACCENT_CYAN};
    border-color: {ACCENT_WHITE};
}}
"""

class InteractiveLegendItem(pg.LegendItem):
    """Refined legend that supports proxy items for toggle logic."""
    def addItem(self, item, name, toggle_target=None):
        try:
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

# --- Legend Helper ---
class HeatmapLegend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 6, 15, 6)
        self.main_layout.setSpacing(2)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")

        self.row_layout = QHBoxLayout()
        self.row_layout.setSpacing(25)

        self.buy_ranges = []
        self.sell_ranges = []
        self.trade_ranges = []
        self.trade_cells = []

        self.row_layout.addLayout(self._create_side("BUY BOOK",  BUY_VOLUME_COLORS,  self.buy_ranges))
        self.row_layout.addLayout(self._create_side("SELL BOOK", SELL_VOLUME_COLORS, self.sell_ranges))
        self.row_layout.addLayout(self._create_trade_side())

        self.row_layout.addStretch()
        self.main_layout.addLayout(self.row_layout)

    def _create_side(self, title, colors, label_list):
        side_layout = QHBoxLayout()
        side_layout.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 7.5pt;")
        side_layout.addWidget(title_lbl)
        for i, color in enumerate(colors):
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch = QLabel()
            swatch.setFixedSize(18, 10)
            swatch.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #333;")
            vbox.addWidget(swatch)
            rlbl = QLabel("0")
            rlbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rlbl.setStyleSheet(f"font-size: 6.5pt; color: {DIM};")
            label_list.append(rlbl)
            vbox.addWidget(rlbl)
            side_layout.addLayout(vbox)
        return side_layout

    def _create_trade_side(self):
        self._trade_layout = QHBoxLayout()
        self._trade_layout.setSpacing(6)
        self._trade_title = QLabel("BOT TRADES")
        self._trade_title.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 7.5pt;")
        self._trade_layout.addWidget(self._trade_title)
        for i, color in enumerate(TRADE_VOLUME_COLORS):
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch = QLabel()
            swatch.setFixedSize(18, 10)
            swatch.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid #333;")
            vbox.addWidget(swatch)
            rlbl = QLabel("")
            rlbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rlbl.setStyleSheet(f"font-size: 6.5pt; color: {DIM};")
            self.trade_ranges.append(rlbl)
            vbox.addWidget(rlbl)
            cell = QWidget()
            cell.setLayout(vbox)
            self.trade_cells.append(cell)
            self._trade_layout.addWidget(cell)
        return self._trade_layout

    def update_ranges(self, max_vol, quantile_edges):
        def update_ob(lbl_list, mv):
            for i in range(10):
                lower = int(np.floor(i * mv / 10))
                upper = int(np.floor((i + 1) * mv / 10))
                lbl_list[i].setText(f"{lower}-{upper}" if i < 9 else f">{lower}")
        update_ob(self.buy_ranges,  max_vol)
        update_ob(self.sell_ranges, max_vol)

        n_buckets = len(quantile_edges) - 1
        for i, (cell, rlbl) in enumerate(zip(self.trade_cells, self.trade_ranges)):
            if i < n_buckets:
                pal_idx = int(round(i * (len(TRADE_VOLUME_COLORS) - 1) / max(n_buckets - 1, 1)))
                c = TRADE_VOLUME_COLORS[pal_idx]
                swatch = cell.findChild(QLabel)
                swatch.setStyleSheet(f"background-color: rgb({c[0]}, {c[1]}, {c[2]}); border: 1px solid #333;")
                lo, hi = quantile_edges[i], quantile_edges[i + 1]
                fmt = lambda v: f"{v:.0f}" if v == int(v) else f"{v:.2f}"
                rlbl.setText(f"{fmt(lo)}-{fmt(hi)}" if i < n_buckets - 1 else f">{fmt(lo)}")
                cell.setVisible(True)
            else:
                cell.setVisible(False)
                rlbl.setText("")

def _is_timestamps_continuous(df):
    if 'day' not in df.columns: return False
    days = df['day'].unique().sort().to_list()
    if len(days) <= 1: return False
    for i in range(len(days) - 1):
        d1_max = df.filter(pl.col('day') == days[i])['timestamp'].max()
        d2_min = df.filter(pl.col('day') == days[i + 1])['timestamp'].min()
        if d2_min > d1_max:
            return True
    return False

class BacktestRunner(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, script_path, backtests_dir):
        super().__init__()
        self.script_path = script_path
        self.backtests_dir = backtests_dir

    def run(self):
        try:
            subprocess.run([sys.executable, self.script_path], cwd=os.path.dirname(self.script_path), check=True)
            logs = [os.path.join(self.backtests_dir, f) for f in os.listdir(self.backtests_dir) if f.endswith('.log')]
            if logs:
                self.finished.emit(max(logs, key=os.path.getmtime))
            else:
                self.error.emit("No .log files found in backtests dir")
        except Exception as e:
            self.error.emit(str(e))

# --- Data Settings Dialog ---
BUILTIN_KEYS = ['__POSITION__', '__PNL__']
BUILTIN_LABELS = {'__POSITION__': '⬡ Position', '__PNL__': '⬡ PnL (Log)'}

class DataSetupDialog(QDialog):
    def __init__(self, keys, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Setup"); self.setMinimumWidth(480)
        self.settings = settings; layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll_content = QWidget(); self.grid = QGridLayout(scroll_content)
        self.grid.setColumnStretch(0, 1)
        self.grid.addWidget(QLabel("<b>Data Key</b>"), 0, 0)
        self.grid.addWidget(QLabel("<b>Main Pane</b>"), 0, 1)
        self.grid.addWidget(QLabel("<b>Generic Pane</b>"), 0, 2)
        self.grid.addWidget(QLabel("<b>Disabled</b>"), 0, 3)
        self.groups = {}
        all_keys = BUILTIN_KEYS + list(keys)
        for i, key in enumerate(all_keys):
            label = BUILTIN_LABELS.get(key, key)
            lbl = QLabel(label)
            if key in BUILTIN_KEYS: lbl.setStyleSheet(f"color: {ACCENT_GOLD}; font-style: italic;")
            self.grid.addWidget(lbl, i+1, 0)
            bm, bg, bd = QRadioButton(), QRadioButton(), QRadioButton()
            grp = QButtonGroup(self); grp.addButton(bm, 0); grp.addButton(bg, 1); grp.addButton(bd, 2)
            self.grid.addWidget(bm, i+1, 1, Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(bg, i+1, 2, Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(bd, i+1, 3, Qt.AlignmentFlag.AlignCenter)
            val = self.settings.get(key, "disabled" if key in BUILTIN_KEYS else "generic")
            if val == "main": bm.setChecked(True)
            elif val == "generic": bg.setChecked(True)
            else: bd.setChecked(True)
            self.groups[key] = grp
        scroll.setWidget(scroll_content); scroll.setWidgetResizable(True); layout.addWidget(scroll)
        btn = QPushButton("Apply"); btn.clicked.connect(self.accept); layout.addWidget(btn)
    def get_results(self):
        out = {}
        for k, g in self.groups.items():
            cid = g.checkedId()
            out[k] = "main" if cid == 0 else ("generic" if cid == 1 else "disabled")
        return out

class CheckableComboBox(QComboBox):
    checkedItemsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Overall")
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.view().viewport().installEventFilter(self)
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.lineEdit() and event.type() == QEvent.Type.MouseButtonPress:
            self.showPopup()
            return True
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            if index.isValid():
                item = self._model.itemFromIndex(index)
                if item:
                    item.setCheckState(
                        Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    self._refresh_text()
                    self.checkedItemsChanged.emit()
            return True  # consume all viewport releases — keeps popup open
        return super().eventFilter(obj, event)

    def _refresh_text(self):
        checked = self.checkedItems()
        self.lineEdit().setText(", ".join(checked) if checked else "")

    def checkedItems(self):
        return [self._model.item(i).text() for i in range(self._model.rowCount())
                if self._model.item(i) and self._model.item(i).checkState() == Qt.CheckState.Checked]

    def addItem(self, text, checked=False):
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._model.appendRow(item)

    def addItems(self, texts):
        for t in texts:
            self.addItem(t)

    def clear(self):
        self._model.clear()
        self.lineEdit().setText("")

    def count(self):
        return self._model.rowCount()

    def findText(self, text, flags=None):
        for i in range(self._model.rowCount()):
            item = self._model.item(i)
            if item and item.text() == text:
                return i
        return -1

    def setCheckedItems(self, texts):
        text_set = set(texts)
        for i in range(self._model.rowCount()):
            item = self._model.item(i)
            if item:
                item.setCheckState(Qt.CheckState.Checked if item.text() in text_set else Qt.CheckState.Unchecked)
        self._refresh_text()


class TokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auth Required"); self.setMinimumWidth(480)
        self.token = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Token missing/expired. Choose method:"))
        self.rb_paste = QRadioButton("Paste idToken"); self.rb_paste.setChecked(True)
        self.rb_refresh = QRadioButton("Paste refreshToken")
        layout.addWidget(self.rb_paste); layout.addWidget(self.rb_refresh)
        layout.addWidget(QLabel("Token:"))
        self.token_input = QLineEdit(); self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.token_input)
        btn = QPushButton("OK"); btn.clicked.connect(self._on_ok); layout.addWidget(btn)

    def _on_ok(self):
        val = self.token_input.text().strip()
        if not val: return
        try:
            from imc_api.submit import cognito_refresh, save_token, token_valid
            if self.rb_refresh.isChecked():
                tok = cognito_refresh(val)
            else:
                tok = val.removeprefix("Bearer ")
            save_token(tok)
            self.token = tok
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class SubmitFileDialog(QDialog):
    def __init__(self, default_path="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Submission File"); self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Algo file to submit:"))
        row = QHBoxLayout()
        self.path_input = QLineEdit(default_path); row.addWidget(self.path_input)
        btn_browse = QPushButton("Browse..."); btn_browse.clicked.connect(self._browse); row.addWidget(btn_browse)
        layout.addLayout(row)
        btns = QHBoxLayout()
        btn_ok = QPushButton("Submit"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok); btns.addWidget(btn_cancel); layout.addLayout(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Algo File", self.path_input.text(), "Python (*.py)")
        if path: self.path_input.setText(path)

    def get_path(self): return self.path_input.text().strip()


class SubmitProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Submission Progress"); self.setMinimumWidth(500); self.setMinimumHeight(260)
        self._finished = False
        layout = QVBoxLayout(self)
        self.lbl_status = QLabel("Starting..."); self.lbl_status.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")
        layout.addWidget(self.lbl_status)
        self.lbl_log = QLabel(); self.lbl_log.setWordWrap(True)
        self.lbl_log.setStyleSheet(f"color: {DIM}; font-size: 8pt;")
        layout.addWidget(self.lbl_log)
        layout.addStretch()
        self.btn_close = QPushButton("Close"); self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept); layout.addWidget(self.btn_close)
        self._lines = []

    def closeEvent(self, event):
        if not self._finished: event.ignore()
        else: super().closeEvent(event)

    def update_status(self, msg):
        self.lbl_status.setText(msg)
        self._lines.append(msg)
        if len(self._lines) > 10: self._lines = self._lines[-10:]
        self.lbl_log.setText('\n'.join(self._lines[:-1]))

    def mark_done(self, success):
        self._finished = True
        self.btn_close.setEnabled(True)
        color = ACCENT_GREEN if success else ACCENT_RED
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold;")


class SubmitWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, token, file_path, round_id, logviz_dir, logs_dir):
        super().__init__()
        self.token = token; self.file_path = file_path
        self.round_id = round_id; self.logviz_dir = logviz_dir; self.logs_dir = logs_dir

    def run(self):
        import time, urllib.request
        try:
            from imc_api.submit import submit_algo, fetch_zip, unzip_and_move, POLL_INTERVAL, ACTIVE_STATUSES, API_BASE, _headers
            self.status.emit(f"Submitting {os.path.basename(self.file_path)}...")
            sub_id, round_id = submit_algo(self.token, self.file_path)
            self.status.emit(f"Submitted — id={sub_id}  round={round_id}")
            url = f"{API_BASE}/submissions/algo/{round_id}?page=1&pageSize=50"
            while True:
                req = urllib.request.Request(url, headers=_headers(self.token))
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
                sub = next((x for x in data["data"]["items"] if x["id"] == sub_id), None)
                if sub is None: raise RuntimeError(f"Submission {sub_id} not found in response")
                status = sub["status"]
                self.status.emit(f"[{time.strftime('%H:%M:%S')}] {status}")
                if status not in ACTIVE_STATUSES: break
                time.sleep(POLL_INTERVAL)
            if status not in ("DONE", "FINISHED"):
                self.done.emit(False, f"Ended with status: {status}"); return
            self.status.emit("Fetching results zip...")
            zip_path = fetch_zip(self.token, sub_id, self.logs_dir)
            self.status.emit(f"Downloaded: {os.path.basename(str(zip_path))}")
            self.status.emit("Extracting logs to logviz/...")
            unzip_and_move(zip_path, self.logviz_dir)
            self.done.emit(True, "Done!")
        except Exception as e:
            self.done.emit(False, str(e))


class LogVisualizer(QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.setWindowTitle('Prosperity Sandbox Visualizer')
        self.setGeometry(50, 50, 1600, 920)
        self.data, self.current_df, self.ob_res = None, None, None
        self.custom_curves = {}
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
        self._backtest_runner = None
        pg.setConfigOptions(useOpenGL=True, imageAxisOrder='row-major')
        self._build_ui()
        self._restore_window_state()
        if log_path: self._load_file(log_path)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)
        controls = QHBoxLayout(); controls.setContentsMargins(12, 12, 12, 12); controls.setSpacing(12)
        btn_open = QPushButton("📂 Open Log"); btn_open.clicked.connect(self._open_dialog); controls.addWidget(btn_open)
        self.btn_backtest = QPushButton("⟳ Refresh Backtest"); self.btn_backtest.clicked.connect(self._run_backtest); self.btn_backtest.setVisible(False); controls.addWidget(self.btn_backtest)
        self.btn_submit = QPushButton("🚀 Submit"); self.btn_submit.clicked.connect(self._on_submit_clicked); self.btn_submit.setVisible(False); controls.addWidget(self.btn_submit)
        btn_import = QPushButton("📊 Import Data"); btn_import.clicked.connect(self._import_data); controls.addWidget(btn_import)
        controls.addWidget(QLabel("Product:")); self.cb_prod = QComboBox(); controls.addWidget(self.cb_prod)
        controls.addWidget(QLabel("Day:")); self.cb_day = QComboBox(); controls.addWidget(self.cb_day)
        controls.addWidget(QLabel("Tag:")); self.cb_tag = CheckableComboBox(); self.cb_tag.setMinimumWidth(110); controls.addWidget(self.cb_tag)
        self.cb_prod.currentTextChanged.connect(self._process_selection); self.cb_day.currentTextChanged.connect(self._process_selection)
        self.cb_tag.checkedItemsChanged.connect(self._on_tag_changed)
        controls.addStretch()
        btn_setup = QPushButton("⚙️ Data Setup [S]"); btn_setup.clicked.connect(self._open_data_setup); controls.addWidget(btn_setup)
        self.lbl_markup = QLabel("MARKUP: OFF"); self.lbl_markup.setStyleSheet(f"color: {DIM}; font-weight: bold;"); controls.addWidget(self.lbl_markup)
        self.lbl_tooltip = QLabel("TOOLTIP: OFF"); self.lbl_tooltip.setStyleSheet(f"color: {DIM}; font-weight: bold;"); controls.addWidget(self.lbl_tooltip)
        self.lbl_zoom = QLabel("Mode: XY"); self.lbl_zoom.setStyleSheet(f"color: {DIM};"); controls.addWidget(self.lbl_zoom)
        btn_export = QPushButton("💾 Export Custom CSV"); btn_export.clicked.connect(self._export_custom_csv); controls.addWidget(btn_export)
        main_layout.addLayout(controls)
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self._build_dashboard_tab()
        market_container = QWidget(); market_layout = QVBoxLayout(market_container); market_layout.setContentsMargins(0, 0, 0, 0); market_layout.setSpacing(0)
        self.hm_legend = HeatmapLegend(); market_layout.addWidget(self.hm_legend)
        self.gw_m = pg.GraphicsLayoutWidget(); self.gw_m.setBackground(BG); market_layout.addWidget(self.gw_m)
        self.tabs.addTab(market_container, "Market View")
        # Info panel overlay on top of graph
        self.info_panel = QFrame(self.gw_m)
        self.info_panel.setStyleSheet(
            f"QFrame {{ background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
            f"QScrollArea {{ background: transparent; border: none; }}"
        )
        _ip_scroll = QScrollArea(self.info_panel)
        _ip_scroll.setWidgetResizable(True)
        _ip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _ip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _ip_scroll.setStyleSheet("background: transparent; border: none;")
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._info_label.setTextFormat(Qt.TextFormat.RichText)
        self._info_label.setWordWrap(False)
        self._info_label.setStyleSheet(
            f"background: transparent; color: {TEXT}; font-family: 'JetBrains Mono', monospace; font-size: 8pt; padding: 6px 10px;"
        )
        _ip_scroll.setWidget(self._info_label)
        _ip_layout = QVBoxLayout(self.info_panel)
        _ip_layout.setContentsMargins(0, 0, 0, 0)
        _ip_layout.addWidget(_ip_scroll)
        self.info_panel.setFixedWidth(265)
        self.info_panel.setVisible(False)
        self.gw_m.installEventFilter(self)
        self.p_m = self.gw_m.addPlot(row=0, col=0); self.p_m.showGrid(x=True, y=True, alpha=0.3); self.p_m.setDownsampling(auto=True, mode='peak')
        self.p_gen = self.gw_m.addPlot(row=1, col=0); self.p_gen.showGrid(x=True, y=True, alpha=0.3); self.p_gen.setFixedHeight(200); self.p_gen.setXLink(self.p_m); self.p_gen.hideAxis('bottom'); self.p_gen.addLegend()
        self.img_item = pg.ImageItem(); self.img_item.setZValue(0); self.p_m.addItem(self.img_item)
        self.img_orders = pg.ImageItem(); self.img_orders.setZValue(1); self.p_m.addItem(self.img_orders); self.img_orders.setVisible(False)
        self.curve_mid = self.p_m.plot(pen=pg.mkPen(ACCENT_CYAN, width=2), name="Mid Price", clipToView=True)
        self.sc_bot = pg.ScatterPlotItem(symbol='x', size=7, brush=ACCENT_WHITE, name="Bot Trades"); self.sc_bot.setZValue(2)
        self.sc_buy = pg.ScatterPlotItem(symbol='t1', size=10, brush=ACCENT_CYAN, name="My Buy"); self.sc_buy.setZValue(3)
        self.sc_sell = pg.ScatterPlotItem(symbol='t', size=10, brush=ACCENT_ORANGE, name="My Sell"); self.sc_sell.setZValue(3)
        for item in [self.sc_bot, self.sc_buy, self.sc_sell]: self.p_m.addItem(item)
        self.leg_m = InteractiveLegendItem(offset=(10, 10)); self.leg_m.setParentItem(self.p_m.graphicsItem())
        self._heatmap_proxy = pg.PlotDataItem(pen=None, brush=pg.mkBrush(ACCENT_PURPLE))
        self.leg_m.addItem(self._heatmap_proxy, "Heatmap", toggle_target=self.img_item)
        self._orders_proxy = pg.PlotDataItem(pen=None, brush=pg.mkBrush(ACCENT_WHITE))
        self.leg_m.addItem(self._orders_proxy, "Order Placement", toggle_target=self.img_orders)
        self.leg_m.addItem(self.curve_mid, "Mid Price"); self.leg_m.addItem(self.sc_bot, "Bot Trades"); self.leg_m.addItem(self.sc_buy, "My Buy"); self.leg_m.addItem(self.sc_sell, "My Sell")
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(DIM, style=Qt.PenStyle.DashLine))
        self.p_m.addItem(self.v_line, ignoreBounds=True); self.p_m.addItem(self.h_line, ignoreBounds=True)
        self.p_m.scene().sigMouseMoved.connect(self._on_mouse_moved); self.p_m.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # PnL Tab
        pnl_container = QWidget()
        pnl_layout = QVBoxLayout(pnl_container); pnl_layout.setContentsMargins(0, 0, 0, 0); pnl_layout.setSpacing(0)
        pnl_ctrl = QHBoxLayout(); pnl_ctrl.setContentsMargins(12, 8, 12, 8)
        self.cb_pnl_type = QComboBox(); self.cb_pnl_type.addItems(["Log PnL", "Realized PnL", "Valuation PnL"])
        self.cb_pnl_type.currentTextChanged.connect(self._process_selection)
        pnl_ctrl.addWidget(QLabel("PnL Method:")); pnl_ctrl.addWidget(self.cb_pnl_type)
        pnl_ctrl.addWidget(QLabel("Product:")); self.cb_pnl_prod = QComboBox()
        self.cb_pnl_prod.currentTextChanged.connect(self._process_selection)
        pnl_ctrl.addWidget(self.cb_pnl_prod); pnl_ctrl.addStretch()
        pnl_layout.addLayout(pnl_ctrl)
        self.gw_p = pg.GraphicsLayoutWidget(); self.gw_p.setBackground(BG); pnl_layout.addWidget(self.gw_p)
        self.tabs.addTab(pnl_container, "PnL")
        self.p_pnl = self.gw_p.addPlot(); self.p_pnl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_pnl = self.p_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2), clipToView=True)

        self.gw_pos = pg.GraphicsLayoutWidget(); self.gw_pos.setBackground(BG); self.tabs.addTab(self.gw_pos, "Position")
        self.p_pos = self.gw_pos.addPlot(); self.p_pos.showGrid(x=True, y=True, alpha=0.3); self.p_pos.addLegend()
        self.pos_curves = {}

        self.gw_c = pg.GraphicsLayoutWidget(); self.gw_c.setBackground(BG); self.tabs.addTab(self.gw_c, "Custom")

        # Logs Tab
        logs_container = QWidget(); logs_layout = QVBoxLayout(logs_container); logs_layout.setContentsMargins(0, 0, 0, 0); logs_layout.setSpacing(0)
        logs_filter_bar = QHBoxLayout(); logs_filter_bar.setContentsMargins(8, 6, 8, 6)
        self.log_filter_input = QLineEdit(); self.log_filter_input.setPlaceholderText("Filter logs..."); self.log_filter_input.textChanged.connect(self._filter_logs_table)
        logs_filter_bar.addWidget(QLabel("🔍")); logs_filter_bar.addWidget(self.log_filter_input); logs_layout.addLayout(logs_filter_bar)
        self.logs_table = QTableWidget(); self.logs_table.setColumnCount(6); self.logs_table.setHorizontalHeaderLabels(['TS', 'Tag', 'Product', 'Pos', 'PnL', 'Msg'])
        self.logs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch); self.logs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.logs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.logs_table.verticalHeader().setVisible(False); logs_layout.addWidget(self.logs_table)
        self.tabs.addTab(logs_container, "Logs")

        QShortcut(QKeySequence("X"), self).activated.connect(lambda: self._set_zoom("x"))
        QShortcut(QKeySequence("Y"), self).activated.connect(lambda: self._set_zoom("y"))
        QShortcut(QKeySequence("Z"), self).activated.connect(lambda: self._set_zoom("xy"))
        QShortcut(QKeySequence("A"), self).activated.connect(self._autoscale_all)
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_markup)
        QShortcut(QKeySequence("T"), self).activated.connect(self._toggle_tooltip)
        QShortcut(QKeySequence("Shift+T"), self).activated.connect(self._freeze_tooltip)
        QShortcut(QKeySequence("Shift+M"), self).activated.connect(self._toggle_gen_pane)
        QShortcut(QKeySequence("C"), self).activated.connect(self._clear_markup)
        QShortcut(QKeySequence("S"), self).activated.connect(self._open_data_setup)
        QShortcut(QKeySequence("?"), self).activated.connect(self._focus_keybinds_tab)

        self._build_sandbox_tab()
        self._build_keybinds_tab()

    def _open_data_setup(self):
        if not self.data: return
        dlg = DataSetupDialog(sorted(self.data.get('custom',{}).keys()), self.data_settings, self)
        if dlg.exec():
            self.data_settings = dlg.get_results()
            self._save_data_settings()
            for c in self.custom_curves.values():
                try: self.p_m.removeItem(c)
                except: pass
                try: self.p_gen.removeItem(c)
                except: pass
            self.custom_curves = {}
            self._process_selection()

    def _settings_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '.logviz_settings.json')

    def _read_config(self):
        try:
            p = self._settings_path()
            if os.path.exists(p):
                with open(p) as f:
                    data = json.load(f)
                    # migrate old flat data_settings format
                    if data and not any(k in data for k in ('data_settings', 'window')):
                        return {'data_settings': data}
                    return data
        except Exception as e:
            print(f"Config read error: {e}")
        return {}

    def _write_config(self, cfg):
        try:
            with open(self._settings_path(), 'w') as f: json.dump(cfg, f)
        except Exception as e:
            print(f"Config write error: {e}")

    def _save_data_settings(self):
        cfg = self._read_config()
        cfg['data_settings'] = self.data_settings
        self._write_config(cfg)

    def _load_data_settings(self):
        return self._read_config().get('data_settings', {})

    def _save_window_state(self):
        g = self.geometry()
        cfg = self._read_config()
        cfg['window'] = {'x': g.x(), 'y': g.y(), 'w': g.width(), 'h': g.height(), 'tab': self.tabs.currentIndex()}
        if self.data:
            legend_vis = {
                'Heatmap': self.img_item.isVisible(),
                'Order Placement': self.img_orders.isVisible(),
                'Mid Price': self.curve_mid.isVisible(),
                'Bot Trades': self.sc_bot.isVisible(),
                'My Buy': self.sc_buy.isVisible(),
                'My Sell': self.sc_sell.isVisible(),
            }
            for name, curve in self.custom_curves.items():
                legend_vis[f'custom:{name}'] = curve.isVisible()
            cfg['view_state'] = {
                'product': self.cb_prod.currentText(),
                'day': self.cb_day.currentText(),
                'dash_product': self.cb_dash_prod.currentText(),
                'pnl_product': self.cb_pnl_prod.currentText(),
                'tag': self.cb_tag.checkedItems(),
                'dd_pct': self.chk_dd_pct.isChecked(),
                'gen_pane_minimized': self._gen_pane_minimized,
                'legend': legend_vis,
            }
        self._write_config(cfg)

    def _restore_window_state(self):
        w = self._read_config().get('window')
        if not w: return
        self.setGeometry(w.get('x', 50), w.get('y', 50), w.get('w', 1600), w.get('h', 920))
        tab = w.get('tab', 0)
        if 0 <= tab < self.tabs.count():
            self.tabs.setCurrentIndex(tab)

    def closeEvent(self, event):
        self._save_window_state()
        super().closeEvent(event)

    def _build_keybinds_tab(self):
        keybinds = [
            ("X",       "Zoom X axis only"),
            ("Y",       "Zoom Y axis only"),
            ("Z",       "Zoom XY (reset)"),
            ("A",       "Autoscale all plots"),
            ("M",       "Toggle markup mode"),
            ("T",       "Toggle info panel (tooltip)"),
            ("Shift+T", "Freeze tooltip at current coordinates"),
            ("Shift+M", "Toggle secondary (gen) pane"),
            ("C",       "Clear markup lines"),
            ("S",       "Open Data Setup"),
            ("?",       "Go to this tab"),
        ]
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(24, 24, 24, 24); layout.setSpacing(0)
        table = QTableWidget(len(keybinds), 2); table.setHorizontalHeaderLabels(["Key", "Action"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for i, (key, desc) in enumerate(keybinds):
            k_item = QTableWidgetItem(key); k_item.setForeground(QBrush(QColor(ACCENT_CYAN)))
            k_item.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
            k_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 0, k_item); table.setItem(i, 1, QTableWidgetItem(desc))
        table.resizeRowsToContents(); table.setMaximumWidth(500); layout.addWidget(table); layout.addStretch()
        self._keybinds_tab_index = self.tabs.addTab(container, "Keys")

    def _focus_keybinds_tab(self):
        self.tabs.setCurrentIndex(self._keybinds_tab_index)

    def _apply_view_state(self):
        vs = self._read_config().get('view_state', {})
        if not vs: return
        for combo, key in [(self.cb_prod, 'product'), (self.cb_day, 'day'), (self.cb_dash_prod, 'dash_product'), (self.cb_pnl_prod, 'pnl_product')]:
            val = vs.get(key, '')
            if val and combo.findText(val) >= 0:
                combo.blockSignals(True); combo.setCurrentText(val); combo.blockSignals(False)
        saved_tags = vs.get('tag', [])
        if isinstance(saved_tags, list) and saved_tags:
            self.cb_tag.blockSignals(True); self.cb_tag.setCheckedItems(saved_tags); self.cb_tag.blockSignals(False)
        self.chk_dd_pct.blockSignals(True); self.chk_dd_pct.setChecked(vs.get('dd_pct', False)); self.chk_dd_pct.blockSignals(False)
        minimized = vs.get('gen_pane_minimized', False)
        if minimized != self._gen_pane_minimized:
            self._gen_pane_minimized = minimized
            if minimized: self.p_gen.setFixedHeight(0); self.p_gen.setVisible(False)
            else: self.p_gen.setFixedHeight(200)
        legend = vs.get('legend', {})
        fixed = {'Heatmap': self.img_item, 'Order Placement': self.img_orders,
                 'Mid Price': self.curve_mid, 'Bot Trades': self.sc_bot,
                 'My Buy': self.sc_buy, 'My Sell': self.sc_sell}
        for name, item in fixed.items():
            if name in legend: item.setVisible(legend[name])

    def _apply_custom_curve_visibility(self):
        legend = self._read_config().get('view_state', {}).get('legend', {})
        if not legend: return
        for name, curve in self.custom_curves.items():
            key = f'custom:{name}'
            if key in legend: curve.setVisible(legend[key])

    def _apply_saved_settings(self):
        saved = self._load_data_settings()
        if saved and self.data:
            current_keys = set(self.data.get('custom', {}).keys())
            for k, v in saved.items():
                if k in current_keys:
                    self.data_settings[k] = v
        self._apply_view_state()

    def _toggle_gen_pane(self):
        self._gen_pane_minimized = not self._gen_pane_minimized
        if self._gen_pane_minimized:
            self.p_gen.setFixedHeight(0)
            self.p_gen.setVisible(False)
        else:
            self.p_gen.setFixedHeight(200)
            if self.data:
                has_generic = (
                    any(self.data_settings.get(k, 'disabled') == 'generic' for k in BUILTIN_KEYS) or
                    any(self.data_settings.get(k, 'generic') == 'generic' for k in self.data.get('custom', {}))
                )
                self.p_gen.setVisible(has_generic)

    def _is_backtest_log(self, path):
        if not path: return False
        return 'backtests' in os.path.normpath(path).split(os.sep)

    def _is_submission_log(self, path):
        if not path: return False
        logviz_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.abspath(path)) == logviz_dir

    def _get_submit_token(self):
        from imc_api.submit import load_token, token_valid
        token = load_token()
        if token_valid(token): return token
        dlg = TokenDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted: return dlg.token
        return None

    def _on_submit_clicked(self):
        token = self._get_submit_token()
        if not token: return
        self.btn_submit.setEnabled(False); self.btn_submit.setText("Checking...")
        from PyQt6.QtWidgets import QApplication as _QApp
        _QApp.processEvents()
        try:
            from imc_api.submit import find_active_submission
            round_id = self._read_config().get('submit_round', 3)
            active = find_active_submission(token, round_id)
        except Exception as e:
            self.btn_submit.setEnabled(True); self.btn_submit.setText("🚀 Submit")
            QMessageBox.warning(self, "Error", f"Failed to check server:\n{e}"); return
        self.btn_submit.setEnabled(True); self.btn_submit.setText("🚀 Submit")
        if active:
            submitter = active.get('submitter') or active.get('teamName') or active.get('filename') or active.get('id', '?')
            QMessageBox.information(self, "Server Busy",
                f"Server busy — processing submission uploaded by: {submitter}\nStatus: {active.get('status', '?')}"); return
        cfg = self._read_config()
        default_file = cfg.get('last_submit_file', _PROJECT_ROOT)
        dlg = SubmitFileDialog(default_file, self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        file_path = dlg.get_path()
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "Invalid File", f"File not found:\n{file_path}"); return
        cfg['last_submit_file'] = file_path; self._write_config(cfg)
        logviz_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(_PROJECT_ROOT, 'logs')
        self._submit_progress = SubmitProgressDialog(self)
        self._submit_worker = SubmitWorker(token, file_path, round_id, logviz_dir, logs_dir)
        self._submit_worker.status.connect(self._submit_progress.update_status)
        self._submit_worker.done.connect(self._on_submit_done)
        self._submit_worker.start()
        self._submit_progress.exec()

    def _on_submit_done(self, success, message):
        self._submit_progress.update_status(message)
        self._submit_progress.mark_done(success)
        if success: self._load_newest_submission_log()

    def _load_newest_submission_log(self):
        logviz_dir = os.path.dirname(os.path.abspath(__file__))
        logs = [os.path.join(logviz_dir, f) for f in os.listdir(logviz_dir) if f.endswith('.log')]
        if logs: self._load_file(max(logs, key=os.path.getmtime))

    def _run_backtest(self):
        self.btn_backtest.setEnabled(False)
        self.btn_backtest.setText("⟳ Running...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(project_root, 'run_my_strategy.py')
        backtests_dir = os.path.join(project_root, 'backtests')
        self._backtest_runner = BacktestRunner(script, backtests_dir)
        self._backtest_runner.finished.connect(self._on_backtest_done)
        self._backtest_runner.error.connect(self._on_backtest_error)
        self._backtest_runner.start()

    def _on_backtest_done(self, log_path):
        self.btn_backtest.setEnabled(True)
        self.btn_backtest.setText("⟳ Refresh Backtest")
        self._load_file(log_path)

    def _on_backtest_error(self, msg):
        self.btn_backtest.setEnabled(True)
        self.btn_backtest.setText("⟳ Refresh Backtest")
        QMessageBox.warning(self, "Backtest Error", msg)

    def eventFilter(self, obj, event):
        if obj is self.gw_m and event.type() == QEvent.Type.Resize:
            self._reposition_info_panel()
        return super().eventFilter(obj, event)

    def _reposition_info_panel(self):
        if not hasattr(self, 'info_panel'): return
        panel_h = max(40, self.gw_m.height() - 20)
        self.info_panel.setFixedHeight(panel_h)
        x = self.gw_m.width() - self.info_panel.width() - 58
        self.info_panel.move(x, 8)

    def _toggle_tooltip(self):
        self.tooltip_enabled = not self.tooltip_enabled
        self._update_tooltip_label()
        if self.tooltip_enabled:
            self._reposition_info_panel()
            self.info_panel.setVisible(True)
        else:
            self.info_panel.setVisible(False)

    def _freeze_tooltip(self):
        if not self.tooltip_enabled: return
        self.tooltip_frozen = not self.tooltip_frozen
        self._update_tooltip_label()
        if self.tooltip_frozen:
            freeze_pen = pg.mkPen('#4488ff', width=1, style=Qt.PenStyle.DashLine)
            vl = pg.InfiniteLine(pos=self.v_line.value(), angle=90, movable=False, pen=freeze_pen)
            hl = pg.InfiniteLine(pos=self.h_line.value(), angle=0, movable=False, pen=freeze_pen)
            self.p_m.addItem(vl); self.p_m.addItem(hl)
            self._freeze_lines = [vl, hl]
        else:
            for item in self._freeze_lines:
                self.p_m.removeItem(item)
            self._freeze_lines = []

    def _update_tooltip_label(self):
        if not self.tooltip_enabled:
            self.lbl_tooltip.setText("TOOLTIP: OFF")
            self.lbl_tooltip.setStyleSheet(f"color: {DIM}; font-weight: bold;")
        elif self.tooltip_frozen:
            self.lbl_tooltip.setText("TOOLTIP: FROZEN")
            self.lbl_tooltip.setStyleSheet(f"color: {ACCENT_GOLD}; font-weight: bold;")
        else:
            self.lbl_tooltip.setText("TOOLTIP: ON")
            self.lbl_tooltip.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold;")

    def _toggle_markup(self):
        self.markup_enabled = not self.markup_enabled
        self.lbl_markup.setText(f"MARKUP: {'ON' if self.markup_enabled else 'OFF'}")
        self.lbl_markup.setStyleSheet(f"color: {ACCENT_GOLD if self.markup_enabled else DIM}; font-weight: bold;")

    def _clear_markup(self):
        for item in self.markup_lines:
            self.p_m.removeItem(item)
        self.markup_lines = []

    def _on_mouse_clicked(self, event):
        if not self.markup_enabled or event.button() != Qt.MouseButton.LeftButton: return
        pos = event.scenePos()
        if not self.p_m.sceneBoundingRect().contains(pos): return
        mouse_point = self.p_m.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        vl = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
        hl = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(ACCENT_GOLD, width=1, style=Qt.PenStyle.DashLine))
        txt = pg.TextItem(text=f"({x:.0f}, {y:.1f})", color=ACCENT_GOLD, anchor=(0, 1))
        txt.setPos(x, y)
        
        self.p_m.addItem(vl); self.p_m.addItem(hl); self.p_m.addItem(txt)
        self.markup_lines.extend([vl, hl, txt])
        print(f"Markup point added: X={x:.0f}, Y={y:.1f}")

    def _build_dashboard_tab(self):
        dash_container = QWidget()
        dash_layout = QVBoxLayout(dash_container); dash_layout.setContentsMargins(12, 12, 12, 12); dash_layout.setSpacing(12)
        filter_layout = QHBoxLayout(); filter_layout.addWidget(QLabel("Dashboard Product:")); self.cb_dash_prod = QComboBox()
        self.cb_dash_prod.currentTextChanged.connect(self._update_dashboard); filter_layout.addWidget(self.cb_dash_prod); filter_layout.addStretch(); dash_layout.addLayout(filter_layout)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setFrameShape(QFrame.Shape.NoFrame); scroll_area.setStyleSheet(f"background-color: {BG};")
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content); scroll_layout.setContentsMargins(0, 0, 0, 0); scroll_layout.setSpacing(15)
        metrics_frame = QFrame(); metrics_frame.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;"); metrics_layout = QVBoxLayout(metrics_frame)
        self.lbl_sharpe = QLabel("Sharpe Ratio: --"); self.lbl_sharpe.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 11pt;"); metrics_layout.addWidget(self.lbl_sharpe)
        h_metrics = QHBoxLayout(); self.lbl_winrate = QLabel("Win Rate: --"); self.lbl_volume = QLabel("Total Volume Traded: --")
        self.lbl_my_trades = QLabel("My Trades: --"); self.lbl_bot_trades = QLabel("Bot Trades: --")
        for l in [self.lbl_winrate, self.lbl_volume, self.lbl_my_trades, self.lbl_bot_trades]: h_metrics.addWidget(l)
        h_metrics.addStretch(); metrics_layout.addLayout(h_metrics); scroll_layout.addWidget(metrics_frame)
        self.gw_dash_pnl = pg.GraphicsLayoutWidget(); self.gw_dash_pnl.setBackground(BG); self.gw_dash_pnl.setFixedHeight(300)
        self.p_dash_pnl = self.gw_dash_pnl.addPlot(title="Cumulative PnL"); self.curve_dash_pnl = self.p_dash_pnl.plot(pen=pg.mkPen(ACCENT_GREEN, width=2)); scroll_layout.addWidget(self.gw_dash_pnl)
        dd_container = QWidget(); dd_layout = QVBoxLayout(dd_container); dd_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_dd_pct = QCheckBox("Show Percentage %"); self.chk_dd_pct.stateChanged.connect(self._update_dashboard); dd_layout.addWidget(self.chk_dd_pct)
        self.gw_dash_dd = pg.GraphicsLayoutWidget(); self.gw_dash_dd.setBackground(BG); self.gw_dash_dd.setFixedHeight(300)
        self.p_dash_dd = self.gw_dash_dd.addPlot(title="Drawdown"); self.p_dash_dd.setXLink(self.p_dash_pnl); self.curve_dash_dd = self.p_dash_dd.plot(pen=pg.mkPen(ACCENT_RED, width=2, fillLevel=0, brush=(255, 61, 90, 50))); dd_layout.addWidget(self.gw_dash_dd)
        scroll_layout.addWidget(dd_container); scroll_layout.addStretch(); scroll_area.setWidget(scroll_content); dash_layout.addWidget(scroll_area)
        self.tabs.insertTab(0, dash_container, "Dashboard"); self.tabs.setCurrentIndex(0)

    def _build_sandbox_tab(self):
        sandbox_container = QWidget(); sandbox_layout = QVBoxLayout(sandbox_container); sandbox_layout.setContentsMargins(0, 0, 0, 0); sandbox_layout.setSpacing(0)
        sb_filter_bar = QHBoxLayout(); sb_filter_bar.setContentsMargins(8, 6, 8, 6); self.sb_filter_input = QLineEdit()
        self.sb_filter_input.setPlaceholderText("Filter sandbox logs..."); self.sb_filter_input.textChanged.connect(self._filter_sandbox_table)
        sb_filter_bar.addWidget(QLabel("🔍")); sb_filter_bar.addWidget(self.sb_filter_input); sandbox_layout.addLayout(sb_filter_bar)
        sb_content_layout = QHBoxLayout(); self.sb_table = QTableWidget(); self.sb_table.setColumnCount(2); self.sb_table.setHorizontalHeaderLabels(['Message', 'Occurrences'])
        self.sb_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch); self.sb_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sb_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.sb_table.verticalHeader().setVisible(False); self.sb_table.itemSelectionChanged.connect(self._on_sb_selection_changed)
        sb_content_layout.addWidget(self.sb_table, stretch=2); self.sb_detail_panel = QFrame(); self.sb_detail_panel.setStyleSheet(f"background-color: {PANEL_BG}; border-left: 1px solid {BORDER};"); self.sb_detail_panel.setFixedWidth(300)
        detail_layout = QVBoxLayout(self.sb_detail_panel); detail_layout.addWidget(QLabel("<b>Occurrences</b>")); self.sb_detail_text = QLabel("Select message")
        self.sb_detail_text.setWordWrap(True); detail_scroll = QScrollArea(); detail_scroll.setWidgetResizable(True); detail_scroll.setWidget(self.sb_detail_text)
        detail_layout.addWidget(detail_scroll); sb_content_layout.addWidget(self.sb_detail_panel); sandbox_layout.addLayout(sb_content_layout)
        self.tabs.addTab(sandbox_container, "SandboxLog")

    def _on_sb_selection_changed(self):
        items = self.sb_table.selectedItems()
        if not items: return
        msg = items[0].text()
        if msg in self.sandbox_msgs:
            ts_list = self.sandbox_msgs[msg]; txt = f"<b>Total: {len(ts_list)}</b><br><br>" + "<br>".join([str(ts) for ts in ts_list[:50]])
            if len(ts_list) > 50: txt += "<br>..."
            self.sb_detail_text.setText(txt)

    def _filter_sandbox_table(self, text):
        for row in range(self.sb_table.rowCount()):
            item = self.sb_table.item(row, 0)
            if item: self.sb_table.setRowHidden(row, text.lower() not in item.text().lower())

    def _on_mouse_moved(self, pos):
        if not self.p_m.sceneBoundingRect().contains(pos) or self.current_df is None: return
        mouse_point = self.p_m.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        plot_t = self._plot_t if self._plot_t is not None else self.current_df['timestamp'].to_numpy()
        idx = int(np.clip(np.searchsorted(plot_t, x), 0, len(plot_t) - 1))
        ts_plot = plot_t[idx]

        row = self.current_df.row(idx, named=True)
        raw_ts = int(row['timestamp'])
        row_day = row.get('day', None)
        mid = row.get('mid_price', 0) or 0
        pnl = row.get('profit_and_loss', 0) or 0

        self.v_line.setPos(ts_plot); self.h_line.setPos(y)

        if not self.tooltip_enabled: return
        if self.tooltip_frozen: return

        lines = []
        def h(color, text): return f'<span style="color:{color}">{text}</span>'
        def sec(title): lines.append(h(ACCENT_CYAN, f'<b>{title}</b>'))
        def row_line(label, val, color=TEXT): lines.append(f'&nbsp;&nbsp;{h(color, label)}&nbsp;{val}')

        # Cursor
        sec('CURSOR')
        row_line('TS    :', str(raw_ts))
        row_line('Price :', f'{y:.2f}')
        lines.append('')

        # Market
        sec('MARKET')
        row_line('Mid   :', f'{mid:,.2f}')
        row_line('PnL   :', f'{pnl:,.0f}', ACCENT_GREEN if pnl >= 0 else ACCENT_RED)
        lines.append('')

        # Orderbook from prices_df row
        bid_p_cols = sorted([c for c in self.current_df.columns if 'bid_price_' in c], key=lambda c: int(c.split('_')[-1]))
        ask_p_cols = sorted([c for c in self.current_df.columns if 'ask_price_' in c], key=lambda c: int(c.split('_')[-1]))
        if bid_p_cols or ask_p_cols:
            sec('ORDERBOOK')
            for pc in reversed(ask_p_cols):
                lvl = pc.split('_')[-1]; vc = f'ask_volume_{lvl}'
                p = row.get(pc); v = row.get(vc)
                if p and v: row_line(f'ASK{lvl}', f'{p:.0f} &times; {int(v)}', ACCENT_ORANGE)
            lines.append(f'&nbsp;&nbsp;{h(DIM, "&#9472;" * 16)}')
            for pc in bid_p_cols:
                lvl = pc.split('_')[-1]; vc = f'bid_volume_{lvl}'
                p = row.get(pc); v = row.get(vc)
                if p and v: row_line(f'BID{lvl}', f'{p:.0f} &times; {int(v)}', ACCENT_CYAN)
            lines.append('')

        # OB heatmap volume at cursor
        if self.ob_res:
            y_levels = self.ob_res['levels']
            y_idx = int(np.clip(np.searchsorted(y_levels, y), 0, len(y_levels) - 1))
            vol = self.ob_res['raw_vol'][y_idx, idx]
            if vol > 0:
                sec('OB HEATMAP')
                row_line(f'Vol @ {y_levels[y_idx]:.0f}', f':{vol:,.0f}')
                lines.append('')

        # Order placements at this ts
        prod = self.cb_prod.currentText()
        day_sel = self.cb_day.currentText()
        _tag_sel = self.cb_tag.checkedItems() if hasattr(self, 'cb_tag') and self.cb_tag.count() > 0 else []
        orders_here = [o for o in self.data.get('orders', []) if o['ts'] == raw_ts
                       and (o.get('product', '') in ('', prod))
                       and (row_day is None or o.get('day', row_day) == row_day)
                       and (not _tag_sel or ((o.get('tag') or '').strip() or 'Untagged') in _tag_sel)]
        if orders_here:
            sec('ORDERS PLACED')
            for o in orders_here:
                col = ACCENT_CYAN if o['side'] == 'BUY' else ACCENT_ORANGE
                tag = o.get('tag', '')
                row_line(o['side'], f'{o["price"]:.0f} &times; {o["qty"]}  [{tag}]', col)
            lines.append('')

        # Trades at this ts
        trades_here = []
        for tr in self.data.get('trades', []):
            if tr.get('symbol') != prod: continue
            if day_sel != 'All' and 'day' in tr and str(tr['day']) != day_sel: continue
            if tr.get('timestamp') == raw_ts and (row_day is None or tr.get('day', row_day) == row_day):
                trades_here.append(tr)
        if trades_here:
            sec('TRADES')
            for tr in trades_here:
                is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
                is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
                qty = tr.get('quantity', 0); pr = tr.get('price', 0)
                if is_b: row_line('MY BUY ', f'{pr} &times; {qty}', ACCENT_CYAN)
                elif is_s: row_line('MY SELL', f'{pr} &times; {qty}', ACCENT_ORANGE)
                else: row_line('BOT    ', f'{pr} &times; {qty}', DIM)
            lines.append('')

        # Custom data
        custom = self.data.get('custom', {})
        if custom:
            sec('CUSTOM DATA')
            for i, (name, pts) in enumerate(custom.items()):
                if not pts: continue
                ts_arr = np.array([p[0] for p in pts])
                ci = int(np.clip(np.searchsorted(ts_arr, x), 0, len(ts_arr) - 1))
                val = pts[ci][1]
                color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
                row_line(name, f': {val:.5g}', color)

        html = (
            f'<div style="font-family:\'JetBrains Mono\',\'Consolas\',monospace;'
            f'font-size:8pt;line-height:1.5;">' + '<br>'.join(lines) + '</div>'
        )
        self._info_label.setText(html)
        self._reposition_info_panel()
        self.info_panel.setVisible(True)

    def _set_zoom(self, mode):
        self.lbl_zoom.setText(f"Mode: {mode.upper()}"); self.p_m.setMouseEnabled(x=(mode in ['x', 'xy']), y=(mode in ['y', 'xy']))

    def _autoscale_all(self):
        for p in [self.p_m, self.p_pnl, self.p_pos, self.p_dash_pnl, self.p_dash_dd]:
            if p: p.autoRange()

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Log", "", "Log (*.log *.json)")
        if path: self._load_file(path)

    def _export_custom_csv(self):
        custom = (self.data or {}).get('custom', {})
        if not custom: return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "custom.csv", "CSV (*.csv)")
        if not path: return
        all_ts = sorted(set(ts for pts in custom.values() for ts, _ in pts))
        rows = ['timestamp,' + ','.join(custom.keys())]
        ts_map = {k: dict(pts) for k, pts in custom.items()}
        for ts in all_ts: rows.append(','.join([str(int(ts))] + [str(ts_map[k].get(ts, '')) for k in custom]))
        with open(path, 'w') as f: f.write('\n'.join(rows))

    def _import_data(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        if not os.path.exists(data_dir): return
        p_dfs, t_dicts = [], []
        for f in os.listdir(data_dir):
            if not f.endswith('.csv'): continue
            filepath = os.path.join(data_dir, f)
            if f.startswith("prices_"): p_dfs.append(pl.read_csv(filepath, separator=";", null_values=['', 'nan']))
            elif f.startswith("trades_"):
                day_match = re.search(r"day_(-?\d+)", f); df = pl.read_csv(filepath, separator=";", null_values=['', 'nan'])
                if day_match: df = df.with_columns(pl.lit(int(day_match.group(1))).alias("day"))
                t_dicts.extend(df.to_dicts())
        if not p_dfs: return
        df = pl.concat(p_dfs, how='diagonal_relaxed').sort(['day', 'timestamp'])
        self.data = {'prices_df': df, 'trades': t_dicts, 'custom': {}, 'debug': [], 'orders': [], '_continuous_ts': _is_timestamps_continuous(df), '_min_day': df['day'].min() if 'day' in df.columns else 0}
        self.trade_tag_map = {}
        self.cb_prod.blockSignals(True); products = df['product'].unique().sort().to_list()
        self.cb_prod.clear(); self.cb_prod.addItems(products); self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()])
        self.cb_prod.blockSignals(False); self.cb_dash_prod.clear(); self.cb_dash_prod.addItems(['Overall'] + products)
        self.cb_pnl_prod.blockSignals(True); self.cb_pnl_prod.clear(); self.cb_pnl_prod.addItems(['Overall'] + products); self.cb_pnl_prod.blockSignals(False)
        self.cb_tag.blockSignals(True); self.cb_tag.clear(); self.cb_tag.addItems(['Untagged']); self.cb_tag.blockSignals(False)
        self._apply_saved_settings()
        self._build_custom_plots(); self._build_position_plot(); self._build_logs_table(); self._process_selection()
        self._apply_custom_curve_visibility(); self._update_dashboard()

    def _load_file(self, path):
        self._current_log_path = path
        with open(path, encoding='utf-8') as f: raw = json.load(f)
        csv_str = raw.get('activitiesLog', '').replace('\\n', '\n')
        df = pl.read_csv(StringIO(csv_str), separator=';', null_values=['', 'nan'])
        df = df.rename({c: c.strip() for c in df.columns}).sort(['day', 'timestamp'])
        # ts→day lookup so LOGORDER gets the actual day, not ts//1000000
        ts_to_day = dict(df.select(['timestamp', 'day']).unique().iter_rows()) if 'day' in df.columns else {}
        custom, orders, sandbox_msgs, debug_msgs = {}, [], {}, []
        for entry in raw.get('logs', []):
            ts, log = entry.get('timestamp', 0), entry.get('lambdaLog', '') or ''
            sb_log = (entry.get('sandboxLog', '') or '').strip()
            if sb_log: sandbox_msgs.setdefault(sb_log, []).append(ts)
            for line in log.split('\n'):
                if line.startswith('LOGVIZ:'):
                    try:
                        p = json.loads(line[7:])
                        for k, v in p.items(): custom.setdefault(k, []).append((ts, float(v)))
                    except: pass
                elif line.startswith('LOGORDER:'):
                    parts = line.split(':')
                    if len(parts) >= 5:
                        orders.append({'ts': ts, 'day': ts_to_day.get(ts, ts//1000000), 'product': parts[1] if len(parts)==6 else '', 'side': parts[2] if len(parts)==6 else parts[1],
                                     'price': float(parts[3] if len(parts)==6 else parts[2]), 'qty': int(parts[4] if len(parts)==6 else parts[3]), 'tag': parts[-1]})
                elif line.startswith('LOGDBG:'):
                    parts = line[7:].split(':', 2)
                    debug_msgs.append({'ts': ts, 'tag': parts[0], 'product': parts[1] if len(parts)>2 else '', 'msg': parts[-1]})
        if raw.get('error'): sandbox_msgs.setdefault(str(raw['error']).strip(), []).append("N/A")
        self.sandbox_msgs = sandbox_msgs
        min_day = df['day'].min() if 'day' in df.columns else 0
        self.data = {'prices_df': df, 'trades': raw.get('tradeHistory', []), 'custom': custom, 'debug': debug_msgs, 'orders': orders, '_continuous_ts': _is_timestamps_continuous(df), '_min_day': min_day}
        self.trade_tag_map = self._build_trade_tag_map(orders)
        unique_tags = sorted({(o.get('tag') or '').strip() for o in orders if (o.get('tag') or '').strip()})
        self.cb_prod.blockSignals(True); products = df['product'].unique().sort().to_list(); self.cb_prod.clear(); self.cb_prod.addItems(products)
        self.cb_day.clear(); self.cb_day.addItems(['All'] + [str(d) for d in df['day'].unique().sort().to_list()]); self.cb_prod.blockSignals(False)
        self.cb_dash_prod.clear(); self.cb_dash_prod.addItems(['Overall'] + products)
        self.cb_pnl_prod.blockSignals(True); self.cb_pnl_prod.clear(); self.cb_pnl_prod.addItems(['Overall'] + products); self.cb_pnl_prod.blockSignals(False)
        self.cb_tag.blockSignals(True); self.cb_tag.clear(); self.cb_tag.addItems(unique_tags + ['Untagged']); self.cb_tag.blockSignals(False)
        for c in self.custom_curves.values():
            try: self.p_m.removeItem(c)
            except: pass
            try: self.p_gen.removeItem(c)
            except: pass
        self.custom_curves = {}
        self.data_settings = {}
        self._apply_saved_settings()
        self.btn_backtest.setVisible(self._is_backtest_log(path))
        self.btn_submit.setVisible(self._is_submission_log(path))
        self._build_custom_plots(); self._build_position_plot(); self._build_logs_table(); self._update_sandbox_table(); self._process_selection()
        self._apply_custom_curve_visibility(); self._update_dashboard()

    def _update_sandbox_table(self):
        self.sb_table.setRowCount(len(self.sandbox_msgs))
        for i, (msg, ts_list) in enumerate(sorted(self.sandbox_msgs.items(), key=lambda x: len(x[1]), reverse=True)):
            m_item = QTableWidgetItem(msg); m_item.setForeground(QBrush(QColor(ACCENT_RED)))
            c_item = QTableWidgetItem(str(len(ts_list))); c_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sb_table.setItem(i, 0, m_item); self.sb_table.setItem(i, 1, c_item)

    def _build_logs_table(self):
        debug_msgs = self.data.get('debug', []); df = self.data.get('prices_df'); trades = self.data.get('trades', [])
        self.logs_table.setRowCount(len(debug_msgs))
        for i, entry in enumerate(debug_msgs):
            tag = entry['tag'].upper(); row_color = {'ERR':QColor(ACCENT_RED),'WARN':QColor(ACCENT_GOLD),'INFO':QColor(ACCENT_CYAN)}.get(tag, QColor(TEXT))
            items = [str(entry['ts']), tag, entry.get('product',''), '', '', entry['msg']]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text); item.setForeground(QBrush(row_color)); self.logs_table.setItem(i, col, item)

    def _filter_logs_table(self, text):
        for row in range(self.logs_table.rowCount()):
            match = any(text.lower() in (self.logs_table.item(row, c).text() if self.logs_table.item(row, c) else '').lower() for c in range(6))
            self.logs_table.setRowHidden(row, not match)

    def _build_custom_plots(self):
        self.gw_c.clear(); custom = self.data.get('custom', {}); anchor = None
        for i, (name, pts) in enumerate(custom.items()):
            p = self.gw_c.addPlot(row=i, col=0, title=name); p.setDownsampling(auto=True, mode='peak')
            if anchor: p.setXLink(anchor)
            else: anchor = p
            p.plot([x[0] for x in pts], [x[1] for x in pts], pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2))

    def _build_position_plot(self):
        for c in self.pos_curves.values(): self.p_pos.removeItem(c)
        self.pos_curves.clear(); trades = self.data.get('trades', [])
        if not trades: return
        tags = self.cb_tag.checkedItems() if hasattr(self, 'cb_tag') else []
        pos_by_sym = {}; cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        for tr in trades:
            if not self._trade_matches_tag(tr, tags): continue
            is_b = str(tr.get('buyer','')).upper()=='SUBMISSION'; is_s = str(tr.get('seller','')).upper()=='SUBMISSION'
            if not is_b and not is_s: continue
            sym, qty, ts = tr.get('symbol',''), tr.get('quantity',0), tr.get('timestamp',0)
            pts = ts + (tr.get('day', min_day) - min_day)*1000000 if 'day' in tr and not cont_ts else ts
            pos_by_sym.setdefault(sym, []).append((tr.get('day', ts//1000000), pts, qty if is_b else -qty))
        for i, sym in enumerate(sorted(pos_by_sym.keys())):
            events = sorted(pos_by_sym[sym], key=lambda x: x[1]); ts_list, pos_list, cum, last_day = [], [], 0, events[0][0]
            for day, pts, delta in events:
                if day != last_day: cum = 0; last_day = day
                cum += delta; ts_list.append(pts); pos_list.append(cum)
            self.pos_curves[sym] = self.p_pos.plot(ts_list, pos_list, pen=pg.mkPen(CUSTOM_COLORS[i % len(CUSTOM_COLORS)], width=2), name=sym, stepMode='right')

    def _compute_position_series(self, t, prod, day, cont_ts, min_day, tags=None):
        """Return list of cumulative position values aligned to t array."""
        trade_events = []
        for tr in self.data.get('trades', []):
            if tr.get('symbol') != prod: continue
            if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
            if not self._trade_matches_tag(tr, tags): continue
            is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if not is_b and not is_s: continue
            tr_ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr and not cont_ts: tr_ts += (tr['day'] - min_day) * 1000000
            qty = int(tr.get('quantity', 0))
            trade_events.append((tr_ts, qty if is_b else -qty))
        trade_events.sort(key=lambda x: x[0])
        # build ts→cum_pos map (last value wins per ts)
        cum, pos_at_t = 0, {}
        for tr_ts, delta in trade_events:
            cum += delta
            pos_at_t[tr_ts] = cum
        t_list = t.tolist()
        result = []
        last_pos = 0
        for ts_val in t_list:
            if ts_val in pos_at_t: last_pos = pos_at_t[ts_val]
            result.append(last_pos)
        return result

    def _build_trade_tag_map(self, orders):
        """(ts, product, side, price_int) → tag for SUBMISSION trades."""
        m = {}
        for o in orders:
            tag = (o.get('tag') or '').strip() or 'Untagged'
            m[(o['ts'], o['product'], o['side'], int(o['price']))] = tag
        return m

    def _get_trade_tag(self, tr):
        """Return tag for a SUBMISSION trade, or None if bot trade."""
        is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
        is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
        if not is_b and not is_s:
            return None
        side = 'BUY' if is_b else 'SELL'
        key = (tr.get('timestamp', 0), tr.get('symbol', ''), side, int(tr.get('price', 0)))
        return self.trade_tag_map.get(key, 'Untagged')

    def _trade_matches_tag(self, tr, tags):
        """True if trade should be included for the given tag filter (empty list = all)."""
        if not tags:
            return True
        tr_tag = self._get_trade_tag(tr)
        if tr_tag is None:
            return False  # bot trade — excluded when a specific tag is selected
        return tr_tag in tags

    def _compute_valuation_pnl_aligned(self, t, prod, day, mid, cont_ts, min_day, tags):
        """Valuation PnL (cash + pos*mid) aligned to t, filtered by tag."""
        trade_events = []
        for tr in self.data.get('trades', []):
            if tr.get('symbol') != prod: continue
            if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
            if not self._trade_matches_tag(tr, tags): continue
            is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if not is_b and not is_s: continue
            tr_ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr and not cont_ts: tr_ts += (tr['day'] - min_day) * 1000000
            qs = int(tr.get('quantity', 0)) * (1 if is_b else -1)
            trade_events.append((tr_ts, qs, float(tr.get('price', 0))))
        trade_events.sort()
        cash_at, pos_at, cc, cp = {}, {}, 0.0, 0
        for tr_ts, qs, pr in trade_events:
            cc -= qs * pr; cp += qs
            cash_at[tr_ts] = cc; pos_at[tr_ts] = cp
        result, lc, lp = [], 0.0, 0
        for i, ts_val in enumerate(t.tolist()):
            if ts_val in cash_at: lc = cash_at[ts_val]; lp = pos_at[ts_val]
            result.append(lc + lp * (float(mid[i]) if i < len(mid) else 0.0))
        return result

    def _on_tag_changed(self):
        if not self.data:
            return
        self._build_position_plot()
        self._process_selection()

    def _compute_pnl(self, pnl_prod, day, pnl_type, cont_ts, min_day, tags=None):
        df = self.data['prices_df']
        if pnl_prod == 'Overall':
            sub = df
            if day != 'All':
                try: sub = sub.filter(pl.col('day') == int(day))
                except: pass
            if day == 'All' and 'day' in sub.columns and not cont_ts:
                sub = sub.with_columns((pl.col('timestamp') + (pl.col('day') - min_day) * 1000000).alias('_cts'))
                t_col = '_cts'
            else:
                t_col = 'timestamp'
            if pnl_type == 'Log PnL' and not tags:
                if 'profit_and_loss' not in sub.columns:
                    ts_arr = sub.select(t_col).unique().sort(t_col)[t_col].to_numpy()
                    return ts_arr, np.zeros(len(ts_arr))
                agg = sub.group_by(t_col).agg(pl.col('profit_and_loss').sum().alias('pnl')).sort(t_col)
                return agg[t_col].to_numpy(), agg['pnl'].to_numpy()
            if pnl_type == 'Log PnL':
                pnl_type = 'Valuation PnL'  # log PnL can't be split by tag; fall back to trade-based
                ts_arr = sub.select(t_col).unique().sort(t_col)[t_col].to_numpy()
                t_list = ts_arr.tolist()
                combined = np.zeros(len(t_list))
                for ap in sub['product'].unique().to_list():
                    ap_mid = {}
                    if pnl_type == 'Valuation PnL':
                        ap_sub = sub.filter(pl.col('product') == ap)
                        ap_mid = dict(zip(ap_sub[t_col].to_list(), ap_sub['mid_price'].to_list()))
                    trade_events = []
                    for tr in self.data['trades']:
                        if tr.get('symbol') != ap: continue
                        if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
                        if not self._trade_matches_tag(tr, tags): continue
                        tr_ts = tr.get('timestamp', 0)
                        if day == 'All' and 'day' in tr and not cont_ts: tr_ts += (tr['day'] - min_day) * 1000000
                        is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
                        is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
                        if not is_b and not is_s: continue
                        qs = int(tr.get('quantity', 0)) * (1 if is_b else -1)
                        trade_events.append((tr_ts, qs, float(tr.get('price', 0))))
                    trade_events.sort()
                    cash_at, pos_at, cc, cp = {}, {}, 0.0, 0
                    for tr_ts, qs, pr in trade_events:
                        cc -= qs * pr; cp += qs
                        cash_at[tr_ts] = cc; pos_at[tr_ts] = cp
                    lc, lp = 0.0, 0
                    for i, ts_val in enumerate(t_list):
                        if ts_val in cash_at: lc = cash_at[ts_val]; lp = pos_at[ts_val]
                        if pnl_type == 'Realized PnL':
                            combined[i] += lc
                        else:
                            combined[i] += lc + lp * (ap_mid.get(ts_val, 0.0) or 0.0)
                return ts_arr, combined
        else:
            sub = df.filter(pl.col('product') == pnl_prod)
            if day != 'All':
                try: sub = sub.filter(pl.col('day') == int(day))
                except: pass
            pnl_t = sub['timestamp'].to_numpy()
            if day == 'All' and 'day' in sub.columns and not cont_ts:
                pnl_t = pnl_t + (sub['day'].to_numpy() - min_day) * 1000000
            mid = sub['mid_price'].to_numpy()
            if pnl_type == 'Log PnL' and not tags:
                return pnl_t, (sub['profit_and_loss'].to_numpy() if 'profit_and_loss' in sub.columns else np.zeros(len(pnl_t)))
            if pnl_type == 'Log PnL':
                pnl_type = 'Valuation PnL'  # log PnL can't be split by tag; fall back to trade-based
            trade_events = []
            for tr in self.data['trades']:
                if tr.get('symbol') != pnl_prod: continue
                if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
                if not self._trade_matches_tag(tr, tags): continue
                tr_ts = tr.get('timestamp', 0)
                if day == 'All' and 'day' in tr and not cont_ts: tr_ts += (tr['day'] - min_day) * 1000000
                is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
                is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
                if not is_b and not is_s: continue
                qs = int(tr.get('quantity', 0)) * (1 if is_b else -1)
                trade_events.append((tr_ts, qs, float(tr.get('price', 0))))
            trade_events.sort()
            cash_at, pos_at, cc, cp = {}, {}, 0.0, 0
            for tr_ts, qs, pr in trade_events:
                cc -= qs * pr; cp += qs
                cash_at[tr_ts] = cc; pos_at[tr_ts] = cp
            realized = np.zeros(len(pnl_t))
            lc, lp = 0.0, 0
            for i, ts_val in enumerate(pnl_t.tolist()):
                if ts_val in cash_at: lc = cash_at[ts_val]; lp = pos_at[ts_val]
                realized[i] = lc if pnl_type == 'Realized PnL' else lc + lp * (mid[i] if i < len(mid) else 0.0)
            return pnl_t, realized

    def _process_selection(self):
        if not self.data or not self.cb_prod.currentText(): return
        prod, day = self.cb_prod.currentText(), self.cb_day.currentText()
        self.current_df = self.data['prices_df'].filter(pl.col('product') == prod)
        try:
            if day != 'All': self.current_df = self.current_df.filter(pl.col('day') == int(day))
        except: pass
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t = self.current_df['timestamp'].to_numpy()
        if day == 'All' and 'day' in self.current_df.columns and not cont_ts: t = t + (self.current_df['day'].to_numpy() - min_day) * 1000000
        self._plot_t = t
        mid = self.current_df['mid_price'].to_numpy(); self.curve_mid.setData(t, mid)
        tags = self.cb_tag.checkedItems() if hasattr(self, 'cb_tag') and self.cb_tag.count() > 0 else []
        pnl_prod = self.cb_pnl_prod.currentText() if self.cb_pnl_prod.count() > 0 else prod
        pnl_type = self.cb_pnl_type.currentText()
        pnl_t, pnl_data = self._compute_pnl(pnl_prod, day, pnl_type, cont_ts, min_day, tags)
        self.curve_pnl.setData(pnl_t, pnl_data)

        mb_t, mb_p, ms_t, ms_p, bot_raw = [], [], [], [], []
        for tr in self.data['trades']:
            if tr.get('symbol') != prod: continue
            if day != 'All' and 'day' in tr and tr['day'] != int(day): continue
            ts = tr.get('timestamp', 0)
            if day == 'All' and 'day' in tr and not cont_ts: ts += (tr['day'] - min_day) * 1000000
            is_b = str(tr.get('buyer','')).upper()=='SUBMISSION'; is_s = str(tr.get('seller','')).upper()=='SUBMISSION'
            if is_b:
                if self._trade_matches_tag(tr, tags): mb_t.append(ts); mb_p.append(tr['price'])
            elif is_s:
                if self._trade_matches_tag(tr, tags): ms_t.append(ts); ms_p.append(tr['price'])
            else: bot_raw.append((ts, tr['price'], int(tr.get('quantity', 1))))
        
        vols = np.array([v for _, _, v in bot_raw]) if bot_raw else np.array([1])
        q_edges = np.unique(np.percentile(vols, np.linspace(0, 100, 11))) if len(bot_raw)>1 else np.array([0, 1000000])
        n_buckets = len(q_edges)-1
        b_t, b_p, b_brushes, b_sizes = [], [], [], []
        for ts, pr, vol in bot_raw:
            b_t.append(ts); b_p.append(pr); buck = min(np.searchsorted(q_edges[1:], vol), n_buckets-1)
            pal_idx = int(buck * (len(TRADE_VOLUME_COLORS)-1) / max(n_buckets-1, 1))
            b_brushes.append(pg.mkBrush(TRADE_VOLUME_COLORS[pal_idx])); b_sizes.append(6 + buck * 2)
        self.sc_buy.setData(x=mb_t, y=mb_p); self.sc_sell.setData(x=ms_t, y=ms_p); self.sc_bot.setData(x=b_t, y=b_p, brush=b_brushes, size=b_sizes)

        self.ob_res = build_ob_heatmap(self.data['prices_df'], prod, day, cont_ts, BUY_VOLUME_COLORS, SELL_VOLUME_COLORS)
        if self.ob_res:
            self.img_item.setImage(self.ob_res['img'], autoLevels=False)
            self.img_item.setRect(get_rect(t, self.ob_res['levels']))
            self.img_item.setVisible(True)
            filtered_orders = [o for o in self.data.get('orders', []) if not tags or ((o.get('tag') or '').strip() or 'Untagged') in tags]
            o_res = build_order_placement_heatmap(filtered_orders, prod, day, t, cont_ts, min_day, ORDER_BUY_COLORS, ORDER_SELL_COLORS)
            if o_res:
                self.img_orders.setImage(o_res['img'], autoLevels=False)
                self.img_orders.setRect(get_rect(t, o_res['levels']))
            else: self.img_orders.clear()
            self.hm_legend.update_ranges(self.ob_res['max_vol'], q_edges)
        else: self.img_item.setVisible(False); self.img_orders.clear()

        # ── Built-in overlay curves (Position, PnL) ──────────────────────────
        if tags:
            _builtin_pnl = self._compute_valuation_pnl_aligned(t, prod, day, mid, cont_ts, min_day, tags)
        else:
            _builtin_pnl = (self.current_df['profit_and_loss'].to_numpy().tolist()
                            if 'profit_and_loss' in self.current_df.columns else [0.0] * len(t))
        builtin_data = {
            '__POSITION__': (t.tolist(), self._compute_position_series(t, prod, day, cont_ts, min_day, tags)),
            '__PNL__': (t.tolist(), _builtin_pnl),
        }
        builtin_colors = {'__POSITION__': ACCENT_PURPLE, '__PNL__': ACCENT_GREEN}
        builtin_labels = {'__POSITION__': 'Position', '__PNL__': 'PnL (Log)'}

        # Update Custom Data Curves
        custom_data = self.data.get('custom', {})
        t_min, t_max = (t[0], t[-1]) if len(t) > 0 else (0, 1)

        has_generic_data = False

        # --- built-ins ---
        for bkey, (bts, bvals) in builtin_data.items():
            pane = self.data_settings.get(bkey, 'disabled')
            if pane == 'disabled':
                if bkey in self.custom_curves:
                    try: self.p_m.removeItem(self.custom_curves[bkey])
                    except: pass
                    try: self.p_gen.removeItem(self.custom_curves[bkey])
                    except: pass
                    del self.custom_curves[bkey]
                continue
            if pane == 'generic': has_generic_data = True
            target_plot = self.p_m if pane == 'main' else self.p_gen
            color = builtin_colors[bkey]
            label = builtin_labels[bkey]
            if bkey not in self.custom_curves:
                curve = target_plot.plot(pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine), name=label)
                self.custom_curves[bkey] = curve
                if pane == 'main': self.leg_m.addItem(curve, f"[B] {label}")
            curve = self.custom_curves[bkey]
            curve.setData(bts, bvals)

        # --- custom keys ---
        for name, pts in custom_data.items():
            pane = self.data_settings.get(name, 'generic')
            if pane == 'disabled':
                if name in self.custom_curves:
                    try: self.p_m.removeItem(self.custom_curves[name])
                    except: pass
                    try: self.p_gen.removeItem(self.custom_curves[name])
                    except: pass
                    del self.custom_curves[name]
                continue
            target_plot = self.p_m if pane == 'main' else self.p_gen
            if pane == 'generic': has_generic_data = True
            if name not in self.custom_curves:
                color = CUSTOM_COLORS[len([k for k in self.custom_curves if k not in BUILTIN_KEYS]) % len(CUSTOM_COLORS)]
                curve = target_plot.plot(pen=pg.mkPen(color, width=1.5), name=name)
                self.custom_curves[name] = curve
                if pane == 'main': self.leg_m.addItem(curve, f"[C] {name}")
            curve = self.custom_curves[name]
            pts_f = [p for p in pts if t_min <= p[0] <= t_max]
            curve.setData([p[0] for p in pts_f], [p[1] for p in pts_f]) if pts_f else curve.setData([], [])

        self.p_gen.setVisible(has_generic_data and not self._gen_pane_minimized)
        self.p_m.autoRange(); self.p_pnl.autoRange()

    def _update_dashboard(self):
        if not self.data: return
        p, d, df = self.cb_dash_prod.currentText(), self.cb_day.currentText(), self.data['prices_df']
        if p != 'Overall': df = df.filter(pl.col('product') == p)
        cont_ts = self.data.get('_continuous_ts', False); min_day = self.data.get('_min_day', 0)
        t_col = 'cts' if (d=='All' and 'day' in df.columns and not cont_ts) else 'timestamp'
        if t_col == 'cts': df = df.with_columns((pl.col('timestamp') + (pl.col('day') - min_day) * 1000000).alias(t_col))
        df = df.sort(t_col); agg = df.group_by(t_col).agg(pl.col('profit_and_loss').sum().alias('pnl')).sort(t_col)
        t, pnl = agg[t_col].to_numpy(), agg['pnl'].to_numpy()
        self.curve_dash_pnl.setData(t, pnl)
        if len(pnl)>0:
            pk = np.maximum.accumulate(pnl); dd = pk - pnl
            if self.chk_dd_pct.isChecked() and np.any(pk>0): dd = (dd / np.where(pk>0, pk, 1)) * 100
            self.curve_dash_dd.setData(t, dd)
            returns = np.diff(pnl)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))
            else:
                sharpe = float('nan')
            win_rate = np.sum(returns > 0) / len(returns) * 100 if len(returns) > 0 else 0.0
            self.lbl_sharpe.setText(f"Sharpe Ratio: {sharpe:.3f}" if not np.isnan(sharpe) else "Sharpe Ratio: --")
            self.lbl_winrate.setText(f"Win Rate: {win_rate:.1f}%")
        else:
            self.lbl_sharpe.setText("Sharpe Ratio: --")
            self.lbl_winrate.setText("Win Rate: --")
        trades = self.data.get('trades', [])
        my_count = bot_count = total_vol = 0
        for tr in trades:
            if p != 'Overall' and tr.get('symbol') != p: continue
            if d != 'All' and 'day' in tr and str(tr['day']) != d: continue
            qty = abs(int(tr.get('quantity', 0)))
            is_b = str(tr.get('buyer', '')).upper() == 'SUBMISSION'
            is_s = str(tr.get('seller', '')).upper() == 'SUBMISSION'
            if is_b or is_s: my_count += 1; total_vol += qty
            else: bot_count += 1; total_vol += qty
        self.lbl_volume.setText(f"Total Volume: {total_vol:,}")
        self.lbl_my_trades.setText(f"My Trades: {my_count}")
        self.lbl_bot_trades.setText(f"Bot Trades: {bot_count}")
        self.p_dash_pnl.autoRange(); self.p_dash_dd.autoRange()

def _auto_detect_log(script_dir: str):
    search_dirs = [script_dir]
    project_root = os.path.dirname(script_dir)
    backtests_dir = os.path.join(project_root, 'backtests')
    if os.path.isdir(backtests_dir): search_dirs.append(backtests_dir)
    cwd_backtests = os.path.join(os.getcwd(), 'backtests')
    if os.path.isdir(cwd_backtests) and cwd_backtests not in search_dirs: search_dirs.append(cwd_backtests)
    all_logs = []
    for d in search_dirs:
        try:
            for f in os.listdir(d):
                if f.endswith('.log'): all_logs.append(os.path.join(d, f))
        except OSError: continue
    if not all_logs: return None
    return max(all_logs, key=os.path.getmtime)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(APP_STYLE)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    startup_log = sys.argv[1] if len(sys.argv)>1 else _auto_detect_log(script_dir)
    if startup_log is None:
        startup_log, _ = QFileDialog.getOpenFileName(None, "Open Log File", script_dir, "Log (*.log *.json)")
        startup_log = startup_log or None
    win = LogVisualizer(startup_log); win.show(); sys.exit(app.exec())