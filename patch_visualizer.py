import sys
import re

with open("logviz/log_visualizer.py", "r") as f:
    content = f.read()

# Patch 1: fetch_hidden_standalone, set_hidden_standalone, HideRunWorker, and update LeaderboardWorker
import_block_pattern = re.compile(r'def fetch_records_standalone\(repo, token\):.*?return records', re.DOTALL)
new_import_block = """def fetch_records_standalone(repo, token):
    gh = GitHubRepo(repo, token); entries = gh.list_directory("records"); records = []
    for entry in entries:
        if not entry.get("name", "").endswith(".json"): continue
        content = gh.get_file_content(entry["path"])
        if content:
            try: records.append(json.loads(content.decode()))
            except: pass
    return records

def fetch_hidden_standalone(repo, token):
    gh = GitHubRepo(repo, token)
    content = gh.get_file_content("records/.hidden.json")
    if content:
        try: return set(json.loads(content.decode()))
        except: pass
    return set()

def set_hidden_standalone(repo, token, hidden_ids):
    gh = GitHubRepo(repo, token)
    gh.put_file("records/.hidden.json", json.dumps(list(hidden_ids)).encode(), "Update hidden runs")
"""
content = import_block_pattern.sub(new_import_block, content, count=1)

lb_worker_pattern = re.compile(r'class LeaderboardWorker\(QThread\):\n    status = pyqtSignal\(str\)\n    done = pyqtSignal\(list, str\) # runs, error\n\n    def run\(self\):\n        try:\n            repo = TRACKER_REPO; token = TRACKER_TOKEN\n            if not repo or not token: raise RuntimeError\("TRACKER_REPO/TOKEN not set"\)\n            self.status.emit\("Fetching leaderboard..."\)\n            runs = fetch_records_standalone\(repo, token\)\n            self.done.emit\(runs, ""\)\n        except Exception as e: self.done.emit\(\[\], str\(e\)\)', re.DOTALL)
new_lb_worker = """class LeaderboardWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(list, set, str) # runs, hidden, error

    def run(self):
        try:
            repo = TRACKER_REPO; token = TRACKER_TOKEN
            if not repo or not token: raise RuntimeError("TRACKER_REPO/TOKEN not set")
            self.status.emit("Fetching leaderboard...")
            runs = fetch_records_standalone(repo, token)
            self.status.emit("Fetching hidden list...")
            hidden = fetch_hidden_standalone(repo, token)
            self.done.emit(runs, hidden, "")
        except Exception as e: self.done.emit([], set(), str(e))

class HideRunWorker(QThread):
    done = pyqtSignal(bool, str)
    def __init__(self, hidden_ids):
        super().__init__()
        self.hidden_ids = hidden_ids
    def run(self):
        try:
            set_hidden_standalone(TRACKER_REPO, TRACKER_TOKEN, self.hidden_ids)
            self.done.emit(True, "Hidden list updated.")
        except Exception as e:
            self.done.emit(False, str(e))
"""
content = lb_worker_pattern.sub(new_lb_worker, content, count=1)

# Patch 2: RunDetailsDialog
run_details_pattern = re.compile(r'class RunDetailsDialog\(QDialog\):.*?btn_close = QPushButton\("Close"\); btn_close.clicked.connect\(self.accept\); layout.addWidget\(btn_close\)', re.DOTALL)
new_run_details = """class RunDetailsDialog(QDialog):
    def __init__(self, run, is_hidden, parent=None):
        super().__init__(parent)
        self.run = run
        self.setWindowTitle(f"Run Details: {run.get('strategy_name', 'Unknown')}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(480)
        layout = QVBoxLayout(self)
        
        info_layout = QHBoxLayout()
        left_form = QVBoxLayout()
        def add_field(label, val, color=TEXT):
            row = QHBoxLayout(); row.addWidget(QLabel(f"<b>{label}:</b>")); lbl = QLabel(str(val))
            lbl.setStyleSheet(f"color: {color};"); row.addWidget(lbl); row.addStretch(); left_form.addLayout(row)

        add_field("Author", run.get("author", "Unknown"), ACCENT_CYAN)
        add_field("Strategy", run.get("strategy_name", "Unknown"), ACCENT_GOLD)
        add_field("Uploaded", run.get("uploaded_at", "Unknown")[:19].replace('T', ' '))
        add_field("Round", run.get("round", "Unknown"))
        add_field("ID", run.get("run_id", "Unknown"), DIM)
        info_layout.addLayout(left_form)
        
        m_frame = QFrame(); m_frame.setStyleSheet(f"background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")
        m_layout = QGridLayout(m_frame); info_layout.addWidget(m_frame)
        metrics = run.get("metrics", {})
        pnl_val = metrics.get('total_pnl', metrics.get('total_profit', 0))
        fields = [
            ("Score", f"{pnl_val:,.0f}", ACCENT_GREEN if pnl_val >= 0 else ACCENT_RED),
            ("Sharpe", f"{metrics.get('sharpe', 0):.3f}", ACCENT_CYAN),
            ("Max DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%", ACCENT_RED if metrics.get('max_drawdown_pct', 0) > 10 else ACCENT_ORANGE),
            ("Trades", f"{metrics.get('submission_trades', 0):,}", TEXT),
            ("Volume", f"{metrics.get('submission_volume', 0):,}", TEXT),
            ("Products", f"{len(metrics.get('products', []))}", TEXT),
        ]
        for i, (l, v, c) in enumerate(fields):
            m_layout.addWidget(QLabel(l), i//2, (i%2)*2); vl = QLabel(v); vl.setStyleSheet(f"color: {c}; font-weight: bold;")
            m_layout.addWidget(vl, i//2, (i%2)*2 + 1)
        layout.addLayout(info_layout)
        
        self.gw = pg.GraphicsLayoutWidget(); self.gw.setBackground(BG); self.gw.setFixedHeight(200)
        self.pnl_plot = self.gw.addPlot(title="PnL Curve")
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.3)
        pnl_series = metrics.get('pnl_series', [])
        if pnl_series:
            color = ACCENT_GREEN if pnl_series[-1] >= 0 else ACCENT_RED
            self.pnl_plot.plot(pnl_series, fillLevel=0, brush=pg.mkBrush(color + '40'), pen=pg.mkPen(color, width=2))
        layout.addWidget(self.gw)

        layout.addWidget(QLabel("<b>Notes:</b>"))
        notes = QLabel(run.get("notes", "No notes provided.")); notes.setWordWrap(True)
        notes.setStyleSheet(f"background: {PANEL_BG}; padding: 8px; border-radius: 4px; color: {TEXT};")
        layout.addWidget(notes)

        btns = QHBoxLayout()
        self.btn_hide = QPushButton("Unhide Run" if is_hidden else "Hide Run")
        self.btn_hide.setStyleSheet(f"background-color: {ACCENT_RED if not is_hidden else DIM}; color: {ACCENT_WHITE}; font-weight: bold;")
        btns.addWidget(self.btn_hide); btns.addStretch()
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.accept); btns.addWidget(btn_close)
        layout.addLayout(btns)"""
content = run_details_pattern.sub(new_run_details, content, count=1)

# Patch 3: LeaderboardTab features
lb_tab_pattern = re.compile(r'    def _build_leaderboard_tab\(self\):.*?(?=    def _build_dashboard_tab\(self\):)', re.DOTALL)

new_lb_tab = """    def _build_leaderboard_tab(self):
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(8)
        
        header = QHBoxLayout()
        self.btn_refresh_lb = QPushButton("⟳ Refresh Leaderboard"); self.btn_refresh_lb.clicked.connect(self._on_leaderboard_refresh); header.addWidget(self.btn_refresh_lb)
        btn_upload = QPushButton("🚀 Upload Local Run"); btn_upload.clicked.connect(self._on_tracker_clicked); header.addWidget(btn_upload)
        header.addSpacing(20)
        
        header.addWidget(QLabel("Product:")); self.lb_cb_prod = QComboBox(); self.lb_cb_prod.currentTextChanged.connect(self._apply_lb_filters); header.addWidget(self.lb_cb_prod)
        header.addWidget(QLabel("Author:")); self.lb_cb_auth = QComboBox(); self.lb_cb_auth.currentTextChanged.connect(self._apply_lb_filters); header.addWidget(self.lb_cb_auth)
        header.addWidget(QLabel("Round:")); self.lb_cb_round = QComboBox(); self.lb_cb_round.currentTextChanged.connect(self._apply_lb_filters); header.addWidget(self.lb_cb_round)
        self.lb_chk_hidden = QCheckBox("Show Hidden"); self.lb_chk_hidden.stateChanged.connect(self._apply_lb_filters); header.addWidget(self.lb_chk_hidden)
        
        self.lbl_lb_status = QLabel(""); self.lbl_lb_status.setStyleSheet(f"color: {DIM};"); header.addWidget(self.lbl_lb_status)
        header.addStretch(); layout.addLayout(header)

        cards_layout = QHBoxLayout()
        self.lb_cards = {}
        for title in ["TOTAL RUNS", "BEST PnL", "TOP AUTHOR", "BEST SHARPE", "PRODUCTS SEEN"]:
            f = QFrame(); f.setStyleSheet(f"background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;")
            fl = QVBoxLayout(f); fl.addWidget(QLabel(f"<span style='color:{TEXT};font-size:8pt;'>{title}</span>"))
            val_lbl = QLabel("—"); val_lbl.setStyleSheet(f"font-size:12pt;font-weight:bold;color:{TEXT};")
            fl.addWidget(val_lbl); cards_layout.addWidget(f); self.lb_cards[title] = val_lbl
        layout.addLayout(cards_layout)

        self.lb_gw_chart = pg.GraphicsLayoutWidget(); self.lb_gw_chart.setBackground(BG); self.lb_gw_chart.setFixedHeight(120)
        self.lb_chart = self.lb_gw_chart.addPlot(); self.lb_chart.hideAxis('left'); self.lb_chart.hideAxis('bottom')
        self.lb_chart.setMouseEnabled(x=False, y=False); layout.addWidget(self.lb_gw_chart)

        self.lb_table = QTableWidget(); self.lb_table.setColumnCount(10)
        self.lb_table.setHorizontalHeaderLabels(["Rank", "Score", "Sharpe", "MaxDD", "Author", "Strategy", "Round", "Trades", "Volume", "Time"])
        self.lb_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.lb_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.lb_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.lb_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lb_table.verticalHeader().setVisible(False); self.lb_table.itemDoubleClicked.connect(self._on_lb_row_double_clicked)
        self.lb_table.horizontalHeader().sectionClicked.connect(self._on_lb_header_clicked)
        layout.addWidget(self.lb_table)

        self._lb_runs = []
        self._lb_hidden_ids = set()
        self._lb_sort_col = 1
        self._lb_sort_asc = False
        self.tabs.addTab(container, "Leaderboard")

    def _on_leaderboard_refresh(self):
        self.btn_refresh_lb.setEnabled(False); self.lbl_lb_status.setText("Fetching...")
        self._lb_worker = LeaderboardWorker()
        self._lb_worker.status.connect(self.lbl_lb_status.setText)
        self._lb_worker.done.connect(self._on_leaderboard_done)
        self._lb_worker.start()

    def _on_leaderboard_done(self, runs, hidden_ids, error):
        self.btn_refresh_lb.setEnabled(True); self.lbl_lb_status.setText("")
        if error: QMessageBox.warning(self, "Leaderboard Error", f"Failed to fetch data: {error}"); return
        self._lb_runs = runs; self._lb_hidden_ids = hidden_ids
        
        prods, authors, rounds = set(), set(), set()
        for r in runs:
            prods.update(r.get('metrics', {}).get('products', []))
            if r.get('author'): authors.add(r['author'])
            if r.get('round') is not None: rounds.add(str(r['round']))
            
        def update_cb(cb, items):
            curr = cb.currentText(); cb.blockSignals(True); cb.clear()
            cb.addItems(["All"] + sorted(list(items))); cb.setCurrentText(curr if curr in items else "All"); cb.blockSignals(False)
            
        update_cb(self.lb_cb_prod, prods); update_cb(self.lb_cb_auth, authors); update_cb(self.lb_cb_round, rounds)
        self._apply_lb_filters()

    def _on_lb_header_clicked(self, logicalIndex):
        if self._lb_sort_col == logicalIndex: self._lb_sort_asc = not self._lb_sort_asc
        else: self._lb_sort_col = logicalIndex; self._lb_sort_asc = logicalIndex in [0, 3]
        self._apply_lb_filters()

    def _apply_lb_filters(self, *args):
        if not hasattr(self, '_lb_runs') or not self._lb_runs: return
        p_filter, a_filter, r_filter = self.lb_cb_prod.currentText(), self.lb_cb_auth.currentText(), self.lb_cb_round.currentText()
        show_hidden = self.lb_chk_hidden.isChecked()
        
        filtered = []
        for r in self._lb_runs:
            if not show_hidden and r.get('run_id') in self._lb_hidden_ids: continue
            if p_filter != "All" and p_filter not in r.get('metrics', {}).get('products', []): continue
            if a_filter != "All" and r.get('author') != a_filter: continue
            if r_filter != "All" and str(r.get('round', '')) != r_filter: continue
            filtered.append(r)
            
        pnls = [r.get('metrics', {}).get('total_profit', r.get('metrics', {}).get('total_pnl', 0)) for r in filtered]
        self.lb_cards["TOTAL RUNS"].setText(str(len(filtered))); self.lb_cards["TOTAL RUNS"].setStyleSheet(f"color: {ACCENT_CYAN}; font-size:12pt;font-weight:bold;")
        
        if filtered:
            best_pnl = max(pnls)
            self.lb_cards["BEST PnL"].setText(f"{best_pnl:,.0f}"); self.lb_cards["BEST PnL"].setStyleSheet(f"color: {ACCENT_GREEN if best_pnl >= 0 else ACCENT_RED}; font-size:12pt;font-weight:bold;")
            best_sh = max([r.get('metrics', {}).get('sharpe', 0) for r in filtered])
            self.lb_cards["BEST SHARPE"].setText(f"{best_sh:.3f}"); self.lb_cards["BEST SHARPE"].setStyleSheet(f"color: {ACCENT_PURPLE}; font-size:12pt;font-weight:bold;")
            author_pnls = {}
            for r in filtered: a = r.get('author', '—'); author_pnls[a] = author_pnls.get(a, 0) + r.get('metrics', {}).get('total_profit', r.get('metrics', {}).get('total_pnl', 0))
            top_a = max(author_pnls, key=author_pnls.get) if author_pnls else "—"
            self.lb_cards["TOP AUTHOR"].setText(top_a); self.lb_cards["TOP AUTHOR"].setStyleSheet(f"color: {ACCENT_GOLD}; font-size:12pt;font-weight:bold;")
            all_p = set()
            for r in filtered: all_p.update(r.get('metrics', {}).get('products', []))
            self.lb_cards["PRODUCTS SEEN"].setText(str(len(all_p))); self.lb_cards["PRODUCTS SEEN"].setStyleSheet(f"color: {ACCENT_CYAN}; font-size:12pt;font-weight:bold;")
        else:
            for k in ["BEST PnL", "TOP AUTHOR", "BEST SHARPE", "PRODUCTS SEEN"]:
                self.lb_cards[k].setText("—"); self.lb_cards[k].setStyleSheet(f"color: {TEXT}; font-size:12pt;font-weight:bold;")

        self.lb_chart.clear()
        if filtered:
            x, y = np.arange(len(filtered)), np.array(pnls)
            brushes = [pg.mkBrush(ACCENT_GREEN if v >= 0 else ACCENT_RED) for v in y]
            bg = pg.BarGraphItem(x=x, height=y, width=0.8, brushes=brushes); self.lb_chart.addItem(bg); self.lb_chart.autoRange()

        def get_val(r):
            m, c = r.get('metrics', {}), self._lb_sort_col
            if c == 1: return m.get('total_profit', m.get('total_pnl', 0))
            if c == 2: return m.get('sharpe', 0)
            if c == 3: return m.get('max_drawdown_pct', 0)
            if c == 4: return str(r.get('author', ''))
            if c == 5: return str(r.get('strategy_name', ''))
            if c == 6: return int(r.get('round', 0)) if str(r.get('round', 0)).isdigit() else 0
            if c == 7: return m.get('submission_trades', 0)
            if c == 8: return m.get('submission_volume', 0)
            if c == 9: return str(r.get('uploaded_at', ''))
            return m.get('total_profit', m.get('total_pnl', 0))
            
        filtered.sort(key=get_val, reverse=not self._lb_sort_asc)
        self._lb_filtered_runs = filtered
        
        self.lb_table.setRowCount(len(filtered))
        for i, run in enumerate(filtered):
            m, is_hid = run.get('metrics', {}), run.get('run_id') in self._lb_hidden_ids
            pnl = m.get('total_profit', m.get('total_pnl', 0))
            rank_str = f"∅{i+1}" if is_hid else ("🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}") if self._lb_sort_col in [0,1] and not self._lb_sort_asc else f"#{i+1}"
            items = [rank_str, f"{pnl:,.0f}", f"{m.get('sharpe', 0):.3f}", f"{m.get('max_drawdown_pct', 0):.2f}%", run.get('author', '??'), run.get('strategy_name', '??'), str(run.get('round', '—')), f"{m.get('submission_trades', 0):,}", f"{m.get('submission_volume', 0):,}", run.get('uploaded_at', '??').split('T')[0]]
            for col, txt in enumerate(items):
                it = QTableWidgetItem(txt)
                color = DIM if is_hid else (ACCENT_GOLD if i==0 and col==0 else ACCENT_GREEN if col==1 and pnl>=0 else ACCENT_RED if col==1 and pnl<0 else ACCENT_CYAN if col==2 else ACCENT_RED if col==3 and m.get('max_drawdown_pct', 0)>10 else ACCENT_ORANGE if col==3 else TEXT)
                it.setForeground(QBrush(QColor(color))); self.lb_table.setItem(i, col, it)
        self.lb_table.resizeRowsToContents()

    def _on_lb_row_double_clicked(self, item):
        row = item.row()
        if hasattr(self, '_lb_filtered_runs') and 0 <= row < len(self._lb_filtered_runs):
            run = self._lb_filtered_runs[row]
            is_hid = run.get('run_id') in self._lb_hidden_ids
            dlg = RunDetailsDialog(run, is_hid, self)
            dlg.btn_hide.clicked.connect(lambda: self._toggle_hide_run(run.get('run_id'), dlg))
            dlg.exec()

    def _toggle_hide_run(self, run_id, dlg):
        if run_id in self._lb_hidden_ids: self._lb_hidden_ids.remove(run_id)
        else: self._lb_hidden_ids.add(run_id)
        dlg.btn_hide.setEnabled(False); dlg.btn_hide.setText("Updating...")
        self._hide_worker = HideRunWorker(self._lb_hidden_ids)
        self._hide_worker.done.connect(lambda s, m: self._on_hide_done(s, m, dlg))
        self._hide_worker.start()

    def _on_hide_done(self, success, msg, dlg):
        if success: self._apply_lb_filters(); dlg.accept()
        else: QMessageBox.warning(self, "Error", f"Failed to update hidden state: {msg}"); dlg.btn_hide.setEnabled(True); dlg.btn_hide.setText("Retry")

"""

content = lb_tab_pattern.sub(new_lb_tab, content, count=1)

with open("logviz/log_visualizer.py", "w") as f:
    f.write(content)

print("Patch applied successfully.")
