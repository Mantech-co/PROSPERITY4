from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED DEFAULTS — fill these in once
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO  = "ManukrishnanP/prosperity4-strategy-tracker"
DEFAULT_TOKEN = "github_pat_14mn07r3v34L1ngmy70k3n"
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ROUND SCHEDULE — edit cutoff dates (UTC); each value is when that round opens
# ─────────────────────────────────────────────────────────────────────────────
ROUND_CUTOFFS: dict = {
    1: "2026-04-17 10:00",
    2: "2026-04-20 10:00",
    3: "2026-04-26 10:00",
    4: "2026-04-28 10:00",
    5: "2026-04-30 10:00",
}
# ─────────────────────────────────────────────────────────────────────────────

"""
strategy_tracker.py — Team Strategy Performance Tracker
========================================================
Run with:  python strategy_tracker.py

On launch, a file explorer opens immediately for zip selection.
Fill in Author, Strategy Name, and optional Notes, then Upload.
Switch to the Leaderboard tab to compare all team runs.

Repo layout on GitHub
─────────────────────
  artifacts/<run_id>.zip    ← raw zip (strategy + log + config)
  records/<run_id>.json     ← performance record (append-only)

DEPENDENCIES
────────────
  Standard library only — tkinter, json, csv, zipfile, urllib, etc.
"""

import base64
import csv
import http.client
import io
import json
import os
import re
import tempfile
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk
from typing import Any


def _get_current_round() -> int:
    """Return current round based on ROUND_CUTOFFS and current UTC time."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    current = min(ROUND_CUTOFFS)
    for rnd, cutoff_str in sorted(ROUND_CUTOFFS.items()):
        if now >= datetime.strptime(cutoff_str, "%Y-%m-%d %H:%M"):
            current = rnd
    return current


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — LOG PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_activities_csv(csv_str: str) -> list[dict]:
    csv_str = csv_str.replace("\\n", "\n").strip()
    reader = csv.DictReader(io.StringIO(csv_str), delimiter=";")
    return [{k.strip(): v for k, v in row.items()} for row in reader]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _compute_sharpe(pnl_series: list[float]) -> float:
    if len(pnl_series) < 2:
        return 0.0
    deltas = [pnl_series[i] - pnl_series[i - 1] for i in range(1, len(pnl_series))]
    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / n
    std = variance ** 0.5
    return (mean / std * 1000.0) if std > 0 else 0.0


def _compute_drawdown(pnl_series: list[float]) -> tuple[float, float]:
    if not pnl_series:
        return 0.0, 0.0
    peak = pnl_series[0]
    max_dd_abs = max_dd_pct = 0.0
    for v in pnl_series:
        if v > peak:
            peak = v
        dd_abs = peak - v
        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs
        if peak > 0:
            dd_pct = dd_abs / peak * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    return max_dd_abs, max_dd_pct


def extract_metrics(log_path: str) -> dict:
    with open(log_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    csv_str = raw.get("activitiesLog", "") or ""
    rows = _parse_activities_csv(csv_str) if csv_str.strip() else []

    products_seen: set[str] = set()
    days_seen: set[int] = set()
    ts_product_pnl: dict[tuple, dict[str, float]] = {}

    for row in rows:
        product = (row.get("product") or "").strip()
        if not product:
            continue
        products_seen.add(product)
        ts  = _safe_float(row.get("timestamp", 0))
        pnl = _safe_float(row.get("profit_and_loss", 0))
        raw_day = row.get("day")
        day_val: int | None = None
        if raw_day is not None and str(raw_day).strip() not in ("", "nan", "None"):
            try:
                day_val = int(float(raw_day))
                days_seen.add(day_val)
            except (ValueError, TypeError):
                pass
        key = (day_val if day_val is not None else 0, int(ts))
        ts_product_pnl.setdefault(key, {})[product] = pnl

    sorted_keys    = sorted(ts_product_pnl.keys())
    agg_pnl_series = [sum(ts_product_pnl[k].values()) for k in sorted_keys]

    total_pnl              = agg_pnl_series[-1] if agg_pnl_series else 0.0
    sharpe                 = _compute_sharpe(agg_pnl_series)
    max_dd_abs, max_dd_pct = _compute_drawdown(agg_pnl_series)

    # Downsample PnL series for storage (max 500 points to keep record size small)
    pnl_series_storage = []
    if agg_pnl_series:
        if len(agg_pnl_series) > 500:
            step = len(agg_pnl_series) / 500.0
            pnl_series_storage = [round(agg_pnl_series[int(i * step)], 2) for i in range(500)]
            # Ensure the last point is included
            if round(agg_pnl_series[-1], 2) != pnl_series_storage[-1]:
                pnl_series_storage.append(round(agg_pnl_series[-1], 2))
        else:
            pnl_series_storage = [round(v, 2) for v in agg_pnl_series]

    per_product_pnl: dict[str, float] = {}
    for key in sorted_keys:
        for prod, pnl in ts_product_pnl[key].items():
            per_product_pnl[prod] = pnl

    trade_history     = raw.get("tradeHistory", []) or []
    total_trades      = len(trade_history)
    submission_trades = submission_volume = 0
    for tr in trade_history:
        buyer  = str(tr.get("buyer",  "")).upper()
        seller = str(tr.get("seller", "")).upper()
        if buyer == "SUBMISSION" or seller == "SUBMISSION":
            submission_trades += 1
            try:
                submission_volume += int(tr.get("quantity", 0))
            except (TypeError, ValueError):
                pass

    return {
        "total_pnl":         round(total_pnl, 4),
        "sharpe":            round(sharpe, 6),
        "max_drawdown_abs":  round(max_dd_abs, 4),
        "max_drawdown_pct":  round(max_dd_pct, 4),
        "products":          sorted(products_seen),
        "days":              sorted(days_seen),
        "per_product_pnl":   {k: round(v, 4) for k, v in per_product_pnl.items()},
        "pnl_series":        pnl_series_storage,
        "total_trades":      total_trades,
        "submission_trades": submission_trades,
        "submission_volume": submission_volume,
        "has_error":         bool(raw.get("error")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — ZIP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_EXTENSIONS = {".py", ".log", ".json"}


def _validate_and_extract_log(zip_path: str) -> tuple[dict, dict]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        by_ext: dict[str, list[str]] = {}
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            by_ext.setdefault(ext, []).append(n)

        missing = _REQUIRED_EXTENSIONS - set(by_ext)
        if missing:
            raise ValueError(f"Zip missing file types: {missing}")

        log_bytes    = zf.read(by_ext[".log"][0])
        config_bytes = zf.read(by_ext[".json"][0])

    tmp_fd, tmp = tempfile.mkstemp(suffix=".log", prefix="_tracker_")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(log_bytes)
        metrics = extract_metrics(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    try:
        config_data = json.loads(config_bytes.decode())
    except Exception:
        config_data = {}

    return metrics, config_data


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — GITHUB API
# ══════════════════════════════════════════════════════════════════════════════

class GitHubRepo:
    API_BASE = "https://api.github.com"

    def __init__(self, repo: str, token: str):
        self.repo  = repo
        self.token = token

    def _headers(self) -> dict:
        return {
            "Authorization":        f"token {self.token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type":         "application/json",
        }

    def _safe_read(self, resp) -> bytes:
        try:
            return resp.read()
        except http.client.IncompleteRead as e:
            return e.partial

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url  = f"{self.API_BASE}/repos/{self.repo}/contents/{path}"
        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw_data = self._safe_read(resp)
                return json.loads(raw_data.decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            raise RuntimeError(f"GitHub {method} {path} → {e.code}: {detail}") from e

    def get_file_sha(self, path: str) -> str | None:
        url = f"{self.API_BASE}/repos/{self.repo}/contents/{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                raw_data = self._safe_read(resp)
                return json.loads(raw_data.decode()).get("sha")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def put_file(self, path: str, content_bytes: bytes, message: str) -> str:
        sha  = self.get_file_sha(path)
        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content_bytes).decode(),
        }
        if sha:
            body["sha"] = sha
        return self._request("PUT", path, body)["content"]["sha"]

    def list_directory(self, path: str) -> list[dict]:
        url = f"{self.API_BASE}/repos/{self.repo}/contents/{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                raw_data = self._safe_read(resp)
                return json.loads(raw_data.decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            raise

    def get_file_content(self, path: str) -> bytes | None:
        url = f"{self.API_BASE}/repos/{self.repo}/contents/{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                raw_data = self._safe_read(resp)
                data     = json.loads(raw_data.decode())
                if "content" in data:
                    return base64.b64decode(data["content"].replace("\n", ""))
                return None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — UPLOAD & LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

def upload_run(zip_path: str, author: str, strategy_name: str,
               notes: str, metrics: dict, config_data: dict,
               repo: str, token: str, round_num: int = 1) -> str:
    run_ts = datetime.now(timezone.utc).isoformat()
    slug   = re.sub(r"[^a-z0-9_]", "_", author.lower())
    run_id = f"{slug}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    record = {
        "run_id":        run_id,
        "author":        author,
        "strategy_name": strategy_name,
        "round":         round_num,
        "notes":         notes,
        "uploaded_at":   run_ts,
        "config":        config_data,
        "metrics":       metrics,
        "artifact_path": f"artifacts/{run_id}.zip",
    }

    gh = GitHubRepo(repo, token)
    with open(zip_path, "rb") as fh:
        zip_bytes = fh.read()

    gh.put_file(f"artifacts/{run_id}.zip", zip_bytes,
                f"artifact: {author}/{strategy_name} [{run_id}]")
    gh.put_file(f"records/{run_id}.json",
                json.dumps(record, indent=2).encode(),
                f"record: {author}/{strategy_name} [{run_id}]")
    return run_id


def fetch_records(repo: str, token: str) -> list[dict]:
    gh      = GitHubRepo(repo, token)
    entries = gh.list_directory("records")
    records = []
    for entry in entries:
        if not entry.get("name", "").endswith(".json"):
            continue
        content = gh.get_file_content(entry["path"])
        if content:
            try:
                records.append(json.loads(content.decode()))
            except json.JSONDecodeError:
                pass
    return records


def fetch_hidden(repo: str, token: str) -> set:
    gh      = GitHubRepo(repo, token)
    content = gh.get_file_content("hidden/index.json")
    if content:
        try:
            return set(json.loads(content.decode()).get("hidden_run_ids", []))
        except Exception:
            pass
    return set()


def set_hidden(repo: str, token: str, hidden_ids: set) -> None:
    gh   = GitHubRepo(repo, token)
    data = json.dumps({"hidden_run_ids": sorted(hidden_ids)}, indent=2).encode()
    gh.put_file("hidden/index.json", data, "update hidden run list")


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

BG        = "#0a0c10"
BG2       = "#0f1218"
PANEL     = "#141820"
PANEL2    = "#1a1f2c"
BORDER    = "#1e2535"
BORDER2   = "#252d40"
TEXT      = "#b8c4d8"
TEXT2     = "#7a8499"
CYAN      = "#00e5ff"
CYAN_DIM  = "#007a8a"
GREEN     = "#00ff88"
GREEN_DIM = "#006644"
RED       = "#ff4466"
RED_DIM   = "#660022"
GOLD      = "#ffcc00"
GOLD_DIM  = "#554400"
PURPLE    = "#bb88ff"

FONT_MONO  = ("Consolas", 9)
FONT_MONO2 = ("Consolas", 8)
FONT_BOLD  = ("Consolas", 9, "bold")
FONT_TTL   = ("Consolas", 14, "bold")
FONT_LG    = ("Consolas", 11, "bold")
FONT_XL    = ("Consolas", 20, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# TTK STYLE
# ══════════════════════════════════════════════════════════════════════════════

def apply_theme(root: tk.Tk) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")

    s.configure(".", background=BG, foreground=TEXT,
                fieldbackground=PANEL, bordercolor=BORDER,
                troughcolor=PANEL, font=FONT_MONO)

    # Notebook
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
    s.configure("TNotebook.Tab", background=BG2, foreground=TEXT2,
                padding=[20, 8], borderwidth=0, font=FONT_BOLD)
    s.map("TNotebook.Tab",
          background=[("selected", BG)],
          foreground=[("selected", CYAN)])

    # Frames
    s.configure("TFrame",       background=BG)
    s.configure("Panel.TFrame", background=PANEL)
    s.configure("Card.TFrame",  background=PANEL2)

    # Labels
    s.configure("TLabel",        background=BG,     foreground=TEXT,   font=FONT_MONO)
    s.configure("Dim.TLabel",    background=BG,     foreground=TEXT2,  font=FONT_MONO2)
    s.configure("Title.TLabel",  background=BG,     foreground=CYAN,   font=FONT_TTL)
    s.configure("Section.TLabel",background=BG,     foreground=TEXT,   font=FONT_LG)
    s.configure("Cyan.TLabel",   background=BG,     foreground=CYAN,   font=FONT_MONO)
    s.configure("Panel.TLabel",  background=PANEL,  foreground=TEXT,   font=FONT_MONO)
    s.configure("Panel2.TLabel", background=PANEL2, foreground=TEXT,   font=FONT_MONO)
    s.configure("Dim2.TLabel",   background=PANEL2, foreground=TEXT2,  font=FONT_MONO2)
    s.configure("Green.TLabel",  background=PANEL2, foreground=GREEN,  font=FONT_BOLD)
    s.configure("Red.TLabel",    background=PANEL2, foreground=RED,    font=FONT_BOLD)
    s.configure("Gold.TLabel",   background=PANEL2, foreground=GOLD,   font=FONT_BOLD)
    s.configure("Purple.TLabel", background=PANEL2, foreground=PURPLE, font=FONT_BOLD)
    s.configure("GreenBG.TLabel",background=BG,     foreground=GREEN,  font=FONT_MONO)
    s.configure("RedBG.TLabel",  background=BG,     foreground=RED,    font=FONT_MONO)

    # Entry
    s.configure("TEntry", fieldbackground=PANEL2, foreground=TEXT,
                bordercolor=BORDER2, insertcolor=CYAN, relief="flat", padding=8,
                font=FONT_MONO)
    s.map("TEntry", bordercolor=[("focus", CYAN)])

    # Button
    s.configure("TButton", background=PANEL2, foreground=TEXT,
                bordercolor=BORDER2, relief="flat", padding=[12, 6], font=FONT_MONO)
    s.map("TButton",
          background=[("active", BORDER2)],
          foreground=[("active", CYAN)])

    s.configure("Accent.TButton", background=CYAN, foreground=BG,
                font=FONT_BOLD, padding=[16, 8], relief="flat")
    s.map("Accent.TButton", background=[("active", "#00b8cc"), ("disabled", PANEL2)],
          foreground=[("disabled", TEXT2)])

    s.configure("Danger.TButton", background=RED_DIM, foreground=RED,
                font=FONT_MONO, padding=[12, 6], relief="flat")
    s.map("Danger.TButton", background=[("active", "#440011")])

    s.configure("Ghost.TButton", background=BG, foreground=TEXT2,
                bordercolor=BORDER, relief="flat", padding=[10, 5], font=FONT_MONO)
    s.map("Ghost.TButton",
          background=[("active", PANEL)],
          foreground=[("active", CYAN)])

    # Treeview
    s.configure("Treeview", background=BG, foreground=TEXT,
                fieldbackground=BG, rowheight=26, borderwidth=0, font=FONT_MONO)
    s.configure("Treeview.Heading", background=PANEL, foreground=CYAN,
                borderwidth=0, relief="flat", font=FONT_BOLD, padding=[8, 6])
    s.map("Treeview",
          background=[("selected", PANEL2)],
          foreground=[("selected", CYAN)])

    # Combobox
    s.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2,
                foreground=TEXT, bordercolor=BORDER2, arrowcolor=CYAN,
                selectbackground=BORDER2, font=FONT_MONO)

    # Scrollbar
    s.configure("TScrollbar", background=PANEL, troughcolor=BG,
                bordercolor=BORDER, arrowcolor=TEXT2, relief="flat", width=8)

    # Separator
    s.configure("TSeparator", background=BORDER)

    # Progressbar
    s.configure("TProgressbar", background=CYAN, troughcolor=PANEL,
                bordercolor=BORDER, lightcolor=CYAN, darkcolor=CYAN)


# ══════════════════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def sep(parent, **kw) -> ttk.Separator:
    return ttk.Separator(parent, orient="horizontal", **kw)


def lbl(parent, text, style="TLabel", **kw) -> ttk.Label:
    return ttk.Label(parent, text=text, style=style, **kw)


class DropZone(tk.Frame):
    """Clickable file drop zone with dashed border."""

    def __init__(self, parent, on_file, accept=".zip", **kw):
        super().__init__(parent, bg=PANEL2, cursor="hand2",
                         highlightthickness=1, highlightbackground=BORDER2,
                         **kw)
        self._on_file = on_file
        self._accept  = accept
        self._build()
        self.bind("<Button-1>", lambda _: self._pick())
        for child in self.winfo_children():
            child.bind("<Button-1>", lambda _: self._pick())

    def _build(self):
        tk.Label(self, text="⊕", font=("Consolas", 28), bg=PANEL2,
                 fg=CYAN_DIM).pack(pady=(20, 4))
        self._main_lbl = tk.Label(self, text="Click to select a .zip file",
                                   font=FONT_BOLD, bg=PANEL2, fg=TEXT2)
        self._main_lbl.pack()
        self._sub_lbl = tk.Label(self, text="strategy.py  ·  log file  ·  config.json",
                                  font=FONT_MONO2, bg=PANEL2, fg=TEXT2)
        self._sub_lbl.pack(pady=(2, 20))

    def _pick(self):
        path = filedialog.askopenfilename(
            title="Select Strategy Zip",
            filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")])
        if path:
            self._on_file(path)

    def set_file(self, path: str):
        name = os.path.basename(path)
        self._main_lbl.configure(text=f"✓  {name}", fg=GREEN)
        self._sub_lbl.configure(text="Click to change file", fg=TEXT2)
        self.configure(highlightbackground=GREEN_DIM)

    def reset(self):
        self._main_lbl.configure(text="Click to select a .zip file", fg=TEXT2)
        self._sub_lbl.configure(text="strategy.py  ·  log file  ·  config.json",
                                 fg=TEXT2)
        self.configure(highlightbackground=BORDER2)


class MetricCard(tk.Frame):
    """Single stat tile."""

    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, bg=PANEL2,
                         highlightthickness=1, highlightbackground=BORDER,
                         **kw)
        tk.Label(self, text=title, font=FONT_MONO2, bg=PANEL2, fg=TEXT2
                 ).pack(anchor="w", padx=12, pady=(10, 2))
        self._val_lbl = tk.Label(self, text="—", font=("Consolas", 13, "bold"),
                                  bg=PANEL2, fg=TEXT2)
        self._val_lbl.pack(anchor="w", padx=12, pady=(0, 10))

    def set(self, value: str, color: str = TEXT):
        self._val_lbl.configure(text=value, fg=color)


class PillTag(tk.Frame):
    """Colored product pill."""

    COLORS = [CYAN, GREEN, PURPLE, GOLD, "#ff8844", "#44aaff"]

    def __init__(self, parent, text: str, idx: int = 0, **kw):
        c = self.COLORS[idx % len(self.COLORS)]
        super().__init__(parent, bg=PANEL2, **kw)
        tk.Label(self, text=text, font=FONT_MONO2, bg=PANEL2, fg=c,
                 padx=6, pady=2).pack()


class StatusBar(tk.Frame):
    """Bottom status bar."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, height=28, **kw)
        self._lbl = tk.Label(self, text="Ready.", font=FONT_MONO2,
                              bg=BG2, fg=TEXT2, anchor="w")
        self._lbl.pack(side="left", fill="x", expand=True, padx=12, pady=4)

    def set(self, msg: str, color: str = TEXT2):
        self._lbl.configure(text=msg, fg=color)


class LabeledEntry(tk.Frame):
    """Label + entry with optional placeholder behavior."""

    def __init__(self, parent, label: str, placeholder: str = "", **kw):
        super().__init__(parent, bg=BG, **kw)
        tk.Label(self, text=label, font=FONT_MONO2, bg=BG,
                 fg=TEXT2).pack(anchor="w", pady=(0, 3))
        self.var = tk.StringVar()
        self._entry = ttk.Entry(self, textvariable=self.var, font=FONT_MONO)
        self._entry.pack(fill="x")
        if placeholder:
            self._ph = placeholder
            self._entry.insert(0, placeholder)
            self._entry.configure(foreground=TEXT2)
            self._entry.bind("<FocusIn>",  self._clear_ph)
            self._entry.bind("<FocusOut>", self._restore_ph)

    def _clear_ph(self, _=None):
        if self._entry.get() == self._ph:
            self._entry.delete(0, "end")
            self._entry.configure(foreground=TEXT)

    def _restore_ph(self, _=None):
        if not self._entry.get():
            self._entry.insert(0, self._ph)
            self._entry.configure(foreground=TEXT2)

    def get(self) -> str:
        v = self.var.get()
        return "" if v == getattr(self, "_ph", None) else v

    def set(self, v: str):
        self.var.set(v)


def _draw_pnl_on_canvas(canvas, pnl_series, final_pnl=None):
    canvas.update_idletasks()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w <= 1:
        return
    canvas.delete("all")
    if not pnl_series:
        canvas.create_text(w // 2, h // 2, text="(no PnL series available)",
                           font=FONT_MONO2, fill=TEXT2)
        return

    # Use reported PnL if the series is inconsistent or missing its tail
    display_pnl = final_pnl if final_pnl is not None else pnl_series[-1]

    min_pnl = min(pnl_series + [display_pnl])
    max_pnl = max(pnl_series + [display_pnl])
    # Add buffer
    span = max_pnl - min_pnl
    if span == 0: span = 1
    
    y_min = min_pnl - 0.1 * abs(span)
    y_max = max_pnl + 0.1 * abs(span)
    y_span = y_max - y_min
    if y_span == 0: y_span = 1

    def to_y(val):
        return h - 10 - ((val - y_min) / y_span) * (h - 20)

    # Baseline (zero)
    if y_min <= 0 <= y_max:
        zy = to_y(0)
        canvas.create_line(10, zy, w - 10, zy, fill=BORDER2, width=1, dash=(2, 2))

    pts = []
    n = len(pnl_series)
    for i, v in enumerate(pnl_series):
        x = 10 + (i / (n - 1) if n > 1 else 0) * (w - 20)
        y = to_y(v)
        pts.append(x)
        pts.append(y)

    if len(pts) >= 4:
        color = GREEN if display_pnl >= 0 else RED
        canvas.create_line(*pts, fill=color, width=2)
        
        # Fill under curve
        fill_pts = [pts[0], to_y(0)] + pts + [pts[-1], to_y(0)]
        canvas.create_polygon(*fill_pts, fill=color, stipple="gray25", outline="")

    # Stats Overlay
    canvas.create_text(15, 15, text=f"STRATEGY PERFORMANCE",
                       anchor="nw", font=FONT_BOLD, fill=CYAN)
    canvas.create_text(w-15, 15, text=f"{display_pnl:+,.0f}",
                       anchor="ne", font=FONT_BOLD, fill=TEXT)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD TAB
# ══════════════════════════════════════════════════════════════════════════════

class UploadTab(ttk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, style="TFrame")
        self._app      = app
        self._zip_path = ""
        self._metrics  = None
        self._config   = {}
        self._build()

    # ─── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        # ── Top header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=0)
        tk.Label(hdr, text="UPLOAD RUN", font=FONT_TTL,
                 bg=BG, fg=CYAN, anchor="w").pack(side="left", padx=28, pady=(22, 0))
        tk.Label(hdr, text="submit a strategy zip to the leaderboard",
                 font=FONT_MONO2, bg=BG, fg=TEXT2).pack(side="left", padx=8, pady=(26, 0))

        sep(self).grid(row=1, column=0, sticky="ew", padx=28, pady=0)

        # ── Main scroll area ───────────────────────────────────────────────────
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb    = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        canvas.grid(row=2, column=0, sticky="nsew", padx=0)
        vsb.grid(row=2, column=1, sticky="ns")
        self.rowconfigure(2, weight=1)

        inner = tk.Frame(canvas, bg=BG)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.itemconfigure(win, width=e.width)
        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        self._build_inner(inner)

    def _build_inner(self, parent):
        parent.columnconfigure(0, weight=1)

        pad = dict(padx=28, pady=0)

        # ── Drop zone ─────────────────────────────────────────────────────────
        tk.Label(parent, text="STRATEGY ZIP", font=FONT_MONO2,
                 bg=BG, fg=TEXT2).grid(row=0, column=0, sticky="w",
                                        padx=28, pady=(24, 4))
        self._drop = DropZone(parent, on_file=self._load_zip, height=100)
        self._drop.grid(row=1, column=0, sticky="ew", padx=28, pady=0)

        # ── Metrics preview cards ─────────────────────────────────────────────
        cards_frame = tk.Frame(parent, bg=BG)
        cards_frame.grid(row=2, column=0, sticky="ew", padx=28, pady=(16, 0))
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self._card_pnl    = MetricCard(cards_frame, "TOTAL PnL")
        self._card_sharpe = MetricCard(cards_frame, "SHARPE RATIO")
        self._card_dd     = MetricCard(cards_frame, "MAX DRAWDOWN %")
        self._card_trades = MetricCard(cards_frame, "SUBMISSION TRADES")

        for i, c in enumerate([self._card_pnl, self._card_sharpe,
                                self._card_dd, self._card_trades]):
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))

        # per-product pills row
        self._prod_frame = tk.Frame(parent, bg=BG)
        self._prod_frame.grid(row=3, column=0, sticky="ew", padx=28, pady=(8, 0))

        # ── PnL Preview Canvas ────────────────────────────────────────────────
        self._preview_canvas = tk.Canvas(parent, bg=PANEL2, height=140,
                                          highlightthickness=0)
        self._preview_canvas.grid(row=4, column=0, sticky="ew", padx=28, pady=(16, 0))
        self._preview_canvas.bind("<Configure>", lambda _: self._refresh_cards())

        # ── Divider ───────────────────────────────────────────────────────────
        sep(parent).grid(row=5, column=0, sticky="ew", padx=28, pady=20)

        # ── Form fields ───────────────────────────────────────────────────────
        form = tk.Frame(parent, bg=BG)
        form.grid(row=6, column=0, sticky="ew", padx=28, pady=0)
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=0)

        self._f_author = LabeledEntry(form, "AUTHOR", placeholder="e.g. jane_doe")
        self._f_author.grid(row=0, column=0, sticky="ew", padx=(0, 16), pady=(0, 16))

        self._f_strat = LabeledEntry(form, "STRATEGY NAME", placeholder="e.g. mean_reversion_v3")
        self._f_strat.grid(row=0, column=1, sticky="ew", pady=(0, 16))

        self._f_round_var = tk.StringVar(value=str(_get_current_round()))
        round_frame = tk.Frame(form, bg=BG)
        round_frame.grid(row=0, column=2, sticky="ew", padx=(16, 0), pady=(0, 16))
        tk.Label(round_frame, text="ROUND", font=FONT_MONO2, bg=BG, fg=TEXT2
                 ).pack(anchor="w", pady=(0, 3))
        tk.Spinbox(round_frame, from_=1, to=10, textvariable=self._f_round_var,
                   width=5, bg=PANEL2, fg=TEXT, insertbackground=CYAN,
                   buttonbackground=PANEL2, relief="flat", font=FONT_MONO
                   ).pack(fill="x")



        # Notes
        tk.Label(parent, text="NOTES  (optional)", font=FONT_MONO2,
                 bg=BG, fg=TEXT2).grid(row=7, column=0, sticky="w",
                                        padx=28, pady=(4, 4))
        self._notes = tk.Text(parent, height=4, bg=PANEL2, fg=TEXT,
                               insertbackground=CYAN, font=FONT_MONO, relief="flat",
                               highlightthickness=1, highlightbackground=BORDER,
                               highlightcolor=CYAN_DIM, padx=10, pady=8)
        self._notes.grid(row=8, column=0, sticky="ew", padx=28)

        # ── Action row ────────────────────────────────────────────────────────
        sep(parent).grid(row=9, column=0, sticky="ew", padx=28, pady=20)

        action = tk.Frame(parent, bg=BG)
        action.grid(row=10, column=0, sticky="ew", padx=28, pady=(0, 28))
        action.columnconfigure(0, weight=1)

        self._status_var = tk.StringVar(value="Select a zip to begin.")
        self._status_lbl = tk.Label(action, textvariable=self._status_var,
                                     font=FONT_MONO, bg=BG, fg=TEXT2, anchor="w")
        self._status_lbl.grid(row=0, column=0, sticky="w")

        self._upload_btn = ttk.Button(action, text="⬆  UPLOAD RUN",
                                       style="Accent.TButton",
                                       command=self._start_upload)
        self._upload_btn.grid(row=0, column=1, sticky="e")

    # ─── Logic ────────────────────────────────────────────────────────────────

    def _load_zip(self, path: str):
        self._zip_path = path
        self._drop.set_file(path)
        self._status("Parsing zip…", CYAN)
        self.update_idletasks()
        try:
            self._metrics, self._config = _validate_and_extract_log(path)
            self._refresh_cards()
            msg = "✓  Ready to upload."
            if self._metrics["has_error"]:
                msg += "  ⚠  Log contains error flag."
            self._status(msg, GREEN if not self._metrics["has_error"] else GOLD)
        except Exception as exc:
            self._metrics = None
            self._config  = {}
            self._reset_cards()
            self._status(f"✗  {exc}", RED)

    def _refresh_cards(self):
        m = self._metrics
        if not m:
            self._reset_cards()
            return
        pnl_color = GREEN if m["total_pnl"] >= 0 else RED
        self._card_pnl.set(f"{m['total_pnl']:,.2f}", pnl_color)
        s_color = GREEN if m["sharpe"] >= 0 else RED
        self._card_sharpe.set(f"{m['sharpe']:.4f}", s_color)
        self._card_dd.set(f"{m['max_drawdown_pct']:.2f}%", RED if m["max_drawdown_pct"] > 5 else GOLD)
        self._card_trades.set(str(m["submission_trades"]), CYAN)

        _draw_pnl_on_canvas(self._preview_canvas, m.get("pnl_series", []), 
                           final_pnl=m.get("total_pnl"))

        for w in self._prod_frame.winfo_children():
            w.destroy()

        if m["per_product_pnl"]:
            tk.Label(self._prod_frame, text="per-product PnL  →",
                     font=FONT_MONO2, bg=BG, fg=TEXT2).pack(side="left", padx=(0, 8))
            for i, (prod, pnl) in enumerate(sorted(m["per_product_pnl"].items())):
                c = PillTag.COLORS[i % len(PillTag.COLORS)]
                pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
                tag_txt = f" {prod}  {pnl_str} "
                tk.Label(self._prod_frame, text=tag_txt, font=FONT_MONO2, bg=PANEL2,
                         fg=c, relief="flat", padx=2).pack(side="left", padx=4)

    def _reset_cards(self):
        for c in [self._card_pnl, self._card_sharpe, self._card_dd, self._card_trades]:
            c.set("—", TEXT2)
        _draw_pnl_on_canvas(self._preview_canvas, [])
        for w in self._prod_frame.winfo_children():
            w.destroy()

    def _status(self, msg: str, color: str = TEXT2):
        self._status_var.set(msg)
        self._status_lbl.configure(fg=color)
        self._app.status.set(msg, color)

    def _start_upload(self):
        author = self._f_author.get().strip()
        strat  = self._f_strat.get().strip()
        notes  = self._notes.get("1.0", "end").strip()
        repo   = self._app.get_repo()
        token  = self._app.get_token()

        if not self._zip_path:
            messagebox.showwarning("No file", "Select a zip file first.")
            return
        if not author:
            messagebox.showwarning("Missing field", "Enter Author name.")
            return
        if not strat:
            messagebox.showwarning("Missing field", "Enter Strategy Name.")
            return
        if not repo or not token:
            messagebox.showerror("Not configured",
                                 "Set DEFAULT_REPO and DEFAULT_TOKEN at the top of strategy_tracker.py.")
            return
        if self._metrics is None:
            messagebox.showerror("No metrics", "Zip could not be parsed.")
            return

        round_num = _get_current_round()
        try:
            round_num = int(self._f_round_var.get())
        except (ValueError, AttributeError):
            pass

        self._upload_btn.configure(state="disabled")
        self._status("Uploading to GitHub…", CYAN)

        def _worker():
            try:
                run_id = upload_run(
                    zip_path=self._zip_path, author=author,
                    strategy_name=strat, notes=notes,
                    metrics=self._metrics, config_data=self._config,
                    repo=repo, token=token, round_num=round_num)
                self.after(0, self._on_success, run_id)
            except Exception as exc:
                self.after(0, self._on_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_success(self, run_id: str):
        self._upload_btn.configure(state="normal")
        self._status(f"✓  Uploaded  —  run_id: {run_id}", GREEN)
        self._app.trigger_leaderboard_refresh()

    def _on_error(self, msg: str):
        self._upload_btn.configure(state="normal")
        self._status(f"✗  Upload failed: {msg}", RED)

    def open_picker(self):
        """Called on startup to immediately show file dialog."""
        self._drop._pick()


# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD TAB
# ══════════════════════════════════════════════════════════════════════════════

_COLS = [
    ("Rank",       45,  "center"),
    ("Round",      55,  "center"),
    ("Author",    120,  "w"),
    ("Strategy",  170,  "w"),
    ("Total PnL", 110,  "e"),
    ("Sharpe",     90,  "e"),
    ("Max DD %",   88,  "e"),
    ("Trades",     75,  "e"),
    ("Volume",     90,  "e"),
    ("Products",  160,  "w"),
    ("Date",       96,  "center"),
    ("Notes",     180,  "w"),
]

_SORT_KEYS: dict[str, tuple] = {
    "Total PnL": ("metrics", "total_pnl"),
    "Sharpe":    ("metrics", "sharpe"),
    "Max DD %":  ("metrics", "max_drawdown_pct"),
    "Trades":    ("metrics", "submission_trades"),
    "Volume":    ("metrics", "submission_volume"),
    "Date":      ("uploaded_at",),
    "Round":     ("round",),
}
_LOWER_IS_BETTER = {"Max DD %"}


def _get_pnl_series(record: dict) -> list[float]:
    """Helper to extract PnL series from record, with fallback for older runs."""
    m = record.get("metrics", {})
    if m.get("pnl_series"):
        return m["pnl_series"]
    
    # Fallback 1: graphLog (timestamp;value)
    cfg = record.get("config", {})
    gl = cfg.get("graphLog", "")
    if gl and isinstance(gl, str):
        try:
            lines = gl.replace("\\n", "\n").strip().split("\n")
            if len(lines) > 1:
                series = []
                for line in lines[1:]:
                    parts = line.split(";")
                    if len(parts) >= 2:
                        series.append(float(parts[1]))
                if series: return series
        except Exception:
            pass

    # Fallback 2: activitiesLog (full csv)
    al = cfg.get("activitiesLog", "")
    if al and isinstance(al, str):
        try:
            rows = _parse_activities_csv(al)
            ts_pnl: dict[int, float] = {}
            for r in rows:
                ts = int(_safe_float(r.get("timestamp", 0)))
                pnl = _safe_float(r.get("profit_and_loss", 0))
                ts_pnl[ts] = ts_pnl.get(ts, 0) + pnl
            if ts_pnl:
                return [ts_pnl[t] for t in sorted(ts_pnl.keys())]
        except Exception:
            pass
            
    return []


class LeaderboardTab(ttk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, style="TFrame")
        self._app        = app
        self._records:    list[dict] = []
        self._hidden_ids: set        = set()
        self._sort_col   = "Total PnL"
        self._sort_asc   = False
        self._build()

    # ─── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 0))
        hdr.columnconfigure(99, weight=1)

        tk.Label(hdr, text="LEADERBOARD", font=FONT_TTL,
                 bg=BG, fg=CYAN).grid(row=0, column=0, sticky="w")

        # Controls
        ctl = tk.Frame(hdr, bg=BG)
        ctl.grid(row=0, column=99, sticky="e")

        tk.Label(ctl, text="Sort", font=FONT_MONO2, bg=BG, fg=TEXT2
                 ).pack(side="left", padx=(0, 6))
        self._sort_var = tk.StringVar(value=self._sort_col)
        sort_cb = ttk.Combobox(ctl, textvariable=self._sort_var,
                               values=list(_SORT_KEYS), state="readonly", width=13)
        sort_cb.pack(side="left")
        sort_cb.bind("<<ComboboxSelected>>", lambda _: self.after(0, self._sort_changed))

        tk.Label(ctl, text="  Product", font=FONT_MONO2, bg=BG, fg=TEXT2
                 ).pack(side="left", padx=(12, 6))
        self._prod_var = tk.StringVar(value="All")
        self._prod_cb  = ttk.Combobox(ctl, textvariable=self._prod_var,
                                       values=["All"], state="readonly", width=14)
        self._prod_cb.pack(side="left")
        self._prod_cb.bind("<<ComboboxSelected>>", lambda _: self.after(0, self._apply))

        tk.Label(ctl, text="  Author", font=FONT_MONO2, bg=BG, fg=TEXT2
                 ).pack(side="left", padx=(12, 6))
        self._author_var = tk.StringVar(value="All")
        self._author_cb  = ttk.Combobox(ctl, textvariable=self._author_var,
                                         values=["All"], state="readonly", width=14)
        self._author_cb.pack(side="left")
        self._author_cb.bind("<<ComboboxSelected>>", lambda _: self.after(0, self._apply))

        tk.Label(ctl, text="  Round", font=FONT_MONO2, bg=BG, fg=TEXT2
                 ).pack(side="left", padx=(12, 6))
        self._round_var = tk.StringVar(value="All")
        self._round_cb  = ttk.Combobox(ctl, textvariable=self._round_var,
                                        values=["All"], state="readonly", width=6)
        self._round_cb.pack(side="left")
        self._round_cb.bind("<<ComboboxSelected>>", lambda _: self.after(0, self._apply))

        self._show_hidden_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctl, text=" Hidden", variable=self._show_hidden_var,
                       command=self._apply,
                       bg=BG, fg=TEXT2, selectcolor=PANEL2,
                       activebackground=BG, activeforeground=CYAN,
                       font=FONT_MONO2).pack(side="left", padx=(10, 0))

        ttk.Button(ctl, text="⟳  Refresh", style="Ghost.TButton",
                   command=self._refresh).pack(side="left", padx=(16, 0))

        # ── Separator ─────────────────────────────────────────────────────────
        sep(self).grid(row=1, column=0, sticky="ew", padx=28, pady=(12, 0))

        # ── Summary cards ─────────────────────────────────────────────────────
        sf = tk.Frame(self, bg=BG)
        sf.grid(row=2, column=0, sticky="ew", padx=28, pady=(16, 0))
        sf.columnconfigure((0, 1, 2, 3, 4), weight=1)

        self._sc_count   = MetricCard(sf, "TOTAL RUNS")
        self._sc_best    = MetricCard(sf, "BEST PnL")
        self._sc_author  = MetricCard(sf, "TOP AUTHOR")
        self._sc_sharpe  = MetricCard(sf, "BEST SHARPE")
        self._sc_prods   = MetricCard(sf, "PRODUCTS SEEN")

        for i, c in enumerate([self._sc_count, self._sc_best, self._sc_author,
                                self._sc_sharpe, self._sc_prods]):
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))

        # ── Chart area (inline bar chart canvas) ──────────────────────────────
        chart_wrap = tk.Frame(self, bg=BG)
        chart_wrap.grid(row=3, column=0, sticky="ew", padx=28, pady=(14, 0))
        chart_wrap.columnconfigure(0, weight=1)

        tk.Label(chart_wrap, text="PnL DISTRIBUTION", font=FONT_MONO2,
                 bg=BG, fg=TEXT2).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self._chart = tk.Canvas(chart_wrap, bg=PANEL2, height=80,
                                 highlightthickness=0)
        self._chart.grid(row=1, column=0, sticky="ew")
        chart_wrap.bind("<Configure>", lambda e: self._draw_chart())

        # ── Treeview ──────────────────────────────────────────────────────────
        tree_frame = tk.Frame(self, bg=BG)
        tree_frame.grid(row=4, column=0, sticky="nsew", padx=28, pady=(14, 0))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        cols = [c[0] for c in _COLS]
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                   selectmode="browse")
        for col, width, anchor in _COLS:
            self._tree.heading(col, text=col,
                               command=lambda c=col: self._heading_click(c))
            self._tree.column(col, width=width, anchor=anchor, minwidth=30)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self._tree.tag_configure("gold",  foreground=GOLD)
        self._tree.tag_configure("silver",foreground="#aabbcc")
        self._tree.tag_configure("bronze",foreground="#cc8844")
        self._tree.tag_configure("neg",   foreground=RED)
        self._tree.tag_configure("dim",   foreground=TEXT2)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── Detail panel (bottom) ─────────────────────────────────────────────
        self._detail = tk.Frame(self, bg=PANEL, height=0)
        self._detail.grid(row=5, column=0, sticky="ew", padx=28, pady=(8, 0))
        self._detail.columnconfigure(0, weight=2)
        self._detail.columnconfigure(1, weight=3)
        self._detail_visible = False
        self._detail_lbl = tk.Label(self._detail, text="", font=FONT_MONO,
                                     bg=PANEL, fg=TEXT, anchor="w", justify="left",
                                     wraplength=400)
        self._detail_lbl.grid(row=0, column=0, sticky="nsew", padx=14, pady=10)

        self._detail_canvas = tk.Canvas(self._detail, bg=PANEL2, height=120,
                                         highlightthickness=0)
        self._detail_canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 14), pady=10)
        self._detail_canvas.bind("<Configure>", lambda _: self._on_select()) # Redraw on resize

        btn_row = tk.Frame(self._detail, bg=PANEL)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 8))
        self._hide_btn = ttk.Button(btn_row, text="Hide Run",
                                     style="Danger.TButton",
                                     command=self._toggle_hide_selected)
        self._hide_btn.pack(side="left")

        # ── Status ────────────────────────────────────────────────────────────
        self._lb_status = tk.StringVar(value="Click ⟳ Refresh to load records.")
        tk.Label(self, textvariable=self._lb_status, font=FONT_MONO2,
                 bg=BG, fg=TEXT2, anchor="w"
                 ).grid(row=6, column=0, sticky="w", padx=30, pady=(6, 12))

    # ─── Data ─────────────────────────────────────────────────────────────────

    def refresh(self):
        self._refresh()

    def _refresh(self):
        repo  = self._app.get_repo()
        token = self._app.get_token()
        if not repo or not token:
            messagebox.showerror("Not configured",
                                 "Set GitHub Repo and Token in the Upload tab.")
            return
        self._lb_status.set("Fetching records from GitHub…")
        self._app.status.set("Fetching leaderboard…", CYAN)

        def _worker():
            try:
                records    = fetch_records(repo, token)
                hidden_ids = fetch_hidden(repo, token)
                self.after(0, self._on_loaded, records, hidden_ids)
            except Exception as exc:
                self.after(0, self._on_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded(self, records: list[dict], hidden_ids: set = None):
        self._records    = records
        self._hidden_ids = hidden_ids or set()
        all_prods:   set[str] = set()
        all_authors: set[str] = set()
        all_rounds:  set[int] = set()
        for r in records:
            all_prods.update(r.get("metrics", {}).get("products", []))
            if r.get("author"):
                all_authors.add(r["author"])
            if r.get("round") is not None:
                all_rounds.add(int(r["round"]))
        self._prod_cb.configure(  values=["All"] + sorted(all_prods))
        self._author_cb.configure(values=["All"] + sorted(all_authors))
        self._round_cb.configure( values=["All"] + [str(rn) for rn in sorted(all_rounds)])
        self._apply()
        self._lb_status.set(f"{len(records)} record(s) loaded.")
        self._app.status.set(f"Leaderboard: {len(records)} runs loaded.", GREEN)

    def _on_error(self, msg: str):
        self._lb_status.set(f"Error: {msg}")
        self._app.status.set(f"Leaderboard error: {msg}", RED)

    # ─── Sort & filter ────────────────────────────────────────────────────────

    def _heading_click(self, col: str):
        if col in _SORT_KEYS:
            if self._sort_var.get() == col:
                self._sort_asc = not self._sort_asc
            else:
                self._sort_var.set(col)
                self._sort_asc = col in _LOWER_IS_BETTER
            self._apply()

    def _sort_changed(self):
        col = self._sort_var.get()
        self._sort_asc = col in _LOWER_IS_BETTER
        self._apply()

    def _apply(self):
        col          = self._sort_var.get()
        prod_filter  = self._prod_var.get()
        auth_filter  = self._author_var.get()
        round_filter = self._round_var.get()
        show_hidden  = self._show_hidden_var.get()
        records      = list(self._records)

        if not show_hidden:
            records = [r for r in records if r.get("run_id") not in self._hidden_ids]
        if prod_filter and prod_filter != "All":
            records = [r for r in records
                       if prod_filter in r.get("metrics", {}).get("products", [])]
        if auth_filter and auth_filter != "All":
            records = [r for r in records if r.get("author") == auth_filter]
        if round_filter and round_filter != "All":
            records = [r for r in records if str(r.get("round", "")) == round_filter]

        path = _SORT_KEYS.get(col, ("metrics", "total_pnl"))

        def _key(r):
            cur = r
            for k in path:
                cur = cur.get(k, "") if isinstance(cur, dict) else ""
            if isinstance(cur, (int, float)):
                return cur
            try:
                return float(cur)
            except Exception:
                return str(cur)

        records.sort(key=_key, reverse=not self._sort_asc)
        self._render(records)
        self._update_summary(records)
        self._draw_chart_data(records)

    # ─── Render ───────────────────────────────────────────────────────────────

    def _render(self, records: list[dict]):
        for row in self._tree.get_children():
            self._tree.delete(row)

        for rank, r in enumerate(records, 1):
            run_id    = r.get("run_id", str(rank))
            is_hidden = run_id in self._hidden_ids
            m         = r.get("metrics", {})
            pnl       = m.get("total_pnl", 0.0)
            date      = (r.get("uploaded_at", "")[:10])
            prods     = ", ".join(m.get("products", []))

            pnl_str = f"{pnl:+,.2f}" if pnl != 0 else "0.00"
            sh_str  = f"{m.get('sharpe', 0):.3f}"
            dd_str  = f"{m.get('max_drawdown_pct', 0):.2f}%"

            rank_str = f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else str(rank)}"
            if is_hidden:
                rank_str = f"∅{rank}"

            tag = ("dim" if is_hidden else
                   "gold" if rank == 1 else
                   "silver" if rank == 2 else
                   "bronze" if rank == 3 else
                   "neg" if pnl < 0 else "dim")

            self._tree.insert("", "end", iid=run_id,
                              values=(rank_str,
                                      r.get("round", "—"),
                                      r.get("author", "—"),
                                      r.get("strategy_name", "—"),
                                      pnl_str, sh_str, dd_str,
                                      m.get("submission_trades", 0),
                                      f"{m.get('submission_volume', 0):,}",
                                      prods, date,
                                      r.get("notes", "")[:60]),
                              tags=(tag,))

    # ─── Summary stats ────────────────────────────────────────────────────────

    def _update_summary(self, records: list[dict]):
        if not records:
            for c in [self._sc_count, self._sc_best, self._sc_author,
                      self._sc_sharpe, self._sc_prods]:
                c.set("—", TEXT2)
            return

        pnls    = [r.get("metrics", {}).get("total_pnl", 0) for r in records]
        sharpes = [r.get("metrics", {}).get("sharpe", 0) for r in records]
        all_prods: set[str] = set()
        for r in records:
            all_prods.update(r.get("metrics", {}).get("products", []))

        best_pnl_r = max(records, key=lambda r: r.get("metrics", {}).get("total_pnl", 0))
        best_sh_r  = max(records, key=lambda r: r.get("metrics", {}).get("sharpe", 0))

        # Top author by total pnl sum
        author_pnl: dict[str, float] = {}
        for r in records:
            a = r.get("author", "—")
            author_pnl[a] = author_pnl.get(a, 0) + r.get("metrics", {}).get("total_pnl", 0)
        top_author = max(author_pnl, key=author_pnl.get) if author_pnl else "—"

        self._sc_count.set(str(len(records)), CYAN)
        best = best_pnl_r.get("metrics", {}).get("total_pnl", 0)
        self._sc_best.set(f"{best:+,.2f}", GREEN if best >= 0 else RED)
        self._sc_author.set(top_author, GOLD)
        self._sc_sharpe.set(f"{best_sh_r.get('metrics',{}).get('sharpe',0):.4f}", PURPLE)
        self._sc_prods.set(str(len(all_prods)), CYAN)

    # ─── Chart ────────────────────────────────────────────────────────────────

    def _draw_chart(self):
        if self._records:
            recs = list(self._records)
            self._draw_chart_data(recs)

    def _draw_chart_data(self, records: list[dict]):
        self._chart.update_idletasks()
        w = self._chart.winfo_width()
        h = self._chart.winfo_height()
        if w <= 1:
            return
        self._chart.delete("all")
        if not records:
            return

        pnls   = [r.get("metrics", {}).get("total_pnl", 0) for r in records]
        labels = [f"{r.get('author','?')[:8]}/{r.get('strategy_name','?')[:8]}"
                  for r in records]

        max_abs = max(abs(p) for p in pnls) if pnls else 1
        if max_abs == 0:
            max_abs = 1

        bar_w   = max(4, (w - 40) // len(pnls) - 2)
        mid_y   = h // 2
        x_start = 20

        # Baseline
        self._chart.create_line(x_start, mid_y, w - 20, mid_y,
                                 fill=BORDER2, width=1)

        for i, (pnl, label) in enumerate(zip(pnls, labels)):
            x = x_start + i * (bar_w + 2)
            ratio = pnl / max_abs
            bar_h = int(abs(ratio) * (mid_y - 8))
            color = GREEN if pnl >= 0 else RED

            if pnl >= 0:
                self._chart.create_rectangle(x, mid_y - bar_h, x + bar_w, mid_y,
                                              fill=color, outline="", width=0)
            else:
                self._chart.create_rectangle(x, mid_y, x + bar_w, mid_y + bar_h,
                                              fill=color, outline="", width=0)

            if bar_w >= 20:
                self._chart.create_text(x + bar_w // 2, h - 4, text=label[:6],
                                         font=FONT_MONO2, fill=TEXT2, angle=0,
                                         anchor="s")

    # ─── Detail pane ─────────────────────────────────────────────────────────

    def _on_select(self, _=None):
        sel = self._tree.selection()
        if not sel:
            return
        run_id = sel[0]
        record = next((r for r in self._records if r.get("run_id") == run_id), None)
        if not record:
            return
        m = record.get("metrics", {})
        lines = [
            f"  RUN  {record.get('run_id', '—')}",
            f"  Author: {record.get('author', '—')}",
            f"  Strategy: {record.get('strategy_name', '—')}",
            f"  Uploaded: {record.get('uploaded_at', '')[:19]}",
            f"  PnL: {m.get('total_pnl',0):+,.4f}",
            f"  Sharpe: {m.get('sharpe',0):.6f}",
            f"  MaxDD: {m.get('max_drawdown_pct',0):.4f}%",
            f"  SubTrades: {m.get('submission_trades',0)}",
        ]
        if m.get("per_product_pnl"):
            pp = "  per-product → " + ", ".join(
                f"{k}:{v:+,.0f}" for k, v in sorted(m["per_product_pnl"].items()))
            lines.append(pp)
        if record.get("notes"):
            lines.append(f"  Notes: {record['notes']}")
        
        self._detail_lbl.configure(text="\n".join(lines))
        _draw_pnl_on_canvas(self._detail_canvas, _get_pnl_series(record),
                           final_pnl=m.get("total_pnl"))
        self._update_hide_btn()

    def _update_hide_btn(self):
        sel = self._tree.selection()
        if not sel:
            return
        if sel[0] in self._hidden_ids:
            self._hide_btn.configure(text="Unhide Run", style="Ghost.TButton")
        else:
            self._hide_btn.configure(text="Hide Run",   style="Danger.TButton")

    def _toggle_hide_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        run_id = sel[0]
        repo  = self._app.get_repo()
        token = self._app.get_token()
        if run_id in self._hidden_ids:
            self._hidden_ids.discard(run_id)
        else:
            self._hidden_ids.add(run_id)
        self._update_hide_btn()
        self._apply()
        self._app.status.set("Saving hidden list…", CYAN)
        def _worker():
            try:
                set_hidden(repo, token, self._hidden_ids)
                self.after(0, lambda: self._app.status.set("Hidden list saved.", GREEN))
            except Exception as exc:
                self.after(0, lambda: self._app.status.set(f"Error saving hidden: {exc}", RED))
        threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ══════════════════════════════════════════════════════════════════════════════

class SettingsTab(ttk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, style="TFrame")
        self._app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)

        tk.Label(self, text="ABOUT", font=FONT_TTL, bg=BG, fg=CYAN, anchor="w"
                 ).grid(row=0, column=0, sticky="w", padx=28, pady=(22, 0))
        sep(self).grid(row=1, column=0, sticky="ew", padx=28, pady=(12, 0))

        info = tk.Frame(self, bg=BG)
        info.grid(row=2, column=0, sticky="ew", padx=28, pady=20)
        info.columnconfigure(0, weight=1)

        lines = [
            ("REPO",  DEFAULT_REPO  or "(not set — fill DEFAULT_REPO in script)"),
            ("TOKEN", "●●●●●●●●" if DEFAULT_TOKEN else "(not set — fill DEFAULT_TOKEN in script)"),
        ]
        for i, (k, v) in enumerate(lines):
            tk.Label(info, text=k, font=FONT_MONO2, bg=BG, fg=TEXT2
                     ).grid(row=i, column=0, sticky="w", pady=(0, 2))
            tk.Label(info, text=v, font=FONT_MONO, bg=BG,
                     fg=GREEN if (DEFAULT_REPO if k == "REPO" else DEFAULT_TOKEN) else RED
                     ).grid(row=i, column=1, sticky="w", padx=(12, 0), pady=(0, 2))

        sep(info).grid(row=len(lines), column=0, columnspan=2, sticky="ew", pady=16)

        shortcuts = [
            ("Ctrl+U", "Switch to Upload tab"),
            ("Ctrl+L", "Switch to Leaderboard tab"),
            ("Ctrl+R", "Refresh Leaderboard data"),
            ("Ctrl+Enter", "Upload Run (when on Upload tab)"),
        ]
        tk.Label(info, text="KEYBOARD SHORTCUTS", font=FONT_MONO2,
                 bg=BG, fg=TEXT2).grid(row=len(lines)+1, column=0, columnspan=2,
                                        sticky="w", pady=(0, 6))
        for i, (key, desc) in enumerate(shortcuts):
            tk.Label(info, text=key, font=FONT_BOLD, bg=BG, fg=CYAN
                     ).grid(row=len(lines)+2+i, column=0, sticky="w")
            tk.Label(info, text=desc, font=FONT_MONO, bg=BG, fg=TEXT
                     ).grid(row=len(lines)+2+i, column=1, sticky="w", padx=(12, 0))


# ══════════════════════════════════════════════════════════════════════════════
# APP SHELL
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Strategy Tracker")
        self.configure(bg=BG)
        self.geometry("1100x780")
        self.minsize(860, 600)

        self._repo  = DEFAULT_REPO
        self._token = DEFAULT_TOKEN

        apply_theme(self)
        self._build()

        # Open file picker immediately on launch
        self.after(200, self._upload_tab.open_picker)

    def _build(self):
        # ── Title strip ───────────────────────────────────────────────────────
        strip = tk.Frame(self, bg=BG2, height=48)
        strip.pack(fill="x", side="top")
        strip.pack_propagate(False)

        tk.Label(strip, text="◈  STRATEGY TRACKER",
                 font=("Consolas", 12, "bold"), bg=BG2, fg=CYAN
                 ).pack(side="left", padx=20, pady=12)
        tk.Label(strip, text="v2.0  ·  team performance dashboard",
                 font=FONT_MONO2, bg=BG2, fg=TEXT2
                 ).pack(side="left", padx=4, pady=16)

        # ── Notebook ──────────────────────────────────────────────────────────
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, side="top")

        self._upload_tab    = UploadTab(self._nb, self)
        self._lb_tab        = LeaderboardTab(self._nb, self)
        self._settings_tab  = SettingsTab(self._nb, self)

        self._nb.add(self._upload_tab,   text="  ⬆  Upload  ")
        self._nb.add(self._lb_tab,       text="  ◈  Leaderboard  ")
        self._nb.add(self._settings_tab, text="  ⚙  Settings  ")

        # ── Status bar ────────────────────────────────────────────────────────
        self.status = StatusBar(self)
        self.status.pack(fill="x", side="bottom")

        # Keyboard shortcut
        self.bind("<Control-l>", lambda _: self._nb.select(1))
        self.bind("<Control-u>", lambda _: self._nb.select(0))
        self.bind("<Control-r>", lambda _: self._lb_tab.refresh())
        self.bind("<Control-Return>", lambda _: self._on_ctrl_enter())

    def _on_ctrl_enter(self):
        # If on Upload tab, trigger upload
        if self._nb.index("current") == 0:
            self._upload_tab._start_upload()

    # ─── Shared state ─────────────────────────────────────────────────────────

    def get_repo(self) -> str:
        return self._repo

    def get_token(self) -> str:
        return self._token

    def set_repo(self, v: str):
        self._repo = v

    def set_token(self, v: str):
        self._token = v

    def trigger_leaderboard_refresh(self):
        self._nb.select(1)
        self._lb_tab.refresh()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
