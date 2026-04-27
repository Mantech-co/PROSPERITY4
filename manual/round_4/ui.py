import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import gammaln

# ── palette ──────────────────────────────────────────────────────────────────
BG    = "#1a1a2e"
ROW1  = "#16213e"
ROW2  = "#0f3460"
FG    = "#e0e0e0"
GREEN = "#00ff88"
RED   = "#ff4d6d"
BLUE  = "#4fc3f7"
GOLD  = "#ffd700"
EBG   = "#0d0d1a"

# ── contracts ─────────────────────────────────────────────────────────────────
CONTRACTS = [
    {"name": "AC",        "expiry": "N/A",    "kind": "underlying", "K": 50,  "bid": 49.975, "ask": 50.025, "max": 200, "csv": None,                          "aon": None},
    {"name": "AC_50_P",   "expiry": "T+21",   "kind": "put",        "K": 50,  "bid": 12,     "ask": 12.05,  "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_50_C",   "expiry": "T+21",   "kind": "call",       "K": 50,  "bid": 12,     "ask": 12.05,  "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_35_P",   "expiry": "T+21",   "kind": "put",        "K": 35,  "bid": 4.33,   "ask": 4.35,   "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_40_P",   "expiry": "T+21",   "kind": "put",        "K": 40,  "bid": 6.5,    "ask": 6.55,   "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_45_P",   "expiry": "T+21",   "kind": "put",        "K": 45,  "bid": 9.05,   "ask": 9.1,    "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_60_C",   "expiry": "T+21",   "kind": "call",       "K": 60,  "bid": 8.8,    "ask": 8.85,   "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_50_P_2", "expiry": "T+14",   "kind": "put",        "K": 50,  "bid": 9.7,    "ask": 9.75,   "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_50_C_2", "expiry": "T+14",   "kind": "call",       "K": 50,  "bid": 9.7,    "ask": 9.75,   "max": 50,  "csv": None,                          "aon": None},
    {"name": "AC_50_CO",  "expiry": "T+14/21","kind": "csv",        "K": 50,  "bid": 22.2,   "ask": 22.3,   "max": 50,  "csv": "chooser_payout_grouped.csv",  "aon": None},
    {"name": "AC_40_BP",  "expiry": "T+21",   "kind": "aon_put",    "K": 40,  "bid": 5.0,    "ask": 5.1,    "max": 50,  "csv": None,                          "aon": 10},
    {"name": "AC_45_KO",  "expiry": "T+21",   "kind": "csv",        "K": 45,  "bid": 0.15,   "ask": 0.175,  "max": 500, "csv": "ko_payout_B35_grouped.csv",   "aon": None},
]

# preload CSVs
for c in CONTRACTS:
    if c["csv"]:
        d = np.loadtxt(c["csv"], delimiter=",", skiprows=1)
        c["spots"], c["payoffs"] = d[:, 0], d[:, 1]

def _binomial_probs(S0, N, sigma, s):
    n  = N * s
    dt = 1.0 / s
    u  = np.exp(sigma * math.sqrt(dt))
    d  = 1.0 / u
    p  = (1.0 - d) / (u - d)
    j  = np.arange(n + 1)
    lp = (gammaln(n+1) - gammaln(j+1) - gammaln(n-j+1)
          + j*math.log(p) + (n-j)*math.log(1-p))
    net = 2*j - n
    return np.where(net >= 0, S0*u**net, S0*d**(-net)), np.exp(lp)

def _payout(c, S, side, vol):
    """Compute payout array for contract c, given side (Buy/Sell) and volume."""
    prem = c["ask"] if side == "Buy" else c["bid"]
    qty  = vol if side == "Buy" else -vol
    kind = c["kind"]
    K    = c["K"]
    if kind == "call":
        raw = np.maximum(S - K, 0)
    elif kind == "put":
        raw = np.maximum(K - S, 0)
    elif kind == "aon_put":
        raw = np.where(S < K, float(c["aon"]), 0.0)
    elif kind == "underlying":
        raw = S
    elif kind == "csv":
        raw = np.interp(S, c["spots"], c["payoffs"],
                        left=c["payoffs"][0], right=c["payoffs"][-1])
    return qty * (raw - prem)


class App:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg=BG)
        self._pending = None

        # per-row state: (side_var, vol_var)
        self.rows = []
        for c in CONTRACTS:
            sv = tk.StringVar(value="Choose")
            vv = tk.StringVar(value="0")
            sv.trace_add("write", lambda *_: self._schedule())
            vv.trace_add("write", lambda *_: self._schedule())
            self.rows.append((sv, vv))

        self._build_ui()
        self._redraw()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        left  = tk.Frame(self.root, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4), pady=8)

        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4,8), pady=8)

        # title
        tk.Label(left, text="AVAILABLE OPTION CONTRACTS", bg=BG, fg=FG,
                 font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0,6))

        # model params strip
        pf = tk.Frame(left, bg=BG)
        pf.pack(fill=tk.X, pady=(0, 6))
        self.v_S0    = self._mini(pf, "S0",    "50")
        self.v_N     = self._mini(pf, "N",     "15")
        self.v_sigma = self._mini(pf, "σ",     "0.1582")
        self.v_s     = self._mini(pf, "s",     "4")
        self.v_Smin  = self._mini(pf, "Smin",  "20")
        self.v_Smax  = self._mini(pf, "Smax",  "90")

        # table header
        self._header(left)

        # contract rows
        for i, c in enumerate(CONTRACTS):
            self._contract_row(left, i, c)

        # matplotlib
        plt.style.use("dark_background")
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 6.5))
        self.fig.patch.set_facecolor(BG)
        cv = FigureCanvasTkAgg(self.fig, master=right)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = cv

    def _mini(self, frame, lbl, val):
        tk.Label(frame, text=lbl, bg=BG, fg=FG,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(4,0))
        v = tk.StringVar(value=val)
        tk.Entry(frame, textvariable=v, width=6, bg=EBG, fg=GREEN,
                 insertbackground=FG, relief=tk.FLAT,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(1,4))
        v.trace_add("write", lambda *_: self._schedule())
        return v

    def _header(self, parent):
        hdr = tk.Frame(parent, bg=ROW2)
        hdr.pack(fill=tk.X, pady=(0,1))
        cols = [("OPTION",8),("EXPIRY",8),("SIZE",5),("BID",6),("ASK",6),("SIZE",5),("BUY/SELL",10),("VOLUME",8)]
        for txt, w in cols:
            tk.Label(hdr, text=txt, bg=ROW2, fg=FG,
                     width=w, font=("Helvetica", 8, "bold"),
                     anchor="center").pack(side=tk.LEFT, padx=1)

    def _contract_row(self, parent, i, c):
        bg = ROW1 if i % 2 == 0 else BG
        row = tk.Frame(parent, bg=bg)
        row.pack(fill=tk.X, pady=1)

        sv, vv = self.rows[i]

        # name (highlighted for CSV options)
        name_fg = BLUE if c["kind"] == "csv" else (GOLD if c["kind"] == "aon_put" else FG)
        tk.Label(row, text=c["name"],   bg=bg, fg=name_fg, width=8,  font=("Helvetica",8,"bold"), anchor="w").pack(side=tk.LEFT, padx=2)
        tk.Label(row, text=c["expiry"], bg=bg, fg=FG,      width=8,  font=("Helvetica",8),         anchor="center").pack(side=tk.LEFT)
        tk.Label(row, text=c["max"],    bg=bg, fg=GREEN,   width=5,  font=("Helvetica",8),         anchor="center").pack(side=tk.LEFT)
        tk.Label(row, text=c["bid"],    bg=bg, fg=GREEN,   width=6,  font=("Helvetica",8),         anchor="center").pack(side=tk.LEFT)
        tk.Label(row, text=c["ask"],    bg=bg, fg=RED,     width=6,  font=("Helvetica",8),         anchor="center").pack(side=tk.LEFT)
        tk.Label(row, text=c["max"],    bg=bg, fg=RED,     width=5,  font=("Helvetica",8),         anchor="center").pack(side=tk.LEFT)

        # buy/sell dropdown
        om = tk.OptionMenu(row, sv, "Choose", "Buy", "Sell")
        om.config(bg=EBG, fg=FG, activebackground=ROW2, activeforeground=FG,
                  highlightthickness=0, relief=tk.FLAT, width=6,
                  font=("Helvetica", 8))
        om["menu"].config(bg=EBG, fg=FG, font=("Helvetica", 8))
        om.pack(side=tk.LEFT, padx=2)

        # volume entry
        ve = tk.Entry(row, textvariable=vv, width=6, bg=EBG, fg=FG,
                      insertbackground=FG, relief=tk.FLAT, font=("Helvetica",8))
        ve.pack(side=tk.LEFT, padx=2)

        # colour dropdown border based on selection
        def _on_side(*_, _om=om, _sv=sv, _vv=vv, _max=c["max"]):
            side = _sv.get()
            _om.config(fg=GREEN if side=="Buy" else (RED if side=="Sell" else FG))
            if side != "Choose" and _vv.get() in ("0",""):
                _vv.set(str(_max))
            elif side == "Choose":
                _vv.set("0")

        sv.trace_add("write", _on_side)

    # ── redraw ────────────────────────────────────────────────────────────────

    def _schedule(self):
        if self._pending:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(120, self._redraw)

    def _redraw(self):
        try:
            S0    = float(self.v_S0.get())
            N     = int(float(self.v_N.get()))
            sigma = float(self.v_sigma.get())
            s     = int(float(self.v_s.get()))
            S_min = float(self.v_Smin.get())
            S_max = float(self.v_Smax.get())
        except ValueError:
            return

        S     = np.linspace(S_min, S_max, 600)
        total = np.zeros(len(S))

        self.ax1.cla(); self.ax2.cla()
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(BG)

        for c, (sv, vv) in zip(CONTRACTS, self.rows):
            side = sv.get()
            if side == "Choose":
                continue
            try:
                vol = float(vv.get())
                if vol <= 0:
                    continue
                vol = min(vol, c["max"])
            except ValueError:
                continue

            y = _payout(c, S, side, vol)
            lbl = f"{'L' if side=='Buy' else 'S'} {c['name']}"
            self.ax1.plot(S, y, linewidth=1, linestyle="--", alpha=0.7, label=lbl)
            total += y

        self.ax1.plot(S, total, linewidth=2.2, color=GOLD, label="Total")
        self.ax1.axhline(0, color="gray", linewidth=0.6)
        self.ax1.set_ylabel("Payout", color=FG)
        self.ax1.tick_params(colors=FG)
        self.ax1.legend(fontsize=7, loc="upper left")
        self.ax1.grid(True, alpha=0.12)

        try:
            levels, probs = _binomial_probs(S0, N, sigma, s)
            ev = np.interp(levels, S, total) * probs
            colors = [GREEN if v >= 0 else RED for v in ev]
            bw = (levels[1] - levels[0]) * 0.8
            self.ax2.bar(levels, ev, width=bw, color=colors, alpha=0.85)
            self.ax2.axhline(0, color="gray", linewidth=0.6)
            self.ax2.set_xlabel("Spot at expiry", color=FG)
            self.ax2.set_ylabel("Payout × P(S)", color=FG)
            self.ax2.set_title(f"Expected Value = {ev.sum():.4f}", color=FG, fontsize=9)
            self.ax2.tick_params(colors=FG)
            self.ax2.grid(True, alpha=0.12)
        except Exception:
            pass

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Option Portfolio Builder")
    root.geometry("1150x740")
    App(root)
    root.mainloop()
