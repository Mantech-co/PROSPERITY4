import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

DATA_DIRS = [
    "/media/manukrishnan/Mk/prosperity_4/data",
    "/media/manukrishnan/Mk/prosperity_4/data/round4",
    "/media/manukrishnan/Mk/prosperity_4/data/old",
    "/media/manukrishnan/Mk/prosperity_4/round_4",
]

def load_prices():
    frames = []
    seen = set()
    for d in DATA_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "prices_*.csv"))):
            key = os.path.basename(f)
            if key in seen:
                continue
            seen.add(key)
            df = pd.read_csv(f, sep=";")
            # derive round/day label from filename
            base = os.path.splitext(key)[0]  # prices_round_4_day_1
            parts = base.split("_")
            try:
                rnd = int(parts[2])
                day_val = int(parts[4])
            except (IndexError, ValueError):
                rnd, day_val = 0, 0
            df["_round"] = rnd
            df["_day_raw"] = day_val
            frames.append(df)
    if not frames:
        raise RuntimeError("No price CSVs found")
    prices = pd.concat(frames, ignore_index=True)
    # global timestamp: encode round + day_raw + timestamp
    # offset days so they are monotonically increasing across rounds
    prices["gts"] = (prices["_round"] * 10 + prices["_day_raw"]) * 1_000_000 + prices["timestamp"]
    if "mid_price" not in prices.columns:
        prices["mid_price"] = (prices["bid_price_1"] + prices["ask_price_1"]) / 2
    return prices

prices = load_prices()
ALL_PRODUCTS = sorted(prices["product"].unique())

# ── state ────────────────────────────────────────────────────────────────────
selected = [ALL_PRODUCTS[0], ALL_PRODUCTS[1] if len(ALL_PRODUCTS) > 1 else ALL_PRODUCTS[0]]
COLORS = ["#1f77b4", "#ff7f0e"]
ZOOM_FACTOR = 0.3
PAN_FRACTION = 0.2

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#1a1a2e")

# layout: main plot takes most width, right panel for controls
ax = fig.add_axes([0.05, 0.18, 0.68, 0.75])
ax.set_facecolor("#16213e")
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color("white")
ax.set_title("Mid Price Viewer", color="white", fontsize=13)
ax.set_xlabel("Global Timestamp", color="#aaa")
ax.set_ylabel("Mid Price", color="#aaa")

lines = [None, None]

def get_series(product):
    sub = prices[prices["product"] == product].sort_values("gts")
    return sub["gts"].values, sub["mid_price"].values

def redraw():
    ax.cla()
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")
    ax.set_xlabel("Global Timestamp", color="#aaa")
    ax.set_ylabel("Mid Price", color="#aaa")
    ax.set_title("Mid Price Viewer — arrow keys: pan | +/-: zoom | r: reset", color="white", fontsize=11)

    plotted = []
    for i, prod in enumerate(selected):
        if prod and visible[i]:
            x, y = get_series(prod)
            ax.plot(x, y, color=COLORS[i], lw=1, label=prod, alpha=0.9)
            plotted.append((x, y))

    if plotted:
        xs = np.concatenate([p[0] for p in plotted])
        ys = np.concatenate([p[1] for p in plotted])
        if view_xlim[0] is None:
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min() - 1, ys.max() + 1)
        else:
            ax.set_xlim(*view_xlim)
            ax.set_ylim(*view_ylim)

    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="#444")
    ax.grid(alpha=0.15, color="#888")
    fig.canvas.draw_idle()

view_xlim = [None, None]
view_ylim = [None, None]
visible = [True, True]

def save_view():
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    view_xlim[0], view_xlim[1] = xl
    view_ylim[0], view_ylim[1] = yl

# ── product selector buttons (cycle through products) ────────────────────────
ax_p1 = fig.add_axes([0.75, 0.82, 0.22, 0.05])
ax_p2 = fig.add_axes([0.75, 0.75, 0.22, 0.05])

ax_p1.set_facecolor("#0f3460")
ax_p2.set_facecolor("#0f3460")

btn_p1 = Button(ax_p1, f"[1] {selected[0][:20]}", color="#0f3460", hovercolor="#16213e")
btn_p2 = Button(ax_p2, f"[2] {selected[1][:20]}", color="#0f3460", hovercolor="#16213e")
btn_p1.label.set_color("white")
btn_p2.label.set_color(COLORS[1])

_idx = [ALL_PRODUCTS.index(selected[0]) if selected[0] in ALL_PRODUCTS else 0,
        ALL_PRODUCTS.index(selected[1]) if selected[1] in ALL_PRODUCTS else 1]

def cycle_product(slot, direction=1):
    _idx[slot] = (_idx[slot] + direction) % len(ALL_PRODUCTS)
    selected[slot] = ALL_PRODUCTS[_idx[slot]]
    label = f"[{slot+1}] {selected[slot][:20]}"
    if slot == 0:
        btn_p1.label.set_text(label)
    else:
        btn_p2.label.set_text(label)
    save_view()
    redraw()

def on_p1(event): cycle_product(0)
def on_p2(event): cycle_product(1)
btn_p1.on_clicked(on_p1)
btn_p2.on_clicked(on_p2)

# ── visibility toggles ───────────────────────────────────────────────────────
ax_chk = fig.add_axes([0.75, 0.60, 0.22, 0.12])
ax_chk.set_facecolor("#1a1a2e")
chk = CheckButtons(ax_chk, ["Show P1", "Show P2"], [True, True])
chk.set_check_props({"facecolor": "#0f3460"})
for txt in chk.labels:
    txt.set_color("white")

def on_toggle(label):
    idx = 0 if label == "Show P1" else 1
    visible[idx] = not visible[idx]
    save_view()
    redraw()

chk.on_clicked(on_toggle)

# ── zoom buttons ─────────────────────────────────────────────────────────────
ax_zi = fig.add_axes([0.75, 0.50, 0.10, 0.05])
ax_zo = fig.add_axes([0.87, 0.50, 0.10, 0.05])
ax_rs = fig.add_axes([0.75, 0.43, 0.22, 0.05])

btn_zi = Button(ax_zi, "Zoom +", color="#0f3460", hovercolor="#16213e")
btn_zo = Button(ax_zo, "Zoom -", color="#0f3460", hovercolor="#16213e")
btn_rs = Button(ax_rs, "Reset View", color="#163d60", hovercolor="#1f5080")
for b in [btn_zi, btn_zo, btn_rs]:
    b.label.set_color("white")

def zoom(factor):
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())
    cx = (xl[0] + xl[1]) / 2
    cy = (yl[0] + yl[1]) / 2
    xr = (xl[1] - xl[0]) * factor / 2
    yr = (yl[1] - yl[0]) * factor / 2
    view_xlim[0], view_xlim[1] = cx - xr, cx + xr
    view_ylim[0], view_ylim[1] = cy - yr, cy + yr
    redraw()

def on_zi(e): zoom(1 - ZOOM_FACTOR)
def on_zo(e): zoom(1 + ZOOM_FACTOR)
btn_zi.on_clicked(on_zi)
btn_zo.on_clicked(on_zo)

def on_reset(e):
    view_xlim[0] = view_xlim[1] = None
    view_ylim[0] = view_ylim[1] = None
    redraw()

btn_rs.on_clicked(on_reset)

# ── crosshair info ───────────────────────────────────────────────────────────
info_text = ax.text(0.01, 0.97, "", transform=ax.transAxes,
                    color="white", fontsize=8, va="top",
                    bbox=dict(facecolor="#0f3460", alpha=0.7, edgecolor="none"))
vline = ax.axvline(x=0, color="#888", lw=0.7, ls="--", visible=False)

def on_motion(event):
    if event.inaxes != ax:
        vline.set_visible(False)
        info_text.set_text("")
        fig.canvas.draw_idle()
        return
    x = event.xdata
    vline.set_xdata([x])
    vline.set_visible(True)
    msgs = []
    for i, prod in enumerate(selected):
        if prod and visible[i]:
            gts, mid = get_series(prod)
            idx = np.searchsorted(gts, x)
            idx = min(max(idx, 0), len(gts) - 1)
            msgs.append(f"P{i+1} {prod}: {mid[idx]:.2f}")
    info_text.set_text("  |  ".join(msgs))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_motion)

# ── keyboard controls ─────────────────────────────────────────────────────────
def on_key(event):
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())
    span_x = xl[1] - xl[0]
    span_y = yl[1] - yl[0]
    pan = span_x * PAN_FRACTION

    if event.key == "right":
        view_xlim[0] = xl[0] + pan; view_xlim[1] = xl[1] + pan
        view_ylim[0], view_ylim[1] = yl
    elif event.key == "left":
        view_xlim[0] = xl[0] - pan; view_xlim[1] = xl[1] - pan
        view_ylim[0], view_ylim[1] = yl
    elif event.key == "up":
        view_xlim[0], view_xlim[1] = xl
        view_ylim[0] = yl[0] + span_y * PAN_FRACTION
        view_ylim[1] = yl[1] + span_y * PAN_FRACTION
    elif event.key == "down":
        view_xlim[0], view_xlim[1] = xl
        view_ylim[0] = yl[0] - span_y * PAN_FRACTION
        view_ylim[1] = yl[1] - span_y * PAN_FRACTION
    elif event.key in ("+", "="):
        zoom(1 - ZOOM_FACTOR); return
    elif event.key == "-":
        zoom(1 + ZOOM_FACTOR); return
    elif event.key == "r":
        on_reset(None); return
    elif event.key == "1":
        cycle_product(0); return
    elif event.key == "2":
        cycle_product(1); return
    elif event.key == "shift+1":
        cycle_product(0, -1); return
    elif event.key == "shift+2":
        cycle_product(1, -1); return
    else:
        return
    redraw()

fig.canvas.mpl_connect("key_press_event", on_key)

# ── help text ────────────────────────────────────────────────────────────────
help_ax = fig.add_axes([0.75, 0.05, 0.22, 0.35])
help_ax.set_facecolor("#0f3460")
help_ax.axis("off")
help_lines = [
    "CONTROLS",
    "─────────────────",
    "← → : pan X",
    "↑ ↓ : pan Y",
    "+ / - : zoom",
    "r : reset view",
    "1 / 2 : next product",
    "⇧1 / ⇧2 : prev product",
    "Click P1/P2 buttons",
    "  to cycle products",
    "Checkboxes: show/hide",
    "Hover: crosshair values",
]
help_ax.text(0.05, 0.95, "\n".join(help_lines),
             transform=help_ax.transAxes, va="top",
             color="white", fontsize=8, family="monospace")

redraw()
plt.show()
