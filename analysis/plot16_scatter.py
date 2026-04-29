import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/media/manukrishnan/Mk/prosperity_4/analysis_output/deep/"

DARK = "#1a1a2e"; MID = "#16213e"; ACCENT = "#f39c12"
plt.rcParams.update({"figure.facecolor": DARK, "axes.facecolor": MID,
                     "text.color": "white", "axes.labelcolor": "#aaa",
                     "xtick.color": "#aaa", "ytick.color": "#aaa",
                     "axes.edgecolor": "#444", "font.size": 8})

GROUPS = {
    "Galaxy Sounds":  ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES",
                       "GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS",
                       "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "Sleep Pods":     ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER",
                       "SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "Microchips":     ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE",
                       "MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "Pebbles":        ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "Robots":         ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES",
                       "ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UV Visors":      ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE",
                       "UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "Translators":    ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK",
                       "TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST",
                       "TRANSLATOR_VOID_BLUE"],
    "Panels":         ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "Oxygen Shakes":  ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH",
                       "OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "Snack Packs":    ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
                       "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}
GCOLORS = dict(zip(GROUPS, plt.cm.tab10(np.linspace(0, 1, 10))))
PROD_COLORS = plt.cm.Set1(np.linspace(0, 1, 5))

prices = pd.concat([
    pd.read_csv(f"/media/manukrishnan/Mk/prosperity_4/data/prices_round_5_day_{d}.csv", sep=";")
    for d in [2, 3, 4]], ignore_index=True)
prices["gts"] = (prices["day"] - 2) * 1_000_000 + prices["timestamp"]
prices.sort_values(["product", "gts"], inplace=True)

pivot = prices.pivot_table(index="gts", columns="product", values="mid_price")
pivot.ffill(inplace=True)

fig, axes = plt.subplots(5, 2, figsize=(20, 28), facecolor=DARK)
axes = axes.flatten()

for ax, (gname, gprods) in zip(axes, GROUPS.items()):
    ax.set_facecolor(MID); ax.spines[:].set_color("#333")
    gc = GCOLORS[gname]
    avail = [p for p in gprods if p in pivot.columns]

    for k, prod in enumerate(avail):
        ret = pivot[prod].pct_change().dropna().values
        # x-axis: return value, y-axis: observation index (spread vertically per product)
        y_jitter = k + np.random.uniform(-0.3, 0.3, size=len(ret))
        c = PROD_COLORS[k % len(PROD_COLORS)]
        ax.scatter(ret, y_jitter, s=0.8, alpha=0.15, color=c,
                   label=prod.split("_")[-1], rasterized=True)

    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlim(-0.005, 0.005)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels([p.split("_")[-1] for p in avail], fontsize=7, color="white")
    ax.set_title(gname, color="white", fontsize=10)
    ax.set_xlabel("Return", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    leg = ax.legend(fontsize=6.5, markerscale=6, ncol=2,
                    facecolor="#0f3460", edgecolor="#444", labelcolor="white")

fig.suptitle("Returns Distribution — Scatter per Group\n(each row = one product, x = return value)",
             color="white", fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig(OUT + "16_returns_scatter_per_group.png", dpi=120,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("saved 16_returns_scatter_per_group.png")
