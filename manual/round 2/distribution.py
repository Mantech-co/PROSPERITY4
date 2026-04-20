"""
Mixture distribution on the strict domain [0, 100].

Construction rule
-----------------
Each component is defined by its natural (untruncated) PDF/CDF. To enforce
the domain [0, 100] we TRUNCATE each component at the boundaries (drop the
mass that falls outside) and then RESCALE the truncated curve so its area
equals the component's target weight. The shape inside [0, 100] is not
renormalised — only multiplied by a constant s_i.

Components and weights
----------------------
  c1. Skew-normal, mode at 26.03, fat LEFT tail   w = 0.30
  c2. Normal, mu = 37, sigma = 4.0                 w = 0.30
  c3. Normal, mu = 50, sigma = 3.5                 w = 0.10
  c4. Normal, mu = 70, sigma = 3.5                 w = 0.10
  c5. Normal, mu = 90, sigma = 3.5                 w = 0.10
  c6. Uniform noise on [0, 100]                    w = 0.10

Mixture CDF
-----------
Let F_i be the natural CDF of component i and define
    A_i = F_i(100) - F_i(0),   s_i = w_i / A_i
Then for 0 <= x <= 100:
    F(x) = sum_i  s_i * [ F_i(x) - F_i(0) ]
with F(x) = 0 for x < 0 and F(x) = 1 for x > 100.

Since s_i * A_i = w_i and sum_i w_i = 1, we automatically get F(100) = 1.
"""

import numpy as np
from scipy.stats import skewnorm, norm
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Component specification
# --------------------------------------------------------------------------
LO, HI = 0.0, 100.0

# Skew-normal: find loc so that the MODE sits exactly at 26.03.
# alpha < 0  => fat LEFT tail. Skew-normal mode has no closed form; we find
# the mode of the unshifted pdf numerically and then shift.
_tt = np.linspace(-200, 200, 40001)

_alpha1, _scale1 = -4.0, 12.0
_loc1 = 26.03 - _tt[np.argmax(skewnorm.pdf(_tt, _alpha1, loc=0.0, scale=_scale1))]

_alpha_r1, _scale_r1 = 4.0, 5.0
_loc_r1 = 0.0 - _tt[np.argmax(skewnorm.pdf(_tt, _alpha_r1, loc=0.0, scale=_scale_r1))]

_alpha_r2, _scale_r2 = 6.0, 8.0
_loc_r2 = 50.0 - _tt[np.argmax(skewnorm.pdf(_tt, _alpha_r2, loc=0.0, scale=_scale_r2))]
_loc_r3 = 41.0 - _tt[np.argmax(skewnorm.pdf(_tt, _alpha_r2, loc=0.0, scale=_scale_r2))]

COMPONENTS = [
    dict(key="c01", kind="norm",     mu=38.0,  sigma=4.0,    w=0.1057, label="Normal μ=36"),
    dict(key="c20", kind="norm",     mu=17.5,  sigma=3.0,    w=0.02, label="Normal μ=36"),
    dict(key="c02", kind="norm",     mu=30.0,  sigma=3.0,    w=0.0323, label="Normal μ=30 (narrow)"),
    dict(key="c03", kind="norm",     mu=30.0,  sigma=6.0,    w=0.1077, label="Normal μ=30 (wide)"),
    dict(key="c04", kind="skewnorm", alpha=_alpha_r1, loc=_loc_r1, scale=_scale_r1,
         w=0.0411, label="Skew-normal fat-right ~0"),
    dict(key="c05", kind="norm",     mu=80.0,  sigma=3.5,    w=0.0137, label="Normal μ=80 (minor)"),
    dict(key="c06", kind="norm",     mu=36.0,  sigma=4.0,    w=0.1377,  label="Normal μ=40"),
    dict(key="c07", kind="norm",     mu=50.0,  sigma=3.5,    w=0.0543,  label="Normal μ=50"),
    dict(key="c08", kind="skewnorm", alpha=_alpha_r2, loc=_loc_r2, scale=_scale_r2,
         w=0.0189, label="Skew-normal fatter-right ~50"),
    *[dict(key=f"c09_{m:02d}", kind="norm", mu=float(m), sigma=3.0,
           w=0.2269 / 10, label=f"Normal μ={m} (×10)") for m in range(10, 110, 10)],
    dict(key="c10", kind="uniform", a=LO, b=HI, w=0.2156, label="Uniform [0,100]"),
    dict(key="c12", kind="skewnorm", alpha=_alpha_r2, loc=_loc_r3, scale=_scale_r2,
         w=0.0274, label="Skew-normal fatter-right ~41"),
    dict(key="c11", kind="skewnorm", alpha=_alpha1, loc=_loc1, scale=_scale1,
         w=0.1064, label="Skew-normal fat-left ~26.03"),
]

_w_total = sum(p["w"] for p in COMPONENTS)
for p in COMPONENTS:
    p["w"] /= _w_total


# --------------------------------------------------------------------------
# 2. Natural component PDF / CDF
# --------------------------------------------------------------------------
def component_pdf(p, x):
    """Natural (untruncated) PDF of component p at x."""
    if p["kind"] == "skewnorm":
        return skewnorm.pdf(x, p["alpha"], loc=p["loc"], scale=p["scale"])
    if p["kind"] == "norm":
        return norm.pdf(x, loc=p["mu"], scale=p["sigma"])
    if p["kind"] == "uniform":
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        out[(x >= p["a"]) & (x <= p["b"])] = 1.0 / (p["b"] - p["a"])
        return out
    raise ValueError(p["kind"])


def component_cdf(p, x):
    """Natural (untruncated) CDF of component p at x."""
    if p["kind"] == "skewnorm":
        return skewnorm.cdf(x, p["alpha"], loc=p["loc"], scale=p["scale"])
    if p["kind"] == "norm":
        return norm.cdf(x, loc=p["mu"], scale=p["sigma"])
    if p["kind"] == "uniform":
        x = np.asarray(x, dtype=float)
        return np.clip((x - p["a"]) / (p["b"] - p["a"]), 0.0, 1.0)
    raise ValueError(p["kind"])


# --------------------------------------------------------------------------
# 3. Rescale factors s_i = w_i / A_i where A_i = F_i(HI) - F_i(LO)
# --------------------------------------------------------------------------
for p in COMPONENTS:
    p["cdf_lo"] = float(component_cdf(p, LO))
    A = float(component_cdf(p, HI) - p["cdf_lo"])
    p["A"] = A
    p["s"] = p["w"] / A


# --------------------------------------------------------------------------
# 4. Mixture PDF and CDF
# --------------------------------------------------------------------------
def mixture_pdf(x):
    """Mixture PDF on [0, 100]; 0 elsewhere."""
    x = np.asarray(x, dtype=float)
    inside = (x >= LO) & (x <= HI)
    total = np.zeros_like(x, dtype=float)
    for p in COMPONENTS:
        total = total + p["s"] * component_pdf(p, x)
    return np.where(inside, total, 0.0)


def mixture_cdf(x):
    """
    Mixture CDF:
        F(x) = 0                                  for x < 0
        F(x) = sum_i s_i * [F_i(x) - F_i(0)]      for 0 <= x <= 100
        F(x) = 1                                  for x > 100
    """
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, LO, HI)
    out = np.zeros_like(xc, dtype=float)
    for p in COMPONENTS:
        out = out + p["s"] * (component_cdf(p, xc) - p["cdf_lo"])
    out = np.where(x < LO, 0.0, out)
    out = np.where(x > HI, 1.0, out)
    return out


# --------------------------------------------------------------------------
# 5. Plot: two-panel figure (PDF + CDF) saved to a PDF file
# --------------------------------------------------------------------------
def make_figure():
    from matplotlib import cm
    x = np.linspace(LO, HI, 4001)
    pdf_vals = mixture_pdf(x)
    cdf_vals = mixture_cdf(x)
    colors = [cm.tab20(i / len(COMPONENTS)) for i in range(len(COMPONENTS))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # ---- PDF panel ----
    for p, c in zip(COMPONENTS, colors):
        y = np.where((x >= LO) & (x <= HI), p["s"] * component_pdf(p, x), 0.0)
        ax1.plot(x, y, color=c, lw=1.4, alpha=0.9, label=f"{p['label']} (w={p['w']:.4f})")
        ax1.fill_between(x, 0, y, color=c, alpha=0.10)
    ax1.plot(x, pdf_vals, color="black", lw=2.2, label="Mixture PDF")
    ax1.set_xlim(LO, HI); ax1.set_ylim(bottom=0)
    ax1.set_xlabel("x"); ax1.set_ylabel("density")
    ax1.set_title("Mixture PDF — components truncated, rescaled to target weights")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    # ---- CDF panel ----
    ax2.plot(x, cdf_vals, color="black", lw=2.0, label="Mixture CDF F(x)")
    for p, c in zip(COMPONENTS, colors):
        contrib = p["s"] * (component_cdf(p, x) - p["cdf_lo"])
        ax2.plot(x, contrib, color=c, lw=1.0, alpha=0.7, ls="--",
                 label=f"{p['key']} (w={p['w']:.4f})")
    ax2.set_xlim(LO, HI); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("x"); ax2.set_ylabel("F(x)")
    ax2.set_title("Mixture CDF (solid) and component contributions (dashed)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 6. Run as a script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Print scale factors
    print(f"{'comp':6s} {'kind':10s} {'A_i':>10s} {'w_i':>6s} {'s_i':>10s}")
    for p in COMPONENTS:
        print(f"{p['key']:6s} {p['kind']:10s} {p['A']:10.6f} {p['w']:6.2f} {p['s']:10.6f}")

    # Sanity checks
    xgrid = np.linspace(LO, HI, 4001)
    area = np.trapezoid(mixture_pdf(xgrid), xgrid)
    print(f"\nPDF area on [0,100] = {area:.6f}   (should be ~1)")
    print(f"F(0)   = {mixture_cdf(np.array([LO]))[0]:.6f}   (should be 0)")
    print(f"F(100) = {mixture_cdf(np.array([HI]))[0]:.6f}   (should be 1)")

    make_figure()