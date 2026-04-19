"""
Build a probability density function on x ∈ [0, 1] from a mixture of
components and visualise it.

Components:
    15595 at 0                                               (delta)
    269   at 0                                               (delta)
    396   at 0.36                                            (delta)
    99    skew-normal around 0.36                            (continuous)
    168   at 0.3                                             (delta)
    43    skew-normal around 0.3                             (continuous)
    562   at 0.3                                             (delta)
    141   skew-normal around 0.3                             (continuous)
    89    Normal μ=0.8, 2σ=0.3 (so σ=0.15)                   (continuous)
    123   Normal μ=0.5, 2σ=0.4 (so σ=0.20)                   (continuous)
    1733  Uniform noise on [0, 1]                            (continuous)
    1480  Nice-number grid (0.05 spacing, 2:1 peak ratio)    (discrete)

Skew-normal convention:
    right-skewed (positive α); scale chosen so that
    pdf(peak + 0.2) / pdf(peak) = 0.01   →  "decays 99% at 0.2 from peak"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
from scipy.optimize import brentq

SEED = 42
rng  = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Skew-normal parameterisation
# ---------------------------------------------------------------------------
def fit_skewnorm(peak, alpha=4.0, decay_offset=0.2, decay_ratio=0.01):
    """
    Return (loc, scale, alpha) for a skew-normal whose MODE is at `peak` and
    whose pdf drops to `decay_ratio` of the peak value at peak+decay_offset.
    """
    xs    = np.linspace(-3, 5, 20001)
    p_std = skewnorm.pdf(xs, alpha)
    mode_std = xs[np.argmax(p_std)]
    peak_pdf = p_std.max()

    def residual(scale):
        x_tail = mode_std + decay_offset / scale
        return skewnorm.pdf(x_tail, alpha) / peak_pdf - decay_ratio

    scale = brentq(residual, 1e-4, 2.0)
    loc   = peak - mode_std * scale
    return loc, scale, alpha


# ---------------------------------------------------------------------------
# Component specification
# ---------------------------------------------------------------------------
components = [
    dict(kind='delta',   n=269,   loc=0.0,   color='#17becf', label='Δ @ 0     (n=269)'),
    dict(kind='delta',   n=396,   loc=0.36,  color='#ff7f0e', label='Δ @ 0.36  (n=396)'),
    dict(kind='skew',    n=99,    peak=0.36, color='#ffbb78', label='SkewN ~0.36  (n=99)'),
    dict(kind='delta',   n=168,   loc=0.3,   color='#2ca02c', label='Δ @ 0.3   (n=168)'),
    dict(kind='skew',    n=43,    peak=0.3,  color='#98df8a', label='SkewN ~0.3   (n=43)'),
    dict(kind='delta',   n=562,   loc=0.3,   color='#d62728', label='Δ @ 0.3   (n=562)'),
    dict(kind='skew',    n=141,   peak=0.3,  color='#ff9896', label='SkewN ~0.3   (n=141)'),
    dict(kind='normal',  n=89,    mean=0.8,  two_sigma=0.3, color='#9467bd',
         label='Normal μ=0.8 2σ=0.3 (n=89)'),
    dict(kind='normal',  n=123,   mean=0.5,  two_sigma=0.4, color='#8c564b',
         label='Normal μ=0.5 2σ=0.4 (n=123)'),
    dict(kind='uniform', n=1733,  color='#7f7f7f', label='Uniform noise (n=1733)'),
    dict(kind='nice',    n=1480,  color='#bcbd22', label='Nice grid 2:1 (n=1480)'),
]
N_TOTAL = sum(c['n'] for c in components)


# ---------------------------------------------------------------------------
# generate_weights — importable, self-contained, stochastic per seed
# ---------------------------------------------------------------------------
def generate_weights(seed=42, bin_w=0.001):
    """
    Draw fresh samples with `seed` and return a histogram as {x: count} dict.
    x values are bin centres in [0, 1] at `bin_w` spacing.
    """
    rng_fn = np.random.default_rng(seed)
    comps = [
        dict(kind='delta',   n=269,  loc=0.0),
        dict(kind='delta',   n=396,  loc=0.36),
        dict(kind='skew',    n=99,   peak=0.36),
        dict(kind='delta',   n=168,  loc=0.3),
        dict(kind='skew',    n=43,   peak=0.3),
        dict(kind='delta',   n=562,  loc=0.3),
        dict(kind='skew',    n=141,  peak=0.3),
        dict(kind='normal',  n=89,   mean=0.8,  two_sigma=0.3),
        dict(kind='normal',  n=123,  mean=0.5,  two_sigma=0.4),
        dict(kind='uniform', n=1733),
        dict(kind='nice',    n=1480),
    ]
    samples_list = []
    for c in comps:
        k, n = c['kind'], c['n']
        if k == 'delta':
            samples_list.append(np.full(n, c['loc']))
        elif k == 'skew':
            loc, scale, alpha = fit_skewnorm(c['peak'])
            samples_list.append(skewnorm.rvs(alpha, loc=loc, scale=scale,
                                              size=n, random_state=rng_fn))
        elif k == 'normal':
            sigma = c['two_sigma'] / 2.0
            samples_list.append(rng_fn.normal(c['mean'], sigma, n))
        elif k == 'uniform':
            samples_list.append(rng_fn.uniform(0.0, 1.0, n))
        elif k == 'nice':
            vals_01  = np.round(np.arange(0.00, 1.01, 0.1), 2)
            vals_005 = np.round(np.arange(0.05, 1.00, 0.1), 2)
            vals = np.concatenate([vals_01, vals_005])
            w    = np.concatenate([np.full(len(vals_01),  2.0),
                                   np.full(len(vals_005), 1.0)])
            samples_list.append(rng_fn.choice(vals, size=n, p=w / w.sum()))

    all_s = np.clip(np.concatenate(samples_list), 0.0, 1.0)
    dec = max(3, int(round(-np.log10(bin_w))))
    centres = np.round(np.arange(0.0, 1.0 + 1e-9, bin_w), dec)
    edges = np.append(centres - bin_w / 2, centres[-1] + bin_w / 2)
    counts, _ = np.histogram(all_s, bins=edges)
    return {round(float(cx), dec): float(cnt)
            for cx, cnt in zip(centres, counts) if cnt > 0}


# ===========================================================================
# Main — sample drawing, plots, CSV
# ===========================================================================
if __name__ == "__main__":

    # --- Draw samples ---
    for c in components:
        k, n = c['kind'], c['n']
        if k == 'delta':
            c['samples'] = np.full(n, c['loc'])

        elif k == 'skew':
            loc, scale, alpha = fit_skewnorm(c['peak'])
            c['skew_params'] = (loc, scale, alpha)
            c['samples'] = skewnorm.rvs(alpha, loc=loc, scale=scale,
                                        size=n, random_state=rng)

        elif k == 'normal':
            sigma = c['two_sigma'] / 2.0
            c['sigma']   = sigma
            c['samples'] = rng.normal(c['mean'], sigma, n)

        elif k == 'uniform':
            c['samples'] = rng.uniform(0.0, 1.0, n)

        elif k == 'nice':
            vals_01  = np.round(np.arange(0.00, 1.01, 0.1), 2)
            vals_005 = np.round(np.arange(0.05, 1.00, 0.1), 2)
            vals = np.concatenate([vals_01, vals_005])
            w    = np.concatenate([np.full(len(vals_01),  2.0),
                                   np.full(len(vals_005), 1.0)])
            c['nice_vals']    = vals
            c['nice_weights'] = w
            c['samples']      = rng.choice(vals, size=n, p=w / w.sum())

    all_samples = np.clip(np.concatenate([c['samples'] for c in components]), 0.0, 1.0)

    print(f'Total samples: {N_TOTAL}')
    for c in components:
        s = c['samples']
        print(f"  {c['label']:<36s}  min={s.min():.3f}  max={s.max():.3f}")


    # =======================================================================
    # PLOT 1 – individual component histograms
    # =======================================================================
    ncols = 3
    nrows = int(np.ceil(len(components) / ncols))
    bins  = np.linspace(0, 1, 1001)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.7 * nrows))
    axes = axes.flatten()
    for ax, c in zip(axes, components):
        ax.hist(np.clip(c['samples'], 0, 1), bins=bins,
                color=c['color'], edgecolor='black', linewidth=0.2)
        ax.set_title(c['label'], fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(components):]:
        ax.axis('off')
    fig.suptitle('1. Individual component distributions',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('./01_individual.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)


    # =======================================================================
    # PLOT 2 – final PDF (analytical: continuous mixture + point masses)
    # =======================================================================
    x = np.linspace(0, 1, 2001)
    pdf_cont = np.zeros_like(x)
    point_mass = {}

    for c in components:
        w = c['n'] / N_TOTAL
        k = c['kind']
        if k == 'delta':
            point_mass[c['loc']] = point_mass.get(c['loc'], 0.0) + w
        elif k == 'skew':
            loc, scale, alpha = c['skew_params']
            pdf_cont += w * skewnorm.pdf(x, alpha, loc=loc, scale=scale)
        elif k == 'normal':
            pdf_cont += w * norm.pdf(x, c['mean'], c['sigma'])
        elif k == 'uniform':
            pdf_cont += w * 1.0
        elif k == 'nice':
            probs = c['nice_weights'] / c['nice_weights'].sum()
            for v, p in zip(c['nice_vals'], probs):
                point_mass[round(v, 2)] = point_mass.get(round(v, 2), 0.0) + w * p

    locs_sorted = sorted(point_mass.keys())
    mass_sorted = [point_mass[l] for l in locs_sorted]

    fig, (ax_cont, ax_full) = plt.subplots(2, 1, figsize=(13, 9))

    ax_cont.plot(x, pdf_cont, color='steelblue', lw=2)
    ax_cont.fill_between(x, pdf_cont, alpha=0.3, color='steelblue')
    ax_cont.set_title('2a. Continuous part of the PDF  (skew-normals + normals + uniform)',
                      fontsize=12, fontweight='bold')
    ax_cont.set_xlabel('x')
    ax_cont.set_ylabel('density f(x)')
    ax_cont.set_xlim(0, 1)
    ax_cont.grid(True, alpha=0.3)

    for cx, text in [(0.3, 'skew-normals'), (0.36, 'skew @0.36'),
                     (0.5, 'Normal 0.5'), (0.8, 'Normal 0.8')]:
        ax_cont.axvline(cx, color='grey', ls=':', alpha=0.4)

    ax_full.plot(x, pdf_cont, color='steelblue', lw=1.3, label='continuous density')
    ax_full.fill_between(x, pdf_cont, alpha=0.2, color='steelblue')
    ax_full.set_xlabel('x')
    ax_full.set_ylabel('density f(x)', color='steelblue')
    ax_full.tick_params(axis='y', labelcolor='steelblue')
    ax_full.set_xlim(0, 1)
    ax_full.grid(True, alpha=0.3)

    ax_mass = ax_full.twinx()
    ax_mass.stem(locs_sorted, mass_sorted,
                 linefmt='r-', markerfmt='ro', basefmt=' ')
    ax_mass.set_ylabel('point-mass probability', color='r')
    ax_mass.tick_params(axis='y', labelcolor='r')
    ax_mass.set_ylim(0, max(mass_sorted) * 1.15)

    for l, m in zip(locs_sorted, mass_sorted):
        if m > 0.01:
            ax_mass.annotate(f'{m:.1%}\n@{l}', xy=(l, m),
                             xytext=(l + 0.012, m),
                             fontsize=8, color='red', va='center')

    ax_full.set_title('2b. Full PDF = continuous density (blue) + discrete masses (red stems)',
                      fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig('./02_final_pdf.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)


    # =======================================================================
    # PLOT 3 – final histogram (linear + log)
    # =======================================================================
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(13, 8))

    ax_lin.hist(all_samples, bins=bins,
                color='steelblue', edgecolor='black', lw=0.3)
    ax_lin.set_title(f'3a. Final histogram – linear scale  (N={N_TOTAL})',
                     fontsize=12, fontweight='bold')
    ax_lin.set_xlabel('x')
    ax_lin.set_ylabel('count')
    ax_lin.set_xlim(0, 1)
    ax_lin.grid(True, alpha=0.3)

    ax_log.hist(all_samples, bins=bins,
                color='steelblue', edgecolor='black', lw=0.3)
    ax_log.set_yscale('log')
    ax_log.set_title(f'3b. Final histogram – log scale  (N={N_TOTAL})',
                     fontsize=12, fontweight='bold')
    ax_log.set_xlabel('x')
    ax_log.set_ylabel('count (log)')
    ax_log.set_xlim(0, 1)
    ax_log.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig('./03_histogram.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)


    # =======================================================================
    # PLOT 4 – final empirical CDF
    # =======================================================================
    fig, ax = plt.subplots(figsize=(13, 6))
    sorted_s = np.sort(all_samples)
    y_cdf    = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.step(sorted_s, y_cdf, where='post', color='darkgreen', lw=1.3)
    ax.set_title(f'4. Final empirical CDF  (N={N_TOTAL})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    for l, m in point_mass.items():
        if m > 0.02:
            idx  = np.searchsorted(sorted_s, l, side='right') - 1
            yval = y_cdf[idx] if 0 <= idx < len(sorted_s) else 0
            ax.annotate(f'jump @ x={l}\n(Δ={m:.1%})',
                        xy=(l, yval),
                        xytext=(l + 0.03, max(yval - 0.1, 0.05)),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                        fontsize=9, color='red')

    fig.tight_layout()
    fig.savefig('./04_cdf.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)

    print('\nSaved:')
    print('  ./01_individual.png')
    print('  ./02_final_pdf.png')
    print('  ./03_histogram.png')
    print('  ./04_cdf.png')


    # =======================================================================
    # DISCRETISED DISTRIBUTION — histogram-based, bin width = 0.001
    # =======================================================================
    BIN_W   = 0.001
    centres = np.round(np.arange(0.0, 1.0 + 1e-9, BIN_W), 3)
    edges   = np.append(centres - BIN_W / 2, centres[-1] + BIN_W / 2)
    counts, _ = np.histogram(all_samples, bins=edges)
    weights = counts.astype(float)

    print('\n' + '=' * 44)
    print(f'{"x":>7s}   {"weight":>12s}')
    print('-' * 44)
    for x_val, w_val in zip(centres, weights):
        print(f'{x_val:7.3f}   {w_val:12.3f}')
    print('-' * 44)
    print(f'{"Σ":>7s}   {weights.sum():12.3f}   (expected {N_TOTAL})')

    csv_path = './distribution_xw.csv'
    with open(csv_path, 'w') as f:
        f.write('x,weight\n')
        for x_val, w_val in zip(centres, weights):
            f.write(f'{x_val:.3f},{w_val:.6f}\n')
    print(f'\nCSV saved: {csv_path}')
