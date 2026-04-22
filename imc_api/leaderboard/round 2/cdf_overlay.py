import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../../manual/round 2'))
from distribution import mixture_cdf

data = [{"speed":0,"players":453},{"speed":1.00,"players":125},{"speed":2.00,"players":84},{"speed":3.00,"players":52},{"speed":4.00,"players":27},{"speed":5.00,"players":129},{"speed":6.00,"players":27},{"speed":7.00,"players":31},{"speed":8.00,"players":30},{"speed":9.00,"players":12},{"speed":10.0,"players":249},{"speed":11.00,"players":53},{"speed":12.00,"players":29},{"speed":13.00,"players":17},{"speed":14.00,"players":5},{"speed":15.00,"players":137},{"speed":16.00,"players":35},{"speed":17.00,"players":22},{"speed":18.00,"players":23},{"speed":19.00,"players":13},{"speed":20.0,"players":281},{"speed":21.00,"players":78},{"speed":22.00,"players":44},{"speed":23.00,"players":33},{"speed":24.00,"players":22},{"speed":25.00,"players":172},{"speed":26.00,"players":63},{"speed":27.00,"players":54},{"speed":28.00,"players":22},{"speed":29.00,"players":14},{"speed":30.0,"players":242},{"speed":31.00,"players":57},{"speed":32.00,"players":42},{"speed":33.00,"players":68},{"speed":34.00,"players":99},{"speed":34.0100,"players":1},{"speed":35.00,"players":155},{"speed":36.00,"players":184},{"speed":37.00,"players":118},{"speed":38.00,"players":75},{"speed":39.00,"players":37},{"speed":40.0,"players":227},{"speed":41.00,"players":139},{"speed":42.00,"players":119},{"speed":42.100,"players":1},{"speed":43.00,"players":89},{"speed":44.00,"players":40},{"speed":45.00,"players":100},{"speed":46.00,"players":69},{"speed":47.00,"players":42},{"speed":48.00,"players":20},{"speed":49.00,"players":14},{"speed":50.0,"players":86},{"speed":51.00,"players":62},{"speed":52.00,"players":56},{"speed":53.00,"players":40},{"speed":54.00,"players":22},{"speed":55.00,"players":33},{"speed":56.00,"players":21},{"speed":57.00,"players":17},{"speed":58.00,"players":17},{"speed":59.00,"players":4},{"speed":60.0,"players":30},{"speed":61.00,"players":14},{"speed":61.100,"players":1},{"speed":62.00,"players":3},{"speed":63.00,"players":7},{"speed":64.00,"players":8},{"speed":65.00,"players":9},{"speed":66.00,"players":5},{"speed":67.00,"players":4},{"speed":68.00,"players":3},{"speed":69.00,"players":3},{"speed":70.0,"players":8},{"speed":71.00,"players":9},{"speed":72.00,"players":2},{"speed":74.00,"players":1},{"speed":75.00,"players":1},{"speed":76.00,"players":1},{"speed":77.00,"players":2},{"speed":78.00,"players":1},{"speed":79.00,"players":1},{"speed":80.0,"players":4},{"speed":81.00,"players":2},{"speed":85.00,"players":1},{"speed":90.0,"players":1},{"speed":91.00,"players":1},{"speed":100,"players":2}]

data_sorted = sorted(data, key=lambda x: x['speed'])
spd = np.array([d['speed'] for d in data_sorted])
cum = np.cumsum([d['players'] for d in data_sorted])
empirical_cdf = cum / cum[-1]

x_smooth = np.linspace(0, 100, 2000)
model_cdf = mixture_cdf(x_smooth)

fig, ax = plt.subplots(figsize=(12, 6))
ax.step(spd, empirical_cdf, where='post', color='steelblue', linewidth=2, label='Empirical CDF (leaderboard data)')
ax.plot(x_smooth, model_cdf, color='darkorange', linewidth=2, linestyle='--', label='Model CDF (distribution.py)')
ax.fill_between(spd, empirical_cdf, step='post', alpha=0.1, color='steelblue')

ax.set_xlabel('Speed', fontsize=13)
ax.set_ylabel('Cumulative Fraction of Players', fontsize=13)
ax.set_title('CDF Overlay — Empirical vs Model — Round 2', fontsize=15, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.grid(alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
