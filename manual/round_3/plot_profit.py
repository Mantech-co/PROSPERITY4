import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 670
U = 920
DIFF = U - L

eq_b1 = L + DIFF/3
eq_b2 = L + 2*DIFF/3
eq_avg_b2 = eq_b2

def expected_profit(b1, b2, avg_b2):
    p_b1 = np.clip((b1 - L) / DIFF, 0, 1)
    profit_b1 = U - b1
    exp_p1 = p_b1 * profit_b1

    p_b2 = np.clip((b2 - np.maximum(b1, L)) / DIFF, 0, 1)
    gain_b2 = U - b2
    if b2 >= avg_b2:
        profit_b2 = gain_b2
    else:
        # Penalty factor < 1 when b2 < avg_b2
        penalty = ((U - avg_b2) / (U - b2))**3
        profit_b2 = gain_b2 * penalty
        
    exp_p2 = p_b2 * profit_b2
    return exp_p1 + exp_p2

v_expected_profit = np.vectorize(expected_profit)
eq_profit = expected_profit(eq_b1, eq_b2, eq_avg_b2)

def generate_surface_views():
    fig = plt.figure(figsize=(24, 18))
    
    b_range_coarse = np.linspace(L, U, 50)
    X, Y = np.meshgrid(b_range_coarse, b_range_coarse)
    
    # Combinations and labels
    # 1: b1, b2 (fixed avg_b2)
    # 2: b1, avg_b2 (fixed b2)
    # 3: b2, avg_b2 (fixed b1)
    configs = [
        (lambda x, y: v_expected_profit(x, y, eq_avg_b2), 'b1', 'b2', eq_b1, eq_b2, 'Vary b1, b2 (avg_b2 fixed)'),
        (lambda x, y: v_expected_profit(x, eq_b2, y), 'b1', 'avg_b2', eq_b1, eq_avg_b2, 'Vary b1, avg_b2 (b2 fixed)'),
        (lambda x, y: v_expected_profit(eq_b1, x, y), 'b2', 'avg_b2', eq_b2, eq_avg_b2, 'Vary b2, avg_b2 (b1 fixed)')
    ]
    
    views = [(30, -60), (0, -90), (90, 0)] # (elev, azim)
    
    plot_idx = 1
    for row_idx, (func, xlabel, ylabel, eq_x, eq_y, title) in enumerate(configs):
        Z = func(X, Y)
        for view_idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, plot_idx, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.scatter([eq_x], [eq_y], [eq_profit], color='red', s=100, label='Equilibrium', depthshade=False)
            
            ax.set_title(f"{title}\nView: elev={elev}, azim={azim}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('Profit')
            ax.view_init(elev=elev, azim=azim)
            if plot_idx == 1:
                ax.legend()
            plot_idx += 1

    plt.tight_layout()
    plt.savefig('profit_surface_views.png')
    print("Surface views saved to profit_surface_views.png")

if __name__ == "__main__":
    generate_surface_views()
