"""
PnL Plotter for hyperparameter tuning.

Visualizes hyperparameter search results from results_[product].csv.
Supports dimensionality reduction via PCA or direct selection of two parameters.

Usage:
    python tuning/pnl_plotter.py --product EMERALDS --mode selection --x m_slope --y slope_threshold
    python tuning/pnl_plotter.py --product TOMATOES --mode pca
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Necessary for environments without a display
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_TEMPLATE = PROJECT_ROOT / "tuning" / "results_{product}.csv"
DEFAULT_OUTPUT_TEMPLATE = PROJECT_ROOT / "tuning" / "pnl_visualization_{product}.png"

def load_data(csv_path: Path):
    """Load results CSV into a pandas DataFrame."""
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
    return pd.read_csv(csv_path)

def plot_3d(df, x_col, y_col, z_col, product, title, output_path):
    """Create a 3D scatter plot and a 2D heatmap."""
    fig = plt.figure(figsize=(16, 7))
    
    # 3D Scatter
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(df[x_col], df[y_col], df[z_col], c=df[z_col], cmap='viridis', s=50)
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_zlabel(z_col)
    ax1.set_title(f"3D Surface ({product}): {title}")
    plt.colorbar(sc, ax=ax1, label='PnL', pad=0.1)

    # 2D Heatmap / Interpolated
    ax2 = fig.add_subplot(122)
    
    # We use tricontourf if we have enough points, otherwise scatter
    if len(df) > 10:
        try:
            cntr = ax2.tricontourf(df[x_col], df[y_col], df[z_col], levels=20, cmap='viridis')
            plt.colorbar(cntr, ax=ax2, label='PnL')
            ax2.plot(df[x_col], df[y_col], 'ko', ms=1, alpha=0.3) # plot points
        except Exception as e:
            print(f"Warning: Could not create contour plot: {e}. Falling back to scatter.")
            sc2 = ax2.scatter(df[x_col], df[y_col], c=df[z_col], cmap='viridis', s=50)
            plt.colorbar(sc2, ax=ax2, label='PnL')
    else:
        sc2 = ax2.scatter(df[x_col], df[y_col], c=df[z_col], cmap='viridis', s=50)
        plt.colorbar(sc2, ax=ax2, label='PnL')

    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title(f"2D Projection ({product}): {title}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def run_pca_reduction(df, target_col):
    """Reduce hyperparameter dimensions to 2 using PCA."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Select features (all columns except target)
    features = [c for c in df.columns if c != target_col]
    if not features:
        print("Error: Not enough features for PCA.")
        sys.exit(1)
        
    x = df[features].values
    
    # Standardize
    x = StandardScaler().fit_transform(x)
    
    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df[target_col] = df[target_col].values
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return pca_df, 'PC1', 'PC2'

def main():
    parser = argparse.ArgumentParser(description="Visualize tuning results for any commodity")
    parser.add_argument(
        "--product", type=str, default="TOMATOES",
        help="Product name (default: TOMATOES)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to results_[product].csv",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save plot",
    )
    parser.add_argument(
        "--mode", type=str, choices=['selection', 'pca'], default='selection',
        help="Plotting mode",
    )
    parser.add_argument("--x", type=str, help="Parameter for X axis (selection mode)")
    parser.add_argument("--y", type=str, help="Parameter for Y axis (selection mode)")
    
    args = parser.parse_args()
    
    product = args.product
    input_path = Path(args.input) if args.input else Path(str(DEFAULT_RESULTS_TEMPLATE).format(product=product.lower()))
    output_path = Path(args.output) if args.output else Path(str(DEFAULT_OUTPUT_TEMPLATE).format(product=product.lower()))
    
    df = load_data(input_path)
    df = df.dropna()
    
    target = f"{product.lower()}_pnl"
    if target not in df.columns:
        # Fallback to any column ending in _pnl
        pnl_cols = [c for c in df.columns if '_pnl' in c]
        if pnl_cols:
            target = pnl_cols[0]
        else:
            print(f"Error: Could not find PnL column (expected {target} in columns {df.columns.tolist()})")
            sys.exit(1)
            
    if args.mode == 'pca':
        print(f"Running PCA dimensionality reduction for {product}...")
        plot_df, x_col, y_col = run_pca_reduction(df, target)
        title = "PCA Reduction"
    else:
        # Selection mode
        params = [c for c in df.columns if c != target]
        if not params:
            print(f"Error: No hyperparameter columns found for {product}.")
            sys.exit(1)
            
        x_col = args.x if args.x in params else params[0]
        y_col = args.y if args.y in params else (params[1] if len(params) > 1 else params[0])
        
        print(f"Plotting {x_col} vs {y_col} against {target} ({product})")
        plot_df = df
        title = f"{x_col} vs {y_col}"

    plot_3d(plot_df, x_col, y_col, target, product, title, output_path)

if __name__ == "__main__":
    main()
