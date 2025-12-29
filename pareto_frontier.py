"""
================================================================================
PARETO FRONTIER ANALYSIS - Efficiency Frontier in (RoT, Volatility) Space
================================================================================

This module finds and visualizes the Pareto efficiency frontier for portfolios
optimized across Risk of Technology (RoT) and Volatility.

The efficiency frontier consists of portfolios where you cannot improve one
metric without worsening the other (Pareto-optimal solutions).

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json


# =============================================================================
# PARETO FRONTIER CALCULATION
# =============================================================================

def find_pareto_frontier_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    minimize_x: bool = True,
    minimize_y: bool = True
) -> np.ndarray:
    """
    Find Pareto-optimal points in 2D space.
    
    A point is Pareto-optimal if no other point dominates it:
    - For minimization: dominates means (x2 <= x1 AND y2 < y1) OR (x2 < x1 AND y2 <= y1)
    - For maximization: dominates means (x2 >= x1 AND y2 > y1) OR (x2 > x1 AND y2 >= y1)
    
    Args:
        x_values: Array of x-axis values
        y_values: Array of y-axis values (same length as x_values)
        minimize_x: Whether to minimize x (True) or maximize x (False)
        minimize_y: Whether to minimize y (True) or maximize y (False)
        
    Returns:
        Boolean array indicating which points are on the Pareto frontier
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            
            # Check if point j dominates point i
            if minimize_x and minimize_y:
                # Minimization: j dominates i if (xj <= xi AND yj < yi) OR (xj < xi AND yj <= yi)
                if (x_values[j] <= x_values[i] and y_values[j] < y_values[i]) or \
                   (x_values[j] < x_values[i] and y_values[j] <= y_values[i]):
                    is_pareto[i] = False
                    break
            elif minimize_x and not minimize_y:
                # Minimize x, maximize y: j dominates i if (xj < xi AND yj >= yi) OR (xj <= xi AND yj > yi)
                if (x_values[j] < x_values[i] and y_values[j] >= y_values[i]) or \
                   (x_values[j] <= x_values[i] and y_values[j] > y_values[i]):
                    is_pareto[i] = False
                    break
            elif not minimize_x and minimize_y:
                # Maximize x, minimize y: j dominates i if (xj > xi AND yj <= yi) OR (xj >= xi AND yj < yi)
                if (x_values[j] > x_values[i] and y_values[j] <= y_values[i]) or \
                   (x_values[j] >= x_values[i] and y_values[j] < y_values[i]):
                    is_pareto[i] = False
                    break
            else:
                # Maximization: j dominates i if (xj >= xi AND yj > yi) OR (xj > xi AND yj >= yi)
                if (x_values[j] >= x_values[i] and y_values[j] > y_values[i]) or \
                   (x_values[j] > x_values[i] and y_values[j] >= y_values[i]):
                    is_pareto[i] = False
                    break
    
    return is_pareto


def find_pareto_frontier_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    minimize_x: bool = True,
    minimize_y: bool = True,
    minimize_z: bool = False  # Typically maximize return
) -> np.ndarray:
    """
    Find Pareto-optimal points in 3D space.
    
    A point is Pareto-optimal if no other point dominates it in all three dimensions.
    
    Dominance: Point j dominates point i if j is better or equal in all dimensions,
    and strictly better in at least one.
    
    Args:
        x_values: Array of x-axis values (e.g., RoT)
        y_values: Array of y-axis values (e.g., Volatility)
        z_values: Array of z-axis values (e.g., Return)
        minimize_x: Whether to minimize x (True) or maximize x (False)
        minimize_y: Whether to minimize y (True) or maximize y (False)
        minimize_z: Whether to minimize z (True) or maximize z (False)
        
    Returns:
        Boolean array indicating which points are on the Pareto frontier
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)
    
    # Convert to "lower is better" for all dimensions
    x_adj = x_values if minimize_x else -x_values
    y_adj = y_values if minimize_y else -y_values
    z_adj = z_values if minimize_z else -z_values
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            
            # Check if point j dominates point i
            # j dominates i if: j <= i in all dimensions AND j < i in at least one
            j_leq_i_x = x_adj[j] <= x_adj[i]
            j_leq_i_y = y_adj[j] <= y_adj[i]
            j_leq_i_z = z_adj[j] <= z_adj[i]
            
            j_lt_i_x = x_adj[j] < x_adj[i]
            j_lt_i_y = y_adj[j] < y_adj[i]
            j_lt_i_z = z_adj[j] < z_adj[i]
            
            # j dominates i if j is <= in all AND < in at least one
            if j_leq_i_x and j_leq_i_y and j_leq_i_z:
                if j_lt_i_x or j_lt_i_y or j_lt_i_z:
                    is_pareto[i] = False
                    break
    
    return is_pareto


def extract_pareto_frontier_3d_from_df(
    df: pd.DataFrame,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility',
    z_col: str = 'z_return',
    minimize_x: bool = True,
    minimize_y: bool = True,
    minimize_z: bool = False  # Maximize return
) -> pd.DataFrame:
    """
    Extract 3D Pareto-optimal portfolios from a results DataFrame.
    
    Args:
        df: DataFrame with portfolio metrics
        x_col: Column name for x-axis (default: 'x_RoT')
        y_col: Column name for y-axis (default: 'y_volatility')
        z_col: Column name for z-axis (default: 'z_return')
        minimize_x: Whether to minimize x (True) or maximize x (False)
        minimize_y: Whether to minimize y (True) or maximize y (False)
        minimize_z: Whether to minimize z (True) or maximize z (False)
        
    Returns:
        DataFrame containing only 3D Pareto-optimal portfolios
    """
    # Extract values
    x_values = df[x_col].values
    y_values = df[y_col].values
    z_values = df[z_col].values
    
    # Remove any NaN or inf values
    valid_mask = (np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values))
    df_valid = df[valid_mask].copy()
    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    z_valid = z_values[valid_mask]
    
    # Find 3D Pareto-optimal points
    is_pareto = find_pareto_frontier_3d(
        x_valid, y_valid, z_valid,
        minimize_x=minimize_x, minimize_y=minimize_y, minimize_z=minimize_z
    )
    
    # Extract Pareto-optimal portfolios
    pareto_df = df_valid[is_pareto].copy()
    
    # Sort by x_col for easier visualization
    pareto_df = pareto_df.sort_values(x_col)
    
    return pareto_df


def extract_pareto_frontier_from_df(
    df: pd.DataFrame,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility',
    minimize_x: bool = True,
    minimize_y: bool = True
) -> pd.DataFrame:
    """
    Extract Pareto-optimal portfolios from a results DataFrame.
    
    Args:
        df: DataFrame with portfolio metrics (must contain x_col and y_col)
        x_col: Column name for x-axis (default: 'x_RoT')
        y_col: Column name for y-axis (default: 'y_volatility')
        minimize_x: Whether to minimize x (True) or maximize x (False)
        minimize_y: Whether to minimize y (True) or maximize y (False)
        
    Returns:
        DataFrame containing only Pareto-optimal portfolios, sorted by x_col
    """
    # Extract values
    x_values = df[x_col].values
    y_values = df[y_col].values
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    df_valid = df[valid_mask].copy()
    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    
    # Find Pareto-optimal points
    is_pareto = find_pareto_frontier_2d(
        x_valid, y_valid, minimize_x=minimize_x, minimize_y=minimize_y
    )
    
    # Extract Pareto-optimal portfolios
    pareto_df = df_valid[is_pareto].copy()
    
    # Sort by x_col for easier visualization
    pareto_df = pareto_df.sort_values(x_col)
    
    return pareto_df


def load_frontier_from_csv(
    csv_path: str,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility',
    minimize_x: bool = True,
    minimize_y: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio results from CSV and extract Pareto frontier.
    
    Args:
        csv_path: Path to CSV file with portfolio results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        minimize_x: Whether to minimize x
        minimize_y: Whether to minimize y
        
    Returns:
        Tuple of (all_results_df, pareto_frontier_df)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract Pareto frontier
    pareto_df = extract_pareto_frontier_from_df(
        df, x_col=x_col, y_col=y_col,
        minimize_x=minimize_x, minimize_y=minimize_y
    )
    
    return df, pareto_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_efficiency_frontier(
    all_results_df: pd.DataFrame,
    pareto_frontier_df: pd.DataFrame,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility',
    x_label: str = 'Risk of Technology (RoT)',
    y_label: str = 'Price Volatility (CHF/MWh)',
    title: str = 'Efficiency Frontier: RoT vs Volatility',
    color_by: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    show_all_points: bool = True,
    alpha_all: float = 0.3,
    alpha_frontier: float = 0.9
) -> plt.Figure:
    """
    Plot the efficiency frontier in 2D space.
    
    Args:
        all_results_df: DataFrame with all portfolio results
        pareto_frontier_df: DataFrame with Pareto-optimal portfolios only
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        color_by: Optional column name to color points by (e.g., 'z_return')
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        show_all_points: Whether to show all points (gray) behind frontier
        alpha_all: Transparency for all points
        alpha_frontier: Transparency for frontier points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points (background)
    if show_all_points:
        ax.scatter(
            all_results_df[x_col],
            all_results_df[y_col],
            c='lightgray',
            alpha=alpha_all,
            s=30,
            label='All Portfolios',
            edgecolors='none'
        )
    
    # Plot Pareto frontier
    if color_by is not None and color_by in pareto_frontier_df.columns:
        scatter = ax.scatter(
            pareto_frontier_df[x_col],
            pareto_frontier_df[y_col],
            c=pareto_frontier_df[color_by],
            cmap='viridis',
            alpha=alpha_frontier,
            s=80,
            label='Efficiency Frontier',
            edgecolors='black',
            linewidths=1,
            zorder=10
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by, fontsize=11, fontweight='bold')
    else:
        ax.scatter(
            pareto_frontier_df[x_col],
            pareto_frontier_df[y_col],
            c='red',
            alpha=alpha_frontier,
            s=80,
            label='Efficiency Frontier',
            edgecolors='black',
            linewidths=1,
            zorder=10
        )
    
    # Connect frontier points with line (sorted by x_col)
    pareto_sorted = pareto_frontier_df.sort_values(x_col)
    ax.plot(
        pareto_sorted[x_col],
        pareto_sorted[y_col],
        'r--',
        linewidth=2,
        alpha=0.5,
        label='Frontier Line',
        zorder=5
    )
    
    # Labels and title
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Statistics text box
    stats_text = (
        f"Frontier Portfolios: {len(pareto_frontier_df)}\n"
        f"Total Portfolios: {len(all_results_df)}\n"
        f"RoT Range: [{pareto_frontier_df[x_col].min():.4f}, {pareto_frontier_df[x_col].max():.4f}]\n"
        f"Volatility Range: [{pareto_frontier_df[y_col].min():.1f}, {pareto_frontier_df[y_col].max():.1f}]"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Efficiency frontier plot saved to: {save_path}")
    
    return fig


# =============================================================================
# FRONTIER ANALYSIS
# =============================================================================

def analyze_frontier(
    pareto_frontier_df: pd.DataFrame,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility'
) -> Dict[str, Any]:
    """
    Analyze the Pareto frontier and return statistics.
    
    Args:
        pareto_frontier_df: DataFrame with Pareto-optimal portfolios
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        
    Returns:
        Dictionary with frontier statistics
    """
    stats = {
        'n_portfolios': len(pareto_frontier_df),
        'x_min': pareto_frontier_df[x_col].min(),
        'x_max': pareto_frontier_df[x_col].max(),
        'x_mean': pareto_frontier_df[x_col].mean(),
        'y_min': pareto_frontier_df[y_col].min(),
        'y_max': pareto_frontier_df[y_col].max(),
        'y_mean': pareto_frontier_df[y_col].mean(),
    }
    
    # Find extreme points
    stats['lowest_rot'] = pareto_frontier_df.loc[pareto_frontier_df[x_col].idxmin()]
    stats['lowest_volatility'] = pareto_frontier_df.loc[pareto_frontier_df[y_col].idxmin()]
    
    return stats


def print_frontier_summary(
    pareto_frontier_df: pd.DataFrame,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility'
):
    """
    Print a summary of the Pareto frontier.
    
    Args:
        pareto_frontier_df: DataFrame with Pareto-optimal portfolios
        x_col: Column name for x-axis
        y_col: Column name for y-axis
    """
    stats = analyze_frontier(pareto_frontier_df, x_col, y_col)
    
    print("="*60)
    print("PARETO FRONTIER SUMMARY")
    print("="*60)
    print(f"\nFrontier Portfolios: {stats['n_portfolios']}")
    print(f"\n{x_col} Range: [{stats['x_min']:.4f}, {stats['x_max']:.4f}] (mean: {stats['x_mean']:.4f})")
    print(f"{y_col} Range: [{stats['y_min']:.2f}, {stats['y_max']:.2f}] (mean: {stats['y_mean']:.2f})")
    
    print(f"\n📍 Lowest {x_col}:")
    print(f"   {x_col}: {stats['lowest_rot'][x_col]:.4f}")
    print(f"   {y_col}: {stats['lowest_rot'][y_col]:.2f}")
    
    print(f"\n📍 Lowest {y_col}:")
    print(f"   {x_col}: {stats['lowest_volatility'][x_col]:.4f}")
    print(f"   {y_col}: {stats['lowest_volatility'][y_col]:.2f}")
    print("="*60)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def find_and_plot_efficiency_frontier(
    csv_path: str,
    x_col: str = 'x_RoT',
    y_col: str = 'y_volatility',
    color_by: Optional[str] = 'z_return',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Complete workflow: load CSV, find frontier, plot, and return results.
    
    Args:
        csv_path: Path to CSV file with portfolio results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_by: Optional column name to color points by
        save_path: Optional path to save the figure
        verbose: Print summary statistics
        
    Returns:
        Tuple of (all_results_df, pareto_frontier_df, figure)
    """
    # Load and extract frontier
    all_df, pareto_df = load_frontier_from_csv(
        csv_path, x_col=x_col, y_col=y_col,
        minimize_x=True, minimize_y=True
    )
    
    if verbose:
        print_frontier_summary(pareto_df, x_col, y_col)
    
    # Plot
    fig = plot_efficiency_frontier(
        all_df, pareto_df,
        x_col=x_col, y_col=y_col,
        color_by=color_by,
        save_path=save_path
    )
    
    return all_df, pareto_df, fig


if __name__ == "__main__":
    # Example usage
    csv_path = "data/result_plots/portfolio_3d_results_5Runs_v2.csv"
    
    all_df, pareto_df, fig = find_and_plot_efficiency_frontier(
        csv_path=csv_path,
        color_by='z_return',
        save_path="data/result_plots/efficiency_frontier.png",
        verbose=True
    )
    
    plt.show()
    
    print(f"\n✅ Found {len(pareto_df)} Pareto-optimal portfolios out of {len(all_df)} total")

