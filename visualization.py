"""
================================================================================
VISUALIZATION - Plotting Functions for Optimization Results
================================================================================

This module provides plotting functions for visualizing the results of the
Swiss Energy Storage Optimization project.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette (distinctive, not generic AI slop)
COLORS = {
    'primary': '#1a365d',      # Deep navy
    'secondary': '#c53030',    # Crimson
    'accent': '#d69e2e',       # Gold
    'success': '#276749',      # Forest green
    'warning': '#dd6b20',      # Burnt orange
    'info': '#2b6cb0',         # Ocean blue
    'light': '#f7fafc',        # Off-white
    'dark': '#1a202c',         # Near-black
    
    # PPU type colors
    'solar': '#ecc94b',        # Sunflower
    'wind': '#4299e1',         # Sky blue
    'hydro': '#38b2ac',        # Teal
    'thermal': '#fc8181',      # Salmon
    'storage': '#9f7aea',      # Lavender
    'h2': '#68d391',           # Mint
    'bio': '#f6ad55',          # Peach
}


# =============================================================================
# GA EVOLUTION PLOTS
# =============================================================================

def plot_fitness_evolution(
    best_history: List[float],
    mean_history: Optional[List[float]] = None,
    title: str = "GA Fitness Evolution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot the evolution of fitness over generations.
    
    Args:
        best_history: List of best fitness per generation
        mean_history: Optional list of mean fitness per generation
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    generations = np.arange(len(best_history))
    
    # Best fitness line
    ax.plot(generations, best_history, 
            color=COLORS['primary'], linewidth=2.5, 
            label='Best Fitness', marker='o', markersize=4)
    
    # Mean fitness line (if provided)
    if mean_history:
        ax.plot(generations, mean_history,
                color=COLORS['secondary'], linewidth=1.5, 
                linestyle='--', alpha=0.7,
                label='Mean Fitness')
    
    # Fill between best and mean
    if mean_history:
        ax.fill_between(generations, best_history, mean_history,
                       color=COLORS['info'], alpha=0.1)
    
    # Highlight improvements
    improvements = []
    for i in range(1, len(best_history)):
        if best_history[i] < best_history[i-1]:
            improvements.append(i)
    
    if improvements:
        ax.scatter([generations[i] for i in improvements],
                  [best_history[i] for i in improvements],
                  color=COLORS['success'], s=100, zorder=5,
                  marker='v', label='Improvement')
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness (CHF)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# PORTFOLIO VISUALIZATION
# =============================================================================

def plot_portfolio_composition(
    portfolio_counts: Dict[str, int],
    title: str = "Portfolio Composition",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot the composition of a PPU portfolio.
    
    Args:
        portfolio_counts: Dictionary of PPU name to count
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Filter non-zero counts
    active = {k: v for k, v in portfolio_counts.items() if v > 0}
    
    if not active:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No PPUs in portfolio', 
                ha='center', va='center', fontsize=16)
        return fig
    
    # Categorize PPUs
    categories = {
        'Solar': ['PV'],
        'Wind': ['WD_ON', 'WD_OFF'],
        'Hydro': ['HYD_R', 'HYD_S', 'PHS'],
        'Thermal': ['THERM', 'THERM_CH4', 'BIO_OIL_ICE', 'PALM_ICE'],
        'Hydrogen': ['H2_G', 'H2_GL', 'H2_L', 'H2P_G', 'H2P_L'],
        'Synthesis': ['SYN_FT', 'SYN_METH', 'SYN_CRACK', 'NH3_FULL', 'NH3_P'],
        'Bio': ['BIO_WOOD', 'CH4_BIO', 'IMP_BIOG'],
        'Other': ['SOL_SALT', 'SOL_STEAM', 'SOL_SALT_STORE'],
    }
    
    cat_colors = {
        'Solar': COLORS['solar'],
        'Wind': COLORS['wind'],
        'Hydro': COLORS['hydro'],
        'Thermal': COLORS['thermal'],
        'Hydrogen': COLORS['h2'],
        'Synthesis': COLORS['storage'],
        'Bio': COLORS['bio'],
        'Other': COLORS['accent'],
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    ppus = list(active.keys())
    counts = list(active.values())
    
    # Assign colors based on category
    bar_colors = []
    for ppu in ppus:
        for cat, ppu_list in categories.items():
            if ppu in ppu_list:
                bar_colors.append(cat_colors[cat])
                break
        else:
            bar_colors.append(COLORS['info'])
    
    bars = ax1.barh(ppus, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax1.set_title('PPU Unit Counts', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Pie chart by category
    cat_totals = {}
    for ppu, count in active.items():
        for cat, ppu_list in categories.items():
            if ppu in ppu_list:
                cat_totals[cat] = cat_totals.get(cat, 0) + count
                break
        else:
            cat_totals['Other'] = cat_totals.get('Other', 0) + count
    
    cat_names = list(cat_totals.keys())
    cat_values = list(cat_totals.values())
    pie_colors = [cat_colors.get(c, COLORS['info']) for c in cat_names]
    
    wedges, texts, autotexts = ax2.pie(
        cat_values, labels=cat_names, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(linewidth=2, edgecolor='white'),
        textprops=dict(fontsize=11, fontweight='bold'),
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
    
    ax2.set_title('Category Distribution', fontsize=13, fontweight='bold')
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# DISPATCH VISUALIZATION
# =============================================================================

def plot_dispatch_scenario(
    overflow_series: np.ndarray,
    demand_series: Optional[np.ndarray] = None,
    renewable_series: Optional[np.ndarray] = None,
    spot_series: Optional[np.ndarray] = None,
    title: str = "Dispatch Scenario",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Plot dispatch scenario results.
    
    Args:
        overflow_series: Deficit/surplus time series (MW)
        demand_series: Optional demand time series
        renewable_series: Optional renewable production series
        spot_series: Optional spot price series
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_plots = 2
    if spot_series is not None:
        n_plots = 3
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    hours = np.arange(len(overflow_series)) * 0.25  # 15-min intervals
    
    # Plot 1: Demand vs Renewable Production
    ax1 = axes[0]
    if demand_series is not None:
        ax1.fill_between(hours, demand_series, 
                        color=COLORS['primary'], alpha=0.3, label='Demand')
        ax1.plot(hours, demand_series, 
                color=COLORS['primary'], linewidth=1.5)
    
    if renewable_series is not None:
        ax1.fill_between(hours, renewable_series,
                        color=COLORS['success'], alpha=0.3, label='Renewable')
        ax1.plot(hours, renewable_series,
                color=COLORS['success'], linewidth=1.5)
    
    ax1.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax1.set_title('Energy Balance', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Deficit/Surplus
    ax2 = axes[1]
    
    # Color based on sign
    positive_mask = overflow_series > 0
    negative_mask = overflow_series < 0
    
    ax2.fill_between(hours, 0, overflow_series,
                    where=positive_mask,
                    color=COLORS['secondary'], alpha=0.4, label='Deficit')
    ax2.fill_between(hours, 0, overflow_series,
                    where=negative_mask,
                    color=COLORS['success'], alpha=0.4, label='Surplus')
    
    ax2.plot(hours, overflow_series, color=COLORS['dark'], linewidth=0.8)
    ax2.axhline(y=0, color=COLORS['dark'], linewidth=1, linestyle='-')
    
    ax2.set_ylabel('Balance (MW)', fontsize=11, fontweight='bold')
    ax2.set_title('Deficit (-) / Surplus (+)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spot Prices (if provided)
    if spot_series is not None:
        ax3 = axes[2]
        
        # Color gradient based on price
        norm = plt.Normalize(spot_series.min(), spot_series.max())
        colors = plt.cm.RdYlGn_r(norm(spot_series))
        
        for i in range(len(hours) - 1):
            ax3.fill_between(hours[i:i+2], 0, [spot_series[i], spot_series[i+1]],
                            color=colors[i], alpha=0.6)
        
        ax3.plot(hours, spot_series, color=COLORS['dark'], linewidth=1)
        
        ax3.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax3.set_title('Spot Prices', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# STORAGE STATE VISUALIZATION
# =============================================================================

def plot_storage_states(
    storage_soc_history: Dict[str, List[float]],
    title: str = "Storage State of Charge",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot storage state of charge over time.
    
    Args:
        storage_soc_history: Dict mapping storage name to SoC time series
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if not storage_soc_history:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No storage data', ha='center', va='center', fontsize=16)
        return fig
    
    n_storages = len(storage_soc_history)
    n_cols = min(3, n_storages)
    n_rows = (n_storages + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    storage_colors = {
        'Lake': COLORS['hydro'],
        'Fuel Tank': COLORS['thermal'],
        'H2 UG 200bar': COLORS['h2'],
        'Liquid H2': COLORS['h2'],
        'Solar salt': COLORS['solar'],
        'Biooil': COLORS['bio'],
        'Palm oil': COLORS['bio'],
        'Biogas': COLORS['bio'],
        'CH4 200bar': COLORS['storage'],
        'Ammonia': COLORS['storage'],
    }
    
    for idx, (name, soc_series) in enumerate(storage_soc_history.items()):
        ax = axes[idx]
        hours = np.arange(len(soc_series)) * 0.25
        
        color = storage_colors.get(name, COLORS['info'])
        
        ax.fill_between(hours, 0, soc_series, color=color, alpha=0.4)
        ax.plot(hours, soc_series, color=color, linewidth=2)
        
        # Target line at 60%
        ax.axhline(y=0.6, color=COLORS['accent'], linestyle='--', 
                  linewidth=1.5, label='Target SoC')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('SoC', fontsize=10)
        ax.set_xlabel('Hours', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_storages, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# COST ANALYSIS VISUALIZATION
# =============================================================================

def plot_cost_breakdown(
    cost_breakdown: Dict[str, float],
    title: str = "Cost Breakdown",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot cost breakdown by category.
    
    Args:
        cost_breakdown: Dictionary of category to cost
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Filter and sort
    active = {k: v for k, v in cost_breakdown.items() if v != 0}
    
    if not active:
        ax1.text(0.5, 0.5, 'No costs', ha='center', va='center')
        return fig
    
    sorted_items = sorted(active.items(), key=lambda x: abs(x[1]), reverse=True)
    categories = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    # Colors based on positive/negative
    colors = [COLORS['secondary'] if v > 0 else COLORS['success'] for v in values]
    
    # Horizontal bar chart
    bars = ax1.barh(categories, values, color=colors, edgecolor='white')
    
    ax1.axvline(x=0, color=COLORS['dark'], linewidth=1)
    ax1.set_xlabel('Cost (CHF)', fontsize=11, fontweight='bold')
    ax1.set_title('Cost by Category', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Pie chart (absolute values)
    abs_values = [abs(v) for v in values]
    pie_colors = [COLORS['secondary'] if v > 0 else COLORS['success'] for v in values]
    
    wedges, texts, autotexts = ax2.pie(
        abs_values, labels=categories, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(linewidth=2, edgecolor='white'),
    )
    
    ax2.set_title('Cost Distribution', fontsize=12, fontweight='bold')
    
    # Legend for colors
    legend_elements = [
        Patch(facecolor=COLORS['secondary'], label='Cost (expense)'),
        Patch(facecolor=COLORS['success'], label='Revenue'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def create_optimization_dashboard(
    best_fitness_history: List[float],
    portfolio_counts: Dict[str, int],
    final_metrics: Dict[str, float],
    title: str = "Optimization Results Dashboard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> plt.Figure:
    """
    Create a comprehensive dashboard of optimization results.
    
    Args:
        best_fitness_history: Best fitness per generation
        portfolio_counts: Final portfolio composition
        final_metrics: Final performance metrics
        title: Dashboard title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98,
                color=COLORS['dark'])
    
    # 1. Fitness Evolution (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    generations = np.arange(len(best_fitness_history))
    ax1.plot(generations, best_fitness_history, 
            color=COLORS['primary'], linewidth=2.5, marker='o', markersize=3)
    ax1.fill_between(generations, best_fitness_history, alpha=0.2, color=COLORS['primary'])
    ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fitness (CHF)', fontsize=11, fontweight='bold')
    ax1.set_title('Fitness Evolution', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Key Metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    metrics_text = []
    if 'annual_production_twh' in final_metrics:
        metrics_text.append(f"Production: {final_metrics['annual_production_twh']:.1f} TWh/year")
    if 'mean_cost' in final_metrics:
        metrics_text.append(f"Mean Cost: {final_metrics['mean_cost']:,.0f} CHF")
    if 'cvar' in final_metrics:
        metrics_text.append(f"CVaR: {final_metrics['cvar']:,.0f} CHF")
    if 'is_sovereign' in final_metrics:
        status = "✓ Yes" if final_metrics['is_sovereign'] else "✗ No"
        metrics_text.append(f"Sovereign: {status}")
    
    for i, text in enumerate(metrics_text):
        ax2.text(0.1, 0.85 - i*0.2, text, fontsize=14, fontweight='bold',
                transform=ax2.transAxes, color=COLORS['dark'])
    
    ax2.set_title('Key Metrics', fontsize=13, fontweight='bold')
    
    # 3. Portfolio Composition (middle, full width)
    ax3 = fig.add_subplot(gs[1, :])
    active = {k: v for k, v in portfolio_counts.items() if v > 0}
    
    if active:
        ppus = list(active.keys())
        counts = list(active.values())
        
        # Simple color assignment
        bar_colors = [COLORS['info']] * len(ppus)
        for i, ppu in enumerate(ppus):
            if 'PV' in ppu or 'SOL' in ppu:
                bar_colors[i] = COLORS['solar']
            elif 'WD' in ppu or 'WIND' in ppu:
                bar_colors[i] = COLORS['wind']
            elif 'HYD' in ppu or 'PHS' in ppu:
                bar_colors[i] = COLORS['hydro']
            elif 'H2' in ppu:
                bar_colors[i] = COLORS['h2']
            elif 'BIO' in ppu or 'PALM' in ppu:
                bar_colors[i] = COLORS['bio']
        
        ax3.bar(ppus, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
        ax3.set_ylabel('Units', fontsize=11, fontweight='bold')
        ax3.set_title('Portfolio Composition', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Summary text (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    total_units = sum(portfolio_counts.values())
    summary = (
        f"Total PPU Units: {total_units}  |  "
        f"Final Fitness: {best_fitness_history[-1]:,.0f} CHF  |  "
        f"Generations: {len(best_fitness_history)}"
    )
    
    ax4.text(0.5, 0.5, summary, fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                     edgecolor=COLORS['primary'], linewidth=2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo plots
    print("Visualization module loaded successfully")
    
    # Test fitness evolution plot
    test_history = [1e9, 5e8, 2e8, 1e8, 8e7, 7e7, 6.5e7, 6.3e7, 6.2e7, 6.1e7]
    fig = plot_fitness_evolution(test_history, title="Test Evolution")
    plt.show()

