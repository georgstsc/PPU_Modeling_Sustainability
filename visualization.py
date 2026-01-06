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
    'danger': '#e53e3e',       # Red (for errors/deficits)
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
        'Thermal': ['THERM', 'THERM_CH4', 'PALM_ICE'],  # BIO_OIL_ICE removed
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


# =============================================================================
# FULL YEAR EVALUATION PLOTS
# =============================================================================

def plot_full_year_overview(
    full_year_results,
    title: str = "Full Year Evaluation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive overview of the full year evaluation.
    
    Args:
        full_year_results: FullYearResults object
        title: Main title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    r = full_year_results
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.25)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Production vs Demand over the year (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    days = np.arange(len(r.demand)) / 24
    
    ax1.fill_between(days, r.demand / 1e3, alpha=0.3, color=COLORS['primary'], label='Demand')
    ax1.plot(days, r.total_production / 1e3, color=COLORS['success'], 
             linewidth=0.8, alpha=0.9, label='Production')
    
    ax1.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Power (GW)', fontsize=11, fontweight='bold')
    ax1.set_title('Annual Production vs Demand', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(r.demand) / 24)
    
    # 2. Monthly breakdown (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    n_months = len(r.monthly_production)
    x = np.arange(n_months)
    width = 0.35
    
    ax2.bar(x - width/2, r.monthly_demand[:n_months], width, 
            label='Demand', color=COLORS['primary'], alpha=0.7)
    ax2.bar(x + width/2, r.monthly_production[:n_months], width,
            label='Production', color=COLORS['success'], alpha=0.7)
    
    ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Energy (TWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Monthly Energy Balance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months[:n_months], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Key metrics (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    metrics = [
        ("Total Demand", f"{r.total_demand_twh:.1f} TWh"),
        ("Total Production", f"{r.total_production_twh:.1f} TWh"),
        ("Coverage Ratio", f"{r.coverage_ratio*100:.1f}%"),
        ("", ""),
        ("Spot Bought", f"{r.total_spot_bought_mwh/1e6:.2f} TWh"),
        ("Spot Sold", f"{r.total_spot_sold_mwh/1e6:.2f} TWh"),
        ("Net Spot Cost", f"{r.total_spot_cost_chf/1e6:,.1f} M CHF"),
        ("", ""),
        ("Hours in Deficit", f"{r.hours_in_deficit} ({r.hours_in_deficit/len(r.demand)*100:.1f}%)"),
        ("Peak Deficit", f"{r.peak_deficit_mw:,.0f} MW"),
    ]
    
    for i, (label, value) in enumerate(metrics):
        if label:
            ax3.text(0.1, 0.95 - i*0.1, f"{label}:", fontsize=11, fontweight='bold',
                    transform=ax3.transAxes, color=COLORS['dark'])
            ax3.text(0.6, 0.95 - i*0.1, value, fontsize=11,
                    transform=ax3.transAxes, color=COLORS['info'])
    
    ax3.set_title('Summary Metrics', fontsize=13, fontweight='bold')
    
    # 4. Deficit/Surplus distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    net_balance = r.total_production - r.demand
    ax4.hist(net_balance / 1e3, bins=50, color=COLORS['info'], 
             edgecolor='white', alpha=0.7)
    ax4.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2, label='Zero balance')
    
    ax4.set_xlabel('Net Balance (GW)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Hours', fontsize=11, fontweight='bold')
    ax4.set_title('Hourly Net Balance Distribution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Spot market activity (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Daily aggregates
    n_days = len(r.spot_bought) // 24
    daily_bought = np.array([np.sum(r.spot_bought[i*24:(i+1)*24]) for i in range(n_days)])
    daily_sold = np.array([np.sum(r.spot_sold[i*24:(i+1)*24]) for i in range(n_days)])
    days_idx = np.arange(n_days)
    
    ax5.fill_between(days_idx, -daily_sold / 1e3, alpha=0.5, 
                     color=COLORS['success'], label='Sold')
    ax5.fill_between(days_idx, daily_bought / 1e3, alpha=0.5,
                     color=COLORS['danger'], label='Bought')
    ax5.axhline(y=0, color='black', linewidth=0.5)
    
    ax5.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Daily Energy (GWh)', fontsize=11, fontweight='bold')
    ax5.set_title('Daily Spot Market Transactions', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, n_days)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_full_year_storage(
    full_year_results,
    title: str = "Storage State of Charge - Full Year",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot storage state of charge over the full year.
    
    Args:
        full_year_results: FullYearResults object
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    r = full_year_results
    storage_data = r.storage_soc
    
    if not storage_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No storage data available", 
                ha='center', va='center', fontsize=14)
        return fig
    
    n_storages = len(storage_data)
    fig, axes = plt.subplots(n_storages, 1, figsize=figsize, sharex=True,
                             facecolor=COLORS['light'])
    
    if n_storages == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    days = np.arange(len(r.demand)) / 24
    
    storage_colors = {
        'PHS': COLORS['hydro'],
        'H2': COLORS['h2'],
        'NH3': COLORS['warning'],
        'BIO': COLORS['bio'],
        'PALM': COLORS['bio'],
        'SYN': COLORS['secondary'],
        'SOL_SALT': COLORS['solar'],
    }
    
    for ax, (storage_name, soc) in zip(axes, storage_data.items()):
        # Find matching color
        color = COLORS['info']
        for key, c in storage_colors.items():
            if key in storage_name.upper():
                color = c
                break
        
        ax.fill_between(days, soc * 100, alpha=0.6, color=color)
        ax.plot(days, soc * 100, color=color, linewidth=0.8)
        
        ax.set_ylabel(f'{storage_name}\nSoC (%)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=60, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    axes[-1].set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    axes[-1].set_xlim(0, len(r.demand) / 24)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_individual_storage(
    full_year_results,
    save_folder: str = "data/result_plots/scenario_evolution",
    figsize: Tuple[int, int] = (12, 4),
) -> List[plt.Figure]:
    """
    Create and save individual plots for each storage system.
    
    Args:
        full_year_results: FullYearResults object
        save_folder: Folder to save individual plots
        figsize: Figure size for each plot
        
    Returns:
        List of matplotlib Figures
    """
    import os
    os.makedirs(save_folder, exist_ok=True)
    
    r = full_year_results
    storage_data = r.storage_soc
    
    if not storage_data:
        print("No storage data available")
        return []
    
    storage_colors = {
        'Lake': COLORS['hydro'],
        'H2': COLORS['h2'],
        'NH3': COLORS['warning'],
        'Biogas': '#2ecc71',
        'Biooil': '#27ae60',
        'Palm': '#16a085',
        'Solar': COLORS['solar'],
        'Fuel': COLORS['secondary'],
        'CH4': '#9b59b6',
        'Ammonia': '#e67e22',
        'Liquid': COLORS['info'],
    }
    
    days = np.arange(len(r.demand)) / 24
    figures = []
    
    for storage_name, soc in storage_data.items():
        fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['light'])
        
        # Find matching color
        color = COLORS['info']
        for key, c in storage_colors.items():
            if key.lower() in storage_name.lower():
                color = c
                break
        
        # Convert to percentage
        soc_pct = np.array(soc) * 100
        
        # Plot
        ax.fill_between(days, soc_pct, alpha=0.4, color=color)
        ax.plot(days, soc_pct, color=color, linewidth=1.5, label='State of Charge')
        
        # Add reference lines
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Full')
        ax.axhline(y=60, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial (60%)')
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Low')
        
        # Stats
        min_soc = np.min(soc_pct)
        max_soc = np.max(soc_pct)
        final_soc = soc_pct[-1]
        changes = np.sum(np.abs(np.diff(soc)) > 0.001)
        
        stats_text = f"Min: {min_soc:.0f}%  Max: {max_soc:.0f}%  Final: {final_soc:.0f}%  Cycles: {changes}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{storage_name} - State of Charge Over Year', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Day of Year', fontsize=11)
        ax.set_ylabel('State of Charge (%)', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, days[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        safe_name = storage_name.replace(' ', '_').replace('/', '_')
        save_path = f"{save_folder}/storage_{safe_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
        
        figures.append(fig)
        plt.close(fig)  # Close to free memory
    
    print(f"\n✅ Saved {len(figures)} individual storage plots to {save_folder}/")
    return figures


def plot_energy_flow(
    dispatch_results: Dict[str, Any],
    portfolio_counts: Dict[str, int],
    title: str = "Energy Flow by PPU Type",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create horizontal diverging bar chart showing energy flow through each PPU type.
    
    Left (negative) = Energy stored/absorbed
    Right (positive) = Energy produced/discharged
    
    Args:
        dispatch_results: Results dict from run_dispatch_simulation
        portfolio_counts: PPU counts from portfolio
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Define PPU groupings by type
    ppu_groups = {
        'INCIDENCE - Solar': {
            'ppus': ['PV'],
            'color': COLORS['solar'],
            'type': 'incidence',
        },
        'INCIDENCE - Wind Onshore': {
            'ppus': ['WD_ON'],
            'color': COLORS['wind'],
            'type': 'incidence',
        },
        'INCIDENCE - Wind Offshore': {
            'ppus': ['WD_OFF'],
            'color': '#1a5276',
            'type': 'incidence',
        },
        'GENERATOR - Hydro Run-of-River': {
            'ppus': ['HYD_R'],
            'color': COLORS['hydro'],
            'type': 'generator',
        },
        'GENERATOR - Hydro Storage': {
            'ppus': ['HYD_S'],
            'color': '#2980b9',
            'type': 'generator',
        },
        'GENERATOR - Thermal': {
            'ppus': ['THERM', 'THERM_CH4'],
            'color': '#e74c3c',
            'type': 'generator',
        },
        'GENERATOR - Hydrogen Power': {
            'ppus': ['H2P_G', 'H2P_L'],
            'color': COLORS['h2'],
            'type': 'generator',
        },
        'GENERATOR - Biomass': {
            'ppus': ['BIO_WOOD', 'PALM_ICE', 'IMP_BIOG'],  # BIO_OIL_ICE removed
            'color': COLORS['bio'],
            'type': 'generator',
        },
        'GENERATOR - Solar Thermal': {
            'ppus': ['SOL_SALT', 'SOL_STEAM'],
            'color': '#f39c12',
            'type': 'generator',
        },
        'GENERATOR - Ammonia': {
            'ppus': ['NH3_P'],
            'color': '#9b59b6',
            'type': 'generator',
        },
        'STORAGE - Lake (PHS)': {
            'storage': 'Lake',
            'color': COLORS['hydro'],
            'type': 'storage',
        },
        'STORAGE - Solar Salt': {
            'storage': 'Solar salt',
            'color': '#f39c12',
            'type': 'storage',
        },
        'STORAGE - H2 Underground': {
            'storage': 'H2 UG 200bar',
            'color': COLORS['h2'],
            'type': 'storage',
        },
        'STORAGE - Liquid H2': {
            'storage': 'Liquid H2',
            'color': '#5dade2',
            'type': 'storage',
        },
        'STORAGE - Biogas': {
            'storage': 'Biogas',
            'color': '#27ae60',
            'type': 'storage',
        },
        'STORAGE - Fuel Tank': {
            'storage': 'Fuel Tank',
            'color': '#8e44ad',
            'type': 'storage',
        },
        'STORAGE - CH4': {
            'storage': 'CH4 200bar',
            'color': '#e67e22',
            'type': 'storage',
        },
        'STORAGE - Ammonia': {
            'storage': 'Ammonia',
            'color': '#9b59b6',
            'type': 'storage',
        },
    }
    
    # Calculate energy for each group
    flow_data = []
    
    # Production from PPUs
    ppu_production = dispatch_results.get('ppu_production', {})
    renewable_mwh = dispatch_results.get('total_renewable_mwh', 0)
    
    for group_name, group_info in ppu_groups.items():
        if group_info['type'] == 'incidence':
            # Incidence-based production (estimate from renewable total)
            # This is approximate - in reality we'd need per-PPU tracking
            ppus = group_info['ppus']
            total_count = sum(portfolio_counts.get(p, 0) for p in ppus)
            if total_count > 0 and 'Solar' in group_name:
                # Solar gets ~70% of renewable (rough estimate based on capacity)
                energy = renewable_mwh * 0.7 / 1e6  # TWh
            elif total_count > 0 and 'Wind' in group_name:
                # Wind gets ~30% of renewable
                energy = renewable_mwh * 0.15 / 1e6  # TWh per wind type
            else:
                energy = 0
            
            if energy > 0:
                flow_data.append({
                    'name': group_name,
                    'produced': energy,
                    'stored': 0,
                    'color': group_info['color'],
                    'type': group_info['type'],
                })
        
        elif group_info['type'] == 'generator':
            ppus = group_info['ppus']
            energy = sum(ppu_production.get(p, 0) for p in ppus) / 1e6  # TWh
            
            if energy > 0:
                flow_data.append({
                    'name': group_name,
                    'produced': energy,
                    'stored': 0,
                    'color': group_info['color'],
                    'type': group_info['type'],
                })
        
        elif group_info['type'] == 'storage':
            storage_name = group_info['storage']
            storage_soc = dispatch_results.get('storage_soc', {}).get(storage_name, [])
            
            if len(storage_soc) > 1:
                # Get storage capacity from config
                from config import DEFAULT_CONFIG
                storage_defs = DEFAULT_CONFIG.storage.STORAGE_DEFINITIONS
                capacity = storage_defs.get(storage_name, {}).get('capacity_MWh', 0)
                
                # Calculate net flow: positive = net discharge, negative = net charge
                initial_soc = storage_soc[0]
                final_soc = storage_soc[-1]
                
                # Net change in stored energy (negative = discharged, positive = charged)
                net_stored = (final_soc - initial_soc) * capacity / 1e6  # TWh
                
                # Estimate total throughput from SoC changes
                soc_changes = np.diff(storage_soc)
                charged = np.sum(soc_changes[soc_changes > 0]) * capacity / 1e6
                discharged = -np.sum(soc_changes[soc_changes < 0]) * capacity / 1e6
                
                if charged > 0.001 or discharged > 0.001:
                    flow_data.append({
                        'name': group_name,
                        'produced': discharged,  # Positive = produced energy
                        'stored': charged,       # Positive = absorbed energy (shown left)
                        'color': group_info['color'],
                        'type': group_info['type'],
                    })
    
    # Sort by type then by total energy
    type_order = {'incidence': 0, 'generator': 1, 'storage': 2}
    flow_data.sort(key=lambda x: (type_order[x['type']], -(x['produced'] + x['stored'])))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['light'])
    
    if not flow_data:
        ax.text(0.5, 0.5, "No energy flow data available", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Plot horizontal bars
    y_positions = np.arange(len(flow_data))
    bar_height = 0.6
    
    names = [d['name'] for d in flow_data]
    produced = [d['produced'] for d in flow_data]
    stored = [-d['stored'] for d in flow_data]  # Negative for left side
    colors = [d['color'] for d in flow_data]
    
    # Production bars (right, positive)
    bars_prod = ax.barh(y_positions, produced, height=bar_height, 
                        color=colors, alpha=0.8, label='Produced/Discharged')
    
    # Storage bars (left, negative)  
    bars_stor = ax.barh(y_positions, stored, height=bar_height,
                        color=colors, alpha=0.4, hatch='///', label='Stored/Charged')
    
    # Add value labels
    for i, (prod, stor, name) in enumerate(zip(produced, stored, names)):
        if prod > 0.01:
            ax.text(prod + 0.2, i, f'{prod:.2f} TWh', va='center', fontsize=9, fontweight='bold')
        if stor < -0.01:
            ax.text(stor - 0.2, i, f'{-stor:.2f} TWh', va='center', ha='right', fontsize=9)
    
    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Energy (TWh)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.5)
    
    # Add section labels
    current_type = None
    for i, d in enumerate(flow_data):
        if d['type'] != current_type:
            current_type = d['type']
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, label='→ Produced / Discharged'),
        Patch(facecolor='gray', alpha=0.4, hatch='///', label='← Stored / Charged'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add annotations
    ax.text(0.02, 0.98, '← Energy Absorbed', transform=ax.transAxes, 
            fontsize=10, va='top', color='gray')
    ax.text(0.98, 0.98, 'Energy Produced →', transform=ax.transAxes,
            fontsize=10, va='top', ha='right', color='gray')
    
    ax.set_xlim(min(stored) * 1.3 if min(stored) < 0 else -1, max(produced) * 1.2)
    ax.invert_yaxis()  # Top to bottom
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_full_year_production_by_ppu(
    full_year_results,
    title: str = "Production by PPU Origin - Full Year",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Plot stacked production decomposed by every individual PPU origin.
    
    Args:
        full_year_results: FullYearResults object
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    r = full_year_results
    n_hours = len(r.demand)
    n_days = n_hours // 24
    
    # Get all unique PPU names that actually produced something
    # Note: Storage INPUT PPUs (PHS, H2_G, etc.) consume energy to charge storage,
    # they are NOT included here because they don't produce energy to grid
    active_ppus = {}
    for ppu_name, prod in r.ppu_production.items():
        if isinstance(prod, (int, float, np.number)):
            prod_hourly = np.full(n_hours, prod / n_hours)
        else:
            prod_hourly = prod
            
        if np.sum(prod_hourly) > 1e-3: # Filter out near-zero producers
            active_ppus[ppu_name] = prod_hourly

    if not active_ppus:
        print("⚠️ No active PPUs found for decomposition plot.")
        return plt.figure()

    # Define a custom sorting order for aesthetic stacking
    # Incidence at bottom, then flex/storage extraction
    def get_sort_key(name):
        name_up = name.upper()
        if 'HYD_R' in name_up or 'RIVER' in name_up: return 0
        if 'PV' in name_up or 'SOLAR' in name_up and 'SALT' not in name_up: return 1
        if 'WD' in name_up or 'WIND' in name_up: return 2
        if 'HYD_S' in name_up or 'LAKE' in name_up: return 3
        if 'SOL_SALT' in name_up or 'SOL_STEAM' in name_up: return 4
        if 'BIO' in name_up or 'PALM' in name_up: return 5
        if 'H2' in name_up or 'NH3' in name_up or 'CH4' in name_up: return 6
        return 10 # Everything else

    sorted_names = sorted(active_ppus.keys(), key=get_sort_key)
    
    # Generate distinct colors for many PPUs
    cmap = plt.cm.get_cmap('tab20', len(sorted_names))
    ppu_colors = {name: cmap(i) for i, name in enumerate(sorted_names)}
    
    # Calculate daily averages for each PPU
    daily_prod = {}
    for name in sorted_names:
        prod = active_ppus[name]
        daily_prod[name] = np.array([np.mean(prod[i*24:(i+1)*24]) for i in range(n_days)]) / 1000.0 # to GW
        
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['light'])
    days = np.arange(n_days)
    bottom = np.zeros(n_days)
    
    # Stack plot
    for name in sorted_names:
        ax.fill_between(days, bottom, bottom + daily_prod[name],
                       alpha=0.8, color=ppu_colors[name], label=name)
        bottom += daily_prod[name]
        
    # Demand line
    daily_demand = np.array([np.mean(r.demand[i*24:(i+1)*24]) for i in range(n_days)]) / 1000.0
    ax.plot(days, daily_demand, color='black', linewidth=2, linestyle='--', 
            label='Demand', zorder=10)
            
    ax.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Power (GW)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Handle legend (might be large)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=min(len(sorted_names), 5), fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_days)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
        
    return fig


def plot_full_year_production_by_source(
    full_year_results,
    title: str = "Production by Source - Full Year",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot stacked production by PPU type over the full year.
    
    Args:
        full_year_results: FullYearResults object
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    r = full_year_results
    
    # Group PPU production by type
    # More specific matching to avoid 'SOL_SALT' (dispatchable) appearing as direct 'Solar'
    categories = {
        'Solar (Direct)': ['PV'],
        'Wind': ['WD_ON', 'WD_OFF', 'WIND'],
        'Hydro (River)': ['HYD_R'],
        'Hydro (Lake)': ['HYD_S', 'Lake'],
        'Solar (Storage)': ['SOL_SALT', 'SOL_STEAM', 'Solar salt'],
        'Synthetic Fuels': ['H2', 'THERM', 'NH3', 'CH4', 'Fuel Tank', 'Ammonia'],
        'Biomass': ['BIO', 'PALM', 'Palm oil', 'Biogas'],  # Biooil removed
        'Other': [],
    }
    
    category_colors = {
        'Solar (Direct)': COLORS['solar'],
        'Wind': COLORS['wind'],
        'Hydro (River)': COLORS['hydro'],
        'Hydro (Lake)': '#0055A4', # Deeper blue
        'Solar (Storage)': '#FFD700', # Gold
        'Synthetic Fuels': COLORS['h2'],
        'Biomass': COLORS['bio'],
        'Other': COLORS['secondary'],
    }
    
    # Aggregate by category
    n_hours = len(r.demand)
    category_production = {cat: np.zeros(n_hours) for cat in categories}
    
    # Order for stacking (bottom to top)
    stack_order = [
        'Hydro (River)', 'Solar (Direct)', 'Wind', 
        'Hydro (Lake)', 'Solar (Storage)', 
        'Biomass', 'Synthetic Fuels', 'Other'
    ]
    
    # Note: Only production PPUs are shown here. Storage INPUT PPUs (PHS, H2_G, etc.)
    # consume energy to charge storage - they are tracked in ppu_consumption separately
    for ppu_name, prod in r.ppu_production.items():
        # Check if prod is an array or scalar
        if isinstance(prod, (int, float, np.number)):
            prod_hourly = np.full(n_hours, prod / n_hours)
        else:
            prod_hourly = prod
            
        assigned = False
        for cat in stack_order:
            keywords = categories[cat]
            if any(kw in ppu_name.upper() for kw in keywords):
                category_production[cat] += prod_hourly
                assigned = True
                break
        if not assigned:
            category_production['Other'] += prod_hourly
    
    # Daily average for visualization
    n_days = n_hours // 24
    daily_prod = {cat: np.array([np.mean(prod[i*24:(i+1)*24]) for i in range(n_days)]) / 1e3
                  for cat, prod in category_production.items()}
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['light'])
    
    days = np.arange(n_days)
    bottom = np.zeros(n_days)
    
    # Stack plot in the defined order
    for cat in stack_order:
        if np.sum(daily_prod[cat]) > 0:
            ax.fill_between(days, bottom, bottom + daily_prod[cat],
                           alpha=0.7, color=category_colors[cat], label=cat)
            bottom += daily_prod[cat]
    
    # Demand line
    daily_demand = np.array([np.mean(r.demand[i*24:(i+1)*24]) for i in range(n_days)]) / 1e3
    ax.plot(days, daily_demand, color='black', linewidth=2, linestyle='--', 
            label='Demand', zorder=10)
    
    ax.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Power (GW)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_days)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved: {save_path}")
    
    return fig




# =============================================================================
# DEMAND VOLATILITY VISUALIZATION
# =============================================================================

def plot_demand_volatility(
    demand_data: np.ndarray,
    title: str = "Energy Demand Volatility - Full Year 2024",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive visualization of demand volatility across the year.
    
    Shows:
    1. Full year hourly demand with rolling statistics
    2. Daily patterns (box plot by hour of day)
    3. Monthly aggregations
    4. Volatility metrics (rolling standard deviation)
    5. Statistics summary
    
    Args:
        demand_data: Hourly demand array (MW) - typically 8784 hours for 2024
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Convert to GW for readability
    demand_gw = demand_data / 1000.0
    n_hours = len(demand_gw)
    n_days = n_hours // 24
    
    # Create datetime index for better plotting
    dates = pd.date_range(start='2024-01-01', periods=n_hours, freq='H')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # ===== SUBPLOT 1: Full Year Hourly Demand with Rolling Statistics =====
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot hourly demand (thin line, semi-transparent)
    ax1.plot(dates, demand_gw, color=COLORS['primary'], alpha=0.3, linewidth=0.5, label='Hourly Demand')
    
    # Rolling mean (7-day window)
    window_days = 7
    window_hours = window_days * 24
    rolling_mean = pd.Series(demand_gw).rolling(window=window_hours, center=True).mean()
    ax1.plot(dates, rolling_mean, color=COLORS['secondary'], linewidth=2, label=f'{window_days}-Day Rolling Mean')
    
    # Rolling min/max bands (7-day window)
    rolling_min = pd.Series(demand_gw).rolling(window=window_hours, center=True).min()
    rolling_max = pd.Series(demand_gw).rolling(window=window_hours, center=True).max()
    ax1.fill_between(dates, rolling_min, rolling_max, alpha=0.2, color=COLORS['info'], 
                     label=f'{window_days}-Day Min/Max Band')
    
    # Overall statistics
    overall_mean = np.mean(demand_gw)
    overall_std = np.std(demand_gw)
    ax1.axhline(overall_mean, color=COLORS['accent'], linestyle='--', linewidth=2, 
                label=f'Annual Mean: {overall_mean:.2f} GW')
    ax1.fill_between(dates, overall_mean - overall_std, overall_mean + overall_std, 
                     alpha=0.1, color=COLORS['accent'], label=f'±1 Std Dev: {overall_std:.2f} GW')
    
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Demand (GW)', fontsize=11, fontweight='bold')
    ax1.set_title('Full Year Hourly Demand with Rolling Statistics', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 2: Daily Pattern (Box Plot by Hour of Day) =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Reshape to (n_days, 24) for hourly analysis
    # Trim to exact multiple of 24 hours
    n_complete_days = n_hours // 24
    daily_patterns = demand_gw[:n_complete_days*24].reshape(n_complete_days, 24)
    
    # Box plot for each hour of day
    box_data = [daily_patterns[:, h] for h in range(24)]
    bp = ax2.boxplot(box_data, positions=range(24), widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['info'])
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Demand (GW)', fontsize=11, fontweight='bold')
    ax2.set_title('Daily Demand Pattern (Box Plot by Hour)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 3: Monthly Aggregations =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Group by month
    df = pd.DataFrame({'demand': demand_gw, 'date': dates})
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    
    monthly_stats = df.groupby('month').agg({
        'demand': ['mean', 'std', 'min', 'max'],
        'month_name': 'first'  # Get month name for each month
    }).reset_index()
    
    months = monthly_stats[('month_name', 'first')].values
    x_pos = np.arange(len(months))
    
    # Plot bars with error bars
    means = monthly_stats[('demand', 'mean')].values
    stds = monthly_stats[('demand', 'std')].values
    mins = monthly_stats[('demand', 'min')].values
    maxs = monthly_stats[('demand', 'max')].values
    
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=COLORS['success'], edgecolor=COLORS['dark'], linewidth=1.5)
    
    # Add min/max markers
    ax3.scatter(x_pos, mins, color=COLORS['danger'], marker='_', s=100, zorder=5, label='Min')
    ax3.scatter(x_pos, maxs, color=COLORS['warning'], marker='_', s=100, zorder=5, label='Max')
    
    ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Demand (GW)', fontsize=11, fontweight='bold')
    ax3.set_title('Monthly Demand Statistics', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(months, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 4: Rolling Volatility (Standard Deviation) =====
    ax4 = fig.add_subplot(gs[2, :])
    
    # Calculate rolling standard deviation
    window_sizes = [24, 168, 720]  # 1 day, 1 week, 1 month
    window_labels = ['1 Day', '1 Week', '1 Month']
    colors_vol = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    for window, label, color in zip(window_sizes, window_labels, colors_vol):
        rolling_std = pd.Series(demand_gw).rolling(window=window, center=True).std()
        ax4.plot(dates, rolling_std, color=color, linewidth=2, label=f'{label} Rolling Std Dev')
    
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Standard Deviation (GW)', fontsize=11, fontweight='bold')
    ax4.set_title('Demand Volatility Over Time (Rolling Standard Deviation)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 5: Statistics Summary Table =====
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    # Calculate comprehensive statistics
    stats_text = f"""
    DEMAND STATISTICS SUMMARY
    {'='*80}
    
    Overall Statistics:
      • Annual Mean:           {overall_mean:>8.2f} GW
      • Annual Std Dev:        {overall_std:>8.2f} GW
      • Coefficient of Variation: {overall_std/overall_mean*100:>6.2f}%
      • Minimum:               {np.min(demand_gw):>8.2f} GW  (at hour {np.argmin(demand_gw)})
      • Maximum:               {np.max(demand_gw):>8.2f} GW  (at hour {np.argmax(demand_gw)})
      • Range:                 {np.max(demand_gw) - np.min(demand_gw):>8.2f} GW
      • Peak-to-Mean Ratio:    {np.max(demand_gw)/overall_mean:>6.2f}x
    
    Daily Patterns:
      • Peak Hour (avg):       Hour {np.argmax([np.mean(daily_patterns[:, h]) for h in range(24)]):>2d}  ({np.max([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} GW)
      • Minimum Hour (avg):    Hour {np.argmin([np.mean(daily_patterns[:, h]) for h in range(24)]):>2d}  ({np.min([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} GW)
      • Daily Peak-to-Trough:  {np.max([np.mean(daily_patterns[:, h]) for h in range(24)]) - np.min([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} GW
    
    Monthly Variations:
      • Highest Month:         {months[np.argmax(means)]}  ({np.max(means):.2f} GW avg)
      • Lowest Month:          {months[np.argmin(means)]}  ({np.min(means):.2f} GW avg)
      • Monthly Range:         {np.max(means) - np.min(means):.2f} GW
    
    Volatility Metrics:
      • 1-Day Rolling Std Dev: {pd.Series(demand_gw).rolling(24).std().mean():.2f} GW (avg)
      • 1-Week Rolling Std Dev: {pd.Series(demand_gw).rolling(168).std().mean():.2f} GW (avg)
      • 1-Month Rolling Std Dev: {pd.Series(demand_gw).rolling(720).std().mean():.2f} GW (avg)
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved demand volatility plot: {save_path}")
    
    return fig


def plot_spot_price_volatility(
    spot_data: np.ndarray,
    title: str = "Spot Price Volatility - Full Year 2024",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive visualization of spot price volatility across the year.
    
    Shows:
    1. Full year hourly prices with rolling statistics
    2. Daily patterns (box plot by hour of day)
    3. Monthly aggregations
    4. Volatility metrics (rolling standard deviation)
    5. Statistics summary
    
    Args:
        spot_data: Hourly spot prices array (CHF/MWh) - typically 8784 hours for 2024
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    prices = spot_data.copy()
    n_hours = len(prices)
    n_days = n_hours // 24
    
    # Create datetime index for better plotting
    dates = pd.date_range(start='2024-01-01', periods=n_hours, freq='H')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # ===== SUBPLOT 1: Full Year Hourly Prices with Rolling Statistics =====
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot hourly prices (thin line, semi-transparent)
    ax1.plot(dates, prices, color=COLORS['accent'], alpha=0.3, linewidth=0.5, label='Hourly Price')
    
    # Rolling mean (7-day window)
    window_days = 7
    window_hours = window_days * 24
    rolling_mean = pd.Series(prices).rolling(window=window_hours, center=True).mean()
    ax1.plot(dates, rolling_mean, color=COLORS['secondary'], linewidth=2, label=f'{window_days}-Day Rolling Mean')
    
    # Rolling min/max bands (7-day window)
    rolling_min = pd.Series(prices).rolling(window=window_hours, center=True).min()
    rolling_max = pd.Series(prices).rolling(window=window_hours, center=True).max()
    ax1.fill_between(dates, rolling_min, rolling_max, alpha=0.2, color=COLORS['info'], 
                     label=f'{window_days}-Day Min/Max Band')
    
    # Overall statistics
    overall_mean = np.mean(prices)
    overall_std = np.std(prices)
    ax1.axhline(overall_mean, color=COLORS['primary'], linestyle='--', linewidth=2, 
                label=f'Annual Mean: {overall_mean:.1f} CHF/MWh')
    ax1.fill_between(dates, overall_mean - overall_std, overall_mean + overall_std, 
                     alpha=0.1, color=COLORS['primary'], label=f'±1 Std Dev: {overall_std:.1f} CHF/MWh')
    
    # Mark zero line (important for spot prices)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Spot Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax1.set_title('Full Year Hourly Spot Prices with Rolling Statistics', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 2: Daily Pattern (Box Plot by Hour of Day) =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Reshape to (n_days, 24) for hourly analysis
    n_complete_days = n_hours // 24
    daily_patterns = prices[:n_complete_days*24].reshape(n_complete_days, 24)
    
    # Box plot for each hour of day
    box_data = [daily_patterns[:, h] for h in range(24)]
    bp = ax2.boxplot(box_data, positions=range(24), widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['accent'])
        patch.set_alpha(0.7)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Daily Price Pattern (Box Plot by Hour)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 3: Monthly Aggregations =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Group by month
    df = pd.DataFrame({'price': prices, 'date': dates})
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    
    monthly_stats = df.groupby('month').agg({
        'price': ['mean', 'std', 'min', 'max'],
        'month_name': 'first'
    }).reset_index()
    
    months = monthly_stats[('month_name', 'first')].values
    x_pos = np.arange(len(months))
    
    means = monthly_stats[('price', 'mean')].values
    stds = monthly_stats[('price', 'std')].values
    mins = monthly_stats[('price', 'min')].values
    maxs = monthly_stats[('price', 'max')].values
    
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1.5)
    
    ax3.scatter(x_pos, mins, color=COLORS['danger'], marker='_', s=100, zorder=5, label='Min')
    ax3.scatter(x_pos, maxs, color=COLORS['warning'], marker='_', s=100, zorder=5, label='Max')
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Monthly Price Statistics', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(months, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 4: Rolling Volatility (Standard Deviation) =====
    ax4 = fig.add_subplot(gs[2, :])
    
    # Calculate rolling standard deviation
    window_sizes = [24, 168, 720]  # 1 day, 1 week, 1 month
    window_labels = ['1 Day', '1 Week', '1 Month']
    colors_vol = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    for window, label, color in zip(window_sizes, window_labels, colors_vol):
        rolling_std = pd.Series(prices).rolling(window=window, center=True).std()
        ax4.plot(dates, rolling_std, color=color, linewidth=2, label=f'{label} Rolling Std Dev')
    
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Standard Deviation (CHF/MWh)', fontsize=11, fontweight='bold')
    ax4.set_title('Price Volatility Over Time (Rolling Standard Deviation)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 5: Statistics Summary Table =====
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    # Calculate comprehensive statistics
    negative_hours = np.sum(prices < 0)
    zero_hours = np.sum(prices == 0)
    high_hours = np.sum(prices > 100)
    
    stats_text = f"""
    SPOT PRICE STATISTICS SUMMARY
    {'='*80}
    
    Overall Statistics:
      • Annual Mean:           {overall_mean:>10.2f} CHF/MWh
      • Annual Std Dev:        {overall_std:>10.2f} CHF/MWh
      • Coefficient of Variation: {overall_std/overall_mean*100:>6.2f}%
      • Minimum:               {np.min(prices):>10.2f} CHF/MWh  (at hour {np.argmin(prices)})
      • Maximum:               {np.max(prices):>10.2f} CHF/MWh  (at hour {np.argmax(prices)})
      • Range:                 {np.max(prices) - np.min(prices):>10.2f} CHF/MWh
      • Median:                {np.median(prices):>10.2f} CHF/MWh
    
    Price Distribution:
      • Hours with negative prices: {negative_hours:>6d} ({negative_hours/n_hours*100:.2f}%)
      • Hours with zero prices:     {zero_hours:>6d} ({zero_hours/n_hours*100:.2f}%)
      • Hours above 100 CHF/MWh:    {high_hours:>6d} ({high_hours/n_hours*100:.2f}%)
    
    Daily Patterns:
      • Peak Hour (avg):       Hour {np.argmax([np.mean(daily_patterns[:, h]) for h in range(24)]):>2d}  ({np.max([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} CHF/MWh)
      • Minimum Hour (avg):    Hour {np.argmin([np.mean(daily_patterns[:, h]) for h in range(24)]):>2d}  ({np.min([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} CHF/MWh)
      • Daily Peak-to-Trough:  {np.max([np.mean(daily_patterns[:, h]) for h in range(24)]) - np.min([np.mean(daily_patterns[:, h]) for h in range(24)]):.2f} CHF/MWh
    
    Monthly Variations:
      • Highest Month:         {months[np.argmax(means)]}  ({np.max(means):.2f} CHF/MWh avg)
      • Lowest Month:          {months[np.argmin(means)]}  ({np.min(means):.2f} CHF/MWh avg)
    
    Volatility Metrics:
      • 1-Day Rolling Std Dev: {pd.Series(prices).rolling(24).std().mean():.2f} CHF/MWh (avg)
      • 1-Week Rolling Std Dev: {pd.Series(prices).rolling(168).std().mean():.2f} CHF/MWh (avg)
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved spot price volatility plot: {save_path}")
    
    return fig


def plot_spot_price_distribution(
    spot_data: np.ndarray,
    title: str = "Spot Price Distribution - Full Year 2024",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create a comprehensive distribution visualization of spot prices.
    
    Shows:
    1. Histogram with KDE
    2. CDF (Cumulative Distribution Function)
    3. Box plot with percentiles
    4. Price percentile table
    
    Args:
        spot_data: Hourly spot prices array (CHF/MWh)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    prices = spot_data.copy()
    n_hours = len(prices)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ===== SUBPLOT 1: Histogram with KDE =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create histogram
    n_bins = 100
    counts, bins, patches = ax1.hist(prices, bins=n_bins, density=True, alpha=0.7, 
                                      color=COLORS['accent'], edgecolor='white', linewidth=0.5)
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(prices)
    x_kde = np.linspace(prices.min(), prices.max(), 500)
    ax1.plot(x_kde, kde(x_kde), color=COLORS['secondary'], linewidth=2.5, label='KDE')
    
    # Mark key statistics
    ax1.axvline(np.mean(prices), color=COLORS['primary'], linestyle='--', linewidth=2, label=f'Mean: {np.mean(prices):.1f}')
    ax1.axvline(np.median(prices), color=COLORS['success'], linestyle='-.', linewidth=2, label=f'Median: {np.median(prices):.1f}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Spot Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Price Distribution (Histogram with KDE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: CDF (Cumulative Distribution Function) =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    sorted_prices = np.sort(prices)
    cdf = np.arange(1, n_hours + 1) / n_hours
    
    ax2.plot(sorted_prices, cdf, color=COLORS['primary'], linewidth=2)
    ax2.fill_between(sorted_prices, 0, cdf, alpha=0.2, color=COLORS['primary'])
    
    # Mark percentiles
    percentiles = [5, 25, 50, 75, 95]
    colors_pct = [COLORS['danger'], COLORS['warning'], COLORS['success'], COLORS['warning'], COLORS['danger']]
    for p, c in zip(percentiles, colors_pct):
        val = np.percentile(prices, p)
        ax2.axhline(p/100, color=c, linestyle=':', alpha=0.7)
        ax2.axvline(val, color=c, linestyle=':', alpha=0.7)
        ax2.scatter([val], [p/100], color=c, s=50, zorder=5)
        ax2.annotate(f'P{p}: {val:.1f}', (val, p/100), textcoords='offset points', 
                    xytext=(5, 5), fontsize=8)
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Spot Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # ===== SUBPLOT 3: Box Plot with Violin =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create violin plot
    parts = ax3.violinplot([prices], positions=[0], showmeans=True, showmedians=True, showextrema=True)
    
    # Color the violin
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['accent'])
        pc.set_alpha(0.7)
    
    # Add box plot on top
    bp = ax3.boxplot([prices], positions=[0], widths=0.15, patch_artist=True,
                     showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': COLORS['secondary']})
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['info'])
        patch.set_alpha(0.8)
    
    # Add individual percentile annotations
    percentile_values = [np.percentile(prices, p) for p in [5, 25, 50, 75, 95]]
    percentile_labels = ['P5', 'P25', 'P50', 'P75', 'P95']
    
    for val, label in zip(percentile_values, percentile_labels):
        ax3.annotate(f'{label}: {val:.1f}', (0.3, val), fontsize=9)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Spot Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Price Distribution (Violin + Box Plot)', fontsize=12, fontweight='bold')
    ax3.set_xticks([0])
    ax3.set_xticklabels(['Full Year'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 4: Statistics Summary Table =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate comprehensive statistics
    negative_hours = np.sum(prices < 0)
    high_hours = np.sum(prices > 100)
    very_high_hours = np.sum(prices > 200)
    
    # Create table data
    table_data = [
        ['Statistic', 'Value', 'Statistic', 'Value'],
        ['Mean', f'{np.mean(prices):.2f} CHF/MWh', 'Std Dev', f'{np.std(prices):.2f} CHF/MWh'],
        ['Median', f'{np.median(prices):.2f} CHF/MWh', 'IQR', f'{np.percentile(prices, 75) - np.percentile(prices, 25):.2f} CHF/MWh'],
        ['Minimum', f'{np.min(prices):.2f} CHF/MWh', 'Maximum', f'{np.max(prices):.2f} CHF/MWh'],
        ['P5', f'{np.percentile(prices, 5):.2f} CHF/MWh', 'P95', f'{np.percentile(prices, 95):.2f} CHF/MWh'],
        ['P10', f'{np.percentile(prices, 10):.2f} CHF/MWh', 'P90', f'{np.percentile(prices, 90):.2f} CHF/MWh'],
        ['P25', f'{np.percentile(prices, 25):.2f} CHF/MWh', 'P75', f'{np.percentile(prices, 75):.2f} CHF/MWh'],
        ['', '', '', ''],
        ['Negative Prices', f'{negative_hours} hrs ({negative_hours/n_hours*100:.1f}%)', 
         '>100 CHF/MWh', f'{high_hours} hrs ({high_hours/n_hours*100:.1f}%)'],
        ['Total Hours', f'{n_hours}', '>200 CHF/MWh', f'{very_high_hours} hrs ({very_high_hours/n_hours*100:.1f}%)'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Price Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved spot price distribution plot: {save_path}")
    
    return fig


def calculate_hourly_production_cost(
    ppu_production: Dict[str, Any],
    ppu_definitions: Dict[str, Any],
    total_production_series: Optional[np.ndarray] = None,
    renewable_production_series: Optional[np.ndarray] = None,
    dispatchable_production_series: Optional[np.ndarray] = None,
    storage_discharge: Optional[Dict[str, np.ndarray]] = None,
    storage_costs: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calculate the weighted average production cost per MWh at each hour.
    
    Handles two cases:
    1. ppu_production contains hourly arrays -> use directly
    2. ppu_production contains scalar totals -> use production time series with weighted costs
    
    Args:
        ppu_production: Dict mapping PPU name to production (array or scalar MWh)
        ppu_definitions: Dict of PPU definitions with cost_per_mwh attribute
        total_production_series: Optional hourly total production array (MW)
        renewable_production_series: Optional hourly renewable production array (MW)
        dispatchable_production_series: Optional hourly dispatchable production array (MW)
        storage_discharge: Optional dict of storage discharge per hour (MWh)
        storage_costs: Optional dict of storage effective costs (CHF/MWh)
        
    Returns:
        Array of hourly production costs (CHF/MWh)
    """
    # Check if ppu_production contains arrays or scalars
    has_arrays = False
    has_scalars = False
    
    for val in ppu_production.values():
        if isinstance(val, np.ndarray) and len(val) > 1:
            has_arrays = True
        elif isinstance(val, (int, float)):
            has_scalars = True
    
    # Get PPU costs
    # NOTE: cost_per_mwh in ppu_definitions is stored in CHF/kWh (not CHF/MWh as the name suggests)
    # Conversion: 1 CHF/kWh × 1000 kWh/MWh = 1000 CHF/MWh
    CHF_KWH_TO_CHF_MWH = 1000.0
    
    def get_ppu_cost(ppu_name: str) -> float:
        """Get PPU cost in CHF/MWh. CRITICAL: No fallback prices allowed."""
        ppu_def = ppu_definitions.get(ppu_name)
        if ppu_def is None:
            raise ValueError(
                f"CRITICAL: PPU '{ppu_name}' not found in ppu_definitions. "
                f"Cannot use fallback price as it would falsify costs. "
                f"Please ensure all PPUs are properly defined."
            )
        elif hasattr(ppu_def, 'cost_per_mwh'):
            # Convert from CHF/kWh to CHF/MWh
            return ppu_def.cost_per_mwh * CHF_KWH_TO_CHF_MWH
        elif isinstance(ppu_def, dict):
            if 'cost_per_mwh' not in ppu_def:
                raise ValueError(
                    f"CRITICAL: PPU '{ppu_name}' missing 'cost_per_mwh' in definition. "
                    f"Cannot use fallback price as it would falsify costs."
                )
            # Convert from CHF/kWh to CHF/MWh
            return ppu_def['cost_per_mwh'] * CHF_KWH_TO_CHF_MWH
        else:
            raise ValueError(
                f"CRITICAL: Cannot extract cost from PPU '{ppu_name}' definition. "
                f"Cannot use fallback price as it would falsify costs."
            )
    
    if has_arrays:
        # Original logic: use hourly arrays directly
        n_hours = 0
        for arr in ppu_production.values():
            if isinstance(arr, np.ndarray) and len(arr) > 1:
                n_hours = len(arr)
                break
        
        if n_hours == 0:
            return np.array([])
        
        hourly_costs = np.zeros(n_hours)
        total_prod = np.zeros(n_hours)
        
        for ppu_name, production_arr in ppu_production.items():
            if not isinstance(production_arr, np.ndarray):
                continue
            if len(production_arr) != n_hours:
                continue
            
            cost_per_mwh = get_ppu_cost(ppu_name)
            hourly_costs += production_arr * cost_per_mwh
            total_prod += production_arr
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_cost = np.where(total_prod > 0, hourly_costs / total_prod, 0.0)
        
        return avg_cost
    
    elif has_scalars and total_production_series is not None:
        # Use production time series with weighted average PPU cost
        n_hours = len(total_production_series)
        
        # Calculate total energy and weighted average cost from ppu_production
        total_energy = 0.0
        weighted_cost = 0.0
        
        for ppu_name, energy_mwh in ppu_production.items():
            if not isinstance(energy_mwh, (int, float)) or energy_mwh <= 0:
                continue
            cost_per_mwh = get_ppu_cost(ppu_name)
            weighted_cost += energy_mwh * cost_per_mwh
            total_energy += energy_mwh
        
        # Average cost from dispatchable sources
        if total_energy > 0:
            dispatchable_avg_cost = weighted_cost / total_energy
        else:
            # CRITICAL: Cannot calculate average without data - raise error
            raise ValueError(
                "CRITICAL: Cannot calculate dispatchable average cost - no dispatchable production data. "
                "Cannot use fallback price as it would falsify costs."
            )
        
        # Calculate average renewable cost (typically lower)
        renewable_cost = 0.0
        renewable_ppus = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R', 'BIO_WOOD']
        for ppu_name in renewable_ppus:
            if ppu_name in ppu_definitions:
                renewable_cost += get_ppu_cost(ppu_name)
        renewable_avg_cost = renewable_cost / len(renewable_ppus) if renewable_ppus else 30.0
        
        # Create hourly cost based on renewable vs dispatchable mix
        hourly_costs = np.zeros(n_hours)
        
        if renewable_production_series is not None and dispatchable_production_series is not None:
            # Use actual split between renewable and dispatchable
            total_prod = renewable_production_series + dispatchable_production_series
            
            for i in range(n_hours):
                if total_prod[i] > 0:
                    ren_frac = renewable_production_series[i] / total_prod[i]
                    disp_frac = dispatchable_production_series[i] / total_prod[i]
                    hourly_costs[i] = ren_frac * renewable_avg_cost + disp_frac * dispatchable_avg_cost
                else:
                    hourly_costs[i] = 0.0
        else:
            # Use uniform average cost
            avg_all = (renewable_avg_cost + dispatchable_avg_cost) / 2
            hourly_costs = np.where(total_production_series > 0, avg_all, 0.0)
        
        return hourly_costs
    
    else:
        # Fallback: return empty or zeros
        if total_production_series is not None:
            return np.zeros(len(total_production_series))
        return np.array([])


def plot_price_comparison_volatility(
    spot_prices: np.ndarray,
    production_costs: np.ndarray,
    title: str = "Price Comparison: Spot Market vs Portfolio Production Cost",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14),
) -> plt.Figure:
    """
    Compare spot market prices with portfolio production costs over time.
    
    Shows:
    1. Full year comparison (rolling averages)
    2. Daily patterns comparison
    3. Monthly comparison
    4. Price difference analysis
    5. Statistics summary
    
    Args:
        spot_prices: Hourly spot prices array (CHF/MWh)
        production_costs: Hourly portfolio production costs (CHF/MWh)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    n_hours = len(spot_prices)
    dates = pd.date_range(start='2024-01-01', periods=n_hours, freq='H')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # ===== SUBPLOT 1: Full Year Comparison with Rolling Averages =====
    ax1 = fig.add_subplot(gs[0, :])
    
    window_hours = 168  # 7-day rolling window
    
    # Rolling averages
    spot_rolling = pd.Series(spot_prices).rolling(window=window_hours, center=True).mean()
    prod_rolling = pd.Series(production_costs).rolling(window=window_hours, center=True).mean()
    
    ax1.plot(dates, spot_rolling, color=COLORS['secondary'], linewidth=2, 
             label=f'Spot Price (7-day avg): {np.mean(spot_prices):.1f} CHF/MWh mean')
    ax1.plot(dates, prod_rolling, color=COLORS['primary'], linewidth=2, 
             label=f'Production Cost (7-day avg): {np.mean(production_costs):.1f} CHF/MWh mean')
    
    # Fill between to show difference
    ax1.fill_between(dates, spot_rolling, prod_rolling, 
                     where=(spot_rolling > prod_rolling),
                     alpha=0.3, color=COLORS['success'], label='Savings (Prod < Spot)')
    ax1.fill_between(dates, spot_rolling, prod_rolling, 
                     where=(spot_rolling <= prod_rolling),
                     alpha=0.3, color=COLORS['danger'], label='Premium (Prod > Spot)')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax1.set_title('Full Year Price Comparison (7-Day Rolling Average)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 2: Daily Pattern Comparison =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    n_complete_days = n_hours // 24
    spot_daily = spot_prices[:n_complete_days*24].reshape(n_complete_days, 24)
    prod_daily = production_costs[:n_complete_days*24].reshape(n_complete_days, 24)
    
    hours = np.arange(24)
    spot_hourly_mean = [np.mean(spot_daily[:, h]) for h in range(24)]
    prod_hourly_mean = [np.mean(prod_daily[:, h]) for h in range(24)]
    spot_hourly_std = [np.std(spot_daily[:, h]) for h in range(24)]
    prod_hourly_std = [np.std(prod_daily[:, h]) for h in range(24)]
    
    ax2.plot(hours, spot_hourly_mean, color=COLORS['secondary'], linewidth=2, 
             marker='o', markersize=4, label='Spot Price')
    ax2.fill_between(hours, 
                     np.array(spot_hourly_mean) - np.array(spot_hourly_std),
                     np.array(spot_hourly_mean) + np.array(spot_hourly_std),
                     alpha=0.2, color=COLORS['secondary'])
    
    ax2.plot(hours, prod_hourly_mean, color=COLORS['primary'], linewidth=2, 
             marker='s', markersize=4, label='Production Cost')
    ax2.fill_between(hours, 
                     np.array(prod_hourly_mean) - np.array(prod_hourly_std),
                     np.array(prod_hourly_mean) + np.array(prod_hourly_std),
                     alpha=0.2, color=COLORS['primary'])
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Daily Pattern Comparison (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Monthly Comparison (Bar Chart) =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    df = pd.DataFrame({
        'spot': spot_prices, 
        'prod': production_costs, 
        'date': dates
    })
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    
    monthly = df.groupby('month').agg({
        'spot': 'mean',
        'prod': 'mean',
        'month_name': 'first'
    }).reset_index()
    
    x_pos = np.arange(len(monthly))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, monthly['spot'], width, 
                    label='Spot Price', color=COLORS['secondary'], alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, monthly['prod'], width, 
                    label='Production Cost', color=COLORS['primary'], alpha=0.8)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Monthly Average Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(monthly['month_name'], rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 4: Price Difference Over Time =====
    ax4 = fig.add_subplot(gs[2, :])
    
    price_diff = spot_prices - production_costs  # Positive = savings
    diff_rolling = pd.Series(price_diff).rolling(window=window_hours, center=True).mean()
    
    ax4.fill_between(dates, 0, diff_rolling, 
                     where=(diff_rolling >= 0),
                     alpha=0.5, color=COLORS['success'], label='Net Savings')
    ax4.fill_between(dates, 0, diff_rolling, 
                     where=(diff_rolling < 0),
                     alpha=0.5, color=COLORS['danger'], label='Net Premium')
    ax4.plot(dates, diff_rolling, color=COLORS['dark'], linewidth=1.5)
    
    ax4.axhline(np.mean(price_diff), color=COLORS['info'], linestyle='--', linewidth=2,
                label=f'Mean Difference: {np.mean(price_diff):.1f} CHF/MWh')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Spot - Production (CHF/MWh)', fontsize=11, fontweight='bold')
    ax4.set_title('Price Difference Over Time (Positive = Savings vs Spot)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 5: Statistics Summary =====
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    # Calculate statistics
    spot_mean = np.mean(spot_prices)
    prod_mean = np.mean(production_costs)
    diff_mean = np.mean(price_diff)
    
    hours_saving = np.sum(price_diff > 0)
    hours_premium = np.sum(price_diff < 0)
    
    total_production_mwh = n_hours * 7.0  # Approximate average hourly production in GW
    annual_savings = diff_mean * total_production_mwh / 1e6  # Million CHF
    
    correlation = np.corrcoef(spot_prices, production_costs)[0, 1]
    
    stats_text = f"""
    PRICE COMPARISON STATISTICS
    {'='*90}
    
    SPOT MARKET                           PORTFOLIO PRODUCTION                 COMPARISON
    {'─'*90}
    Mean:     {spot_mean:>8.2f} CHF/MWh          Mean:     {prod_mean:>8.2f} CHF/MWh          Difference: {diff_mean:>+8.2f} CHF/MWh
    Std Dev:  {np.std(spot_prices):>8.2f} CHF/MWh          Std Dev:  {np.std(production_costs):>8.2f} CHF/MWh          Correlation: {correlation:>7.3f}
    Median:   {np.median(spot_prices):>8.2f} CHF/MWh          Median:   {np.median(production_costs):>8.2f} CHF/MWh          
    Min:      {np.min(spot_prices):>8.2f} CHF/MWh          Min:      {np.min(production_costs):>8.2f} CHF/MWh          Hours Saving: {hours_saving:>6d} ({hours_saving/n_hours*100:.1f}%)
    Max:      {np.max(spot_prices):>8.2f} CHF/MWh          Max:      {np.max(production_costs):>8.2f} CHF/MWh          Hours Premium: {hours_premium:>5d} ({hours_premium/n_hours*100:.1f}%)
    
    {'─'*90}
    INTERPRETATION:
    • Positive difference (Spot > Production) = Portfolio produces energy cheaper than market
    • Negative difference (Spot < Production) = Portfolio costs more than buying from market
    • Annual Impact: ~{annual_savings:+.1f} M CHF {"SAVINGS" if annual_savings > 0 else "PREMIUM"} vs buying all from spot market
    """
    
    ax5.text(0.02, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved price comparison volatility plot: {save_path}")
    
    return fig


def plot_price_comparison_distribution(
    spot_prices: np.ndarray,
    production_costs: np.ndarray,
    title: str = "Price Distribution Comparison: Spot vs Portfolio Production",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Compare distributions of spot prices and production costs.
    
    Shows:
    1. Overlaid histograms with KDE
    2. Overlaid CDFs
    3. Side-by-side box plots
    4. Statistics comparison table
    
    Args:
        spot_prices: Hourly spot prices array (CHF/MWh)
        production_costs: Hourly portfolio production costs (CHF/MWh)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    from scipy import stats
    
    n_hours = len(spot_prices)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ===== SUBPLOT 1: Overlaid Histograms with KDE =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Determine common bin range
    min_val = min(np.min(spot_prices), np.min(production_costs))
    max_val = max(np.max(spot_prices), np.max(production_costs))
    bins = np.linspace(min_val, max_val, 80)
    
    # Histograms
    ax1.hist(spot_prices, bins=bins, density=True, alpha=0.5, 
             color=COLORS['secondary'], label='Spot Price', edgecolor='white')
    ax1.hist(production_costs, bins=bins, density=True, alpha=0.5, 
             color=COLORS['primary'], label='Production Cost', edgecolor='white')
    
    # KDE curves
    x_kde = np.linspace(min_val, max_val, 500)
    
    kde_spot = stats.gaussian_kde(spot_prices)
    kde_prod = stats.gaussian_kde(production_costs[production_costs > 0])  # Exclude zeros for KDE
    
    ax1.plot(x_kde, kde_spot(x_kde), color=COLORS['secondary'], linewidth=2.5, linestyle='-')
    ax1.plot(x_kde, kde_prod(x_kde), color=COLORS['primary'], linewidth=2.5, linestyle='-')
    
    # Mark means
    ax1.axvline(np.mean(spot_prices), color=COLORS['secondary'], linestyle='--', 
                linewidth=2, label=f'Spot Mean: {np.mean(spot_prices):.1f}')
    ax1.axvline(np.mean(production_costs), color=COLORS['primary'], linestyle='--', 
                linewidth=2, label=f'Prod Mean: {np.mean(production_costs):.1f}')
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Price Distributions (Histogram + KDE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: Overlaid CDFs =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    spot_sorted = np.sort(spot_prices)
    prod_sorted = np.sort(production_costs)
    cdf = np.arange(1, n_hours + 1) / n_hours
    
    ax2.plot(spot_sorted, cdf, color=COLORS['secondary'], linewidth=2, label='Spot Price')
    ax2.plot(prod_sorted, cdf, color=COLORS['primary'], linewidth=2, label='Production Cost')
    
    # Mark key percentiles
    for p in [25, 50, 75]:
        spot_val = np.percentile(spot_prices, p)
        prod_val = np.percentile(production_costs, p)
        ax2.axhline(p/100, color='gray', linestyle=':', alpha=0.5)
        ax2.scatter([spot_val], [p/100], color=COLORS['secondary'], s=40, zorder=5)
        ax2.scatter([prod_val], [p/100], color=COLORS['primary'], s=40, zorder=5)
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distribution Functions', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # ===== SUBPLOT 3: Side-by-Side Box Plots =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    bp = ax3.boxplot([spot_prices, production_costs], 
                     labels=['Spot Price', 'Production Cost'],
                     patch_artist=True, showmeans=True,
                     meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markersize': 8})
    
    bp['boxes'][0].set_facecolor(COLORS['secondary'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['primary'])
    bp['boxes'][1].set_alpha(0.7)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Price (CHF/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Price Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentile annotations
    for i, (data, x) in enumerate([(spot_prices, 1), (production_costs, 2)]):
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)
        ax3.annotate(f'P5: {p5:.0f}', (x + 0.2, p5), fontsize=8)
        ax3.annotate(f'P95: {p95:.0f}', (x + 0.2, p95), fontsize=8)
    
    # ===== SUBPLOT 4: Statistics Comparison Table =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    price_diff = spot_prices - production_costs
    
    table_data = [
        ['Metric', 'Spot Price', 'Production Cost', 'Difference'],
        ['Mean', f'{np.mean(spot_prices):.2f}', f'{np.mean(production_costs):.2f}', f'{np.mean(price_diff):+.2f}'],
        ['Median', f'{np.median(spot_prices):.2f}', f'{np.median(production_costs):.2f}', f'{np.median(price_diff):+.2f}'],
        ['Std Dev', f'{np.std(spot_prices):.2f}', f'{np.std(production_costs):.2f}', '-'],
        ['P5', f'{np.percentile(spot_prices, 5):.2f}', f'{np.percentile(production_costs, 5):.2f}', '-'],
        ['P25', f'{np.percentile(spot_prices, 25):.2f}', f'{np.percentile(production_costs, 25):.2f}', '-'],
        ['P75', f'{np.percentile(spot_prices, 75):.2f}', f'{np.percentile(production_costs, 75):.2f}', '-'],
        ['P95', f'{np.percentile(spot_prices, 95):.2f}', f'{np.percentile(production_costs, 95):.2f}', '-'],
        ['Min', f'{np.min(spot_prices):.2f}', f'{np.min(production_costs):.2f}', '-'],
        ['Max', f'{np.max(spot_prices):.2f}', f'{np.max(production_costs):.2f}', '-'],
        ['', '', '', ''],
        ['< 0 CHF/MWh', f'{np.sum(spot_prices < 0)} hrs', f'{np.sum(production_costs < 0)} hrs', '-'],
        ['> 100 CHF/MWh', f'{np.sum(spot_prices > 100)} hrs', f'{np.sum(production_costs > 100)} hrs', '-'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.22, 0.26, 0.26, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor(COLORS['dark'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color code the difference column
    for i in range(1, len(table_data)):
        if table_data[i][3].startswith('+'):
            table[(i, 3)].set_text_props(color=COLORS['success'], fontweight='bold')
        elif table_data[i][3].startswith('-') and table_data[i][3] != '-':
            table[(i, 3)].set_text_props(color=COLORS['danger'], fontweight='bold')
        
        # Alternate row colors
        if i % 2 == 0:
            for j in range(4):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Price Statistics Comparison (all values in CHF/MWh)', 
                  fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved price comparison distribution plot: {save_path}")
    
    return fig


def plot_ppu_cost_breakdown(
    ppu_constructs_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    title: str = "PPU Cost Breakdown by Component",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14),
    top_n_ppus: int = 15,
) -> plt.Figure:
    """
    Visualize cost breakdown by components for each PPU to identify expensive components.
    
    Shows:
    1. Stacked bar chart of component costs per PPU
    2. Top most expensive components across all PPUs
    3. Cost accumulation through chains
    4. Efficiency losses vs costs
    
    Args:
        ppu_constructs_df: DataFrame with PPU definitions (columns: PPU, Components, Category)
        cost_df: Cost table DataFrame (indexed by item name)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        top_n_ppus: Number of top expensive PPUs to show
        
    Returns:
        matplotlib Figure object
    """
    import ast
    
    # Ensure cost_df is indexed by item
    if 'item' in cost_df.columns and cost_df.index.name != 'item':
        cost_df = cost_df.set_index('item', drop=False)
    
    # Calculate cost breakdown for each PPU
    ppu_costs = []
    component_totals = {}
    
    for _, row in ppu_constructs_df.iterrows():
        ppu_name = row['PPU']
        components_val = row.get('Components', '')
        # Handle potential array from duplicate indices
        if hasattr(components_val, '__iter__') and not isinstance(components_val, str):
            components_val = components_val.iloc[0] if hasattr(components_val, 'iloc') else str(components_val)
        if pd.isna(components_val) or components_val == '':
            continue
            
        components = components_val
        if isinstance(components, str):
            components = ast.literal_eval(components)
        
        category = row.get('Category', 'Unknown')
        
        total_cost = 0.0
        cumulative_efficiency = 1.0
        component_breakdown = []
        
        for comp in components:
            if comp in cost_df.index:
                comp_row = cost_df.loc[comp]
                comp_cost = comp_row['cost'] if not pd.isna(comp_row['cost']) else 0.0
                comp_eff = comp_row['efficiency'] if not pd.isna(comp_row['efficiency']) else 1.0
                comp_w = comp_row['w'] if not pd.isna(comp_row.get('w', np.nan)) else 0.0
                
                # Auxiliary energy cost
                aux_cost = 0.0
                if comp_w > 0 and cumulative_efficiency > 0:
                    aux_cost = comp_w / cumulative_efficiency
                
                component_cost_total = comp_cost + aux_cost
                total_cost += component_cost_total
                cumulative_efficiency *= comp_eff
                
                component_breakdown.append({
                    'component': comp,
                    'direct_cost': comp_cost,
                    'aux_cost': aux_cost,
                    'total_cost': component_cost_total,
                    'efficiency': comp_eff,
                })
                
                # Track component totals
                if comp not in component_totals:
                    component_totals[comp] = {'total_cost': 0.0, 'count': 0, 'direct_cost': 0.0, 'aux_cost': 0.0}
                component_totals[comp]['total_cost'] += component_cost_total
                component_totals[comp]['direct_cost'] += comp_cost
                component_totals[comp]['aux_cost'] += aux_cost
                component_totals[comp]['count'] += 1
            else:
                component_breakdown.append({
                    'component': comp,
                    'direct_cost': 0.0,
                    'aux_cost': 0.0,
                    'total_cost': 0.0,
                    'efficiency': 1.0,
                })
        
        ppu_costs.append({
            'ppu': ppu_name,
            'category': category,
            'total_cost_chf_kwh': total_cost,
            'total_cost_chf_mwh': total_cost * 1000,
            'efficiency': cumulative_efficiency,
            'breakdown': component_breakdown,
            'n_components': len(components),
        })
    
    # Sort by total cost
    ppu_costs.sort(key=lambda x: x['total_cost_chf_kwh'], reverse=True)
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1])
    
    # ===== SUBPLOT 1: Stacked Bar Chart - Cost by Component per PPU =====
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get top N PPUs
    top_ppus = ppu_costs[:top_n_ppus]
    ppu_names = [p['ppu'] for p in top_ppus]
    
    # Get all unique components in order of appearance
    all_components = []
    for p in top_ppus:
        for c in p['breakdown']:
            if c['component'] not in all_components:
                all_components.append(c['component'])
    
    # Color palette for components
    n_components = len(all_components)
    cmap = plt.cm.get_cmap('tab20', n_components)
    comp_colors = {comp: cmap(i) for i, comp in enumerate(all_components)}
    
    # Build stacked data
    x = np.arange(len(top_ppus))
    bottoms = np.zeros(len(top_ppus))
    
    for comp in all_components:
        heights = []
        for p in top_ppus:
            cost = 0.0
            for c in p['breakdown']:
                if c['component'] == comp:
                    cost = c['total_cost'] * 1000  # Convert to CHF/MWh
                    break
            heights.append(cost)
        
        ax1.bar(x, heights, bottom=bottoms, label=comp, color=comp_colors[comp], edgecolor='white', linewidth=0.5)
        bottoms += np.array(heights)
    
    ax1.set_xlabel('PPU', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cost (CHF/MWh)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Cost Breakdown by Component - Top {top_n_ppus} Most Expensive PPUs', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ppu_names, rotation=45, ha='right')
    
    # Add total cost labels
    for i, p in enumerate(top_ppus):
        ax1.annotate(f'{p["total_cost_chf_mwh"]:.0f}', 
                     (i, p['total_cost_chf_mwh'] + 5), 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Legend outside
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== SUBPLOT 2: Top Most Expensive Components =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Sort components by total cost contribution
    comp_sorted = sorted(component_totals.items(), key=lambda x: x[1]['total_cost'], reverse=True)[:15]
    comp_names = [c[0] for c in comp_sorted]
    comp_costs = [c[1]['total_cost'] * 1000 for c in comp_sorted]  # CHF/MWh
    comp_direct = [c[1]['direct_cost'] * 1000 for c in comp_sorted]
    comp_aux = [c[1]['aux_cost'] * 1000 for c in comp_sorted]
    
    y_pos = np.arange(len(comp_names))
    
    bars1 = ax2.barh(y_pos, comp_direct, label='Direct Cost', color=COLORS['primary'], alpha=0.8)
    bars2 = ax2.barh(y_pos, comp_aux, left=comp_direct, label='Auxiliary Energy Cost', color=COLORS['secondary'], alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(comp_names, fontsize=9)
    ax2.set_xlabel('Total Cost Contribution (CHF/MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 15 Most Expensive Components (summed across all PPUs)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # ===== SUBPLOT 3: Single Component Cost (Unit Cost) =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Get unit cost per component (not summed across PPUs)
    comp_unit_costs = []
    for comp, totals in component_totals.items():
        if totals['count'] > 0:
            avg_cost = (totals['direct_cost'] / totals['count']) * 1000  # CHF/MWh per instance
            comp_unit_costs.append((comp, avg_cost, totals['count']))
    
    comp_unit_costs.sort(key=lambda x: x[1], reverse=True)
    comp_unit_costs = comp_unit_costs[:15]
    
    comp_names_unit = [c[0] for c in comp_unit_costs]
    comp_costs_unit = [c[1] for c in comp_unit_costs]
    comp_counts = [c[2] for c in comp_unit_costs]
    
    y_pos = np.arange(len(comp_names_unit))
    bars = ax3.barh(y_pos, comp_costs_unit, color=COLORS['accent'], alpha=0.8)
    
    # Add count annotations
    for i, (cost, count) in enumerate(zip(comp_costs_unit, comp_counts)):
        ax3.annotate(f'(used in {count} PPUs)', (cost + 2, i), va='center', fontsize=8, color='gray')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(comp_names_unit, fontsize=9)
    ax3.set_xlabel('Unit Cost (CHF/MWh per instance)', fontsize=11, fontweight='bold')
    ax3.set_title('Top 15 Highest Unit Cost Components', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # ===== SUBPLOT 4: Efficiency vs Cost Trade-off =====
    ax4 = fig.add_subplot(gs[2, 0])
    
    efficiencies = [p['efficiency'] for p in ppu_costs]
    costs = [p['total_cost_chf_mwh'] for p in ppu_costs]
    categories = [p['category'] for p in ppu_costs]
    names = [p['ppu'] for p in ppu_costs]
    
    # Color by category
    cat_colors = {'Production': COLORS['success'], 'Storage': COLORS['info'], 'Unknown': COLORS['warning']}
    colors = [cat_colors.get(c, COLORS['warning']) for c in categories]
    
    scatter = ax4.scatter(efficiencies, costs, c=colors, s=80, alpha=0.7, edgecolors='white')
    
    # Add labels for top 10 expensive
    for i, (eff, cost, name) in enumerate(zip(efficiencies, costs, names)):
        if i < 10:  # Top 10 most expensive
            ax4.annotate(name, (eff, cost), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Chain Efficiency', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Cost (CHF/MWh)', fontsize=11, fontweight='bold')
    ax4.set_title('Efficiency vs Cost Trade-off', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[cat], label=cat) for cat in cat_colors]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # ===== SUBPLOT 5: Summary Statistics Table =====
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    avg_cost = np.mean([p['total_cost_chf_mwh'] for p in ppu_costs])
    max_cost = max([p['total_cost_chf_mwh'] for p in ppu_costs])
    min_cost = min([p['total_cost_chf_mwh'] for p in ppu_costs])
    max_ppu = next(p['ppu'] for p in ppu_costs if p['total_cost_chf_mwh'] == max_cost)
    min_ppu = next(p['ppu'] for p in ppu_costs if p['total_cost_chf_mwh'] == min_cost)
    
    # Most expensive component overall
    most_expensive_comp = comp_sorted[0][0] if comp_sorted else 'N/A'
    most_expensive_comp_cost = comp_sorted[0][1]['total_cost'] * 1000 if comp_sorted else 0
    
    stats_text = f"""
    COST BREAKDOWN SUMMARY
    {'='*60}
    
    PPU Statistics:
      • Total PPUs analyzed:     {len(ppu_costs)}
      • Average PPU cost:        {avg_cost:.1f} CHF/MWh
      • Most expensive PPU:      {max_ppu} ({max_cost:.1f} CHF/MWh)
      • Least expensive PPU:     {min_ppu} ({min_cost:.1f} CHF/MWh)
    
    Component Statistics:
      • Total unique components: {len(component_totals)}
      • Most expensive component:{most_expensive_comp}
        (contributes {most_expensive_comp_cost:.1f} CHF/MWh total)
    
    Top 5 Cost Drivers:
    {'─'*60}
    """
    
    for i, (comp, totals) in enumerate(comp_sorted[:5]):
        stats_text += f"    {i+1}. {comp:30s}: {totals['total_cost']*1000:>8.1f} CHF/MWh\n"
    
    stats_text += f"""
    {'─'*60}
    
    For comparison:
      • Spot market average:     ~76 CHF/MWh
      • Your cheapest PPU:       {min_cost:.1f} CHF/MWh
    """
    
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved PPU cost breakdown plot: {save_path}")
    
    return fig


def plot_portfolio_cost_breakdown(
    portfolio: Dict[str, int],
    ppu_definitions: Dict[str, Any],
    ppu_constructs_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    title: str = "Portfolio Cost Breakdown by Component",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 16),
) -> plt.Figure:
    """
    Visualize cost breakdown for a specific PPU portfolio, weighted by PPU counts.
    
    Shows which components across all PPUs in the portfolio are responsible for costs.
    
    Args:
        portfolio: Dict of PPU name -> count (e.g., {'PV': 500, 'WD_ON': 85})
        ppu_definitions: Dict of PPU definitions with cost and efficiency
        ppu_constructs_df: DataFrame with PPU component definitions
        cost_df: Cost table DataFrame
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    import ast
    
    # Ensure cost_df is indexed by item
    if 'item' in cost_df.columns and cost_df.index.name != 'item':
        cost_df = cost_df.set_index('item', drop=False)
    
    # Build component lookup from constructs
    ppu_components = {}
    for _, row in ppu_constructs_df.iterrows():
        ppu_name = row['PPU']
        components_val = row.get('Components', '')
        if hasattr(components_val, '__iter__') and not isinstance(components_val, str):
            components_val = components_val.iloc[0] if hasattr(components_val, 'iloc') else str(components_val)
        if pd.isna(components_val) or components_val == '':
            continue
        components = components_val
        if isinstance(components, str):
            components = ast.literal_eval(components)
        ppu_components[ppu_name] = components
    
    # Calculate weighted costs per component across portfolio
    component_costs_weighted = {}  # component -> {'direct': total, 'aux': total, 'ppu_count': count}
    ppu_cost_breakdown = {}  # ppu_name -> {'total_cost': x, 'count': n, 'component_costs': {...}}
    
    total_portfolio_cost = 0.0
    total_ppu_count = sum(portfolio.values())
    
    CHF_KWH_TO_CHF_MWH = 1000.0  # Costs stored in CHF/kWh, convert to CHF/MWh
    
    for ppu_name, ppu_count in portfolio.items():
        if ppu_count <= 0:
            continue
            
        if ppu_name not in ppu_components:
            continue
            
        components = ppu_components[ppu_name]
        ppu_def = ppu_definitions.get(ppu_name)
        
        # Calculate component costs for this PPU
        cumulative_efficiency = 1.0
        ppu_component_costs = {}
        ppu_total_cost = 0.0
        
        for comp in components:
            if comp in cost_df.index:
                comp_row = cost_df.loc[comp]
                # Handle duplicate indices
                if hasattr(comp_row, 'iloc') and len(comp_row.shape) > 1:
                    comp_row = comp_row.iloc[0]
                    
                comp_cost = float(comp_row['cost']) if not pd.isna(comp_row['cost']) else 0.0
                comp_eff = float(comp_row['efficiency']) if not pd.isna(comp_row['efficiency']) else 1.0
                comp_w = float(comp_row.get('w', 0)) if not pd.isna(comp_row.get('w', np.nan)) else 0.0
                
                # Auxiliary energy cost
                aux_cost = 0.0
                if comp_w > 0 and cumulative_efficiency > 0:
                    aux_cost = comp_w / cumulative_efficiency
                
                total_comp_cost = (comp_cost + aux_cost) * CHF_KWH_TO_CHF_MWH  # Convert to CHF/MWh
                cumulative_efficiency *= comp_eff
                
                ppu_component_costs[comp] = {
                    'direct': comp_cost * CHF_KWH_TO_CHF_MWH,
                    'aux': aux_cost * CHF_KWH_TO_CHF_MWH,
                    'total': total_comp_cost,
                }
                ppu_total_cost += total_comp_cost
                
                # Accumulate weighted by PPU count
                if comp not in component_costs_weighted:
                    component_costs_weighted[comp] = {'direct': 0.0, 'aux': 0.0, 'total': 0.0, 'ppu_usage': []}
                
                component_costs_weighted[comp]['direct'] += comp_cost * CHF_KWH_TO_CHF_MWH * ppu_count
                component_costs_weighted[comp]['aux'] += aux_cost * CHF_KWH_TO_CHF_MWH * ppu_count
                component_costs_weighted[comp]['total'] += total_comp_cost * ppu_count
                component_costs_weighted[comp]['ppu_usage'].append((ppu_name, ppu_count))
        
        ppu_cost_breakdown[ppu_name] = {
            'total_cost': ppu_total_cost,
            'count': ppu_count,
            'weighted_cost': ppu_total_cost * ppu_count,
            'component_costs': ppu_component_costs,
            'efficiency': cumulative_efficiency,
        }
        
        total_portfolio_cost += ppu_total_cost * ppu_count
    
    # Sort data for plotting
    sorted_components = sorted(component_costs_weighted.items(), 
                               key=lambda x: x[1]['total'], reverse=True)
    sorted_ppus = sorted(ppu_cost_breakdown.items(), 
                         key=lambda x: x[1]['weighted_cost'], reverse=True)
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1.2])
    
    # ===== SUBPLOT 1: Top Component Cost Contributors (Weighted) =====
    ax1 = fig.add_subplot(gs[0, :])
    
    top_components = sorted_components[:20]
    comp_names = [c[0] for c in top_components]
    comp_direct = [c[1]['direct'] for c in top_components]
    comp_aux = [c[1]['aux'] for c in top_components]
    
    x = np.arange(len(comp_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comp_direct, width, label='Direct Cost', color=COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, comp_aux, width, label='Auxiliary Energy Cost', color=COLORS['secondary'], alpha=0.8)
    
    ax1.set_xlabel('Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Weighted Cost (CHF/MWh × PPU count)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 20 Component Cost Contributors Across Your Portfolio', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (name, data) in enumerate(top_components):
        pct = (data['total'] / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        ax1.annotate(f'{pct:.1f}%', (i, data['direct'] + data['aux']), 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ===== SUBPLOT 2: PPU Cost Contribution (Weighted by count) =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    top_ppus = sorted_ppus[:15]
    ppu_names = [f"{p[0]} (×{p[1]['count']})" for p in top_ppus]
    ppu_weighted_costs = [p[1]['weighted_cost'] for p in top_ppus]
    
    colors = [COLORS['success'] if ppu_definitions.get(p[0], {}) 
              and getattr(ppu_definitions.get(p[0]), 'category', '') == 'Production' 
              else COLORS['info'] for p in top_ppus]
    
    y_pos = np.arange(len(ppu_names))
    bars = ax2.barh(y_pos, ppu_weighted_costs, color=colors, alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ppu_names, fontsize=9)
    ax2.set_xlabel('Weighted Cost (CHF/MWh × count)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 15 PPU Cost Contributors in Portfolio', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # ===== SUBPLOT 3: Unit Cost per PPU =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    ppu_unit_costs = [(name, data['total_cost'], data['count']) 
                      for name, data in sorted_ppus]
    ppu_unit_costs.sort(key=lambda x: x[1], reverse=True)
    ppu_unit_costs = ppu_unit_costs[:15]
    
    ppu_names_unit = [f"{p[0]}" for p in ppu_unit_costs]
    ppu_costs_unit = [p[1] for p in ppu_unit_costs]
    ppu_counts = [p[2] for p in ppu_unit_costs]
    
    y_pos = np.arange(len(ppu_names_unit))
    bars = ax3.barh(y_pos, ppu_costs_unit, color=COLORS['accent'], alpha=0.8)
    
    for i, (cost, count) in enumerate(zip(ppu_costs_unit, ppu_counts)):
        ax3.annotate(f'×{count}', (cost + 5, i), va='center', fontsize=9, fontweight='bold')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(ppu_names_unit, fontsize=9)
    ax3.set_xlabel('Unit Cost per PPU (CHF/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('PPU Unit Costs (Most Expensive First)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # ===== SUBPLOT 4: Component Usage Heatmap =====
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Build matrix of component usage across top PPUs
    top_ppu_names = [p[0] for p in sorted_ppus[:10]]
    top_comp_names = [c[0] for c in sorted_components[:15]]
    
    usage_matrix = np.zeros((len(top_ppu_names), len(top_comp_names)))
    
    for i, ppu_name in enumerate(top_ppu_names):
        ppu_data = ppu_cost_breakdown.get(ppu_name, {})
        comp_costs = ppu_data.get('component_costs', {})
        for j, comp_name in enumerate(top_comp_names):
            if comp_name in comp_costs:
                usage_matrix[i, j] = comp_costs[comp_name]['total']
    
    im = ax4.imshow(usage_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(np.arange(len(top_comp_names)))
    ax4.set_yticks(np.arange(len(top_ppu_names)))
    ax4.set_xticklabels(top_comp_names, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(top_ppu_names, fontsize=9)
    ax4.set_title('Component Cost per PPU (CHF/MWh)', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Cost (CHF/MWh)', fontsize=9)
    
    # ===== SUBPLOT 5: Summary Statistics =====
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    avg_ppu_cost = total_portfolio_cost / total_ppu_count if total_ppu_count > 0 else 0
    most_expensive_ppu = sorted_ppus[0] if sorted_ppus else ('N/A', {'total_cost': 0, 'count': 0})
    most_expensive_comp = sorted_components[0] if sorted_components else ('N/A', {'total': 0})
    
    stats_text = f"""
    PORTFOLIO COST BREAKDOWN SUMMARY
    {'='*60}
    
    Portfolio Overview:
      • Total PPU count:           {total_ppu_count:,}
      • Total portfolio cost:      {total_portfolio_cost:,.0f} CHF/MWh (weighted sum)
      • Average cost per PPU:      {avg_ppu_cost:.1f} CHF/MWh
    
    Most Expensive PPU Type:
      • {most_expensive_ppu[0]}: {most_expensive_ppu[1]['total_cost']:.1f} CHF/MWh
        (×{most_expensive_ppu[1]['count']} units = {most_expensive_ppu[1]['weighted_cost']:,.0f} total)
    
    Top 5 Cost-Driving Components:
    {'─'*60}
    """
    
    for i, (comp, data) in enumerate(sorted_components[:5]):
        pct = (data['total'] / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        stats_text += f"    {i+1}. {comp:30s}: {data['total']:>10,.0f} ({pct:>5.1f}%)\n"
    
    stats_text += f"""
    {'─'*60}
    
    Top 5 PPU Types by Total Cost:
    {'─'*60}
    """
    
    for i, (ppu_name, data) in enumerate(sorted_ppus[:5]):
        pct = (data['weighted_cost'] / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        stats_text += f"    {i+1}. {ppu_name:15s} ×{data['count']:>3}: {data['weighted_cost']:>10,.0f} ({pct:>5.1f}%)\n"
    
    stats_text += f"""
    {'─'*60}
    
    For comparison:
      • Spot market average:       ~76 CHF/MWh
      • Portfolio weighted avg:    {avg_ppu_cost:.1f} CHF/MWh
    """
    
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved portfolio cost breakdown plot: {save_path}")
    
    return fig


def plot_energy_balance_distribution(
    demand: np.ndarray,
    production: np.ndarray,
    renewable_production: Optional[np.ndarray] = None,
    spot_bought: Optional[np.ndarray] = None,
    spot_sold: Optional[np.ndarray] = None,
    include_spot_in_balance: bool = False,  # Default to False to see portfolio balance
    title: str = "Energy Balance Distribution - Portfolio Performance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
) -> plt.Figure:
    """
    Visualize the distribution of energy surplus and deficit over time.
    
    Shows:
    1. Time series of net balance (production - demand)
    2. Distribution histogram of surplus vs deficit hours
    3. When renewable (incidence) energy alone satisfies demand
    4. Monthly breakdown of surplus/deficit
    5. Daily pattern analysis
    
    Args:
        demand: Hourly demand array (MW or MWh)
        production: Hourly production array (MW or MWh)
        renewable_production: Optional hourly renewable production array (MW or MWh)
        spot_bought: Optional hourly spot market purchases (MW or MWh)
        spot_sold: Optional hourly spot market sales (MW or MWh)
        include_spot_in_balance: If True, includes spot market in the balance (usually results in 0 balance)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # ... (keep validation logic) ...
    
    # Calculate net balance (positive = surplus, negative = deficit)
    total_supply = production.copy()
    if spot_bought is not None and len(spot_bought) == len(production):
        total_supply = total_supply + spot_bought
    if spot_sold is not None and len(spot_sold) == len(production):
        total_supply = total_supply - spot_sold

    if include_spot_in_balance:
        # Calculate total supply (production + spot purchases - spot sales)
        # This gives the actual energy available to meet demand (usually matches demand)
        net_balance = total_supply - demand
    else:
        # Calculate portfolio balance (production - demand)
        # This shows the actual performance of the portfolio before spot market
        net_balance = production - demand
    
    n_hours = len(net_balance)
    hours = np.arange(n_hours)
    
    # Debug: Check if production and demand are identical (would show 100% balanced)
    if not include_spot_in_balance and np.allclose(production, demand, rtol=1e-5):
        print(f"⚠️ WARNING: production and demand are nearly identical!")
        print(f"   Production range: [{np.min(production):.2f}, {np.max(production):.2f}]")
        print(f"   Demand range: [{np.min(demand):.2f}, {np.max(demand):.2f}]")
        if spot_bought is not None:
            print(f"   Spot bought: {np.sum(spot_bought):.2f} MWh total")
        if spot_sold is not None:
            print(f"   Spot sold: {np.sum(spot_sold):.2f} MWh total")
        print(f"   Using total_supply (production + spot_bought - spot_sold) for balance calculation")
    
    # Classify each hour (use a small tolerance for "balanced")
    tolerance = 1e-3  # 1 kW tolerance
    surplus_mask = net_balance > tolerance
    deficit_mask = net_balance < -tolerance
    balanced_mask = np.abs(net_balance) <= tolerance
    
    surplus_hours = np.sum(surplus_mask)
    deficit_hours = np.sum(deficit_mask)
    balanced_hours = np.sum(balanced_mask)
    
    # Debug output (only if suspicious)
    if balanced_hours > n_hours * 0.9:  # More than 90% balanced is suspicious
        print(f"⚠️ WARNING: {100*balanced_hours/n_hours:.1f}% of hours are 'balanced' - this may indicate a data issue")
        print(f"   DEBUG: n_hours={n_hours}, surplus={surplus_hours}, deficit={deficit_hours}, balanced={balanced_hours}")
        print(f"   DEBUG: Production stats: min={np.min(production):.2f}, max={np.max(production):.2f}, mean={np.mean(production):.2f}")
        print(f"   DEBUG: Total supply stats: min={np.min(total_supply):.2f}, max={np.max(total_supply):.2f}, mean={np.mean(total_supply):.2f}")
        print(f"   DEBUG: Demand stats: min={np.min(demand):.2f}, max={np.max(demand):.2f}, mean={np.mean(demand):.2f}")
        print(f"   DEBUG: Net balance stats: min={np.min(net_balance):.2f}, max={np.max(net_balance):.2f}, mean={np.mean(net_balance):.2f}")
        if spot_bought is not None:
            print(f"   DEBUG: Spot bought total: {np.sum(spot_bought):.2f} MWh")
        if spot_sold is not None:
            print(f"   DEBUG: Spot sold total: {np.sum(spot_sold):.2f} MWh")
    
    # Calculate renewable-only balance if provided
    renewable_sufficient = None
    renewable_sufficient_hours = 0
    renewable_balance = None
    if renewable_production is not None and len(renewable_production) == n_hours:
        renewable_balance = renewable_production - demand
        renewable_sufficient = renewable_balance >= 0  # Renewable alone satisfies demand
        renewable_sufficient_hours = np.sum(renewable_sufficient)
    
    # Create figure - adjust grid for renewable visualization
    fig = plt.figure(figsize=figsize, facecolor=COLORS['light'])
    if renewable_production is not None:
        # 4 rows: time series, histogram, renewable analysis, monthly/daily
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1, 1, 1])
    else:
        # Original 3 rows
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1, 1])
    
    # ===== SUBPLOT 1: Time Series of Net Balance =====
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create color array for fill (use total_supply for display)
    surplus_fill = np.where(net_balance > 0, net_balance, 0)
    deficit_fill = np.where(net_balance < 0, net_balance, 0)
    
    # Convert to days for x-axis
    days = hours / 24
    
    ax1.fill_between(days, 0, surplus_fill / 1000, color=COLORS['success'], alpha=0.6, label='Surplus')
    ax1.fill_between(days, 0, deficit_fill / 1000, color=COLORS['danger'], alpha=0.6, label='Deficit')
    ax1.axhline(y=0, color='black', linewidth=1, linestyle='-')
    
    ax1.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Net Balance (GW)', fontsize=11, fontweight='bold')
    ax1.set_title('Energy Balance Over Time (Production - Demand)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_hours / 24)
    
    # Add month markers
    if n_hours >= 8760:
        month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, (start, name) in enumerate(zip(month_starts, month_names)):
            ax1.axvline(x=start, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
            ax1.text(start + 15, ax1.get_ylim()[1] * 0.95, name, fontsize=8, ha='center', alpha=0.7)
    
    # ===== SUBPLOT 2: Distribution Histogram =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create histogram
    bins = 100
    surplus_data = net_balance[surplus_mask] / 1000  # Convert to GW
    deficit_data = net_balance[deficit_mask] / 1000
    
    if len(surplus_data) > 0:
        ax2.hist(surplus_data, bins=bins//2, color=COLORS['success'], alpha=0.7, 
                 label=f'Surplus hours ({surplus_hours:,})', edgecolor='white')
    if len(deficit_data) > 0:
        ax2.hist(deficit_data, bins=bins//2, color=COLORS['danger'], alpha=0.7,
                 label=f'Deficit hours ({deficit_hours:,})', edgecolor='white')
    
    ax2.axvline(x=0, color='black', linewidth=2, linestyle='-')
    ax2.set_xlabel('Net Balance (GW)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Hours', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Energy Balance', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Pie Chart of Hours Classification =====
    row_idx = 1
    ax3 = fig.add_subplot(gs[row_idx, 1])
    
    sizes = [surplus_hours, deficit_hours, balanced_hours]
    labels = [f'Surplus\n{surplus_hours:,}h ({100*surplus_hours/n_hours:.1f}%)',
              f'Deficit\n{deficit_hours:,}h ({100*deficit_hours/n_hours:.1f}%)',
              f'Balanced\n{balanced_hours:,}h ({100*balanced_hours/n_hours:.1f}%)']
    colors_pie = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    explode = (0.02, 0.02, 0.02)
    
    # Filter out zero values
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors_pie, explode) if s > 0]
    if non_zero:
        sizes, labels, colors_pie, explode = zip(*non_zero)
        ax3.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
                autopct='', shadow=True, startangle=90)
    ax3.set_title('Time Classification', fontsize=11, fontweight='bold')
    
    # ===== SUBPLOT 3B: Renewable Energy Sufficiency =====
    if renewable_production is not None and renewable_balance is not None:
        row_idx = 2
        ax3b = fig.add_subplot(gs[row_idx, :])
        
        # Convert to days for x-axis
        days = hours / 24
        
        # Create color array: green when renewable sufficient, red when not
        renewable_sufficient_fill = np.where(renewable_sufficient, renewable_balance / 1000, 0)
        renewable_insufficient_fill = np.where(~renewable_sufficient, renewable_balance / 1000, 0)
        
        ax3b.fill_between(days, 0, renewable_sufficient_fill, color=COLORS['success'], 
                          alpha=0.6, label=f'Renewable Sufficient ({renewable_sufficient_hours:,}h)')
        ax3b.fill_between(days, 0, renewable_insufficient_fill, color=COLORS['danger'], 
                          alpha=0.6, label=f'Renewable Insufficient ({n_hours - renewable_sufficient_hours:,}h)')
        ax3b.axhline(y=0, color='black', linewidth=1, linestyle='-')
        
        # Add percentage line
        pct_sufficient = 100 * renewable_sufficient_hours / n_hours
        ax3b.axhline(y=0, color='black', linewidth=1, linestyle='-')
        ax3b.text(0.99, 0.95, f'{pct_sufficient:.1f}% of hours\nrenewable-sufficient', 
                 transform=ax3b.transAxes, fontsize=11, fontweight='bold',
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['dark']))
        
        ax3b.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
        ax3b.set_ylabel('Renewable Balance (GW)', fontsize=11, fontweight='bold')
        ax3b.set_title('When Renewable (Incidence) Energy Alone Satisfies Demand', 
                      fontsize=12, fontweight='bold')
        ax3b.legend(loc='upper right', fontsize=10)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_xlim(0, n_hours / 24)
        
        # Add month markers
        if n_hours >= 8760:
            month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, (start, name) in enumerate(zip(month_starts, month_names)):
                ax3b.axvline(x=start, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
                if i % 2 == 0:  # Show every other month to avoid crowding
                    ax3b.text(start + 15, ax3b.get_ylim()[1] * 0.95, name, 
                             fontsize=8, ha='center', alpha=0.7)
    
    # ===== SUBPLOT 4: Monthly Breakdown =====
    row_idx = 3 if renewable_production is not None else 2
    ax4 = fig.add_subplot(gs[row_idx, 0])
    
    # Calculate monthly statistics
    if n_hours >= 8760:
        month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_surplus = []
        monthly_deficit = []
        monthly_net = []
        
        start_idx = 0
        for hours_in_month in month_hours:
            end_idx = start_idx + hours_in_month
            if end_idx > n_hours:
                end_idx = n_hours
            
            month_balance = net_balance[start_idx:end_idx]
            monthly_surplus.append(np.sum(month_balance[month_balance > 0]) / 1e6)  # TWh
            monthly_deficit.append(np.sum(month_balance[month_balance < 0]) / 1e6)  # TWh
            monthly_net.append(np.sum(month_balance) / 1e6)  # TWh
            
            start_idx = end_idx
        
        x = np.arange(len(month_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, monthly_surplus, width, label='Surplus', 
                        color=COLORS['success'], alpha=0.8)
        bars2 = ax4.bar(x + width/2, monthly_deficit, width, label='Deficit', 
                        color=COLORS['danger'], alpha=0.8)
        
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Energy (TWh)', fontsize=11, fontweight='bold')
        ax4.set_title('Monthly Energy Surplus/Deficit', fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(month_names)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Requires full year data\n(8760+ hours)', 
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Monthly Breakdown (N/A)', fontsize=11, fontweight='bold')
    
    # ===== SUBPLOT 5: Daily Pattern (Hour of Day) =====
    row_idx = 3 if renewable_production is not None else 2
    ax5 = fig.add_subplot(gs[row_idx, 1])
    
    # Reshape to days x hours if possible
    n_full_days = n_hours // 24
    if n_full_days > 0:
        daily_balance = net_balance[:n_full_days * 24].reshape(n_full_days, 24)
        
        # Average balance per hour of day
        hourly_avg = np.mean(daily_balance, axis=0) / 1000  # GW
        hourly_std = np.std(daily_balance, axis=0) / 1000
        
        hours_of_day = np.arange(24)
        
        colors_hourly = [COLORS['success'] if v > 0 else COLORS['danger'] for v in hourly_avg]
        ax5.bar(hours_of_day, hourly_avg, color=colors_hourly, alpha=0.8, edgecolor='white')
        ax5.errorbar(hours_of_day, hourly_avg, yerr=hourly_std, fmt='none', 
                     color='gray', alpha=0.5, capsize=2)
        
        ax5.axhline(y=0, color='black', linewidth=1)
        ax5.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Avg Net Balance (GW)', fontsize=11, fontweight='bold')
        ax5.set_title('Daily Pattern: Average Balance by Hour', fontsize=11, fontweight='bold')
        ax5.set_xticks(hours_of_day[::2])
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Requires at least 24 hours', 
                 ha='center', va='center', fontsize=12, transform=ax5.transAxes)
    
    # Add summary statistics text
    total_surplus_twh = np.sum(net_balance[surplus_mask]) / 1e6
    total_deficit_twh = np.sum(net_balance[deficit_mask]) / 1e6
    net_balance_twh = total_surplus_twh + total_deficit_twh
    
    summary_text = f"""
    ENERGY BALANCE SUMMARY
    {'─'*40}
    Total Hours:     {n_hours:,} ({n_hours/24:.0f} days)
    Surplus Hours:   {surplus_hours:,} ({100*surplus_hours/n_hours:.1f}%)
    Deficit Hours:   {deficit_hours:,} ({100*deficit_hours/n_hours:.1f}%)
    """
    
    if renewable_production is not None:
        summary_text += f"""
    Renewable Sufficient: {renewable_sufficient_hours:,}h ({100*renewable_sufficient_hours/n_hours:.1f}%)
    """
    
    summary_text += f"""
    Total Surplus:   {total_surplus_twh:+.2f} TWh
    Total Deficit:   {total_deficit_twh:+.2f} TWh
    Net Balance:     {net_balance_twh:+.2f} TWh
    
    Max Surplus:     {np.max(net_balance)/1000:+.2f} GW
    Max Deficit:     {np.min(net_balance)/1000:+.2f} GW
    """
    
    fig.text(0.98, 0.02, summary_text, fontsize=9, fontfamily='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['dark']))
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['light'])
        print(f"Saved energy balance distribution plot: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo plots
    print("Visualization module loaded successfully")
    
    # Test fitness evolution plot
    test_history = [1e9, 5e8, 2e8, 1e8, 8e7, 7e7, 6.5e7, 6.3e7, 6.2e7, 6.1e7]
    fig = plot_fitness_evolution(test_history, title="Test Evolution")
    plt.show()

