"""
Helper functions for Energy Portfolio Optimization Notebook
Encapsulates common functionality to avoid code duplication
"""

import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from pareto_frontier import extract_pareto_frontier_3d_from_df
from scipy.stats import pearsonr

from config import Config
from ppu_framework import Portfolio, load_all_ppu_data
from optimization import Individual, evaluate_portfolio_full_year, FullYearResults
from data_loader import CachedData
from visualization import (
    plot_full_year_production_by_source,
    plot_full_year_storage,
    plot_energy_balance_distribution
)


# =============================================================================
# PORTFOLIO COMPLIANCE CHECKING
# =============================================================================

def is_portfolio_compliant(portfolio_dict: Dict[str, int], row: pd.Series, config: Config) -> bool:
    """
    Check if portfolio satisfies ALL hard constraints.
    
    Args:
        portfolio_dict: Dictionary of PPU counts
        row: Row from results DataFrame (may contain metrics)
        config: Configuration object with scenario parameters
        
    Returns:
        True if portfolio is compliant, False otherwise
    """
    if isinstance(portfolio_dict, str):
        try:
            portfolio_dict = json.loads(portfolio_dict)
        except:
            try:
                portfolio_dict = ast.literal_eval(portfolio_dict)
            except:
                return False
    
    # 1. Aviation Fuel Constraint
    MIN_THERM_FOR_AVIATION = 263
    therm_count = portfolio_dict.get('THERM', 0)
    syn_ft_count = portfolio_dict.get('SYN_FT', 0)
    syn_crack_count = portfolio_dict.get('SYN_CRACK', 0)
    
    if not (therm_count >= MIN_THERM_FOR_AVIATION and (syn_ft_count > 0 or syn_crack_count > 0)):
        return False
    
    # 2. Electrical Sovereignty
    target_demand = config.energy_system.TARGET_ANNUAL_DEMAND_TWH
    if 'total_domestic_production_twh' in row:
        if row['total_domestic_production_twh'] < target_demand * 0.99:  # 1% tolerance
            return False
    
    # Storage constraint is checked during simulation (cyclic SOC)
    return True


def parse_portfolio_dict(portfolio_str: Any) -> Dict[str, int]:
    """Parse portfolio dictionary from string or dict."""
    if isinstance(portfolio_str, dict):
        return portfolio_str
    try:
        return ast.literal_eval(portfolio_str)
    except:
        try:
            return json.loads(portfolio_str)
        except:
            return {}


# =============================================================================
# EFFICIENCY ANALYSIS
# =============================================================================

def calculate_energy_flows(
    full_year_results: FullYearResults,
    portfolio_dict: Dict[str, int],
    config: Config
) -> Dict[str, float]:
    """
    Calculate comprehensive energy flows from raw incidence to final demand.
    
    Returns dictionary with all energy flow metrics.
    """
    # Get data
    ppu_production = full_year_results.ppu_production
    ppu_consumption = full_year_results.ppu_consumption
    
    renewable_production_twh = np.sum(full_year_results.renewable_production) / 1e6
    total_demand_twh = np.sum(full_year_results.demand) / 1e6
    total_production_twh = np.sum(full_year_results.total_production) / 1e6
    
    # Load PPU definitions
    _, _, ppu_definitions = load_all_ppu_data(config)
    
    # Storage input PPUs
    storage_input_ppus = ['PHS', 'H2_G', 'H2_GL', 'H2_L', 'SYN_FT', 'SYN_METH', 
                          'NH3_FULL', 'SYN_CRACK', 'CH4_BIO', 'SOL_SALT_STORE']
    
    # STEP 1: Estimate raw incidence energy
    renewable_ppus = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R', 'BIO_WOOD']
    total_units = 0
    efficiency_sum = 0.0
    
    for ppu_name in renewable_ppus:
        count = portfolio_dict.get(ppu_name, 0)
        if count > 0 and ppu_name in ppu_definitions:
            ppu_def = ppu_definitions[ppu_name]
            efficiency = ppu_def.efficiency if hasattr(ppu_def, 'efficiency') else 0.84
            total_units += count
            efficiency_sum += efficiency * count
    
    avg_conversion_efficiency = efficiency_sum / total_units if total_units > 0 else 0.84
    raw_incidence_twh = renewable_production_twh / avg_conversion_efficiency if avg_conversion_efficiency > 0 else renewable_production_twh
    production_losses_twh = raw_incidence_twh - renewable_production_twh
    
    # STEP 2: Calculate storage energy flows
    storage_charged = 0.0
    for ppu in storage_input_ppus:
        if ppu in ppu_consumption:
            prod_data = ppu_consumption[ppu]
            if isinstance(prod_data, np.ndarray):
                storage_charged += np.sum(prod_data) / 1e6  # TWh
    
    storage_discharged = 0.0
    if hasattr(full_year_results, 'storage_soc') and full_year_results.storage_soc:
        storage_soc_dict = full_year_results.storage_soc
        if isinstance(storage_soc_dict, dict) and len(storage_soc_dict) > 0:
            from dispatch_engine import initialize_storage_state
            try:
                storages = initialize_storage_state(portfolio_dict, config)
                for storage_name, soc_array in storage_soc_dict.items():
                    if storage_name not in storages:
                        continue
                    storage = storages[storage_name]
                    capacity_mwh = storage.capacity_mwh
                    efficiency_discharge = storage.efficiency_discharge
                    
                    if isinstance(soc_array, np.ndarray) and len(soc_array) > 1:
                        soc_mwh = soc_array * capacity_mwh
                        soc_changes = np.diff(soc_mwh)
                        discharging_changes = -soc_changes[soc_changes < 0]
                        if len(discharging_changes) > 0:
                            energy_input_mwh = np.sum(discharging_changes)
                            total_discharged_mwh = energy_input_mwh * efficiency_discharge
                            storage_discharged += total_discharged_mwh / 1e6  # TWh
            except:
                pass
    
    if storage_discharged == 0 and storage_charged > 0:
        storage_discharged = storage_charged * 0.75  # Estimate 75% round-trip
    
    storage_losses_twh = max(0, storage_charged - storage_discharged)
    
    # STEP 3: Calculate flexible production losses
    flex_ppus = config.ppu.FLEX_PPUS
    flexible_production_twh = 0.0
    flexible_input_energy_twh = 0.0
    
    for ppu_name in flex_ppus:
        count = portfolio_dict.get(ppu_name, 0)
        if count > 0 and ppu_name in ppu_production:
            prod_data = ppu_production[ppu_name]
            if isinstance(prod_data, np.ndarray):
                ppu_prod = np.sum(prod_data) / 1e6  # TWh
                flexible_production_twh += ppu_prod
                
                if ppu_name in ppu_definitions:
                    ppu_def = ppu_definitions[ppu_name]
                    efficiency = ppu_def.efficiency if hasattr(ppu_def, 'efficiency') else 0.80
                    if efficiency > 0:
                        flexible_input_energy_twh += ppu_prod / efficiency
    
    flexible_losses_twh = max(0, flexible_input_energy_twh - flexible_production_twh)
    
    # STEP 4: Track energy flows to demand
    energy_delivered_twh = total_demand_twh
    renewable_direct_to_demand_twh = max(0, renewable_production_twh - storage_charged)
    storage_to_flexible_twh = min(storage_discharged, flexible_input_energy_twh) if flexible_input_energy_twh > 0 else 0
    storage_direct_to_demand_twh = max(0, storage_discharged - storage_to_flexible_twh)
    renewable_curtailed_twh = max(0, renewable_production_twh - storage_charged - renewable_direct_to_demand_twh)
    
    # STEP 5: Calculate metrics
    total_losses_twh = production_losses_twh + storage_losses_twh + flexible_losses_twh + renewable_curtailed_twh
    
    overall_efficiency = (energy_delivered_twh / raw_incidence_twh * 100) if raw_incidence_twh > 0 else 0
    storage_rt_efficiency = (storage_discharged / storage_charged * 100) if storage_charged > 0 else 0
    renewable_utilization = min(100.0, (renewable_production_twh / total_demand_twh * 100)) if total_demand_twh > 0 else 0
    storage_utilization = (storage_charged / renewable_production_twh * 100) if renewable_production_twh > 0 else 0
    flexible_efficiency = (flexible_production_twh/flexible_input_energy_twh*100) if flexible_input_energy_twh > 0 else 0
    
    return {
        'raw_incidence_twh': raw_incidence_twh,
        'renewable_production_twh': renewable_production_twh,
        'total_demand_twh': total_demand_twh,
        'total_production_twh': total_production_twh,
        'energy_delivered_twh': energy_delivered_twh,
        'storage_charged_twh': storage_charged,
        'storage_discharged_twh': storage_discharged,
        'storage_to_flexible_twh': storage_to_flexible_twh,
        'storage_direct_to_demand_twh': storage_direct_to_demand_twh,
        'flexible_production_twh': flexible_production_twh,
        'flexible_input_energy_twh': flexible_input_energy_twh,
        'production_losses_twh': production_losses_twh,
        'storage_losses_twh': storage_losses_twh,
        'flexible_losses_twh': flexible_losses_twh,
        'renewable_curtailed_twh': renewable_curtailed_twh,
        'total_losses_twh': total_losses_twh,
        'renewable_utilization': renewable_utilization,
        'storage_utilization': storage_utilization,
        'storage_rt_efficiency': storage_rt_efficiency,
        'flexible_efficiency': flexible_efficiency,
        'overall_efficiency': overall_efficiency,
    }


# =============================================================================
# PORTFOLIO ANALYSIS HELPERS
# =============================================================================

def analyze_portfolio_decomposition(
    portfolio_dict: Dict[str, int],
    full_year_results: FullYearResults
) -> Dict[str, Any]:
    """
    Analyze portfolio by category (INCIDENCE, FLEX, STORAGE).
    
    Returns dictionary with category breakdowns.
    """
    INCIDENCE_PPUS = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R', 'BIO_WOOD']
    FLEX_PPUS = ['HYD_S', 'THERM', 'H2P_G', 'H2P_L', 'SOL_SALT', 'SOL_STEAM', 
                 'PALM_ICE', 'IMP_BIOG', 'THERM_CH4', 'NH3_P']
    STORAGE_PPUS = ['PHS', 'H2_G', 'H2_GL', 'H2_L', 'SYN_FT', 'SYN_METH', 
                    'NH3_FULL', 'SYN_CRACK', 'CH4_BIO', 'SOL_SALT_STORE']
    
    ppu_production = full_year_results.ppu_production
    ppu_consumption = full_year_results.ppu_consumption
    
    def analyze_category(cat_name, ppu_list, use_consumption=False):
        cat_counts = {p: portfolio_dict.get(p, 0) for p in ppu_list if portfolio_dict.get(p, 0) > 0}
        data_source = ppu_consumption if (use_consumption and ppu_consumption) else ppu_production
        
        cat_energy = {}
        for p in ppu_list:
            if p in data_source:
                energy_mwh = np.sum(data_source.get(p, [0]))
                if energy_mwh > 0:
                    cat_energy[p] = energy_mwh / 1e6  # Convert to TWh
        
        total_units = sum(cat_counts.values())
        total_energy = sum(cat_energy.values())
        return cat_counts, cat_energy, total_units, total_energy
    
    inc_counts, inc_energy, inc_total_units, inc_total_twh = analyze_category(
        "INCIDENCE", INCIDENCE_PPUS, use_consumption=False)
    flex_counts, flex_energy, flex_total_units, flex_total_twh = analyze_category(
        "FLEX", FLEX_PPUS, use_consumption=False)
    stor_counts, stor_energy, stor_total_units, stor_total_twh = analyze_category(
        "STORAGE", STORAGE_PPUS, use_consumption=True)
    
    return {
        'incidence': {'counts': inc_counts, 'energy': inc_energy, 'total_units': inc_total_units, 'total_twh': inc_total_twh},
        'flex': {'counts': flex_counts, 'energy': flex_energy, 'total_units': flex_total_units, 'total_twh': flex_total_twh},
        'storage': {'counts': stor_counts, 'energy': stor_energy, 'total_units': stor_total_units, 'total_twh': stor_total_twh},
    }


def get_frontier_file_paths(scenario: str) -> Tuple[Path, Path]:
    """
    Get file paths for frontier results based on scenario.
    
    Returns:
        (all_results_path, frontier_path)
    """
    base_path = Path('data/result_plots')
    if scenario == "2050":
        all_path = base_path / 'multi_objective_results_2050.csv'
        frontier_path = base_path / 'multi_objective_results_2050_frontier_3d.csv'
    else:  # 2024
        all_path = base_path / 'multi_objective_results.csv'
        frontier_path = base_path / 'multi_objective_results_frontier_3d.csv'
    
    return all_path, frontier_path


# =============================================================================
# EXTERNAL DATA VISUALIZATION
# =============================================================================

def plot_external_data_overview(
    cached_data: CachedData,
    demand_scenario: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive external data overview visualization.
    
    Plots all input data except currency exchange rates (which are handled separately).
    
    Args:
        cached_data: CachedData object with all input data
        demand_scenario: Scenario name ("2024" or "2050")
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Create time indices
    n_hours = len(cached_data.get_demand())
    hours = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    days = pd.date_range('2024-01-01', periods=366, freq='D')
    
    # Get data
    solar_data = cached_data.get_solar_incidence(copy=False)
    wind_data = cached_data.get_wind_incidence(copy=False)
    spot_prices = cached_data.get_spot_prices(copy=False)
    demand = cached_data.get_demand(copy=False)
    water_inflow = cached_data.get_water_inflow(copy=False)
    palm_oil_prices = cached_data.get_palm_oil_prices_daily(copy=False)
    ror_production = cached_data.ror_production  # Monthly RoR
    
    # Calculate spatial averages for solar and wind
    solar_avg = np.nanmean(solar_data, axis=1)  # Average across locations per hour
    wind_avg = np.nanmean(wind_data, axis=1)    # Average across locations per hour
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    
    # Plot 1: Solar Irradiance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hours[:len(solar_avg)], solar_avg, color='#FFB300', alpha=0.7, linewidth=0.3)
    daily_max = pd.Series(solar_avg[:len(hours)], index=hours[:len(solar_avg)]).resample('D').max()
    ax1.fill_between(daily_max.index, daily_max.values, alpha=0.3, color='orange', label='Daily max')
    ax1.set_title('Solar Irradiance (Swiss Average)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Irradiance (kWh/m²)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    stats_text = f"Mean: {np.nanmean(solar_avg):.2f} kWh/m²\nMax: {np.nanmax(solar_avg):.2f} kWh/m²\nTotal: {np.nansum(solar_avg):.0f} kWh/m²/year"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Wind Speed
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hours[:len(wind_avg)], wind_avg, color='#1976D2', alpha=0.7, linewidth=0.3)
    daily_mean_wind = pd.Series(wind_avg[:len(hours)], index=hours[:len(wind_avg)]).resample('D').mean()
    ax2.plot(daily_mean_wind.index, daily_mean_wind.values, color='darkblue', alpha=0.8, linewidth=1, label='Daily mean')
    ax2.set_title('Wind Speed (Swiss Average)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    stats_text = f"Mean: {np.nanmean(wind_avg):.2f} m/s\nMax: {np.nanmax(wind_avg):.2f} m/s"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Run of River (Monthly)
    ax3 = fig.add_subplot(gs[1, 0])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    from matplotlib import cm
    colors_ror = cm.get_cmap('Blues')(np.linspace(0.3, 0.9, 12))
    bars = ax3.bar(months, ror_production[:12], color=colors_ror, edgecolor='navy', linewidth=1)
    ax3.set_title('Run-of-River Hydropower Production', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Production (GWh)')
    ax3.grid(True, alpha=0.3, axis='y')
    total_ror = np.sum(ror_production[:12])
    ax3.annotate(f'Total: {total_ror/1000:.2f} TWh/year', xy=(0.5, 0.95), xycoords='axes fraction',
                 fontsize=11, ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Plot 4: Electricity Spot Price
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(hours[:len(spot_prices)], spot_prices, color='#E53935', alpha=0.6, linewidth=0.3)
    weekly_mean = pd.Series(spot_prices[:len(hours)], index=hours[:len(spot_prices)]).resample('W').mean()
    ax4.plot(weekly_mean.index, weekly_mean.values, color='darkred', alpha=0.9, linewidth=1.5, label='Weekly mean')
    ax4.set_title('Electricity Spot Price (CHF/MWh)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price (CHF/MWh)')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    stats_text = f"Mean: {np.mean(spot_prices):.1f} CHF/MWh\nStd: {np.std(spot_prices):.1f} CHF/MWh\nMax: {np.max(spot_prices):.1f} CHF/MWh"
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 5: Palm Oil Price
    ax5 = fig.add_subplot(gs[2, 0])
    days_palm = days[:len(palm_oil_prices)]
    ax5.plot(days_palm, palm_oil_prices, color='#8D6E63', linewidth=2, marker='', label='Palm Oil (CHF/MWh)')
    ax5.fill_between(days_palm, palm_oil_prices, alpha=0.3, color='#A1887F')
    ax5.set_title('Palm Oil Price (CHF/MWh)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Price (CHF/MWh)')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    ax5.grid(True, alpha=0.3)
    stats_text = f"Mean: {np.mean(palm_oil_prices):.1f} CHF/MWh\nMin: {np.min(palm_oil_prices):.1f} CHF/MWh\nMax: {np.max(palm_oil_prices):.1f} CHF/MWh"
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 6: Water Inflow to Lakes
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(hours[:len(water_inflow)], water_inflow/1000, color='#0288D1', alpha=0.7, linewidth=0.5)  # Convert to GWh
    weekly_sum = pd.Series(water_inflow[:len(hours)], index=hours[:len(water_inflow)]).resample('W').sum()
    ax6_twin = ax6.twinx()
    ax6_twin.bar(weekly_sum.index, weekly_sum.values/1e6, width=5, alpha=0.4, color='navy', label='Weekly total (TWh)')
    ax6.set_title('Water Inflow to Lakes', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Inflow Rate (GWh/hour)', color='#0288D1')
    ax6_twin.set_ylabel('Weekly Total (TWh)', color='navy')
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax6.xaxis.set_major_locator(mdates.MonthLocator())
    ax6.grid(True, alpha=0.3)
    total_inflow_twh = np.sum(water_inflow) / 1e6
    stats_text = f"Total Annual: {total_inflow_twh:.2f} TWh\n(0.9 kWh/m³ conversion)"
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Plot 7: Demand vs Supply Overview (moved to (3,0) since currency plot removed)
    ax8 = fig.add_subplot(gs[3, 0])
    demand_weekly = pd.Series(demand[:len(hours)], index=hours[:len(demand)]).resample('W').sum() / 1e6  # TWh
    ax8.fill_between(demand_weekly.index, demand_weekly.values, alpha=0.6, color='#9C27B0', label='Weekly Demand')
    ax8.plot(demand_weekly.index, demand_weekly.values, color='#6A1B9A', linewidth=2)
    ax8.set_title('Electricity Demand (Weekly)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Demand (TWh/week)')
    ax8.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax8.xaxis.set_major_locator(mdates.MonthLocator())
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)
    total_demand_twh = np.sum(demand) / 1e6
    stats_text = f"Total: {total_demand_twh:.2f} TWh/year\nPeak: {np.max(demand)/1000:.2f} GW\nMean: {np.mean(demand)/1000:.2f} GW"
    ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'External Data Overview for Swiss Energy Portfolio Optimization ({demand_scenario})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def calculate_rolling_correlations(
    cached_data: CachedData,
    window_3month: int = 90 * 24,  # 90 days in hours
    window_1day: int = 24  # 1 day in hours
) -> Dict[str, pd.DataFrame]:
    """
    Calculate rolling correlations between data curves and demand.
    
    Args:
        cached_data: CachedData object with all input data
        window_3month: Window size for 3-month rolling correlation (in hours)
        window_1day: Window size for 1-day rolling correlation (in hours)
        
    Returns:
        Dictionary with correlation DataFrames for each data type
    """
    # Get data
    n_hours = len(cached_data.get_demand())
    hours = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    
    solar_data = cached_data.get_solar_incidence(copy=False)
    wind_data = cached_data.get_wind_incidence(copy=False)
    spot_prices = cached_data.get_spot_prices(copy=False)
    demand = cached_data.get_demand(copy=False)
    water_inflow = cached_data.get_water_inflow(copy=False)
    
    # Calculate spatial averages
    solar_avg = np.nanmean(solar_data, axis=1)
    wind_avg = np.nanmean(wind_data, axis=1)
    
    # Create DataFrame with all time series
    df = pd.DataFrame({
        'demand': demand,
        'solar': solar_avg[:n_hours],
        'wind': wind_avg[:n_hours],
        'spot_price': spot_prices[:n_hours],
        'water_inflow': water_inflow[:n_hours]
    }, index=hours[:n_hours])
    
    # Calculate rolling correlations
    correlations_3month = {}
    correlations_1day = {}
    
    def compute_rolling_corr(series1, series2, window, min_periods=None):
        """Compute rolling correlation between two series."""
        if min_periods is None:
            min_periods = window // 2
        
        # Use vectorized approach with pandas rolling
        # Create aligned series
        aligned = pd.DataFrame({'x': series1, 'y': series2})
        
        # Calculate rolling means and stds
        x_mean = aligned['x'].rolling(window=window, min_periods=min_periods).mean()
        y_mean = aligned['y'].rolling(window=window, min_periods=min_periods).mean()
        x_std = aligned['x'].rolling(window=window, min_periods=min_periods).std()
        y_std = aligned['y'].rolling(window=window, min_periods=min_periods).std()
        
        # Calculate rolling covariance
        xy_mean = (aligned['x'] * aligned['y']).rolling(window=window, min_periods=min_periods).mean()
        cov = xy_mean - x_mean * y_mean
        
        # Calculate correlation
        corr = cov / (x_std * y_std)
        
        return corr
    
    for col in ['solar', 'wind', 'spot_price', 'water_inflow']:
        # 3-month rolling correlation
        corr_3m = compute_rolling_corr(df['demand'], df[col], window_3month, window_3month//2)
        correlations_3month[col] = corr_3m
        
        # 1-day rolling correlation
        corr_1d = compute_rolling_corr(df['demand'], df[col], window_1day, window_1day//2)
        correlations_1day[col] = corr_1d
    
    # Convert to DataFrames
    corr_3m_df = pd.DataFrame(correlations_3month, index=df.index)
    corr_1d_df = pd.DataFrame(correlations_1day, index=df.index)
    
    return {
        '3month': corr_3m_df,
        '1day': corr_1d_df,
        'data': df
    }


def compute_seasonal_correlation(cached_data: CachedData) -> Dict[str, float]:
    """
    Compute TRUE seasonal correlation using monthly averages.
    
    This measures whether months with HIGH values have HIGH/LOW demand.
    Removes diurnal effects by averaging to monthly level.
    
    Args:
        cached_data: CachedData object with all input data
        
    Returns:
        Dictionary with correlation values for each variable
    """
    n_hours = len(cached_data.get_demand())
    hours = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    
    # Get data
    solar_avg = np.nanmean(cached_data.get_solar_incidence(copy=False), axis=1)
    wind_avg = np.nanmean(cached_data.get_wind_incidence(copy=False), axis=1)
    spot_prices = cached_data.get_spot_prices(copy=False)
    demand = cached_data.get_demand(copy=False)
    water_inflow = cached_data.get_water_inflow(copy=False)
    
    # Create DataFrame with datetime index
    df = pd.DataFrame({
        'demand': demand,
        'solar': solar_avg[:n_hours],
        'wind': wind_avg[:n_hours],
        'spot_price': spot_prices[:n_hours],
        'water_inflow': water_inflow[:n_hours]
    }, index=hours[:n_hours])
    
    # Calculate monthly averages
    monthly_avg = df.resample('M').mean()
    
    # Calculate correlation across monthly averages
    correlations = {}
    for col in ['solar', 'wind', 'spot_price', 'water_inflow']:
        mask = ~(np.isnan(monthly_avg[col]) | np.isnan(monthly_avg['demand']))
        if np.sum(mask) > 2:
            correlations[col] = np.corrcoef(
                monthly_avg.loc[mask, col], 
                monthly_avg.loc[mask, 'demand']
            )[0, 1]
        else:
            correlations[col] = np.nan
    
    return correlations


def compute_hourly_correlation(cached_data: CachedData) -> Dict[str, float]:
    """
    Compute hourly correlation: average correlation at each hour of day.
    
    For each hour (0-23), calculates correlation across all days of the year.
    Returns the mean correlation across all hours.
    
    Args:
        cached_data: CachedData object with all input data
        
    Returns:
        Dictionary with mean hourly correlation for each variable
    """
    n_hours = len(cached_data.get_demand())
    n_days = n_hours // 24
    
    if n_days == 0:
        return {'solar': np.nan, 'wind': np.nan, 'spot_price': np.nan, 'water_inflow': np.nan}
    
    # Get data
    solar_avg = np.nanmean(cached_data.get_solar_incidence(copy=False), axis=1)
    wind_avg = np.nanmean(cached_data.get_wind_incidence(copy=False), axis=1)
    spot_prices = cached_data.get_spot_prices(copy=False)
    demand = cached_data.get_demand(copy=False)
    water_inflow = cached_data.get_water_inflow(copy=False)
    
    # Reshape to (n_days, 24)
    demand_hourly = np.array(demand[:n_days*24]).reshape(n_days, 24)
    solar_hourly = np.array(solar_avg[:n_days*24]).reshape(n_days, 24)
    wind_hourly = np.array(wind_avg[:n_days*24]).reshape(n_days, 24)
    spot_hourly = np.array(spot_prices[:n_days*24]).reshape(n_days, 24)
    water_hourly = np.array(water_inflow[:n_days*24]).reshape(n_days, 24)
    
    data_dict = {
        'solar': solar_hourly,
        'wind': wind_hourly,
        'spot_price': spot_hourly,
        'water_inflow': water_hourly
    }
    
    # Calculate correlation for each hour and variable
    correlations = {}
    for var, var_data in data_dict.items():
        hourly_corrs = []
        for h in range(24):
            demand_h = demand_hourly[:, h]
            var_h = var_data[:, h]
            mask = ~(np.isnan(demand_h) | np.isnan(var_h))
            if np.sum(mask) > 10:
                corr_h = np.corrcoef(demand_h[mask], var_h[mask])[0, 1]
                hourly_corrs.append(corr_h)
        
        correlations[var] = np.nanmean(hourly_corrs) if hourly_corrs else np.nan
    
    return correlations


def get_monthly_averages(cached_data: CachedData) -> pd.DataFrame:
    """
    Get monthly averages for all variables.
    
    Args:
        cached_data: CachedData object with all input data
        
    Returns:
        DataFrame with monthly averages
    """
    n_hours = len(cached_data.get_demand())
    hours = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    
    solar_avg = np.nanmean(cached_data.get_solar_incidence(copy=False), axis=1)
    wind_avg = np.nanmean(cached_data.get_wind_incidence(copy=False), axis=1)
    
    df = pd.DataFrame({
        'demand': cached_data.get_demand(copy=False),
        'solar': solar_avg[:n_hours],
        'wind': wind_avg[:n_hours],
        'spot_price': cached_data.get_spot_prices(copy=False)[:n_hours],
        'water_inflow': cached_data.get_water_inflow(copy=False)[:n_hours]
    }, index=hours[:n_hours])
    
    monthly = df.resample('M').mean()
    monthly.index = monthly.index.month
    monthly.index.name = 'month'
    
    return monthly


def plot_correlation_analysis(
    cached_data: CachedData,
    demand_scenario: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling correlation analysis between data curves and demand.
    
    Args:
        cached_data: CachedData object with all input data
        demand_scenario: Scenario name ("2024" or "2050")
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Calculate correlations
    corr_data = calculate_rolling_correlations(cached_data)
    corr_3m_df = corr_data['3month']
    corr_1d_df = corr_data['1day']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Rolling Correlation Analysis with Demand ({demand_scenario})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: 3-month rolling correlations
    ax1 = axes[0, 0]
    for col in corr_3m_df.columns:
        ax1.plot(corr_3m_df.index, corr_3m_df[col], label=col.replace('_', ' ').title(), 
                alpha=0.7, linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_title('3-Month Rolling Window Correlation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Plot 2: 1-day rolling correlations
    ax2 = axes[0, 1]
    for col in corr_1d_df.columns:
        ax2.plot(corr_1d_df.index, corr_1d_df[col], label=col.replace('_', ' ').title(), 
                alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_title('1-Day Rolling Window Correlation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Plot 3: Correlation distribution (3-month)
    ax3 = axes[1, 0]
    for col in corr_3m_df.columns:
        valid_corr = corr_3m_df[col].dropna()
        if len(valid_corr) > 0:
            ax3.hist(valid_corr, bins=30, alpha=0.5, label=col.replace('_', ' ').title(), 
                    density=True)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('3-Month Correlation Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Density')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Correlation statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_data = []
    for col in corr_3m_df.columns:
        valid_corr_3m = corr_3m_df[col].dropna()
        valid_corr_1d = corr_1d_df[col].dropna()
        
        if len(valid_corr_3m) > 0 and len(valid_corr_1d) > 0:
            stats_data.append({
                'Variable': col.replace('_', ' ').title(),
                '3M Mean': f"{valid_corr_3m.mean():.3f}",
                '3M Std': f"{valid_corr_3m.std():.3f}",
                '1D Mean': f"{valid_corr_1d.mean():.3f}",
                '1D Std': f"{valid_corr_1d.std():.3f}"
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        table = ax4.table(cellText=stats_df.values,
                         colLabels=stats_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Correlation Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

