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
        'renewable_direct_to_demand_twh': renewable_direct_to_demand_twh,
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
        'spot_bought_twh': full_year_results.total_spot_bought_mwh / 1e6,
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
    
    Args:
        scenario: "2024" or "2050"
    
    Returns:
        (all_results_path, frontier_path)
    """
    base_path = Path('data/result_plots')
    # Ensure scenario is in filename for both 2024 and 2050
    all_path = base_path / f'multi_objective_results_{scenario}.csv'
    frontier_path = base_path / f'multi_objective_results_{scenario}_frontier_3d.csv'
    
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


# =============================================================================
# COMPREHENSIVE PORTFOLIO ANALYSIS FUNCTIONS
# =============================================================================

def analyze_incidence_vs_demand(
    full_year_results: FullYearResults,
    portfolio_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze renewable incidence vs demand (before storage).
    
    Returns dictionary with statistics and creates plots.
    """
    dem = np.array(full_year_results.demand)
    ren = np.array(full_year_results.renewable_production)
    balance = ren - dem
    
    # Statistics
    total_demand_twh = np.sum(dem) / 1e6
    total_renewable_twh = np.sum(ren) / 1e6
    coverage = (total_renewable_twh / total_demand_twh) * 100
    surplus_h = np.sum(balance > 0)
    deficit_h = np.sum(balance < 0)
    total_surplus_twh = np.sum(balance[balance > 0]) / 1e6
    total_deficit_twh = np.sum(balance[balance < 0]) / 1e6
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    hours = np.arange(len(dem))
    days = hours / 24
    
    # Plot 1: Incidence vs Demand
    ax1 = axes[0]
    ax1.fill_between(days, dem/1e3, alpha=0.3, label='Demand', color='red')
    ax1.fill_between(days, ren/1e3, alpha=0.5, label='Incidence (Renewable)', color='green')
    ax1.set_ylabel('Power (GW)')
    ax1.set_title(f'Portfolio #{portfolio_idx+1}: Incidence Production vs Demand (Before Storage)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Balance (Surplus/Deficit)
    ax2 = axes[1]
    balance_gw = balance / 1e3
    ax2.fill_between(days, balance_gw, where=balance_gw >= 0, alpha=0.7, 
                      label='Surplus (storable)', color='green', interpolate=True)
    ax2.fill_between(days, balance_gw, where=balance_gw < 0, alpha=0.7, 
                      label='Deficit (need storage/dispatch)', color='red', interpolate=True)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Balance (GW)')
    ax2.set_title('Hourly Balance: Incidence - Demand')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Energy Balance
    ax3 = axes[2]
    cumulative_balance = np.cumsum(balance) / 1e6  # TWh cumulative
    ax3.plot(days, cumulative_balance, 'b-', linewidth=1.5, label='Cumulative Balance')
    ax3.fill_between(days, cumulative_balance, where=cumulative_balance >= 0, 
                      alpha=0.3, color='green')
    ax3.fill_between(days, cumulative_balance, where=cumulative_balance < 0, 
                      alpha=0.3, color='red')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Day of Year')
    ax3.set_ylabel('Cumulative Balance (TWh)')
    ax3.set_title('Cumulative Energy Balance (shows seasonal storage need)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add month labels
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax3.set_xticks(month_starts)
    ax3.set_xticklabels(month_labels)
    
    plt.tight_layout()
    plt.show()
    
    seasonal_storage_need = -np.min(cumulative_balance) if len(cumulative_balance) > 0 else 0
    
    return {
        'total_demand_twh': total_demand_twh,
        'total_renewable_twh': total_renewable_twh,
        'coverage': coverage,
        'surplus_hours': surplus_h,
        'deficit_hours': deficit_h,
        'total_surplus_twh': total_surplus_twh,
        'total_deficit_twh': total_deficit_twh,
        'seasonal_storage_need_twh': seasonal_storage_need
    }

def plot_ppu_decomposition(
    portfolio_dict: Dict[str, int],
    full_year_results: FullYearResults,
    portfolio_idx: int = 0
) -> None:
    """
    Plot PPU decomposition by category (INCIDENCE, FLEX, STORAGE).
    """
    decomposition = analyze_portfolio_decomposition(portfolio_dict, full_year_results)
    
    inc_data = decomposition['incidence']
    flex_data = decomposition['flex']
    stor_data = decomposition['storage']
    
    grand_total_units = inc_data['total_units'] + flex_data['total_units'] + stor_data['total_units']
    grand_total_twh = inc_data['total_twh'] + flex_data['total_twh'] + stor_data['total_twh']
    
    # Print numerical decomposition
    print("\n📊 NUMERICAL DECOMPOSITION (Unit Counts):")
    print(f"   {'Category':<12} | {'Units':>8} | {'% of Total':>10} | {'Energy (TWh)':>12} | {'% of Energy':>11}")
    print(f"   {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*11}")
    print(f"   {'INCIDENCE':<12} | {inc_data['total_units']:>8} | {inc_data['total_units']/grand_total_units*100 if grand_total_units else 0:>9.1f}% | {inc_data['total_twh']:>12.2f} | {inc_data['total_twh']/grand_total_twh*100 if grand_total_twh else 0:>10.1f}%")
    print(f"   {'FLEX':<12} | {flex_data['total_units']:>8} | {flex_data['total_units']/grand_total_units*100 if grand_total_units else 0:>9.1f}% | {flex_data['total_twh']:>12.2f} | {flex_data['total_twh']/grand_total_twh*100 if grand_total_twh else 0:>10.1f}%")
    print(f"   {'STORAGE':<12} | {stor_data['total_units']:>8} | {stor_data['total_units']/grand_total_units*100 if grand_total_units else 0:>9.1f}% | {stor_data['total_twh']:>12.2f} | {stor_data['total_twh']/grand_total_twh*100 if grand_total_twh else 0:>10.1f}%")
    print(f"   {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*11}")
    print(f"   {'TOTAL':<12} | {grand_total_units:>8} | {'100.0':>9}% | {grand_total_twh:>12.2f} | {'100.0':>10}%")
    print("\n   ℹ️  Note: STORAGE energy shows electricity consumed for charging (not produced)")
    
    # Detailed breakdown
    def print_category_detail(cat_name, counts, energy, total_units, total_twh, is_storage=False):
        energy_label = "absorbed" if is_storage else "produced"
        print(f"\n   🔹 {cat_name} PPUs ({energy_label}):")
        if not counts:
            print(f"      (none in portfolio)")
            return
        for ppu in sorted(counts.keys(), key=lambda x: -counts[x]):
            unit_pct = counts[ppu] / total_units * 100 if total_units else 0
            energy_twh = energy.get(ppu, 0)
            energy_pct = energy_twh / total_twh * 100 if total_twh else 0
            print(f"      {ppu:<15}: {counts[ppu]:>5} units ({unit_pct:>5.1f}%) | {energy_twh:>6.2f} TWh ({energy_pct:>5.1f}%)")
    
    print_category_detail("INCIDENCE", inc_data['counts'], inc_data['energy'], 
                         inc_data['total_units'], inc_data['total_twh'], is_storage=False)
    print_category_detail("FLEX (Dispatchable)", flex_data['counts'], flex_data['energy'], 
                         flex_data['total_units'], flex_data['total_twh'], is_storage=False)
    print_category_detail("STORAGE (Input)", stor_data['counts'], stor_data['energy'], 
                         stor_data['total_units'], stor_data['total_twh'], is_storage=True)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Pie chart: Units by Category
    ax1 = axes[0]
    cat_units = [inc_data['total_units'], flex_data['total_units'], stor_data['total_units']]
    cat_labels = ['INCIDENCE', 'FLEX', 'STORAGE']
    cat_colors = ['#2ecc71', '#e67e22', '#3498db']
    if sum(cat_units) > 0:
        wedges, texts, autotexts = ax1.pie(cat_units, labels=cat_labels, autopct='%1.1f%%',
                                           colors=cat_colors, explode=[0.02]*3)
        ax1.set_title('Unit Distribution by Category')
    
    # Pie chart: Energy by Category
    ax2 = axes[1]
    cat_twh = [inc_data['total_twh'], flex_data['total_twh'], stor_data['total_twh']]
    if sum(cat_twh) > 0:
        wedges, texts, autotexts = ax2.pie(cat_twh, labels=cat_labels, autopct='%1.1f%%',
                                           colors=cat_colors, explode=[0.02]*3)
        ax2.set_title('Energy Production by Category (TWh)')
    
    # Bar chart: Top 10 PPUs by Energy
    ax3 = axes[2]
    all_energy = {**inc_data['energy'], **flex_data['energy'], **stor_data['energy']}
    sorted_energy = sorted(all_energy.items(), key=lambda x: -x[1])[:10]
    if sorted_energy:
        INCIDENCE_PPUS = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R', 'BIO_WOOD']
        FLEX_PPUS = ['HYD_S', 'THERM', 'H2P_G', 'H2P_L', 'SOL_SALT', 'SOL_STEAM', 
                     'PALM_ICE', 'IMP_BIOG', 'THERM_CH4', 'NH3_P']
        ppus = [x[0] for x in sorted_energy]
        twhs = [x[1] for x in sorted_energy]
        colors_bar = []
        for p in ppus:
            if p in INCIDENCE_PPUS: colors_bar.append('#2ecc71')
            elif p in FLEX_PPUS: colors_bar.append('#e67e22')
            else: colors_bar.append('#3498db')
        ax3.barh(range(len(ppus)), twhs, color=colors_bar)
        ax3.set_yticks(range(len(ppus)))
        ax3.set_yticklabels(ppus)
        ax3.invert_yaxis()
        ax3.set_xlabel('Energy (TWh)')
        ax3.set_title('Top 10 PPUs by Energy Production')
        ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def analyze_hourly_energy_mix(
    full_year_results: FullYearResults,
    portfolio_dict: Dict[str, int],
    portfolio_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze hourly energy mix (daily average profile).
    """
    ppu_production = full_year_results.ppu_production
    n_hours = len(full_year_results.demand)
    n_days = n_hours // 24
    demand_array = np.array(full_year_results.demand)
    
    # Get all PPUs with production
    active_ppus = [p for p in ppu_production.keys() 
                  if isinstance(ppu_production[p], np.ndarray) and len(ppu_production[p]) == n_hours]
    
    # Calculate average production by hour of day for each PPU
    hourly_avg_by_ppu = {}
    for ppu in active_ppus:
        prod = np.array(ppu_production[ppu])[:n_days*24]
        prod_reshaped = prod.reshape(n_days, 24)
        hourly_avg_by_ppu[ppu] = np.mean(prod_reshaped, axis=0)
    
    # Sort PPUs by total daily production
    ppu_daily_totals = {p: np.sum(hourly_avg_by_ppu[p]) for p in hourly_avg_by_ppu}
    sorted_ppus = sorted(ppu_daily_totals.keys(), key=lambda x: -ppu_daily_totals[x])
    
    # Print hourly mix table
    print("\n📊 AVERAGE HOURLY ENERGY MIX (MW by PPU):")
    key_hours = [0, 6, 8, 12, 14, 18, 20, 22]
    
    header = f"   {'PPU':<15}"
    for h in key_hours:
        header += f" | {h:02d}:00"
    print(header)
    print(f"   {'-'*15}" + "-+-" + "-+-".join(["-"*6]*len(key_hours)))
    
    for ppu in sorted_ppus[:12]:
        row = f"   {ppu:<15}"
        for h in key_hours:
            val = hourly_avg_by_ppu[ppu][h]
            if val >= 1000:
                row += f" | {val/1e3:5.1f}k"
            else:
                row += f" | {val:6.0f}"
        print(row)
    
    # Stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    hours_x = np.arange(24)
    top_ppus = sorted_ppus[:8]
    other_ppus = sorted_ppus[8:]
    
    bottom = np.zeros(24)
    colors_hourly = plt.cm.tab20(np.linspace(0, 1, len(top_ppus) + 1))
    
    for i, ppu in enumerate(top_ppus):
        values = hourly_avg_by_ppu[ppu] / 1e3  # GW
        ax.fill_between(hours_x, bottom, bottom + values, 
                       label=ppu, alpha=0.8, color=colors_hourly[i])
        ax.plot(hours_x, bottom + values, color='white', linewidth=0.5)
        bottom += values
    
    if other_ppus:
        other_sum = np.sum([hourly_avg_by_ppu[p] for p in other_ppus], axis=0) / 1e3
        ax.fill_between(hours_x, bottom, bottom + other_sum, 
                       label='Other', alpha=0.6, color='gray')
        bottom += other_sum
    
    demand_hourly_avg = np.array([np.mean(demand_array[h::24]) for h in range(24)]) / 1e3
    ax.plot(hours_x, demand_hourly_avg, 'r--', linewidth=2.5, label='Avg Demand')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Power (GW)', fontsize=12)
    ax.set_title(f'Portfolio #{portfolio_idx+1}: Average Daily Energy Mix by Source', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'hourly_avg_by_ppu': hourly_avg_by_ppu,
        'sorted_ppus': sorted_ppus
    }


def analyze_production_concentration(
    portfolio_dict: Dict[str, int],
    cached_data: CachedData,
    config: Config,
    n_hours: int
) -> Dict[str, Any]:
    """
    Analyze solar/wind production concentration across locations.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    solar_incidence = cached_data.get_solar_incidence(copy=False)
    wind_incidence = cached_data.get_wind_incidence(copy=False)
    
    solar_units = portfolio_dict.get('PV', 0)
    wind_on_units = portfolio_dict.get('WD_ON', 0)
    wind_off_units = portfolio_dict.get('WD_OFF', 0)
    
    mw_per_unit = config.ppu.MW_PER_UNIT
    
    def compute_location_concentration(incidence_data, n_units, n_hours_data):
        n_hours_inc, n_locations = incidence_data.shape
        n_hours_use = min(n_hours_data, n_hours_inc)
        n_locs_used = min(n_units, n_locations)
        if n_locs_used == 0:
            return None, None, None
        
        prod_per_loc = incidence_data[:n_hours_use, :n_locs_used] * mw_per_unit
        productions = []
        pct_for_90 = []
        
        for t in range(n_hours_use):
            loc_prod = prod_per_loc[t, :]
            total_prod = np.sum(loc_prod)
            if total_prod < 1.0:
                continue
            
            sorted_prod = np.sort(loc_prod)[::-1]
            cumsum = np.cumsum(sorted_prod)
            threshold_90 = 0.9 * total_prod
            n_locs_for_90 = np.searchsorted(cumsum, threshold_90) + 1
            pct_locs = (n_locs_for_90 / n_locs_used) * 100
            
            productions.append(total_prod)
            pct_for_90.append(pct_locs)
        
        return np.array(productions), np.array(pct_for_90), True
    
    # Solar analysis
    print("\n🔆 SOLAR PV CONCENTRATION:")
    if solar_units > 0:
        solar_prod_arr, solar_pct_90, solar_valid = compute_location_concentration(
            solar_incidence, solar_units, n_hours)
        if solar_valid and len(solar_prod_arr) > 0:
            print(f"   Units: {solar_units} | Locations used: {min(solar_units, solar_incidence.shape[1])}")
            print(f"   Hours with production: {len(solar_prod_arr):,}")
            print(f"   Avg % locations for 90%: {np.mean(solar_pct_90):.1f}%")
            print(f"   Min % locations for 90%: {np.min(solar_pct_90):.1f}% (most concentrated)")
            print(f"   Max % locations for 90%: {np.max(solar_pct_90):.1f}% (most distributed)")
    else:
        print("   No Solar PV in portfolio")
        solar_prod_arr, solar_pct_90 = None, None
    
    # Wind analysis
    print("\n💨 WIND CONCENTRATION:")
    wind_total_units = wind_on_units + wind_off_units
    if wind_total_units > 0:
        wind_prod_arr, wind_pct_90, wind_valid = compute_location_concentration(
            wind_incidence, wind_total_units, n_hours)
        if wind_valid and len(wind_prod_arr) > 0:
            print(f"   Units: {wind_total_units} | Locations used: {min(wind_total_units, wind_incidence.shape[1])}")
            print(f"   Hours with production: {len(wind_prod_arr):,}")
            print(f"   Avg % locations for 90%: {np.mean(wind_pct_90):.1f}%")
            print(f"   Min % locations for 90%: {np.min(wind_pct_90):.1f}% (most concentrated)")
            print(f"   Max % locations for 90%: {np.max(wind_pct_90):.1f}% (most distributed)")
    else:
        print("   No Wind in portfolio")
        wind_prod_arr, wind_pct_90 = None, None
    
    # 3D plots
    fig_3d = plt.figure(figsize=(16, 6))
    
    def plot_3d_concentration(ax, prod_arr, pct_90_arr, title, color):
        if prod_arr is None or len(prod_arr) == 0:
            ax.text2D(0.5, 0.5, f'No data for {title}', ha='center', va='center', 
                     transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        n_bins_x, n_bins_y = 25, 20
        x_edges = np.linspace(0, np.percentile(prod_arr, 99), n_bins_x + 1)
        y_edges = np.linspace(0, 100, n_bins_y + 1)
        
        hist, x_edges, y_edges = np.histogram2d(
            prod_arr / 1e3, pct_90_arr, bins=[x_edges / 1e3, y_edges])
        
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_width = x_edges[1] - x_edges[0]
        y_width = y_edges[1] - y_edges[0]
        
        xpos, ypos = np.meshgrid(x_centers, y_centers, indexing='ij')
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        
        dx = x_width * 0.8 * np.ones_like(xpos)
        dy = y_width * 0.8 * np.ones_like(ypos)
        dz = hist.flatten()
        
        mask = dz > 0
        if np.sum(mask) > 0:
            colors_3d = plt.cm.viridis(dz[mask] / np.max(dz[mask]))
            ax.bar3d(xpos[mask], ypos[mask], zpos[mask], 
                    dx[mask], dy[mask], dz[mask],
                    color=colors_3d, alpha=0.8, edgecolor='white', linewidth=0.2)
        
        ax.set_xlabel('Production (GW)', fontsize=10)
        ax.set_ylabel('% Locations for 90%', fontsize=10)
        ax.set_zlabel('Hours (freq)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=25, azim=45)
    
    ax1 = fig_3d.add_subplot(121, projection='3d')
    plot_3d_concentration(ax1, solar_prod_arr, solar_pct_90, 'Solar PV: Production Concentration', 'gold')
    
    ax2 = fig_3d.add_subplot(122, projection='3d')
    plot_3d_concentration(ax2, wind_prod_arr, wind_pct_90, 'Wind: Production Concentration', 'skyblue')
    
    plt.tight_layout()
    plt.show()
    
    # 2D heatmaps
    fig_heat, axes_heat = plt.subplots(1, 2, figsize=(14, 5))
    
    def plot_2d_heatmap(ax, prod_arr, pct_90_arr, title):
        if prod_arr is None or pct_90_arr is None or len(prod_arr) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        max_prod = np.percentile(prod_arr, 99)
        if max_prod <= 0 or not np.isfinite(max_prod):
            ax.text(0.5, 0.5, 'No significant production', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        n_bins = 30
        x_edges = np.linspace(0, max_prod, n_bins + 1)
        y_edges = np.linspace(0, 100, n_bins + 1)
        
        hist, _, _ = np.histogram2d(prod_arr / 1e3, pct_90_arr, bins=[x_edges / 1e3, y_edges])
        
        im = ax.imshow(hist.T, origin='lower', aspect='auto',
                      extent=[x_edges[0]/1e3, x_edges[-1]/1e3, 0, 100],
                      cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label='Hours (frequency)')
        ax.set_xlabel('Total Production (GW)')
        ax.set_ylabel('% Locations for 90% of Production')
        ax.set_title(title)
        
        avg_pct = np.mean(pct_90_arr)
        if np.isfinite(avg_pct):
            ax.axhline(avg_pct, color='cyan', linestyle='--', linewidth=2,
                      label=f'Avg: {avg_pct:.0f}%')
            ax.legend(loc='upper right')
    
    plot_2d_heatmap(axes_heat[0], solar_prod_arr, solar_pct_90, 'Solar PV: Concentration Heatmap')
    plot_2d_heatmap(axes_heat[1], wind_prod_arr, wind_pct_90, 'Wind: Concentration Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 INTERPRETATION:")
    print("   - Lower % = Production concentrated in fewer locations (some sites dominate)")
    print("   - Higher % = Production distributed evenly across locations")
    print("   - High production + Low % = Peak hours rely on best sites")
    print("   - Low production + High % = Low-resource hours spread thin")
    
    return {
        'solar': {'prod_arr': solar_prod_arr, 'pct_90': solar_pct_90},
        'wind': {'prod_arr': wind_prod_arr, 'pct_90': wind_pct_90}
    }


def print_annual_performance_summary(
    full_year_results: FullYearResults,
    config: Config
) -> None:
    """
    Print annual performance summary.
    """
    total_demand_twh = np.sum(full_year_results.demand) / 1e6
    total_prod_twh = np.sum(full_year_results.total_production) / 1e6
    total_ren_twh = np.sum(full_year_results.renewable_production) / 1e6
    
    print(f"\n📊 Energy Balance:")
    print(f"   Total Demand:       {total_demand_twh:.2f} TWh")
    print(f"   Total Production:   {total_prod_twh:.2f} TWh")
    print(f"   Renewable Only:     {total_ren_twh:.2f} TWh")
    print(f"   Spot Bought:        {full_year_results.total_spot_bought_mwh/1e6:.2f} TWh")
    print(f"   Spot Sold:          {full_year_results.total_spot_sold_mwh/1e6:.2f} TWh")
    
    print(f"\n💰 Spot Market Financials:")
    print(f"   Net Spot Cost:      {full_year_results.total_spot_cost_chf/1e9:.2f} B CHF")
    
    deficit_hours_count = np.sum(full_year_results.deficit > 0)
    surplus_hours_count = np.sum(full_year_results.surplus > 0)
    print(f"\n⚡ Grid Balance (after dispatch):")
    print(f"   Hours in Deficit:   {deficit_hours_count} h ({deficit_hours_count/8760*100:.1f}%)")
    print(f"   Hours in Surplus:   {surplus_hours_count} h ({surplus_hours_count/8760*100:.1f}%)")
    
    av_consumed = full_year_results.aviation_fuel_consumed_mwh / 1e6
    av_target = config.energy_system.AVIATION_FUEL_DEMAND_TWH_YEAR
    print(f"\n✈️  Aviation Fuel:")
    print(f"   Consumed:           {av_consumed:.2f} TWh")
    print(f"   Target:             {av_target:.2f} TWh")
    print(f"   Constraint Met:     {'✅ Yes' if av_consumed >= av_target * 0.99 else '❌ No'}")


def plot_end_to_end_effectiveness(
    full_year_results: FullYearResults,
    portfolio_dict: Dict[str, int],
    config: Config
) -> None:
    """
    Plot end-to-end effectiveness analysis with energy flows.
    """
    flows = calculate_energy_flows(full_year_results, portfolio_dict, config)
    
    # Print detailed summary
    print(f"\n📊 ENERGY FLOW SUMMARY (From Raw Incidence to Final Demand):")
    print(f"   {'Metric':<45} | {'Value (TWh)':>15} | {'% of Raw Incidence':>18}")
    print(f"   {'-'*45}-+-{'-'*15}-+-{'-'*18}")
    print(f"   {'Raw Incidence Energy':<45} | {flows['raw_incidence_twh']:>15.2f} | {'100.0':>18}%")
    print(f"   {'  → After Production Conversion':<45} | {flows['renewable_production_twh']:>15.2f} | {flows['renewable_production_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'    → Direct to Demand':<45} | {flows['renewable_direct_to_demand_twh']:>15.2f} | {flows['renewable_direct_to_demand_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'    → Storage Charged':<45} | {flows['storage_charged_twh']:>15.2f} | {flows['storage_charged_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'      → Storage Discharged':<45} | {flows['storage_discharged_twh']:>15.2f} | {flows['storage_discharged_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'        → To Flexible PPUs':<45} | {flows['storage_to_flexible_twh']:>15.2f} | {flows['storage_to_flexible_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'        → Direct to Demand':<45} | {flows['storage_direct_to_demand_twh']:>15.2f} | {flows['storage_direct_to_demand_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'Flexible Production (from storage)':<45} | {flows['flexible_production_twh']:>15.2f} | {flows['flexible_production_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'Spot Market Purchases':<45} | {flows['spot_bought_twh']:>15.2f} | {'N/A':>18}")
    print(f"   {'FINAL ENERGY DELIVERED (Demand)':<45} | {flows['energy_delivered_twh']:>15.2f} | {flows['energy_delivered_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    
    print(f"\n📉 ENERGY LOSSES BREAKDOWN:")
    print(f"   {'Loss Category':<45} | {'Value (TWh)':>15} | {'% of Raw Incidence':>18}")
    print(f"   {'-'*45}-+-{'-'*15}-+-{'-'*18}")
    print(f"   {'Production Losses (conversion)':<45} | {flows['production_losses_twh']:>15.2f} | {flows['production_losses_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'Storage Losses (round-trip)':<45} | {flows['storage_losses_twh']:>15.2f} | {flows['storage_losses_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'Flexible Production Losses':<45} | {flows['flexible_losses_twh']:>15.2f} | {flows['flexible_losses_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'Curtailment/Other Losses':<45} | {flows['renewable_curtailed_twh']:>15.2f} | {flows['renewable_curtailed_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    print(f"   {'TOTAL LOSSES':<45} | {flows['total_losses_twh']:>15.2f} | {flows['total_losses_twh']/flows['raw_incidence_twh']*100 if flows['raw_incidence_twh'] > 0 else 0:>17.1f}%")
    
    print(f"\n⚡ EFFECTIVENESS METRICS:")
    print(f"   {'Metric':<50} | {'Value':>10}")
    print(f"   {'-'*50}-+-{'-'*10}")
    print(f"   {'Overall System Efficiency (Raw → Delivered)':<50} | {flows['overall_efficiency']:>9.2f}%")
    print(f"   {'Renewable Utilization (% of demand)':<50} | {flows['renewable_utilization']:>9.2f}%")
    print(f"   {'Storage Utilization (% of renewable)':<50} | {flows['storage_utilization']:>9.2f}%")
    print(f"   {'Storage Round-Trip Efficiency':<50} | {flows['storage_rt_efficiency']:>9.2f}%")
    print(f"   {'Flexible Production Efficiency':<50} | {flows['flexible_efficiency']:>9.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart
    ax1 = axes[0]
    pie_data = {
        'Energy Delivered': flows['energy_delivered_twh'],
        'Production Losses': flows['production_losses_twh'],
        'Storage Losses': flows['storage_losses_twh'],
        'Flexible Production Losses': flows['flexible_losses_twh'],
        'Curtailment/Other': flows['renewable_curtailed_twh']
    }
    pie_data = {k: max(0, v) for k, v in pie_data.items() if v > 0.01}
    
    if pie_data:
        colors_pie = ['#2ecc71', '#e67e22', '#3498db', '#9b59b6', '#95a5a6']
        wedges, texts, autotexts = ax1.pie(
            list(pie_data.values()),
            labels=list(pie_data.keys()),
            autopct=lambda pct: f'{pct:.1f}%\n({pct*flows["raw_incidence_twh"]/100:.1f} TWh)',
            colors=colors_pie[:len(pie_data)],
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        ax1.set_title('Energy Breakdown: Delivered vs Losses\n(From Raw Incidence)', 
                     fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2 = axes[1]
    loss_categories = ['Production\nLosses', 'Storage\nLosses', 'Flexible\nLosses', 'Curtailment']
    loss_values = [flows['production_losses_twh'], flows['storage_losses_twh'], 
                   flows['flexible_losses_twh'], flows['renewable_curtailed_twh']]
    loss_colors = ['#e67e22', '#3498db', '#9b59b6', '#95a5a6']
    
    bars = ax2.bar(loss_categories, loss_values, color=loss_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Energy Losses (TWh)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Losses Breakdown by Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, loss_values):
        if val > 0.01:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} TWh\n({val/flows["raw_incidence_twh"]*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Overall System Efficiency: {flows['overall_efficiency']:.1f}% of raw incidence energy reaches final demand")
    print(f"   • Production Losses: {flows['production_losses_twh']:.1f} TWh ({flows['production_losses_twh']/flows['raw_incidence_twh']*100:.1f}%) lost during conversion")
    if flows['storage_charged_twh'] > 0:
        print(f"   • Storage Losses: {flows['storage_losses_twh']:.1f} TWh ({flows['storage_losses_twh']/flows['raw_incidence_twh']*100:.1f}%) lost in round-trip storage")
        print(f"   • Storage Round-Trip Efficiency: {flows['storage_rt_efficiency']:.1f}%")
    if flows['flexible_production_twh'] > 0:
        print(f"   • Flexible Production Losses: {flows['flexible_losses_twh']:.1f} TWh ({flows['flexible_losses_twh']/flows['raw_incidence_twh']*100:.1f}%) lost in flexible PPU conversion")
    if flows['renewable_curtailed_twh'] > 0:
        print(f"   • Curtailment: {flows['renewable_curtailed_twh']:.1f} TWh ({flows['renewable_curtailed_twh']/flows['raw_incidence_twh']*100:.1f}%) of renewable energy curtailed")
