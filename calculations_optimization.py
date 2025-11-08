"""
Optimization Framework Functions for PPU Portfolio Design
==========================================================

This module contains all functions for the multi-period stochastic optimization
of Power Production Unit (PPU) portfolios.

Step 1: Random Daily Scenario Generator
Step 2: Cost Escalation and Portfolio Encoding
Step 3: Single-Scenario Dispatch Validation
Step 4: Multi-Scenario Portfolio Evaluation
Step 5: Portfolio Search and Optimization
Step 6: Visualization and Frontier Analysis
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import pickle


# ============================================================================
# STEP 1: Random Daily Scenario Generator
# ============================================================================

def load_annual_data(data_dir: str = 'data/') -> Dict[str, Any]:
    """
    Load all 2024 annual datasets at 15-minute resolution.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
        
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'demand_15min': Series with demand in MW (35040 timesteps)
        - 'spot_15min': Series with spot prices in CHF/MWh
        - 'solar_incidence': DataFrame with solar capacity factors by location
        - 'wind_incidence': DataFrame with wind capacity factors by location
        - 'ror_15min': Series with run-of-river production
        - 'timestamp_index': DatetimeIndex for full year
    """
    data_path = Path(data_dir)
    
    # 1. Load hourly demand data - following existing pipeline format
    demand_hourly = pd.read_csv(
        data_path / 'monthly_hourly_load_values_2024.csv',
        sep='\t'
    )
    # Filter for Switzerland (CH) - as in analyze_monthly_hourly_load()
    demand_hourly = demand_hourly[demand_hourly['CountryCode'] == 'CH'].copy()
    # Parse datetime from DateUTC (format: DD-MM-YYYY HH:MM) - as in existing pipeline
    demand_hourly['datetime'] = pd.to_datetime(
        demand_hourly['DateUTC'], 
        format='%d-%m-%Y %H:%M'
    )
    # Drop duplicate timestamps (e.g., daylight saving time transitions)
    demand_hourly = demand_hourly.drop_duplicates(subset=['datetime'], keep='first')
    demand_hourly.set_index('datetime', inplace=True)
    demand_hourly.sort_index(inplace=True)
    # Resample hourly demand to 15-min with linear interpolation
    demand_15min = demand_hourly['Value'].resample('15min').interpolate(method='linear')
    
    # 2. Load spot prices - following existing pipeline format
    spot_df = pd.read_csv(
        data_path / 'spot_price_hourly.csv',
        parse_dates=['time']
    )
    spot_df.set_index('time', inplace=True)
    spot_df.sort_index(inplace=True)
    spot_15min = spot_df['price'].resample('15min').interpolate(method='linear')
    
    # 3. Load solar incidence - following existing pipeline format with lat/lon header rows
    import csv
    with open(data_path / 'solar_incidence_hourly_2024.csv', 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)
    
    # Read the actual data (skip first 3 header rows)
    solar_df = pd.read_csv(
        data_path / 'solar_incidence_hourly_2024.csv',
        skiprows=3
    )
    solar_df.rename(columns={solar_df.columns[0]: "datetime"}, inplace=True)
    solar_df["datetime"] = pd.to_datetime(solar_df["datetime"])
    solar_df.set_index("datetime", inplace=True)
    
    # Get mean solar incidence across all locations (spatial average)
    data_cols = [col for col in solar_df.columns if col != "datetime"]
    solar_hourly = solar_df[data_cols].mean(axis=1)
    solar_15min = solar_hourly.resample('15min').interpolate(method='linear')
    
    # 4. Load wind incidence - following existing pipeline format
    with open(data_path / 'wind_incidence_hourly_2024.csv', 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)
    
    wind_df = pd.read_csv(
        data_path / 'wind_incidence_hourly_2024.csv',
        skiprows=3
    )
    wind_df.rename(columns={wind_df.columns[0]: "datetime"}, inplace=True)
    wind_df["datetime"] = pd.to_datetime(wind_df["datetime"])
    wind_df.set_index("datetime", inplace=True)
    
    # Get mean wind incidence across all locations (spatial average)
    data_cols = [col for col in wind_df.columns if col != "datetime"]
    wind_hourly = wind_df[data_cols].mean(axis=1)
    wind_15min = wind_hourly.resample('15min').interpolate(method='linear')
    
    # 5. Load run-of-river monthly data - following existing pipeline format
    ror_monthly = pd.read_csv(data_path / 'water_monthly_ror_2024.csv')
    ror_monthly['Month'] = pd.to_datetime(ror_monthly['Month'], format='%Y-%m')
    ror_monthly.set_index('Month', inplace=True)
    
    # Create full year timestamp index at 15-min resolution
    timestamp_index = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-12-31 23:45:00',
        freq='15min'
    )
    
    # Distribute monthly RoR energy (GWh) uniformly across each month's 15-min intervals
    # Converting to power (MW): Energy (GWh) * 1000 / (num_intervals * 0.25 hours)
    ror_15min = pd.Series(index=timestamp_index, dtype=float)
    for month_start, row in ror_monthly.iterrows():
        # Get all timestamps in this month
        month_mask = (timestamp_index.year == month_start.year) & \
                     (timestamp_index.month == month_start.month)
        num_intervals = month_mask.sum()
        # Convert GWh to MW for 15-min intervals
        ror_15min[month_mask] = (row['RoR_GWh'] * 1000) / (num_intervals * 0.25)
    
    return {
        'demand_15min': demand_15min,
        'spot_15min': spot_15min,
        'solar_incidence': solar_15min,
        'wind_incidence': wind_15min,
        'ror_15min': ror_15min,
        'timestamp_index': timestamp_index
    }


def generate_random_scenario(
    annual_data: Dict[str, Any],
    num_days: int = 30,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Randomly select num_days individual days from annual data and concatenate.
    
    Parameters
    ----------
    annual_data : dict
        Output from load_annual_data()
    num_days : int
        Number of days to sample (default=30)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    scenario_dict : dict
        - 'demand_MW': Array of concatenated demand
        - 'spot_price': Array of concatenated spot prices
        - 'solar_cf': Array of solar capacity factors
        - 'wind_cf': Array of wind capacity factors
        - 'ror_MW': Array of RoR production
        - 'selected_days': List[int] selected day numbers (0-364)
        - 'num_days': int number of days sampled
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample random days (without replacement)
    n_days_in_year = 365
    day_indices = np.random.choice(n_days_in_year, size=num_days, replace=False)
    day_indices = sorted(day_indices.tolist())  # Sort for easier interpretation
    
    # Extract 96 timesteps per day and concatenate
    timesteps_per_day = 96
    
    # Initialize lists for concatenation
    demand_segments = []
    spot_segments = []
    solar_segments = []
    wind_segments = []
    ror_segments = []
    sampled_dates = []
    
    for day_idx in day_indices:
        start_idx = day_idx * timesteps_per_day
        end_idx = (day_idx + 1) * timesteps_per_day
        
        # Extract segments - handle Series/arrays
        demand_seg = annual_data['demand_15min'].iloc[start_idx:end_idx] if hasattr(annual_data['demand_15min'], 'iloc') else annual_data['demand_15min'][start_idx:end_idx]
        spot_seg = annual_data['spot_15min'].iloc[start_idx:end_idx] if hasattr(annual_data['spot_15min'], 'iloc') else annual_data['spot_15min'][start_idx:end_idx]
        solar_seg = annual_data['solar_incidence'].iloc[start_idx:end_idx] if hasattr(annual_data['solar_incidence'], 'iloc') else annual_data['solar_incidence'][start_idx:end_idx]
        wind_seg = annual_data['wind_incidence'].iloc[start_idx:end_idx] if hasattr(annual_data['wind_incidence'], 'iloc') else annual_data['wind_incidence'][start_idx:end_idx]
        ror_seg = annual_data['ror_15min'].iloc[start_idx:end_idx] if hasattr(annual_data['ror_15min'], 'iloc') else annual_data['ror_15min'][start_idx:end_idx]
        
        demand_segments.append(demand_seg)
        spot_segments.append(spot_seg)
        solar_segments.append(solar_seg)
        wind_segments.append(wind_seg)
        ror_segments.append(ror_seg)
        
        # Record the date
        sampled_dates.append(annual_data['timestamp_index'][start_idx])
    
    # Concatenate all segments - convert to numpy arrays
    scenario_demand = np.concatenate([np.array(s) for s in demand_segments])
    scenario_spot = np.concatenate([np.array(s) for s in spot_segments])
    scenario_solar = np.concatenate([np.array(s) for s in solar_segments])
    scenario_wind = np.concatenate([np.array(s) for s in wind_segments])
    scenario_ror = np.concatenate([np.array(s) for s in ror_segments])
    
    return {
        'demand_MW': scenario_demand,
        'spot_price': scenario_spot,
        'solar_cf': scenario_solar,
        'wind_cf': scenario_wind,
        'ror_MW': scenario_ror,
        'selected_days': day_indices,
        'num_days': num_days,
        'sampled_dates': sampled_dates
    }


def validate_scenario_completeness(
    scenario_dict: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Ensure scenario contains all required data and matches expected structure.
    
    Parameters
    ----------
    scenario_dict : dict
        Output from generate_random_scenario()
        
    Returns
    -------
    is_valid : bool
        True if all checks pass
    message : str
        Validation result message
    """
    # Check required keys
    required_keys = ['demand_MW', 'spot_price', 'solar_cf', 
                     'wind_cf', 'ror_MW', 'selected_days', 'num_days']
    
    missing_keys = [key for key in required_keys if key not in scenario_dict]
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    # Check array lengths match
    num_timesteps = len(scenario_dict['demand_MW'])
    expected_timesteps = scenario_dict['num_days'] * 96  # 96 timesteps per day
    
    for key in ['spot_price', 'solar_cf', 'wind_cf', 'ror_MW']:
        if len(scenario_dict[key]) != num_timesteps:
            return False, f"Length mismatch: {key} has {len(scenario_dict[key])} timesteps, expected {num_timesteps}"
    
    if num_timesteps != expected_timesteps:
        return False, f"Total timesteps {num_timesteps} doesn't match expected {expected_timesteps} for {scenario_dict['num_days']} days"
    
    return True, f"Scenario valid: {scenario_dict['num_days']} days, {num_timesteps} timesteps"
    
    if report['errors']:
        return False, report
    
    # Check lengths
    expected_length = 2880  # 30 days × 96 timesteps/day
    
    for key in ['demand_15min', 'spot_15min', 'ror_15min']:
        if len(scenario_dict[key]) != expected_length:
            report['errors'].append(
                f"{key} has length {len(scenario_dict[key])}, expected {expected_length}"
            )
    
    # Check DataFrame shapes
    if scenario_dict['solar_15min'].shape[0] != expected_length:
        report['errors'].append(
            f"solar_15min has {scenario_dict['solar_15min'].shape[0]} rows, expected {expected_length}"
        )
    
    if scenario_dict['wind_15min'].shape[0] != expected_length:
        report['errors'].append(
            f"wind_15min has {scenario_dict['wind_15min'].shape[0]} rows, expected {expected_length}"
        )
    
    # Check location counts
    if scenario_dict['solar_15min'].shape[1] != expected_locations['solar']:
        report['warnings'].append(
            f"solar_15min has {scenario_dict['solar_15min'].shape[1]} locations, "
            f"expected {expected_locations['solar']}"
        )
    
    if scenario_dict['wind_15min'].shape[1] != expected_locations['wind']:
        report['warnings'].append(
            f"wind_15min has {scenario_dict['wind_15min'].shape[1]} locations, "
            f"expected {expected_locations['wind']}"
        )
    
    # Check for NaN values
    if scenario_dict['demand_15min'].isna().any():
        report['errors'].append("demand_15min contains NaN values")
    
    if scenario_dict['spot_15min'].isna().any():
        report['errors'].append("spot_15min contains NaN values")
    
    # Check demand is positive
    if (scenario_dict['demand_15min'] <= 0).any():
        report['errors'].append("demand_15min contains non-positive values")
    
    # Check capacity factors in [0, 1]
    solar_min = scenario_dict['solar_15min'].min().min()
    solar_max = scenario_dict['solar_15min'].max().max()
    
    if solar_min < 0 or solar_max > 1:
        report['errors'].append(
            f"solar_15min capacity factors out of range [0,1]: [{solar_min}, {solar_max}]"
        )
    
    wind_min = scenario_dict['wind_15min'].min().min()
    wind_max = scenario_dict['wind_15min'].max().max()
    
    if wind_min < 0 or wind_max > 1:
        report['errors'].append(
            f"wind_15min capacity factors out of range [0,1]: [{wind_min}, {wind_max}]"
        )
    
    # Check day_indices
    if len(scenario_dict['day_indices']) != 30:
        report['errors'].append(
            f"day_indices has {len(scenario_dict['day_indices'])} days, expected 30"
        )
    
    if any(d < 0 or d >= 365 for d in scenario_dict['day_indices']):
        report['errors'].append("day_indices contains invalid day numbers (must be 0-364)")
    
    if len(set(scenario_dict['day_indices'])) != len(scenario_dict['day_indices']):
        report['errors'].append("day_indices contains duplicate days")
    
    # Summary
    if not report['errors']:
        report['info'].append(f"Scenario validated successfully: {expected_length} timesteps")
        return True, report
    else:
        return False, report


# ============================================================================
# STEP 2: Cost Escalation and Portfolio Encoding
# ============================================================================

def encode_portfolio(ppu_counts_dict: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert portfolio dictionary to standardized integer vector.
    
    Parameters
    ----------
    ppu_counts_dict : dict
        E.g., {'PV': 3, 'WD_OFF': 2, 'HYD_S': 1}
        
    Returns
    -------
    x : np.ndarray
        Integer vector of PPU counts
    tech_names : list
        Ordered technology names corresponding to x indices
    """
    # Define canonical ordering (alphabetical for consistency)
    tech_names = sorted(ppu_counts_dict.keys())
    
    # Create vector
    x = np.array([ppu_counts_dict[tech] for tech in tech_names], dtype=int)
    
    return x, tech_names


def decode_portfolio(x: np.ndarray, tech_names: List[str]) -> Dict[str, int]:
    """
    Convert integer vector back to portfolio dictionary.
    
    Parameters
    ----------
    x : np.ndarray
        Integer counts
    tech_names : list
        Technology names
        
    Returns
    -------
    ppu_counts_dict : dict
        Reconstructed portfolio (only non-zero entries)
    """
    # Only include technologies with non-zero counts
    return {tech_names[i]: int(x[i]) for i in range(len(x)) if x[i] > 0}


def calculate_portfolio_capex(
    ppu_counts_dict: Dict[str, int],
    cost_table_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute total capital cost with per-unit escalation penalty.
    
    Formula: CAPEX_t(k) = C_t * k * [1 + 0.1 * (k-1) / 2]
    
    Note: Only Store PPUs have cost escalation. Incidence PPUs (PV, WD_ON, WD_OFF)
    do not have escalation as they don't appear in cost table.
    
    Parameters
    ----------
    ppu_counts_dict : dict
        Portfolio specification
    cost_table_df : DataFrame
        Columns=['PPU_Name', 'cost_chf_per_kwh']
        
    Returns
    -------
    result : dict
        - 'total_capex': float, total cost in CHF
        - 'cost_breakdown': dict, per-technology costs
    """
    cost_breakdown = {}
    total_capex = 0.0
    
    # Create lookup dict from cost table
    cost_lookup = dict(zip(cost_table_df['PPU_Name'], cost_table_df['cost_chf_per_kwh']))
    
    for tech, count in ppu_counts_dict.items():
        if count == 0:
            continue
            
        # Only calculate cost if technology is in cost table
        # (Incidence PPUs like PV, WD_ON, WD_OFF are not in cost table)
        if tech in cost_lookup:
            base_cost = cost_lookup[tech]
            
            # Apply escalation formula: sum_{i=1}^{k} C * [1 + 0.1*(i-1)]
            # Simplified: C * k * [1 + 0.1 * (k-1) / 2]
            tech_cost = base_cost * count * (1 + 0.1 * (count - 1) / 2)
            
            cost_breakdown[tech] = tech_cost
            total_capex += tech_cost
    
    return {
        'total_capex': total_capex,
        'cost_breakdown': cost_breakdown
    }


def assign_renewable_locations(
    ppu_counts_dict: Dict[str, int],
    solar_ranking_df: pd.DataFrame,
    wind_ranking_df: pd.DataFrame
) -> Dict[str, int]:
    """
    Assign unique geographic locations to renewable PPUs using best-first ranking.
    
    Parameters
    ----------
    ppu_counts_dict : dict
        Portfolio with renewable counts
    solar_ranking_df : DataFrame
        Columns=['location_id', 'annual_capacity_factor'], sorted descending
    wind_ranking_df : DataFrame
        Similar structure for wind
        
    Returns
    -------
    location_map : dict
        {PPU_ID: location_rank}, e.g., {'PV_1': 1, 'PV_2': 2, ...}
        
    Raises
    ------
    ValueError
        If insufficient locations available
    """
    location_map = {}
    
    # Count solar PPUs (PV)
    solar_ppus = ['PV']
    total_solar = sum(ppu_counts_dict.get(tech, 0) for tech in solar_ppus)
    
    if total_solar > len(solar_ranking_df):
        raise ValueError(
            f"Insufficient solar locations: need {total_solar}, "
            f"have {len(solar_ranking_df)}"
        )
    
    # Assign solar locations
    solar_idx = 0
    for tech in solar_ppus:
        count = ppu_counts_dict.get(tech, 0)
        for i in range(count):
            ppu_id = f"{tech}_{i+1}"
            location_map[ppu_id] = solar_ranking_df.iloc[solar_idx]['location_id']
            solar_idx += 1
    
    # Count wind PPUs (WD_ON, WD_OFF)
    wind_ppus = ['WD_ON', 'WD_OFF']
    total_wind = sum(ppu_counts_dict.get(tech, 0) for tech in wind_ppus)
    
    if total_wind > len(wind_ranking_df):
        raise ValueError(
            f"Insufficient wind locations: need {total_wind}, "
            f"have {len(wind_ranking_df)}"
        )
    
    # Assign wind locations
    wind_idx = 0
    for tech in wind_ppus:
        count = ppu_counts_dict.get(tech, 0)
        for i in range(count):
            ppu_id = f"{tech}_{i+1}"
            location_map[ppu_id] = wind_ranking_df.iloc[wind_idx]['location_id']
            wind_idx += 1
    
    return location_map


# ============================================================================
# STEP 3: Single-Scenario Dispatch Validation
# ============================================================================

# These functions will wrap the existing dispatch simulation from
# calculationPipelineFramework.py

def run_single_scenario_dispatch(
    portfolio_dict: Dict[str, int],
    scenario_dict: Dict[str, Any],
    hyperparams: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute simplified dispatch simulation for a given portfolio over a single scenario.
    
    This implements a greedy heuristic dispatch:
    - Renewables produce at capacity factor
    - RoR produces at available level
    - Storage and conventional plants balance remaining demand
    - Spot market handles final surplus/deficit
    
    Parameters
    ----------
    portfolio_dict : dict
        PPU counts, e.g. {'hydro_storage': 2, 'nuclear_large': 1, ...}
    scenario_dict : dict
        From generate_random_scenario() with keys: demand_MW, spot_price, solar_cf, wind_cf, ror_MW
    hyperparams : dict, optional
        Configuration (defaults provided if None)
        
    Returns
    -------
    results_dict : dict
        - 'net_cost_CHF': Total operational cost (spot transactions + storage losses)
        - 'energy_balance': Array of net energy at each timestep (demand - supply)
        - 'spot_transactions': Array of spot market purchases (MW)
        - 'storage_soc': Dict of SOC trajectories for each storage PPU
        - 'diagnostics': Dictionary with HHI, spot_dependence, etc.
    """
    if hyperparams is None:
        hyperparams = {
            'storage_capacity_mwh': 1000.0,  # MWh per storage PPU
            'storage_efficiency': 0.85,
            'nuclear_capacity_mw': 1000.0,  # MW per nuclear unit
            'hydro_capacity_mw': 500.0,     # MW per hydro unit
        }
    
    n_timesteps = len(scenario_dict['demand_MW'])
    demand = scenario_dict['demand_MW']
    spot_price = scenario_dict['spot_price']
    solar_cf = scenario_dict['solar_cf']
    wind_cf = scenario_dict['wind_cf']
    ror_mw = scenario_dict['ror_MW']
    
    # Initialize arrays
    supply = np.zeros(n_timesteps)
    spot_transactions = np.zeros(n_timesteps)  # Positive = buy, negative = sell
    net_cost = 0.0
    
    # 1. Add renewable production (at capacity factor)
    solar_capacity = portfolio_dict.get('solar_pv', 0) * 100.0  # 100 MW per PV unit
    wind_capacity = portfolio_dict.get('wind_onshore', 0) * 100.0  # 100 MW per wind unit
    
    renewable_supply = solar_capacity * solar_cf + wind_capacity * wind_cf + ror_mw
    
    # 2. Add baseload (nuclear runs at constant capacity)
    nuclear_capacity = portfolio_dict.get('nuclear_large', 0) * hyperparams['nuclear_capacity_mw']
    baseload_supply = np.full(n_timesteps, nuclear_capacity)
    
    # 3. Simple storage dispatch: charge when surplus, discharge when deficit
    storage_units = portfolio_dict.get('hydro_storage', 0)
    storage_capacity_total = storage_units * hyperparams['storage_capacity_mwh']
    storage_soc = storage_capacity_total * 0.5  # Start at 50% SOC
    storage_soc_trajectory = []
    storage_discharge = np.zeros(n_timesteps)
    storage_charge = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        # Calculate available supply before storage/spot
        available_supply = renewable_supply[t] + baseload_supply[t]
        balance = demand[t] - available_supply
        
        if balance > 0:  # Need more energy - discharge storage or buy from spot
            # Try to discharge storage first
            max_discharge_power = storage_units * hyperparams['hydro_capacity_mw']
            max_discharge_energy = storage_soc * hyperparams['storage_efficiency']
            discharge = min(max_discharge_power, max_discharge_energy, balance)
            
            storage_soc -= discharge / hyperparams['storage_efficiency']
            storage_discharge[t] = discharge
            balance -= discharge
            
            # Remaining deficit from spot market
            spot_transactions[t] = balance
            net_cost += balance * spot_price[t] * 0.25  # 0.25 hours per timestep
            
        else:  # Surplus energy - charge storage or sell to spot
            surplus = -balance
            # Try to charge storage first
            max_charge_power = storage_units * hyperparams['hydro_capacity_mw']
            max_charge_capacity = (storage_capacity_total - storage_soc) / hyperparams['storage_efficiency']
            charge = min(max_charge_power, max_charge_capacity, surplus)
            
            storage_soc += charge * hyperparams['storage_efficiency']
            storage_charge[t] = charge
            surplus -= charge
            
            # Sell remaining surplus to spot market  
            spot_transactions[t] = -surplus
            net_cost -= surplus * spot_price[t] * 0.25
        
        storage_soc_trajectory.append(storage_soc)
    
    # Total supply = renewables + baseload + storage discharge - storage charge + spot
    supply = renewable_supply + baseload_supply + storage_discharge - storage_charge + spot_transactions
    
    # Calculate diagnostics
    total_demand = np.sum(demand) * 0.25  # MWh
    total_spot_buy = np.sum(np.maximum(spot_transactions, 0)) * 0.25
    spot_dependence = total_spot_buy / total_demand if total_demand > 0 else 0
    
    # Calculate Herfindahl-Hirschman Index (HHI) for energy mix diversity
    total_renewable = np.sum(renewable_supply) * 0.25
    total_nuclear = np.sum(baseload_supply) * 0.25
    total_storage_discharge = np.sum(storage_discharge) * 0.25
    total_supply = total_renewable + total_nuclear + total_storage_discharge + total_spot_buy
    
    shares = []
    if total_supply > 0:
        if total_renewable > 0:
            shares.append((total_renewable / total_supply) ** 2)
        if total_nuclear > 0:
            shares.append((total_nuclear / total_supply) ** 2)
        if total_storage_discharge > 0:
            shares.append((total_storage_discharge / total_supply) ** 2)
        if total_spot_buy > 0:
            shares.append((total_spot_buy / total_supply) ** 2)
    
    hhi = sum(shares) if shares else 0
    
    return {
        'net_cost_CHF': net_cost,
        'energy_balance': demand - supply,
        'spot_transactions': spot_transactions,
        'storage_soc': {'hydro_storage': storage_soc_trajectory},
        'diagnostics': {
            'hhi': hhi,
            'spot_dependence': spot_dependence,
            'total_demand_mwh': total_demand,
            'total_spot_buy_mwh': total_spot_buy,
            'avg_storage_soc': np.mean(storage_soc_trajectory) / storage_capacity_total if storage_capacity_total > 0 else 0
        }
    }


def validate_energy_balance(
    results: Dict[str, Any],
    scenario: Dict[str, Any],
    epsilon: float = 1e-3
) -> Tuple[bool, float, List[int]]:
    """
    Verify energy conservation at every timestep.
    
    Energy balance equation:
    Production(t) + Storage_Discharge(t) + Spot_Buy(t) = 
        Demand(t) + Storage_Charge(t) + Spot_Sell(t)
    
    Parameters
    ----------
    results : dict
        Output from run_single_scenario_dispatch()
    scenario : dict
        Scenario data
    epsilon : float
        Tolerance for numerical errors (MW)
        
    Returns
    -------
    is_valid : bool
        True if all timesteps balanced
    max_imbalance : float
        Largest absolute imbalance in MW
    imbalance_timesteps : list
        Indices where |imbalance| > epsilon
    """
    # TODO: Implement energy balance validation
    raise NotImplementedError("Energy balance validation not yet implemented")


def validate_storage_bounds(
    raw_energy_storage: List[Dict[str, Any]],
    epsilon: float = 1e-3
) -> Tuple[List[Tuple], bool]:
    """
    Ensure all storage SOC values remain within physical limits.
    
    Constraint: 0 <= SOC_s(t) <= Capacity_s for all s, t
    
    Parameters
    ----------
    raw_energy_storage : list
        Storage state dictionaries from dispatch
    epsilon : float
        Tolerance for bound violations
        
    Returns
    -------
    violations : list
        List of (storage_name, timestep, SOC, capacity, violation_type)
    is_valid : bool
        True if len(violations) == 0
    """
    violations = []
    
    for storage in raw_energy_storage:
        storage_name = storage['storage']
        capacity = storage['value']
        history = storage.get('history', [])
        
        for timestep, soc in history:
            if soc < -epsilon:
                violations.append((
                    storage_name, timestep, soc, capacity, 'underflow'
                ))
            elif soc > capacity + epsilon:
                violations.append((
                    storage_name, timestep, soc, capacity, 'overflow'
                ))
    
    return violations, len(violations) == 0


# ============================================================================
# STEP 4: Multi-Scenario Portfolio Evaluation
# ============================================================================

def evaluate_portfolio_multiscenario(
    portfolio_dict: Dict[str, int],
    data_dict: Dict[str, Any],
    n_scenarios: int = 20,
    seed_base: Optional[int] = None,
    hyperparams: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run portfolio across N random scenarios and aggregate cost statistics.
    
    Parameters
    ----------
    portfolio_dict : dict
        PPU counts
    data_dict : dict
        Annual data from load_annual_data()
    n_scenarios : int
        Number of scenarios to evaluate
    seed_base : int, optional
        Base seed for scenario generation
    hyperparams : dict
        Dispatch configuration
        
    Returns
    -------
    evaluation_results : dict
        - 'portfolio': Input portfolio
        - 'scenario_costs': np.array of length N
        - 'mean_cost': Arithmetic mean
        - 'cvar_95': CVaR at 95% confidence
        - 'diagnostics': Aggregated metrics
    """
    costs = []
    all_diagnostics = []
    
    for i in range(n_scenarios):
        # Generate scenario
        seed = seed_base + i if seed_base is not None else None
        scenario = generate_random_scenario(annual_data=data_dict, num_days=30, seed=seed)
        
        # Run dispatch
        results = run_single_scenario_dispatch(portfolio_dict, scenario, hyperparams)
        
        # Record cost
        costs.append(results['net_cost_CHF'])
        all_diagnostics.append(results['diagnostics'])
    
    # Compute statistics
    scenario_costs = np.array(costs)
    mean_cost = np.mean(scenario_costs)
    cvar_95 = compute_cvar_95(scenario_costs, alpha=0.95)
    
    # Aggregate diagnostics
    aggregated_diagnostics = aggregate_diagnostics(all_diagnostics)
    
    return {
        'portfolio': portfolio_dict,
        'scenario_costs': scenario_costs,
        'mean_cost': mean_cost,
        'cvar_95': cvar_95,
        'diagnostics': aggregated_diagnostics
    }


def compute_cvar_95(costs: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Conditional Value-at-Risk at specified confidence level.
    
    CVaR_α = E[C | C >= VaR_α]
    
    For N=20, α=0.95: worst 5% = top 1 scenario
    VaR_idx = ceil(N * α) - 1
    CVaR = mean(costs[VaR_idx:])
    
    Parameters
    ----------
    costs : np.ndarray
        Scenario cost realizations
    alpha : float
        Confidence level (default=0.95)
        
    Returns
    -------
    cvar : float
        Mean of worst (1-α)% of scenarios
    """
    sorted_costs = np.sort(costs)
    n = len(costs)
    
    # Index of VaR (Value-at-Risk)
    var_idx = int(np.ceil(n * alpha)) - 1
    
    # CVaR is mean of tail (costs >= VaR)
    cvar = np.mean(sorted_costs[var_idx:])
    
    return cvar


def aggregate_diagnostics(scenario_results_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute mean and standard deviation of diagnostic metrics across scenarios.
    
    Parameters
    ----------
    scenario_results_list : list
        List of diagnostic dicts from each scenario
        
    Returns
    -------
    aggregated_diagnostics : dict
        - 'avg_hhi': Mean Herfindahl-Hirschman Index
        - 'std_hhi': Standard deviation of HHI
        - 'avg_spot_dependence': Mean fraction of demand met by spot
        - 'avg_storage_utilization': Mean % of storage capacity used
    """
    if not scenario_results_list:
        return {}
    
    # Extract metrics
    hhi_values = [d.get('hhi', 0) for d in scenario_results_list]
    spot_dep_values = [d.get('spot_dependence', 0) for d in scenario_results_list]
    storage_util_values = [d.get('storage_utilization', 0) for d in scenario_results_list]
    
    return {
        'avg_hhi': np.mean(hhi_values),
        'std_hhi': np.std(hhi_values, ddof=1) if len(hhi_values) > 1 else 0,
        'avg_spot_dependence': np.mean(spot_dep_values),
        'avg_storage_utilization': np.mean(storage_util_values)
    }


# ============================================================================
# STEP 5: Portfolio Search and Optimization
# ============================================================================

def random_portfolio_search(
    n_portfolios: int,
    data_dict: Dict[str, Any],
    n_scenarios_per: int = 20,
    tech_bounds: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
    hyperparams: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Baseline search - generate random portfolios and evaluate.
    
    Parameters
    ----------
    n_portfolios : int
        Number of random portfolios to generate
    data_dict : dict
        Annual data
    n_scenarios_per : int
        Scenarios per portfolio evaluation
    tech_bounds : dict
        Max units per technology, e.g., {'PV': 50, 'WD_ON': 30}
    seed : int, optional
        Random seed
    hyperparams : dict
        Dispatch configuration
        
    Returns
    -------
    results_df : DataFrame
        Columns = ['portfolio_dict', 'mean_cost', 'cvar_95', 'hhi', ...]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if tech_bounds is None:
        tech_bounds = {
            'PV': 20, 'WD_ON': 20, 'WD_OFF': 20,
            'HYD_S': 10, 'PHS': 10, 'H2_G': 10, 'H2_FC': 10
        }
    
    results = []
    
    for i in range(n_portfolios):
        # Generate random portfolio
        portfolio = {}
        for tech, max_count in tech_bounds.items():
            portfolio[tech] = np.random.randint(0, max_count + 1)
        
        # Evaluate
        eval_results = evaluate_portfolio_multiscenario(
            portfolio, data_dict, n_scenarios_per, 
            seed_base=i*1000 if seed is not None else None,
            hyperparams=hyperparams
        )
        
        # Record
        results.append({
            'portfolio_dict': portfolio,
            'mean_cost': eval_results['mean_cost'],
            'cvar_95': eval_results['cvar_95'],
            'hhi': eval_results['diagnostics'].get('avg_hhi', 0),
            'spot_dependence': eval_results['diagnostics'].get('avg_spot_dependence', 0)
        })
    
    return pd.DataFrame(results)


def get_pareto_frontier(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter results to non-dominated solutions in (mean_cost, cvar_95) space.
    
    Portfolio x dominates y if:
    - mean_cost(x) <= mean_cost(y) AND cvar_95(x) <= cvar_95(y)
    - AND at least one inequality is strict
    
    Parameters
    ----------
    results_df : DataFrame
        All evaluated portfolios
        
    Returns
    -------
    pareto_df : DataFrame
        Subset of non-dominated portfolios
    """
    pareto_indices = []
    
    for i in range(len(results_df)):
        is_dominated = False
        
        for j in range(len(results_df)):
            if i == j:
                continue
            
            # Check if j dominates i
            mean_i, cvar_i = results_df.iloc[i][['mean_cost', 'cvar_95']]
            mean_j, cvar_j = results_df.iloc[j][['mean_cost', 'cvar_95']]
            
            if (mean_j <= mean_i and cvar_j <= cvar_i and 
                (mean_j < mean_i or cvar_j < cvar_i)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return results_df.iloc[pareto_indices].copy()


def save_evaluation_result(
    portfolio: Dict[str, int],
    eval_results: Dict[str, Any],
    db_path: str = 'results.pkl'
) -> None:
    """
    Persist portfolio evaluation to disk for incremental search.
    
    Parameters
    ----------
    portfolio : dict
        Portfolio specification
    eval_results : dict
        Output from evaluate_portfolio_multiscenario()
    db_path : str
        File path (pickle format)
    """
    from pathlib import Path
    import pickle
    
    db_file = Path(db_path)
    
    # Load existing results if file exists
    if db_file.exists():
        with open(db_file, 'rb') as f:
            results_list = pickle.load(f)
    else:
        results_list = []
    
    # Append new result
    results_list.append({
        'portfolio': portfolio,
        'eval_results': eval_results
    })
    
    # Save back
    with open(db_file, 'wb') as f:
        pickle.dump(results_list, f)


def load_all_results(db_path: str = 'results.pkl') -> pd.DataFrame:
    """
    Load all saved portfolio evaluations from disk.
    
    Parameters
    ----------
    db_path : str
        File path
        
    Returns
    -------
    results_df : DataFrame
        All saved evaluations
    """
    from pathlib import Path
    import pickle
    
    db_file = Path(db_path)
    
    if not db_file.exists():
        return pd.DataFrame()
    
    with open(db_file, 'rb') as f:
        results_list = pickle.load(f)
    
    # Convert to DataFrame
    rows = []
    for entry in results_list:
        portfolio = entry['portfolio']
        eval_res = entry['eval_results']
        
        rows.append({
            'portfolio_dict': portfolio,
            'mean_cost': eval_res['mean_cost'],
            'cvar_95': eval_res['cvar_95'],
            'hhi': eval_res['diagnostics'].get('avg_hhi', 0),
            'spot_dependence': eval_res['diagnostics'].get('avg_spot_dependence', 0)
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# STEP 6: Visualization and Frontier Analysis
# ============================================================================

def plot_efficient_frontier(
    results_df: pd.DataFrame,
    highlight_portfolios: Optional[List[int]] = None,
    color_by: str = 'hhi'
) -> Any:
    """
    Create scatter plot of mean cost vs CVaR₉₅ with Pareto frontier highlighted.
    
    Parameters
    ----------
    results_df : DataFrame
        All portfolio evaluations
    highlight_portfolios : list, optional
        Indices to annotate
    color_by : str
        Column to use for color mapping (default='hhi')
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Scatter plot object
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get Pareto frontier
    pareto_df = get_pareto_frontier(results_df)
    pareto_indices = pareto_df.index
    
    # Plot all points
    scatter = ax.scatter(
        results_df['mean_cost'],
        results_df['cvar_95'],
        c=results_df[color_by],
        cmap='viridis',
        alpha=0.6,
        s=50,
        label='All portfolios'
    )
    
    # Highlight Pareto frontier
    ax.scatter(
        pareto_df['mean_cost'],
        pareto_df['cvar_95'],
        c='red',
        marker='*',
        s=200,
        edgecolors='black',
        linewidths=1.5,
        label='Pareto frontier',
        zorder=10
    )
    
    # Annotate highlights
    if highlight_portfolios:
        for idx in highlight_portfolios:
            if idx in results_df.index:
                ax.annotate(
                    f'P{idx}',
                    (results_df.loc[idx, 'mean_cost'], results_df.loc[idx, 'cvar_95']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
    
    ax.set_xlabel('Mean Cost (CHF)', fontsize=12)
    ax.set_ylabel('CVaR₉₅ (CHF)', fontsize=12)
    ax.set_title('Portfolio Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_by.upper(), fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_portfolio_composition(
    portfolio_dict: Dict[str, int],
    title: Optional[str] = None
) -> Any:
    """
    Horizontal bar chart showing PPU counts by technology.
    
    Parameters
    ----------
    portfolio_dict : dict
        PPU counts
    title : str, optional
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if not portfolio_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Empty Portfolio', 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by count descending
    sorted_items = sorted(portfolio_dict.items(), key=lambda x: x[1], reverse=True)
    techs, counts = zip(*sorted_items)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(techs))
    bars = ax.barh(y_pos, counts, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.1, i, str(count), 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(techs, fontsize=11)
    ax.set_xlabel('Number of Units', fontsize=12)
    ax.set_title(title or 'Portfolio Composition', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_deficit_surplus_timeline(
    results: Dict[str, Any],
    scenario_idx: int = 0
) -> Any:
    """
    30-day timeline with shaded regions for surplus (green) and deficit (red).
    
    Parameters
    ----------
    results : dict
        Dispatch results with 'overflow_series'
    scenario_idx : int
        Which scenario to plot (default=0)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    overflow = results['overflow_series']
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    timesteps = np.arange(len(overflow))
    
    # Plot line
    ax.plot(timesteps, overflow, color='black', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Shade surplus (positive) in green
    ax.fill_between(timesteps, 0, overflow, 
                     where=(overflow >= 0), 
                     color='green', alpha=0.3, label='Surplus')
    
    # Shade deficit (negative) in red
    ax.fill_between(timesteps, 0, overflow, 
                     where=(overflow < 0), 
                     color='red', alpha=0.3, label='Deficit')
    
    ax.set_xlabel('Timestep (15-min intervals)', fontsize=12)
    ax.set_ylabel('Energy Surplus/Deficit (MW)', fontsize=12)
    ax.set_title('30-Day Energy Balance Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_storage_trajectories(
    results: Dict[str, Any],
    storage_names: Optional[List[str]] = None
) -> Any:
    """
    Multi-panel plot showing SOC over time for all storage types.
    
    Parameters
    ----------
    results : dict
        Dispatch results with 'raw_energy_storage'
    storage_names : list, optional
        Which storages to plot (default=all)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Subplots for each storage
    """
    import matplotlib.pyplot as plt
    
    raw_storage = results['raw_energy_storage']
    
    if storage_names is None:
        storage_names = [s['storage'] for s in raw_storage]
    
    n_storages = len(storage_names)
    
    fig, axes = plt.subplots(n_storages, 1, figsize=(14, 3*n_storages), sharex=True)
    
    if n_storages == 1:
        axes = [axes]
    
    for ax, storage_name in zip(axes, storage_names):
        # Find storage data
        storage_data = next((s for s in raw_storage if s['storage'] == storage_name), None)
        
        if storage_data is None:
            ax.text(0.5, 0.5, f'{storage_name} not found', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        capacity = storage_data['value']
        history = storage_data.get('history', [])
        
        if history:
            timesteps, socs = zip(*history)
            
            # Plot SOC
            ax.plot(timesteps, socs, color='blue', linewidth=1.5, label='SOC')
            
            # Plot capacity line
            ax.axhline(y=capacity, color='red', linestyle='--', 
                      linewidth=2, label=f'Capacity ({capacity:.0f} MWh)')
            
            ax.fill_between(timesteps, 0, socs, alpha=0.2, color='blue')
        
        ax.set_ylabel(f'{storage_name}\nSOC (MWh)', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep (15-min intervals)', fontsize=12)
    fig.suptitle('Storage State of Charge Trajectories', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig


def generate_portfolio_summary_table(
    portfolio_dict: Dict[str, int],
    eval_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create formatted summary table with key metrics.
    
    Parameters
    ----------
    portfolio_dict : dict
        Portfolio specification
    eval_results : dict
        From evaluate_portfolio_multiscenario()
        
    Returns
    -------
    summary_df : DataFrame
        Rows = metrics, single column with values
    """
    # Calculate total capacity (this would need actual capacity data)
    # Placeholder for now
    total_units = sum(portfolio_dict.values())
    
    summary = {
        'Total Units': total_units,
        'Mean Cost (CHF)': f"{eval_results['mean_cost']:,.0f}",
        'CVaR₉₅ (CHF)': f"{eval_results['cvar_95']:,.0f}",
        'HHI': f"{eval_results['diagnostics'].get('avg_hhi', 0):.3f}",
        'Spot Dependence (%)': f"{eval_results['diagnostics'].get('avg_spot_dependence', 0)*100:.1f}%",
        'Storage Utilization (%)': f"{eval_results['diagnostics'].get('avg_storage_utilization', 0)*100:.1f}%"
    }
    
    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
    summary_df.index.name = 'Metric'
    
    return summary_df
