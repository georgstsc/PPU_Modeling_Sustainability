"""
================================================================================
DISPATCH ENGINE - Core Simulation Logic
================================================================================

This module implements the dispatch simulation for the Swiss Energy Storage
Optimization project. It handles:
- Renewable energy collection (incidence-based production)
- Storage charging and discharging
- Grid balancing decisions
- Cost tracking

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from numba import njit, prange
import warnings

from config import Config, DEFAULT_CONFIG


# =============================================================================
# DISPATCH STATE TRACKING
# =============================================================================

@dataclass
class StorageState:
    """State of a single storage system."""
    
    name: str
    capacity_mwh: float  # Maximum capacity
    current_mwh: float  # Current stored energy
    max_power_mw: float  # Maximum charge/discharge rate
    efficiency_charge: float
    efficiency_discharge: float
    
    # Import pricing (for palm oil, biooil)
    import_price_chf_per_mwh: Optional[float] = None
    
    @property
    def soc(self) -> float:
        """State of Charge [0, 1]."""
        if self.capacity_mwh > 0:
            return self.current_mwh / self.capacity_mwh
        return 0.0
    
    def can_discharge(self, amount_mwh: float) -> float:
        """Return how much can actually be discharged."""
        # Limited by current storage and power rating
        max_from_power = self.max_power_mw * 0.25  # 15-min timestep
        max_from_storage = self.current_mwh * self.efficiency_discharge
        return min(amount_mwh, max_from_power, max_from_storage)
    
    def can_charge(self, amount_mwh: float) -> float:
        """Return how much can actually be charged."""
        # Limited by remaining capacity and power rating
        max_from_power = self.max_power_mw * 0.25
        remaining = self.capacity_mwh - self.current_mwh
        max_from_capacity = remaining / self.efficiency_charge if self.efficiency_charge > 0 else 0
        return min(amount_mwh, max_from_power, max_from_capacity)
    
    def discharge(self, amount_mwh: float) -> float:
        """Discharge storage. Returns actual energy delivered."""
        actual = self.can_discharge(amount_mwh)
        energy_withdrawn = actual / self.efficiency_discharge if self.efficiency_discharge > 0 else actual
        self.current_mwh = max(0, self.current_mwh - energy_withdrawn)
        return actual
    
    def charge(self, amount_mwh: float) -> float:
        """Charge storage. Returns actual energy absorbed."""
        actual = self.can_charge(amount_mwh)
        energy_stored = actual * self.efficiency_charge
        self.current_mwh = min(self.capacity_mwh, self.current_mwh + energy_stored)
        return actual


@dataclass
class DispatchState:
    """Complete state for a dispatch simulation."""
    
    storages: Dict[str, StorageState] = field(default_factory=dict)
    
    # Tracking arrays (filled during simulation)
    overflow_series: List[float] = field(default_factory=list)  # Deficit/surplus per timestep
    production_by_ppu: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    spot_bought: List[Tuple[int, float]] = field(default_factory=list)
    spot_sold: List[Tuple[int, float]] = field(default_factory=list)
    
    # Summary statistics
    total_deficit_mwh: float = 0.0
    total_surplus_mwh: float = 0.0
    total_spot_buy_mwh: float = 0.0
    total_spot_sell_mwh: float = 0.0
    total_spot_buy_cost: float = 0.0
    total_spot_sell_revenue: float = 0.0
    
    def copy(self) -> 'DispatchState':
        """Create a deep copy of this state."""
        new_state = DispatchState()
        for name, storage in self.storages.items():
            new_state.storages[name] = StorageState(
                name=storage.name,
                capacity_mwh=storage.capacity_mwh,
                current_mwh=storage.current_mwh,
                max_power_mw=storage.max_power_mw,
                efficiency_charge=storage.efficiency_charge,
                efficiency_discharge=storage.efficiency_discharge,
                import_price_chf_per_mwh=storage.import_price_chf_per_mwh,
            )
        return new_state


# =============================================================================
# DISPATCH INDICES (Decision Support)
# =============================================================================

@njit(cache=True)
def calculate_disposition_index(
    soc: float,
    soc_target: float = 0.60,
    deadband: float = 0.05,
    alpha: float = 1.0
) -> float:
    """
    Calculate disposition index for storage willingness to discharge.
    
    Returns value in [-1, 1]:
    - +1: Storage full → wants to discharge
    - -1: Storage empty → wants to charge
    -  0: At target → neutral
    """
    if soc > soc_target + deadband:
        delta = soc - (soc_target + deadband)
    elif soc < soc_target - deadband:
        delta = soc - (soc_target - deadband)
    else:
        delta = 0.0
    
    max_excursion = max(1.0 - soc_target - deadband, soc_target - deadband)
    if max_excursion > 0:
        delta_norm = delta / max_excursion
    else:
        delta_norm = 0.0
    
    return np.tanh(delta_norm / alpha)


@njit(cache=True)
def calculate_utility_indices(
    phi_t: float,
    phi_smoothed: float,
    alpha_u: float = 1000.0
) -> Tuple[float, float]:
    """
    Calculate utility indices for system-wide dispatch context.
    
    Args:
        phi_t: Net system shortfall (MW). >0 means deficit.
        phi_smoothed: Smoothed shortfall (EMA)
        alpha_u: Scaling parameter
        
    Returns:
        (u_dis, u_chg): Discharge and charge utility indices in [-1, 1]
    """
    u_dis = np.tanh(phi_smoothed / alpha_u)
    
    if phi_t < 0:
        u_chg = np.tanh(-phi_smoothed / alpha_u)
    else:
        u_chg = 0.0
    
    return u_dis, u_chg


@njit(cache=True)
def exponential_moving_average(
    current: float,
    previous_ema: float,
    beta: float = 0.2
) -> float:
    """Calculate EMA for smoothing."""
    return (1.0 - beta) * current + beta * previous_ema


# =============================================================================
# RENEWABLE PRODUCTION CALCULATIONS
# =============================================================================

@njit(cache=True)
def solar_power_mw(irradiance_kwh_m2_h: float, area_m2: float) -> float:
    """
    Calculate solar power output.
    
    Args:
        irradiance_kwh_m2_h: Solar irradiance (kWh/m²/hour)
        area_m2: Total panel area (m²)
        
    Returns:
        Power output in MW for a 15-min interval
    """
    # Convert kWh/m²/hour to MW for 15-min
    # Energy = irradiance * area * 0.25h / 1000 (kWh to MWh)
    return irradiance_kwh_m2_h * area_m2 * 0.25 / 1000.0


@njit(cache=True)
def wind_power_mw(
    wind_speed_m_s: float,
    num_turbines: int,
    rated_power_mw: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0
) -> float:
    """
    Calculate wind power output using standard power curve.
    
    Args:
        wind_speed_m_s: Wind speed (m/s)
        num_turbines: Number of turbines
        rated_power_mw: Rated power per turbine (MW)
        cut_in: Cut-in wind speed (m/s)
        rated_speed: Rated wind speed (m/s)
        cut_out: Cut-out wind speed (m/s)
        
    Returns:
        Power output in MW
    """
    if wind_speed_m_s < cut_in or wind_speed_m_s > cut_out:
        return 0.0
    elif wind_speed_m_s >= rated_speed:
        return rated_power_mw * num_turbines
    else:
        # Cubic power curve between cut-in and rated
        ratio = (wind_speed_m_s - cut_in) / (rated_speed - cut_in)
        return rated_power_mw * num_turbines * (ratio ** 3)


# =============================================================================
# CORE DISPATCH SIMULATION
# =============================================================================

def initialize_storage_state(
    portfolio_counts: Dict[str, int],
    config: Config = DEFAULT_CONFIG
) -> Dict[str, StorageState]:
    """
    Initialize storage states based on portfolio configuration.
    
    Storage capacities are scaled by the number of PPUs that can input to them.
    
    Args:
        portfolio_counts: PPU counts from portfolio
        config: Configuration
        
    Returns:
        Dictionary of StorageState objects
    """
    storage_defs = config.storage.STORAGE_DEFINITIONS
    initial_soc = config.storage.INITIAL_SOC_FRACTION
    
    storages = {}
    
    for storage_name, storage_def in storage_defs.items():
        # Count how many PPUs can input to this storage
        input_ppus = storage_def.get('input_by', [])
        unit_count = sum(portfolio_counts.get(ppu, 0) for ppu in input_ppus)
        
        # Scale capacity by unit count (minimum 1 for base capacity)
        scale_factor = max(1, unit_count)
        
        # Special handling: Lake doesn't scale
        if storage_name == 'Lake':
            scale_factor = 1
        
        capacity = storage_def['capacity_MWh'] * scale_factor
        current = capacity * initial_soc
        
        storages[storage_name] = StorageState(
            name=storage_name,
            capacity_mwh=capacity,
            current_mwh=current,
            max_power_mw=storage_def['max_power_MW'] * scale_factor,
            efficiency_charge=storage_def['efficiency_charge'],
            efficiency_discharge=storage_def['efficiency_discharge'],
            import_price_chf_per_mwh=storage_def.get('import_price_chf_per_mwh'),
        )
    
    return storages


def run_dispatch_timestep(
    t: int,
    demand_mw: float,
    spot_price: float,
    renewable_production_mw: float,
    state: DispatchState,
    ppu_dictionary: pd.DataFrame,
    config: Config,
    phi_smoothed: float,
) -> Tuple[DispatchState, float]:
    """
    Run dispatch for a single timestep.
    
    Args:
        t: Timestep index
        demand_mw: Demand (MW)
        spot_price: Spot price (CHF/MWh)
        renewable_production_mw: Total renewable production (MW)
        state: Current dispatch state
        ppu_dictionary: PPU dictionary DataFrame
        config: Configuration
        phi_smoothed: Smoothed shortfall (EMA)
        
    Returns:
        Tuple of (updated_state, new_phi_smoothed)
    """
    timestep_h = 0.25  # 15 minutes
    epsilon = 1e-6  # Tolerance
    
    # Net system balance
    overflow_mw = demand_mw - renewable_production_mw
    state.overflow_series.append(overflow_mw)
    
    # Update EMA
    phi_smoothed = exponential_moving_average(overflow_mw, phi_smoothed, beta=0.2)
    
    # Calculate utility indices
    u_dis, u_chg = calculate_utility_indices(overflow_mw, phi_smoothed)
    
    if overflow_mw > epsilon:
        # DEFICIT: Need more energy
        remaining_deficit = overflow_mw * timestep_h  # MWh
        state.total_deficit_mwh += remaining_deficit
        
        # Try to discharge from storages (priority order)
        for storage_name in config.dispatch.DISCHARGE_PRIORITY:
            if storage_name not in state.storages:
                continue
            
            storage = state.storages[storage_name]
            d_stor = calculate_disposition_index(storage.soc)
            
            # Only discharge if storage is willing (d_stor > -0.5)
            if d_stor > -0.5 and remaining_deficit > 0:
                discharged = storage.discharge(remaining_deficit)
                remaining_deficit -= discharged
                
                # Track production
                ppu_name = f"DISPATCH_{storage_name}"
                if ppu_name not in state.production_by_ppu:
                    state.production_by_ppu[ppu_name] = []
                state.production_by_ppu[ppu_name].append((t, discharged / timestep_h))
        
        # Buy remaining from spot market
        if remaining_deficit > 0:
            state.spot_bought.append((t, remaining_deficit / timestep_h))
            state.total_spot_buy_mwh += remaining_deficit
            state.total_spot_buy_cost += remaining_deficit * spot_price
    
    elif overflow_mw < -epsilon:
        # SURPLUS: Excess energy
        remaining_surplus = abs(overflow_mw) * timestep_h  # MWh
        state.total_surplus_mwh += remaining_surplus
        
        # Try to charge storages (priority order)
        for storage_name in config.dispatch.CHARGE_PRIORITY:
            if storage_name not in state.storages:
                continue
            
            storage = state.storages[storage_name]
            d_stor = calculate_disposition_index(storage.soc)
            
            # Only charge if storage is willing (d_stor < 0.5)
            if d_stor < 0.5 and remaining_surplus > 0:
                charged = storage.charge(remaining_surplus)
                remaining_surplus -= charged
        
        # Sell remaining to spot market
        if remaining_surplus > 0:
            state.spot_sold.append((t, remaining_surplus / timestep_h))
            state.total_spot_sell_mwh += remaining_surplus
            state.total_spot_sell_revenue += remaining_surplus * spot_price
    
    return state, phi_smoothed


def calculate_renewable_production(
    t: int,
    ppu_dictionary: pd.DataFrame,
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    config: Config,
) -> float:
    """
    Calculate total renewable production at timestep t.
    
    Args:
        t: Timestep index
        ppu_dictionary: PPU dictionary with location assignments
        solar_data: Solar irradiance array (n_hours, n_locations)
        wind_data: Wind speed array (n_hours, n_locations)
        config: Configuration
        
    Returns:
        Total renewable production in MW
    """
    total_mw = 0.0
    mw_per_unit = config.ppu.MW_PER_UNIT
    
    for _, row in ppu_dictionary.iterrows():
        ppu_name = row['PPU_Name']
        components = row['Components']
        location_rank = row.get('Location_Rank', np.nan)
        
        # Solar PV
        if 'PV' in components:
            if not np.isnan(location_rank) and t < len(solar_data):
                loc_idx = min(int(location_rank) - 1, solar_data.shape[1] - 1)
                irradiance = solar_data[t, loc_idx] if solar_data.ndim > 1 else solar_data[t]
                
                # Assume 1000 m² per unit for PV (rough estimate)
                area_m2 = mw_per_unit * 1000  # m² per MW
                power = solar_power_mw(irradiance, area_m2)
                total_mw += power * row.get('Chain_Efficiency', 0.8)
        
        # Wind
        elif 'Wind (onshore)' in components or 'Wind (offshore)' in components:
            if not np.isnan(location_rank) and t < len(wind_data):
                loc_idx = min(int(location_rank) - 1, wind_data.shape[1] - 1)
                wind_speed = wind_data[t, loc_idx] if wind_data.ndim > 1 else wind_data[t]
                
                # 1 unit = ~100 turbines at 3 MW each
                num_turbines = int(mw_per_unit / 3)
                power = wind_power_mw(wind_speed, num_turbines)
                total_mw += power * row.get('Chain_Efficiency', 0.85)
    
    return total_mw


def run_dispatch_simulation(
    scenario_indices: np.ndarray,
    ppu_dictionary: pd.DataFrame,
    demand_data: np.ndarray,
    spot_data: np.ndarray,
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    portfolio_counts: Dict[str, int],
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False,
) -> Tuple[DispatchState, Dict[str, Any]]:
    """
    Run complete dispatch simulation for a scenario.
    
    Args:
        scenario_indices: Array of hour indices to simulate
        ppu_dictionary: PPU dictionary DataFrame
        demand_data: Full demand array
        spot_data: Full spot price array
        solar_data: Full solar irradiance array
        wind_data: Full wind speed array
        portfolio_counts: PPU counts
        config: Configuration
        verbose: Print progress
        
    Returns:
        Tuple of (final_state, results_dict)
    """
    # Initialize storage states
    storages = initialize_storage_state(portfolio_counts, config)
    
    state = DispatchState(storages=storages)
    phi_smoothed = 0.0
    
    n_timesteps = len(scenario_indices)
    
    if verbose:
        print(f"Running dispatch simulation for {n_timesteps} timesteps...")
    
    for i, t in enumerate(scenario_indices):
        # Get data for this timestep
        demand_mw = demand_data[t] if t < len(demand_data) else 0.0
        spot_price = spot_data[t] if t < len(spot_data) else 50.0  # Default price
        
        # Calculate renewable production
        renewable_mw = calculate_renewable_production(
            t, ppu_dictionary, solar_data, wind_data, config
        )
        
        # Run timestep dispatch
        state, phi_smoothed = run_dispatch_timestep(
            t=i,
            demand_mw=demand_mw,
            spot_price=spot_price,
            renewable_production_mw=renewable_mw,
            state=state,
            ppu_dictionary=ppu_dictionary,
            config=config,
            phi_smoothed=phi_smoothed,
        )
    
    # Compile results
    results = {
        'n_timesteps': n_timesteps,
        'total_deficit_mwh': state.total_deficit_mwh,
        'total_surplus_mwh': state.total_surplus_mwh,
        'total_spot_buy_mwh': state.total_spot_buy_mwh,
        'total_spot_sell_mwh': state.total_spot_sell_mwh,
        'total_spot_buy_cost_chf': state.total_spot_buy_cost,
        'total_spot_sell_revenue_chf': state.total_spot_sell_revenue,
        'net_spot_cost_chf': state.total_spot_buy_cost - state.total_spot_sell_revenue,
        'overflow_series': np.array(state.overflow_series),
        'final_storage_soc': {name: s.soc for name, s in state.storages.items()},
    }
    
    if verbose:
        print(f"  Deficit: {state.total_deficit_mwh:,.0f} MWh")
        print(f"  Surplus: {state.total_surplus_mwh:,.0f} MWh")
        print(f"  Net spot cost: {results['net_spot_cost_chf']:,.0f} CHF")
    
    return state, results


# =============================================================================
# COST COMPUTATION
# =============================================================================

def compute_scenario_cost(
    results: Dict[str, Any],
    ppu_dictionary: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> float:
    """
    Compute total cost for a scenario.
    
    Includes:
    - Net spot market cost (buy - sell)
    - PPU operational costs
    
    Args:
        results: Results from dispatch simulation
        ppu_dictionary: PPU dictionary
        config: Configuration
        
    Returns:
        Total cost in CHF
    """
    # Spot market cost
    spot_cost = results['net_spot_cost_chf']
    
    # PPU operational costs (simplified: based on production)
    # In a full implementation, this would track actual production per PPU
    ppu_cost = 0.0
    
    # Estimate based on deficit that was served from storage
    # Assume average cost across all flex PPUs
    if 'total_deficit_mwh' in results:
        avg_flex_cost = 50.0  # CHF/MWh placeholder
        if not ppu_dictionary.empty:
            flex_ppus = ppu_dictionary[ppu_dictionary['PPU_Extract'] == 'Flex']
            if not flex_ppus.empty and 'Cost_CHF_per_MWh' in flex_ppus.columns:
                avg_flex_cost = flex_ppus['Cost_CHF_per_MWh'].mean()
        
        # Deficit served by storage has a cost
        storage_served = results['total_deficit_mwh'] - results['total_spot_buy_mwh']
        ppu_cost = storage_served * avg_flex_cost
    
    return spot_cost + ppu_cost


def compute_portfolio_metrics(
    scenario_results: List[Dict[str, Any]],
    config: Config = DEFAULT_CONFIG,
) -> Dict[str, float]:
    """
    Compute aggregate metrics across multiple scenarios.
    
    Args:
        scenario_results: List of results from multiple scenarios
        config: Configuration
        
    Returns:
        Dictionary of portfolio metrics
    """
    if not scenario_results:
        return {'mean_cost': float('inf'), 'cvar': float('inf')}
    
    costs = [r.get('net_spot_cost_chf', 0) for r in scenario_results]
    
    # Mean cost
    mean_cost = np.mean(costs)
    
    # CVaR (Conditional Value at Risk) - average of worst 5%
    alpha = config.fitness.CVAR_ALPHA
    sorted_costs = np.sort(costs)
    n_worst = max(1, int(len(costs) * (1 - alpha)))
    cvar = np.mean(sorted_costs[-n_worst:])
    
    # Weighted combination
    cvar_weight = config.fitness.CVAR_WEIGHT
    combined = (1 - cvar_weight) * mean_cost + cvar_weight * cvar
    
    return {
        'mean_cost': mean_cost,
        'cvar': cvar,
        'combined_cost': combined,
        'min_cost': np.min(costs),
        'max_cost': np.max(costs),
        'std_cost': np.std(costs),
    }


if __name__ == "__main__":
    # Basic test
    print("Dispatch engine loaded successfully")
    
    # Test disposition index
    for soc in [0.2, 0.5, 0.6, 0.8, 0.95]:
        d = calculate_disposition_index(soc)
        print(f"SoC={soc:.2f} -> d_stor={d:.3f}")

