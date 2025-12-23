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
# PRECOMPUTED RENEWABLE INDICES (for vectorized production calculation)
# =============================================================================

@dataclass
class PrecomputedRenewables:
    """Pre-computed indices for fast renewable production calculation."""
    
    # Solar PV: location indices and efficiencies
    pv_location_indices: np.ndarray  # Array of location indices (0-based)
    pv_efficiencies: np.ndarray      # Chain efficiencies for each PV unit
    pv_area_m2: float                # Area per unit
    
    # Wind: location indices and efficiencies
    wind_location_indices: np.ndarray  # Array of location indices (0-based)
    wind_efficiencies: np.ndarray      # Chain efficiencies for each wind unit
    wind_num_turbines: int             # Turbines per unit
    
    # Wind power curve parameters
    cut_in_speed: float = 3.0
    rated_speed: float = 12.0
    cut_out_speed: float = 25.0
    rated_power_mw: float = 3.0


def precompute_renewable_indices(
    ppu_dictionary: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> PrecomputedRenewables:
    """
    Pre-compute location indices and parameters for fast renewable calculation.
    
    This should be called ONCE before the simulation loop, not every timestep.
    
    Args:
        ppu_dictionary: PPU dictionary with location assignments
        config: Configuration
        
    Returns:
        PrecomputedRenewables object for fast vectorized calculations
    """
    mw_per_unit = config.ppu.MW_PER_UNIT
    
    # Extract solar PV units
    pv_locations = []
    pv_effs = []
    
    # Extract wind units  
    wind_locations = []
    wind_effs = []
    
    for _, row in ppu_dictionary.iterrows():
        components = row['Components']
        location_rank = row.get('Location_Rank', np.nan)
        efficiency = row.get('Chain_Efficiency', 0.8)
        
        if pd.isna(location_rank):
            continue
            
        loc_idx = int(location_rank) - 1  # Convert to 0-based index
        
        # Solar PV
        if 'PV' in components:
            pv_locations.append(loc_idx)
            pv_effs.append(efficiency)
        
        # Wind (onshore or offshore)
        elif 'Wind (onshore)' in components or 'Wind (offshore)' in components:
            wind_locations.append(loc_idx)
            wind_effs.append(efficiency if not pd.isna(efficiency) else 0.85)
    
    return PrecomputedRenewables(
        pv_location_indices=np.array(pv_locations, dtype=np.int32),
        pv_efficiencies=np.array(pv_effs, dtype=np.float64),
        pv_area_m2=mw_per_unit * 1000,  # m² per MW
        wind_location_indices=np.array(wind_locations, dtype=np.int32),
        wind_efficiencies=np.array(wind_effs, dtype=np.float64),
        wind_num_turbines=int(mw_per_unit / 3),
    )


@njit(cache=True)
def _calculate_solar_power_vectorized(
    irradiance_values: np.ndarray,
    efficiencies: np.ndarray,
    area_m2: float,
) -> float:
    """Vectorized solar power calculation using numba.
    
    Args:
        irradiance_values: Solar irradiance in kWh/m²/hour
        efficiencies: Chain efficiencies for each unit
        area_m2: Panel area per unit in m²
        
    Returns:
        Total power in MW
    """
    efficiency_pv = 0.20  # 20% PV panel efficiency
    total_power = 0.0
    for i in range(len(irradiance_values)):
        # P = irradiance * area * PV_efficiency * chain_efficiency
        # Units: kWh/m²/h * m² * efficiency = kW
        # Divide by 1000 to get MW
        power = irradiance_values[i] * area_m2 * efficiency_pv * efficiencies[i] / 1000.0
        total_power += power
    return total_power


@njit(cache=True)
def _calculate_wind_power_vectorized(
    wind_speeds: np.ndarray,
    efficiencies: np.ndarray,
    num_turbines: int,
    rated_power: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
) -> float:
    """Vectorized wind power calculation using numba."""
    total_power = 0.0
    for i in range(len(wind_speeds)):
        ws = wind_speeds[i]
        if ws < cut_in or ws > cut_out:
            power = 0.0
        elif ws >= rated_speed:
            power = rated_power * num_turbines
        else:
            ratio = (ws - cut_in) / (rated_speed - cut_in)
            power = rated_power * num_turbines * (ratio ** 3)
        total_power += power * efficiencies[i]
    return total_power


def calculate_renewable_production_fast(
    t: int,
    precomputed: PrecomputedRenewables,
    solar_data: np.ndarray,
    wind_data: np.ndarray,
) -> float:
    """
    Fast vectorized renewable production calculation.
    
    Uses precomputed indices to avoid iterrows() overhead.
    Handles NaN values in input data by treating them as zero.
    
    Args:
        t: Timestep index
        precomputed: PrecomputedRenewables object
        solar_data: Solar irradiance array (n_hours, n_locations)
        wind_data: Wind speed array (n_hours, n_locations)
        
    Returns:
        Total renewable production in MW
    """
    total_mw = 0.0
    
    # Solar PV production
    if len(precomputed.pv_location_indices) > 0 and t < len(solar_data):
        # Clip indices to valid range
        max_solar_loc = solar_data.shape[1] - 1 if solar_data.ndim > 1 else 0
        pv_locs = np.clip(precomputed.pv_location_indices, 0, max_solar_loc)
        
        if solar_data.ndim > 1:
            irradiance_values = solar_data[t, pv_locs].astype(np.float64)
        else:
            irradiance_values = np.full(len(pv_locs), float(solar_data[t]))
        
        # Replace NaN with 0 (no production)
        irradiance_values = np.nan_to_num(irradiance_values, nan=0.0)
        
        total_mw += _calculate_solar_power_vectorized(
            irradiance_values,
            precomputed.pv_efficiencies,
            precomputed.pv_area_m2,
        )
    
    # Wind production
    if len(precomputed.wind_location_indices) > 0 and t < len(wind_data):
        # Clip indices to valid range
        max_wind_loc = wind_data.shape[1] - 1 if wind_data.ndim > 1 else 0
        wind_locs = np.clip(precomputed.wind_location_indices, 0, max_wind_loc)
        
        if wind_data.ndim > 1:
            wind_speeds = wind_data[t, wind_locs].astype(np.float64)
        else:
            wind_speeds = np.full(len(wind_locs), float(wind_data[t]))
        
        # Replace NaN with 0 (no production)
        wind_speeds = np.nan_to_num(wind_speeds, nan=0.0)
        
        total_mw += _calculate_wind_power_vectorized(
            wind_speeds,
            precomputed.wind_efficiencies,
            precomputed.wind_num_turbines,
        )
    
    return total_mw


# =============================================================================
# DISPATCH STATE TRACKING
# =============================================================================

@dataclass
class StorageState:
    """State of a single storage system."""
    
    name: str
    capacity_mwh: float  # Maximum capacity
    current_mwh: float  # Current stored energy
    max_charge_power_mw: float  # Maximum charge rate (limited by input PPUs)
    max_discharge_power_mw: float  # Maximum discharge rate (limited by extraction PPUs)
    efficiency_charge: float
    efficiency_discharge: float
    
    # Import pricing (for palm oil, biooil)
    import_price_chf_per_mwh: Optional[float] = None
    
    # Legacy compatibility - use discharge power as default
    @property
    def max_power_mw(self) -> float:
        """Legacy: returns max discharge power for backward compatibility."""
        return self.max_discharge_power_mw
    
    @property
    def soc(self) -> float:
        """State of Charge [0, 1]."""
        if self.capacity_mwh > 0:
            return self.current_mwh / self.capacity_mwh
        return 0.0
    
    def can_discharge(self, amount_mwh: float, timestep_h: float = 1.0) -> float:
        """Return how much can actually be discharged."""
        # Limited by current storage and DISCHARGE power rating (extraction PPUs)
        max_from_power = self.max_discharge_power_mw * timestep_h  # Energy in MWh per timestep
        max_from_storage = self.current_mwh * self.efficiency_discharge
        return min(amount_mwh, max_from_power, max_from_storage)
    
    def can_charge(self, amount_mwh: float, timestep_h: float = 1.0) -> float:
        """Return how much can actually be charged."""
        # Limited by remaining capacity and CHARGE power rating (input PPUs)
        max_from_power = self.max_charge_power_mw * timestep_h  # Energy in MWh per timestep
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
# DISPATCHABLE GENERATOR SUPPORT
# =============================================================================

# PPUs that are dispatchable generators (can produce power on demand)
DISPATCHABLE_GENERATORS = {
    'THERM': {'capacity_factor': 0.95, 'cost_per_mwh': 80},
    'THERM_CH4': {'capacity_factor': 0.95, 'cost_per_mwh': 70},
    'H2P_G': {'capacity_factor': 0.90, 'cost_per_mwh': 100},
    'H2P_L': {'capacity_factor': 0.90, 'cost_per_mwh': 100},
    'BIO_WOOD': {'capacity_factor': 0.85, 'cost_per_mwh': 90},
    'BIO_OIL_ICE': {'capacity_factor': 0.90, 'cost_per_mwh': 85},
    'PALM_ICE': {'capacity_factor': 0.90, 'cost_per_mwh': 80},
    'IMP_BIOG': {'capacity_factor': 0.95, 'cost_per_mwh': 75},
    'NH3_P': {'capacity_factor': 0.90, 'cost_per_mwh': 95},
    'SOL_SALT': {'capacity_factor': 0.70, 'cost_per_mwh': 60},  # Solar thermal
    'SOL_STEAM': {'capacity_factor': 0.70, 'cost_per_mwh': 65},
    'HYD_R': {'capacity_factor': 0.50, 'cost_per_mwh': 20},  # Run-of-river
    'HYD_S': {'capacity_factor': 0.40, 'cost_per_mwh': 25},  # Storage hydro
}

# Storage discharge costs (opportunity cost - value of stored energy)
# Higher cost = more reluctant to discharge (save for peak demand)
STORAGE_DISCHARGE_COSTS = {
    'Lake': 150,          # Hydro is precious - save for peaks!
    'Solar salt': 40,     # Easy to recharge from solar
    'Biogas': 50,         # Medium value
    'H2 UG 200bar': 80,   # H2 is valuable
    'CH4 200bar': 70,     # Methane storage
    'Liquid H2': 90,      # Liquid H2 expensive to produce
    'Fuel Tank': 60,      # Synthetic fuel
    'Ammonia': 85,        # Ammonia valuable
    'Biooil': 55,         # Bio-oil
    'Palm oil': 55,       # Palm oil
}


@dataclass
class DispatchableCapacity:
    """Tracks available dispatchable generation capacity."""
    
    ppu_name: str
    max_power_mw: float
    capacity_factor: float
    cost_per_mwh: float
    
    # Track utilization
    energy_produced_mwh: float = 0.0
    
    def available_power(self) -> float:
        """Available power considering capacity factor."""
        return self.max_power_mw * self.capacity_factor
    
    def dispatch(self, requested_mwh: float, timestep_h: float = 0.25) -> float:
        """Dispatch up to requested amount, return actual dispatch."""
        available_mwh = self.available_power() * timestep_h
        dispatched = min(requested_mwh, available_mwh)
        self.energy_produced_mwh += dispatched
        return dispatched


def initialize_dispatchable_generators(
    portfolio_counts: Dict[str, int],
    config: Config = DEFAULT_CONFIG
) -> Dict[str, DispatchableCapacity]:
    """
    Initialize dispatchable generator capacities.
    
    Args:
        portfolio_counts: PPU counts from portfolio
        config: Configuration
        
    Returns:
        Dictionary of DispatchableCapacity objects
    """
    mw_per_unit = config.ppu.MW_PER_UNIT
    dispatchers = {}
    
    for ppu_name, gen_def in DISPATCHABLE_GENERATORS.items():
        count = portfolio_counts.get(ppu_name, 0)
        if count > 0:
            dispatchers[ppu_name] = DispatchableCapacity(
                ppu_name=ppu_name,
                max_power_mw=count * mw_per_unit,
                capacity_factor=gen_def['capacity_factor'],
                cost_per_mwh=gen_def['cost_per_mwh'],
            )
    
    return dispatchers


# =============================================================================
# CORE DISPATCH SIMULATION
# =============================================================================

def initialize_storage_state(
    portfolio_counts: Dict[str, int],
    config: Config = DEFAULT_CONFIG
) -> Dict[str, StorageState]:
    """
    Initialize storage states based on portfolio configuration.
    
    Power limits are determined by the number of PPUs connected:
    - Charge power = MW_PER_UNIT × count of input PPUs
    - Discharge power = MW_PER_UNIT × count of extraction PPUs
    
    Capacity is scaled by input PPUs (more storage input = more capacity needed).
    
    Args:
        portfolio_counts: PPU counts from portfolio
        config: Configuration
        
    Returns:
        Dictionary of StorageState objects
    """
    storage_defs = config.storage.STORAGE_DEFINITIONS
    initial_soc = config.storage.INITIAL_SOC_FRACTION
    mw_per_unit = config.ppu.MW_PER_UNIT  # 10 MW per PPU unit
    
    storages = {}
    
    for storage_name, storage_def in storage_defs.items():
        # Count PPUs that INPUT to this storage (for charging)
        input_ppus = storage_def.get('input_by', [])
        input_ppu_count = sum(portfolio_counts.get(ppu, 0) for ppu in input_ppus)
        
        # Count PPUs that EXTRACT from this storage (for discharging)
        extract_ppus = storage_def.get('extracted_by', [])
        extract_ppu_count = sum(portfolio_counts.get(ppu, 0) for ppu in extract_ppus)
        
        # Calculate power limits based on PPU counts
        # Each PPU unit can handle MW_PER_UNIT (10 MW) of power flow
        charge_power_mw = input_ppu_count * mw_per_unit
        discharge_power_mw = extract_ppu_count * mw_per_unit
        
        # Apply physical power cap if defined (e.g., Lake = 2 GW)
        physical_cap = storage_def.get('physical_power_cap_MW')
        if physical_cap is not None:
            charge_power_mw = min(charge_power_mw, physical_cap)
            discharge_power_mw = min(discharge_power_mw, physical_cap)
        
        # Scale capacity by input PPU count (minimum 1 for base capacity)
        # Special handling: Lake doesn't scale capacity
        if storage_name == 'Lake':
            capacity_scale = 1
        else:
            capacity_scale = max(1, input_ppu_count)
        
        capacity = storage_def['capacity_MWh'] * capacity_scale
        current = capacity * initial_soc
        
        storages[storage_name] = StorageState(
            name=storage_name,
            capacity_mwh=capacity,
            current_mwh=current,
            max_charge_power_mw=charge_power_mw,
            max_discharge_power_mw=discharge_power_mw,
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
    precomputed_renewables: Optional[PrecomputedRenewables] = None,
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
        precomputed_renewables: Optional precomputed indices for fast calculation
        
    Returns:
        Tuple of (final_state, results_dict)
    """
    # Initialize storage states
    storages = initialize_storage_state(portfolio_counts, config)
    
    # Initialize dispatchable generators
    dispatchers = initialize_dispatchable_generators(portfolio_counts, config)
    
    state = DispatchState(storages=storages)
    phi_smoothed = 0.0
    
    n_timesteps = len(scenario_indices)
    timestep_h = 1.0  # 1 hour (data is hourly: 8784 hours/year)
    
    # Initialize time series tracking arrays
    total_production_series = np.zeros(n_timesteps)
    renewable_production_series = np.zeros(n_timesteps)
    dispatchable_production_series = np.zeros(n_timesteps)
    spot_bought_series = np.zeros(n_timesteps)
    spot_sold_series = np.zeros(n_timesteps)
    deficit_series = np.zeros(n_timesteps)
    surplus_series = np.zeros(n_timesteps)
    
    # Track storage SoC over time
    storage_soc_series = {name: np.zeros(n_timesteps) for name in storages.keys()}
    
    # Precompute renewable indices ONCE (big speedup!)
    if precomputed_renewables is None:
        precomputed_renewables = precompute_renewable_indices(ppu_dictionary, config)
    
    # Calculate total dispatchable capacity
    total_dispatchable_mw = sum(d.available_power() for d in dispatchers.values())
    
    if verbose:
        print(f"Running dispatch simulation for {n_timesteps} timesteps...")
        print(f"  Dispatchable capacity: {total_dispatchable_mw:,.0f} MW")
    
    for i, t in enumerate(scenario_indices):
        # Get data for this timestep
        demand_mw = demand_data[t] if t < len(demand_data) else 0.0
        spot_price = spot_data[t] if t < len(spot_data) else 50.0  # Default price
        
        # ===== STEP 1: Calculate renewable production =====
        renewable_mw = calculate_renewable_production_fast(
            t, precomputed_renewables, solar_data, wind_data
        )
        renewable_production_series[i] = renewable_mw
        
        # Calculate initial balance
        balance_mw = renewable_mw - demand_mw  # Positive = surplus, Negative = deficit
        
        storage_discharged_mw = 0.0
        storage_charged_mw = 0.0
        dispatchable_mw = 0.0
        
        if balance_mw < 0:
            # ===== STEP 2: DEFICIT - Use cost-based dispatch with disposition index =====
            remaining_deficit_mwh = abs(balance_mw) * timestep_h
            
            # Build unified supply options: generators + willing storages
            supply_options = []
            
            # Add dispatchable generators
            for dispatcher in dispatchers.values():
                if dispatcher.available_power() > 0:
                    supply_options.append({
                        'type': 'generator',
                        'name': dispatcher.ppu_name,
                        'cost': dispatcher.cost_per_mwh,
                        'object': dispatcher,
                    })
            
            # Add storages (only if willing to discharge based on disposition index)
            for storage_name, storage in state.storages.items():
                d_stor = calculate_disposition_index(storage.soc, soc_target=0.60)
                base_cost = STORAGE_DISCHARGE_COSTS.get(storage_name, 50)
                
                # Disposition penalty: if SoC < target, increase effective cost
                # d_stor ranges from -1 (empty) to +1 (full)
                # When d_stor < 0 (below target), add penalty to discourage discharge
                if d_stor < 0:
                    disposition_penalty = abs(d_stor) * 200  # Big penalty when low
                else:
                    disposition_penalty = 0  # No penalty when above target
                
                effective_cost = base_cost + disposition_penalty
                
                # Only add if storage has energy and is willing (d_stor > -0.8)
                if storage.current_mwh > 0 and d_stor > -0.8:
                    supply_options.append({
                        'type': 'storage',
                        'name': storage_name,
                        'cost': effective_cost,
                        'object': storage,
                        'd_stor': d_stor,
                    })
            
            # Sort by cost (cheapest first)
            supply_options.sort(key=lambda x: x['cost'])
            
            # Dispatch in cost order
            for option in supply_options:
                if remaining_deficit_mwh <= 0:
                    break
                    
                if option['type'] == 'generator':
                    dispatcher = option['object']
                    available = dispatcher.available_power() * timestep_h
                    dispatch = min(available, remaining_deficit_mwh)
                    dispatcher.energy_produced_mwh += dispatch
                    dispatchable_mw += dispatch / timestep_h
                    remaining_deficit_mwh -= dispatch
                    
                elif option['type'] == 'storage':
                    storage = option['object']
                    discharged = storage.discharge(remaining_deficit_mwh)
                    remaining_deficit_mwh -= discharged
                    storage_discharged_mw += discharged / timestep_h
            
            # Track any remaining deficit
            if remaining_deficit_mwh > 0:
                remaining_mw = remaining_deficit_mwh / timestep_h
                state.total_deficit_mwh += remaining_deficit_mwh
                deficit_series[i] = remaining_mw
                # Buy from spot market
                state.spot_bought.append((i, remaining_mw))
                state.total_spot_buy_mwh += remaining_deficit_mwh
                state.total_spot_buy_cost += remaining_deficit_mwh * spot_price
        
        else:
            # ===== STEP 4: SURPLUS - Charge storage =====
            remaining_surplus_mwh = balance_mw * timestep_h
            state.total_surplus_mwh += remaining_surplus_mwh
            
            for storage_name in config.dispatch.CHARGE_PRIORITY:
                if storage_name not in state.storages or remaining_surplus_mwh <= 0:
                    continue
                storage = state.storages[storage_name]
                charged = storage.charge(remaining_surplus_mwh)
                remaining_surplus_mwh -= charged
                storage_charged_mw += charged / timestep_h
            
            # Any remaining surplus could be sold or curtailed
            if remaining_surplus_mwh > 0:
                surplus_series[i] = remaining_surplus_mwh / timestep_h
                # Could sell to spot market (optional)
                state.spot_sold.append((i, remaining_surplus_mwh / timestep_h))
                state.total_spot_sell_mwh += remaining_surplus_mwh
                state.total_spot_sell_revenue += remaining_surplus_mwh * spot_price
        
        # Track total production
        dispatchable_production_series[i] = dispatchable_mw
        total_production_mw = renewable_mw + dispatchable_mw + storage_discharged_mw
        total_production_series[i] = total_production_mw
        
        # Update overflow series for tracking
        state.overflow_series.append(demand_mw - total_production_mw)
        
        # Track spot market transactions from state
        # spot_bought and spot_sold are lists of (t, value) tuples
        for ts, val in state.spot_bought:
            if ts == i:
                spot_bought_series[i] = val
        for ts, val in state.spot_sold:
            if ts == i:
                spot_sold_series[i] = val
        
        # Track storage SoC
        for name, storage in state.storages.items():
            storage_soc_series[name][i] = storage.soc
    
    # Compile dispatchable production summary
    dispatchable_summary = {
        name: d.energy_produced_mwh 
        for name, d in dispatchers.items()
    }
    total_dispatchable_mwh = sum(dispatchable_summary.values())
    total_renewable_mwh = np.sum(renewable_production_series) * timestep_h
    
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
        # Time series data for full year evaluation
        'total_production': total_production_series,
        'renewable_production': renewable_production_series,
        'dispatchable_production': dispatchable_production_series,
        'spot_bought': spot_bought_series,
        'spot_sold': spot_sold_series,
        'deficit': deficit_series,
        'surplus': surplus_series,
        'storage_soc': storage_soc_series,
        'ppu_production': dispatchable_summary,
        # Summary stats
        'total_renewable_mwh': total_renewable_mwh,
        'total_dispatchable_mwh': total_dispatchable_mwh,
        'dispatchable_capacity_mw': total_dispatchable_mw,
    }
    
    if verbose:
        print(f"  Deficit: {state.total_deficit_mwh:,.0f} MWh")
        print(f"  Surplus: {state.total_surplus_mwh:,.0f} MWh")
        print(f"  Net spot cost: {results['net_spot_cost_chf']:,.0f} CHF")
        print(f"  Total renewable production: {np.sum(total_production_series):,.0f} MWh")
    
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

