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
    """Pre-computed parameters for fast renewable production calculation.
    
    DISTRIBUTED DEPLOYMENT MODEL:
    - Solar: Each PPU adds 1000 m² to EVERY location (distributed rooftop solar)
    - Wind: Each PPU adds 1 turbine to EVERY location (distributed wind farms)
    - Production is summed across ALL locations
    """
    
    # Solar PV: count and efficiency (distributed across all locations)
    pv_count: int = 0                # Number of PV PPU units
    pv_efficiency: float = 0.84      # Average chain efficiency
    pv_area_per_location_m2: float = 1000.0  # 1000 m² per PPU per location
    
    # Wind: count and efficiency (distributed across all locations)  
    wind_onshore_count: int = 0      # Number of WD_ON PPU units
    wind_offshore_count: int = 0     # Number of WD_OFF PPU units
    wind_efficiency: float = 0.84    # Average chain efficiency
    turbines_per_location: int = 1   # 1 turbine per PPU per location
    
    # River: PPU indices and monthly production
    river_indices: List[int] = field(default_factory=list)
    ror_production_monthly: np.ndarray = field(default_factory=lambda: np.zeros(12)) # MWh per month
    
    # Solar Thermal: PPU indices
    sol_store_indices: List[int] = field(default_factory=list)
    
    mw_per_unit: float = 10.0

    # Wind power curve parameters
    cut_in_speed: float = 3.0
    rated_speed: float = 12.0
    cut_out_speed: float = 25.0
    rated_power_mw: float = 3.0


def precompute_renewable_indices(
    ppu_dictionary: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    ror_production: Optional[np.ndarray] = None
) -> PrecomputedRenewables:
    """
    Pre-compute parameters for DISTRIBUTED renewable calculation.
    
    DISTRIBUTED DEPLOYMENT MODEL:
    - Solar: Each PPU adds 1000 m² to EVERY location (distributed rooftop solar)
    - Wind: Each PPU adds 1 turbine to EVERY location (distributed wind farms)
    - Production is summed across ALL locations at each timestep
    
    This should be called ONCE before the simulation loop, not every timestep.
    
    Args:
        ppu_dictionary: PPU dictionary with PPU counts
        config: Configuration
        ror_production: Optional monthly river hydro production array
        
    Returns:
        PrecomputedRenewables object for fast distributed calculations
    """
    mw_per_unit = config.ppu.MW_PER_UNIT
    
    # Count PPU units by type
    pv_count = 0
    pv_efficiency_sum = 0.0
    
    wind_onshore_count = 0
    wind_offshore_count = 0
    wind_efficiency_sum = 0.0
    
    # Extract river units
    river_indices = []
    
    # Extract solar thermal store units
    sol_store_indices = []
    
    for i, row in ppu_dictionary.iterrows():
        ppu_name = row['PPU_Name']
        components = row['Components']
        efficiency = row.get('Chain_Efficiency', 0.84)
        
        # River hydro (incidence)
        if 'River' in components or ppu_name == 'HYD_R':
            river_indices.append(i)
            continue
            
        # Solar thermal store (incidence -> storage)
        if 'Solar concentrator' in str(components) or 'SOL_SALT_STORE' in ppu_name:
            sol_store_indices.append(i)
        
        # Solar PV - count units
        if 'PV' in components:
            pv_count += 1
            pv_efficiency_sum += efficiency if not pd.isna(efficiency) else 0.84
        
        # Wind onshore
        elif 'Wind (onshore)' in components:
            wind_onshore_count += 1
            wind_efficiency_sum += efficiency if not pd.isna(efficiency) else 0.84
            
        # Wind offshore
        elif 'Wind (offshore)' in components:
            wind_offshore_count += 1
            wind_efficiency_sum += efficiency if not pd.isna(efficiency) else 0.84
    
    # Calculate average efficiencies
    pv_efficiency = pv_efficiency_sum / pv_count if pv_count > 0 else 0.84
    total_wind = wind_onshore_count + wind_offshore_count
    wind_efficiency = wind_efficiency_sum / total_wind if total_wind > 0 else 0.84
    
    return PrecomputedRenewables(
        pv_count=pv_count,
        pv_efficiency=pv_efficiency,
        pv_area_per_location_m2=1000.0,  # 1000 m² per PPU per location
        wind_onshore_count=wind_onshore_count,
        wind_offshore_count=wind_offshore_count,
        wind_efficiency=wind_efficiency,
        turbines_per_location=1,  # 1 turbine per PPU per location
        river_indices=river_indices,
        ror_production_monthly=ror_production if ror_production is not None else np.zeros(12),
        sol_store_indices=sol_store_indices,
        mw_per_unit=mw_per_unit
    )


@njit(cache=True)
def _calculate_solar_power_distributed(
    irradiance_all_locations: np.ndarray,
    n_ppu: int,
    area_per_location_m2: float,
    chain_efficiency: float,
) -> float:
    """Distributed solar power calculation - sums across ALL locations.
    
    DISTRIBUTED MODEL: Each PPU adds area_per_location_m2 to EVERY location.
    Total area = n_ppu × area_per_location_m2 × n_locations
    
    Args:
        irradiance_all_locations: Solar irradiance at ALL locations (1D array)
        n_ppu: Number of PV PPU units
        area_per_location_m2: Panel area per PPU per location (1000 m²)
        chain_efficiency: PPU chain efficiency
        
    Returns:
        Total power in MW
    """
    if n_ppu == 0:
        return 0.0
    
    efficiency_pv = 0.20  # 20% PV panel efficiency
    total_power = 0.0
    
    # Sum production across ALL locations
    for i in range(len(irradiance_all_locations)):
        # Total area at this location = n_ppu × area_per_location
        total_area_at_loc = n_ppu * area_per_location_m2
        # P = irradiance × area × PV_efficiency × chain_efficiency
        # Units: kWh/m²/h × m² × efficiency = kW → /1000 = MW
        power = irradiance_all_locations[i] * total_area_at_loc * efficiency_pv * chain_efficiency / 1000.0
        total_power += power
    
    return total_power


@njit(cache=True)
def _calculate_wind_power_distributed(
    wind_speeds_all_locations: np.ndarray,
    n_ppu: int,
    turbines_per_location: int,
    chain_efficiency: float,
    rated_power: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
) -> float:
    """Distributed wind power calculation - sums across ALL locations.
    
    DISTRIBUTED MODEL: Each PPU adds turbines_per_location to EVERY location.
    Total turbines = n_ppu × turbines_per_location × n_locations
    
    Args:
        wind_speeds_all_locations: Wind speed at ALL locations (1D array)
        n_ppu: Number of wind PPU units
        turbines_per_location: Turbines per PPU per location (1)
        chain_efficiency: PPU chain efficiency
        rated_power: Rated power per turbine (MW)
        cut_in: Cut-in wind speed (m/s)
        rated_speed: Rated wind speed (m/s)
        cut_out: Cut-out wind speed (m/s)
        
    Returns:
        Total power in MW
    """
    if n_ppu == 0:
        return 0.0
    
    total_power = 0.0
    
    # Total turbines at each location
    total_turbines_at_loc = n_ppu * turbines_per_location
    
    # Sum production across ALL locations
    for i in range(len(wind_speeds_all_locations)):
        ws = wind_speeds_all_locations[i]
        
        # Wind power curve
        if ws < cut_in or ws > cut_out:
            power = 0.0
        elif ws >= rated_speed:
            power = rated_power * total_turbines_at_loc
        else:
            ratio = (ws - cut_in) / (rated_speed - cut_in)
            power = rated_power * total_turbines_at_loc * (ratio ** 3)
        
        total_power += power * chain_efficiency
    
    return total_power


def calculate_renewable_production_fast(
    t: int,
    precomputed: PrecomputedRenewables,
    solar_data: np.ndarray,
    wind_data: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Fast DISTRIBUTED renewable production calculation.
    
    DISTRIBUTED DEPLOYMENT MODEL:
    - Solar: Each PPU adds 1000 m² to EVERY location (distributed rooftop solar)
    - Wind: Each PPU adds 1 turbine to EVERY location (distributed wind farms)
    - Production is summed across ALL locations at each timestep
    
    Args:
        t: Timestep index (absolute hour of year)
        precomputed: PrecomputedRenewables object with PPU counts
        solar_data: Solar irradiance array (n_hours, n_locations)
        wind_data: Wind speed array (n_hours, n_locations)
        
    Returns:
        Tuple of (Total renewable production in MW, breakdown dict)
    """
    total_mw = 0.0
    breakdown = {}
    
    # Calculate month index (0-11) for monthly profiles
    month_idx = min(11, t // (24 * 30))
    
    # =================================================================
    # SOLAR PV - DISTRIBUTED across ALL locations
    # =================================================================
    if precomputed.pv_count > 0 and t < len(solar_data):
        # Get irradiance at ALL locations for this timestep
        if solar_data.ndim > 1:
            irradiance_all = solar_data[t, :].astype(np.float64)
        else:
            irradiance_all = np.array([float(solar_data[t])])
        
        # Replace NaN with 0 (no production)
        irradiance_all = np.nan_to_num(irradiance_all, nan=0.0)
        
        pv_mw = _calculate_solar_power_distributed(
            irradiance_all,
            precomputed.pv_count,
            precomputed.pv_area_per_location_m2,  # 1000 m² per PPU per location
            precomputed.pv_efficiency,
        )
        total_mw += pv_mw
        breakdown['PV'] = pv_mw
    
    # =================================================================
    # RIVER HYDRO (HYD_R) - Incidence based (unchanged)
    # =================================================================
    if len(precomputed.river_indices) > 0:
        n_units = len(precomputed.river_indices)
        monthly_mwh = precomputed.ror_production_monthly[month_idx]
        hourly_mw_total = monthly_mwh / (30 * 24)
        
        SWISS_TOTAL_HYD_R_UNITS = 340.0
        river_mw = hourly_mw_total * (n_units / SWISS_TOTAL_HYD_R_UNITS)
        total_mw += river_mw
        breakdown['HYD_R'] = river_mw
    
    # =================================================================
    # SOLAR THERMAL STORAGE (SOL_SALT_STORE) - Incidence based
    # =================================================================
    if len(precomputed.sol_store_indices) > 0 and t < len(solar_data):
        n_units = len(precomputed.sol_store_indices)
        
        # Use mean irradiance across all locations for thermal
        if solar_data.ndim > 1:
            irrad = np.nanmean(solar_data[t, :])
        else:
            irrad = float(solar_data[t])
        irrad = np.nan_to_num(irrad, nan=0.0)
        
        # Solar thermal: ~10,000 m² concentrator area per unit, 40% efficiency
        sol_thermal_mw = irrad * 10000.0 * 0.40 * n_units / 1000.0
        total_mw += sol_thermal_mw
        breakdown['SOL_SALT_STORE'] = sol_thermal_mw
    
    # =================================================================
    # WIND - DISTRIBUTED across ALL locations
    # =================================================================
    total_wind_count = precomputed.wind_onshore_count + precomputed.wind_offshore_count
    if total_wind_count > 0 and t < len(wind_data):
        # Get wind speed at ALL locations for this timestep
        if wind_data.ndim > 1:
            wind_speeds_all = wind_data[t, :].astype(np.float64)
        else:
            wind_speeds_all = np.array([float(wind_data[t])])
        
        # Replace NaN with 0 (no production)
        wind_speeds_all = np.nan_to_num(wind_speeds_all, nan=0.0)
        
        # Onshore wind (3 MW turbines)
        if precomputed.wind_onshore_count > 0:
            wind_on_mw = _calculate_wind_power_distributed(
                wind_speeds_all,
                precomputed.wind_onshore_count,
                precomputed.turbines_per_location,  # 1 turbine per PPU per location
                precomputed.wind_efficiency,
                rated_power=3.0,  # 3 MW onshore turbines
            )
            total_mw += wind_on_mw
            breakdown['WD_ON'] = wind_on_mw
        
        # Offshore wind (5 MW turbines)
        if precomputed.wind_offshore_count > 0:
            wind_off_mw = _calculate_wind_power_distributed(
                wind_speeds_all,
                precomputed.wind_offshore_count,
                precomputed.turbines_per_location,
                precomputed.wind_efficiency,
                rated_power=5.0,  # 5 MW offshore turbines
        )
            total_mw += wind_off_mw
            breakdown['WD_OFF'] = wind_off_mw
        
    return total_mw, breakdown


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

# PPUs that are standalone dispatchable generators (fuel-based, no dedicated storage)
# These can dispatch without drawing from storage - they use direct fuel input
DISPATCHABLE_GENERATORS = {
    'THERM_M': {'capacity_factor': 0.95, 'cost_per_mwh': 85},   # Methane thermal (direct import)
    'THERM_G': {'capacity_factor': 0.95, 'cost_per_mwh': 23},   # Gas thermal (direct import)
    # BIO_WOOD: Wood → Pyrolysis → Biooil → Steam reforming → CCPP → Grid
    # This is a flexible generator that uses biomass as direct input (not storage-backed)
    'BIO_WOOD': {'capacity_factor': 0.70, 'cost_per_mwh': 215}, # Wood biomass (215 CHF/MWh from PPU costs)
}

# Storage discharge costs (opportunity cost - value of stored energy)
# Higher cost = more reluctant to discharge (save for peak demand)
# This cost should include the OPEX of the extraction PPU (e.g. SOL_SALT)
STORAGE_DISCHARGE_COSTS = {
    'Lake': 150,          # HYD_S extraction
    'Solar salt': 40,     # SOL_SALT / SOL_STEAM extraction
    'Biogas': 50,         # IMP_BIOG extraction
    'H2 UG 200bar': 80,   # H2P_G extraction
    'CH4 200bar': 70,     # THERM_CH4 extraction
    'Liquid H2': 90,      # H2P_L extraction
    'Fuel Tank': 60,      # THERM extraction
    'Ammonia': 85,        # NH3_P extraction
    'Biooil': 55,         # BIO_OIL_ICE extraction
    'Palm oil': 55,       # PALM_ICE extraction
}

# Mapping from storage name to the PPU(s) that extract from it
# Used to attribute storage discharge production to the correct PPU for RoT calculation
STORAGE_TO_EXTRACTING_PPU = {
    'Lake': 'HYD_S',
    'Solar salt': 'SOL_SALT',  # Could also be SOL_STEAM, using primary
    'Biogas': 'IMP_BIOG',
    'H2 UG 200bar': 'H2P_G',
    'CH4 200bar': 'THERM_CH4',
    'Liquid H2': 'H2P_L',
    'Fuel Tank': 'THERM',
    'Ammonia': 'NH3_P',
    'Biooil': 'BIO_OIL_ICE',
    'Palm oil': 'PALM_ICE',
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
        
        # Apply maximum capacity cap if defined
        max_cap = storage_def.get('max_capacity_cap_MWh')
        if max_cap is not None:
            capacity = min(capacity, max_cap)
        
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
    
    # =========================================================================
    # AVIATION FUEL TRACKING
    # =========================================================================
    # Mandatory biooil discharge for aviation: 23 TWh/year = ~2625.57 MWh/hour
    aviation_fuel_required_mwh = config.energy_system.AVIATION_FUEL_HOURLY_MWH * timestep_h
    aviation_fuel_consumed_series = np.zeros(n_timesteps)  # Track actual consumption
    aviation_fuel_shortfall_series = np.zeros(n_timesteps)  # Track any shortfall
    aviation_fuel_import_cost_series = np.zeros(n_timesteps)  # Track import costs
    
    # Biooil import price from storage definition
    biooil_import_price = config.storage.STORAGE_DEFINITIONS.get(
        'Biooil', {}
    ).get('import_price_chf_per_mwh', 67.0)
    
    # Track total aviation fuel metrics
    total_aviation_fuel_consumed_mwh = 0.0
    total_aviation_fuel_shortfall_mwh = 0.0
    total_aviation_fuel_import_cost_chf = 0.0
    
    # Track hourly production per PPU
    # Initialize with arrays of zeros
    ppu_production_hourly = {name: np.zeros(n_timesteps) for name in portfolio_counts.keys() if portfolio_counts[name] > 0}
    
    # Track storage SoC over time
    storage_soc_series = {name: np.zeros(n_timesteps) for name in storages.keys()}
    
    # Precompute renewable indices ONCE (big speedup!)
    if precomputed_renewables is None:
        # Load ror_production from config if available (via data loader)
        # For evaluation, we assume data is already cached
        from data_loader import load_all_data
        cached_data = load_all_data(config)
        precomputed_renewables = precompute_renewable_indices(
            ppu_dictionary, config, ror_production=cached_data.ror_production
        )
    
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
        renewable_mw, renewable_breakdown = calculate_renewable_production_fast(
            t, precomputed_renewables, solar_data, wind_data
        )
        renewable_production_series[i] = renewable_mw
        
        # Track renewable production hourly
        for name, prod in renewable_breakdown.items():
            if name in ppu_production_hourly:
                ppu_production_hourly[name][i] = prod
        
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
                    dispatch_mw = dispatch / timestep_h
                    dispatchable_mw += dispatch_mw
                    remaining_deficit_mwh -= dispatch
                    
                    # Track hourly
                    if dispatcher.ppu_name in ppu_production_hourly:
                        ppu_production_hourly[dispatcher.ppu_name][i] += dispatch_mw
                    
                elif option['type'] == 'storage':
                    storage = option['object']
                    discharged = storage.discharge(remaining_deficit_mwh)
                    remaining_deficit_mwh -= discharged
                    dis_mw = discharged / timestep_h
                    storage_discharged_mw += dis_mw
                    
                    # Track hourly under the EXTRACTING PPU name (not storage name)
                    # This ensures RoT calculation uses correct PPU production volumes
                    storage_name = option['name']
                    extracting_ppu = STORAGE_TO_EXTRACTING_PPU.get(storage_name, storage_name)
                    if extracting_ppu not in ppu_production_hourly:
                        ppu_production_hourly[extracting_ppu] = np.zeros(n_timesteps)
                    ppu_production_hourly[extracting_ppu][i] += dis_mw
            
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
            # ===== STEP 4: SURPLUS - Utility-Based Proportional Storage Charging =====
            # All storages self-regulate around their target SoC (60%)
            # Distribution is proportional to: charge_willingness × efficiency
            
            surplus_mwh = balance_mw * timestep_h
            state.total_surplus_mwh += surplus_mwh
            
            # Calculate charge willingness for each storage
            # disposition_index: -1 (empty, wants charge) to +1 (full, wants discharge)
            # charge_willingness = how much storage wants to charge (0 to 1)
            target_soc = config.storage.TARGET_SOC_FRACTION
            
            charge_weights = {}
            total_charge_weight = 0.0
            
            for storage_name, storage in state.storages.items():
                d_stor = calculate_disposition_index(storage.soc, soc_target=target_soc)
                # Charge willingness: stronger when below target (d_stor < 0)
                # Also consider available capacity
                available_capacity = storage.capacity_mwh * (1.0 - storage.soc)
                if available_capacity > 0 and d_stor < 0.5:  # Only charge if below ~target+deadband
                    # Weight = willingness × efficiency × available capacity factor
                    willingness = max(0.0, 0.5 - d_stor)  # 0 to 1
                    weight = willingness * storage.efficiency_charge
                    charge_weights[storage_name] = weight
                    total_charge_weight += weight
            
            # Distribute surplus proportionally
            remaining_surplus_mwh = surplus_mwh
            
            if total_charge_weight > 0:
                for storage_name, weight in charge_weights.items():
                    if remaining_surplus_mwh <= 0:
                        break
                    
                storage = state.storages[storage_name]
                    proportion = weight / total_charge_weight
                    allocated_mwh = surplus_mwh * proportion
                    
                    # Ghost PPU mechanism for Biooil and Palm oil
                    # They sell surplus electricity on spot → import fuel
                    if storage_name in ('Biooil', 'Palm oil'):
                        import_price = 67.0 if storage_name == 'Biooil' else 87.0
                        
                        # Sell allocated surplus on spot market
                        spot_revenue = allocated_mwh * spot_price
                        
                        # Buy fuel with revenue (convert electricity value to fuel)
                        fuel_mwh = spot_revenue / import_price
                        
                        # Charge storage with imported fuel
                        actually_charged = storage.charge(fuel_mwh)
                        
                        if actually_charged > 0:
                            # Track spot sale (electricity sold by ghost PPU)
                            actual_elec_sold = actually_charged * import_price / spot_price if spot_price > 0 else 0
                            state.total_spot_sell_mwh += actual_elec_sold
                            state.total_spot_sell_revenue += actual_elec_sold * spot_price
                            remaining_surplus_mwh -= actual_elec_sold
                    else:
                        # Regular storage: charge directly with electricity
                        charged = storage.charge(allocated_mwh)
                remaining_surplus_mwh -= charged
                storage_charged_mw += charged / timestep_h
            
            # Any remaining surplus is sold on spot market
            if remaining_surplus_mwh > 0:
                surplus_series[i] = remaining_surplus_mwh / timestep_h
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
        
        # =====================================================================
        # MANDATORY AVIATION FUEL CONSUMPTION
        # =====================================================================
        # Aviation fuel (23 TWh/year biooil) is a HARD constraint
        # We ALWAYS meet it by:
        # 1. First, use biooil from storage (if available)
        # 2. Then, IMPORT any shortfall at import price (67 CHF/MWh)
        # This ensures aviation fuel is always consumed - no shortfall.
        
        from_storage_mwh = 0.0
        imported_mwh = 0.0
        
        if 'Biooil' in state.storages:
            biooil_storage = state.storages['Biooil']
            
            # Try to discharge from storage first
            available_biooil_mwh = biooil_storage.current_mwh
            from_storage_mwh = min(aviation_fuel_required_mwh, available_biooil_mwh)
            
            # Withdraw from storage (no efficiency loss - raw fuel to planes)
            biooil_storage.current_mwh -= from_storage_mwh
        
        # Import any shortfall (automatic purchase at import price)
        imported_mwh = aviation_fuel_required_mwh - from_storage_mwh
        
        # Total consumed = from storage + imported (always meets requirement)
        total_consumed_mwh = from_storage_mwh + imported_mwh
        
        # Track consumption (always equals required)
        aviation_fuel_consumed_series[i] = total_consumed_mwh
        total_aviation_fuel_consumed_mwh += total_consumed_mwh
        
        # Shortfall is now only tracked for info (but should always be 0)
        # We import to cover it, so actual shortfall = 0
        aviation_fuel_shortfall_series[i] = 0.0  # No shortfall - we import
        
        # Calculate import cost for ALL biooil consumed
        # (Biooil must be purchased/imported regardless of source)
        import_cost = total_consumed_mwh * biooil_import_price
        aviation_fuel_import_cost_series[i] = import_cost
        total_aviation_fuel_import_cost_chf += import_cost
    
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
        'ppu_production': ppu_production_hourly,
        # Summary stats
        'total_renewable_mwh': total_renewable_mwh,
        'total_dispatchable_mwh': total_dispatchable_mwh,
        'dispatchable_capacity_mw': total_dispatchable_mw,
        # =====================================================================
        # AVIATION FUEL RESULTS
        # =====================================================================
        'aviation_fuel_consumed_mwh': total_aviation_fuel_consumed_mwh,
        'aviation_fuel_shortfall_mwh': total_aviation_fuel_shortfall_mwh,
        'aviation_fuel_import_cost_chf': total_aviation_fuel_import_cost_chf,
        'aviation_fuel_consumed_series': aviation_fuel_consumed_series,
        'aviation_fuel_shortfall_series': aviation_fuel_shortfall_series,
        'aviation_fuel_import_cost_series': aviation_fuel_import_cost_series,
        'aviation_fuel_required_hourly_mwh': aviation_fuel_required_mwh,
        # Validation: True if all hourly requirements were met
        'aviation_fuel_constraint_met': total_aviation_fuel_shortfall_mwh < 1e-6,
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
    ppu_definitions: Optional[Dict] = None,
) -> float:
    """
    Compute total cost for a scenario using DETAILED per-PPU tracking.
    
    Includes:
    - Net spot market cost (buy - sell)
    - PPU operational costs (based on actual production per PPU)
    - Aviation fuel import costs (biooil for aviation)
    
    This uses the same cost calculation logic as portfolio_metrics.py
    for consistency across the entire pipeline.
    
    Args:
        results: Results from dispatch simulation (must include 'ppu_production')
        ppu_dictionary: PPU dictionary DataFrame
        config: Configuration
        ppu_definitions: Optional dict of PPUDefinition objects for accurate costs
        
    Returns:
        Total cost in CHF
    """
    # Conversion factor: cost_per_mwh in PPUDefinition is CHF/kWh, need CHF/MWh
    CHF_KWH_TO_CHF_MWH = 1000.0
    
    # Spot market cost
    spot_cost = results.get('net_spot_cost_chf', 0.0)
    
    # Aviation fuel import cost (biooil purchased for aviation)
    aviation_fuel_cost = results.get('aviation_fuel_import_cost_chf', 0.0)
    
    # PPU operational costs - DETAILED calculation using actual production per PPU
    ppu_cost = 0.0
    ppu_production = results.get('ppu_production', {})
    
    if ppu_production:
        for ppu_name, production_array in ppu_production.items():
            if not isinstance(production_array, np.ndarray):
                continue
            
            # Total production for this PPU (MWh)
            total_production_mwh = np.sum(production_array)
            
            if total_production_mwh <= 0:
                continue
            
            # Get cost per MWh for this PPU
            cost_per_mwh = 50.0  # Default fallback (CHF/MWh)
            
            # Try to get accurate cost from ppu_definitions first
            if ppu_definitions is not None:
                ppu_def = ppu_definitions.get(ppu_name)
                if ppu_def is not None:
                    if hasattr(ppu_def, 'cost_per_mwh'):
                        # cost_per_mwh is actually in CHF/kWh, convert to CHF/MWh
                        cost_per_mwh = ppu_def.cost_per_mwh * CHF_KWH_TO_CHF_MWH
                    elif isinstance(ppu_def, dict) and 'cost_per_mwh' in ppu_def:
                        cost_per_mwh = ppu_def['cost_per_mwh'] * CHF_KWH_TO_CHF_MWH
            
            # Fallback: try to get from ppu_dictionary DataFrame
            if cost_per_mwh == 50.0 and not ppu_dictionary.empty:
                ppu_rows = ppu_dictionary[ppu_dictionary['PPU_Name'] == ppu_name]
                if not ppu_rows.empty and 'Cost_CHF_per_MWh' in ppu_rows.columns:
                    cost_per_mwh = ppu_rows['Cost_CHF_per_MWh'].values[0]
        
            # Add production cost
            ppu_cost += total_production_mwh * cost_per_mwh
    
    return spot_cost + ppu_cost + aviation_fuel_cost


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

