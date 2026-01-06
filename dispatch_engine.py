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
        # Get PPU unit count from row (critical for correct production!)
        unit_count = row.get('Count', 1)
        
        # River hydro (incidence)
        if 'River' in components or ppu_name == 'HYD_R':
            river_indices.append(i)
            continue
            
        # Solar thermal store (incidence -> storage)
        if 'Solar concentrator' in str(components) or 'SOL_SALT_STORE' in ppu_name:
            sol_store_indices.append(i)
        
        # Solar PV - count UNITS (not rows!)
        if 'PV' in components:
            pv_count += unit_count
            pv_efficiency_sum += (efficiency if not pd.isna(efficiency) else 0.84) * unit_count
        
        # Wind onshore - count UNITS (not rows!)
        elif 'Wind (onshore)' in components:
            wind_onshore_count += unit_count
            wind_efficiency_sum += (efficiency if not pd.isna(efficiency) else 0.84) * unit_count
            
        # Wind offshore - count UNITS (not rows!)
        elif 'Wind (offshore)' in components:
            wind_offshore_count += unit_count
            wind_efficiency_sum += (efficiency if not pd.isna(efficiency) else 0.84) * unit_count
    
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
    chain_efficiency: float,  # NOTE: Not used - see below
) -> float:
    """Distributed solar power calculation - sums across ALL locations.
    
    DISTRIBUTED MODEL: Each PPU adds area_per_location_m2 to EVERY location.
    Total area = n_ppu × area_per_location_m2 × n_locations
    
    IMPORTANT: We use DIRECT-TO-GRID efficiency (PV → Inverter → Grid ≈ 0.95)
    NOT the full chain efficiency which includes Battery (0.795).
    Storage efficiency is applied separately by the dispatch engine when
    charging/discharging to avoid DOUBLE-COUNTING losses.
    
    Args:
        irradiance_all_locations: Solar irradiance at ALL locations (1D array)
        n_ppu: Number of PV PPU units
        area_per_location_m2: Panel area per PPU per location (1000 m²)
        chain_efficiency: PPU chain efficiency (IGNORED to prevent double-counting)
        
    Returns:
        Total power in MW
    """
    if n_ppu == 0:
        return 0.0
    
    efficiency_pv = 0.20  # 20% PV panel efficiency
    efficiency_inverter = 0.95  # 95% inverter efficiency (PV → Grid direct)
    # NOTE: Storage efficiency (Battery ~0.88 round-trip) is NOT applied here
    # It's applied by dispatch_engine when charging/discharging storage
    
    total_power = 0.0
    
    # Sum production across ALL locations
    for i in range(len(irradiance_all_locations)):
        # Total area at this location = n_ppu × area_per_location
        total_area_at_loc = n_ppu * area_per_location_m2
        # P = irradiance × area × PV_efficiency × inverter_efficiency
        # Units: kWh/m²/h × m² × efficiency = kW → /1000 = MW
        power = irradiance_all_locations[i] * total_area_at_loc * efficiency_pv * efficiency_inverter / 1000.0
        total_power += power
    
    return total_power


@njit(cache=True)
def _calculate_wind_power_distributed(
    wind_speeds_all_locations: np.ndarray,
    n_ppu: int,
    turbines_per_location: int,
    chain_efficiency: float,  # NOTE: Not used - see below
    rated_power: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
) -> float:
    """Distributed wind power calculation - sums across ALL locations.
    
    DISTRIBUTED MODEL: Each PPU adds turbines_per_location to EVERY location.
    Total turbines = n_ppu × turbines_per_location × n_locations
    
    IMPORTANT: We use DIRECT-TO-GRID efficiency (Wind → Inverter → Grid ≈ 0.95)
    NOT the full chain efficiency which includes Battery (0.795).
    Storage efficiency is applied separately by the dispatch engine when
    charging/discharging to avoid DOUBLE-COUNTING losses.
    
    Args:
        wind_speeds_all_locations: Wind speed at ALL locations (1D array)
        n_ppu: Number of wind PPU units
        turbines_per_location: Turbines per PPU per location (1)
        chain_efficiency: PPU chain efficiency (IGNORED to prevent double-counting)
        rated_power: Rated power per turbine (MW)
        cut_in: Cut-in wind speed (m/s)
        rated_speed: Rated wind speed (m/s)
        cut_out: Cut-out wind speed (m/s)
        
    Returns:
        Total power in MW
    """
    if n_ppu == 0:
        return 0.0
    
    # Direct-to-grid efficiency (no storage in path)
    # Storage efficiency is applied by dispatch_engine when charging/discharging
    efficiency_direct_to_grid = 0.95  # Wind turbine → Grid (generator + transformer)
    
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
        
        total_power += power * efficiency_direct_to_grid
    
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
    
    # Import pricing (for palm oil - dynamic from rea_holdings_share_prices.csv)
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
                max_charge_power_mw=storage.max_charge_power_mw,
                max_discharge_power_mw=storage.max_discharge_power_mw,
                efficiency_charge=storage.efficiency_charge,
                efficiency_discharge=storage.efficiency_discharge,
                import_price_chf_per_mwh=storage.import_price_chf_per_mwh,
            )
        return new_state


# =============================================================================
# DISPATCH INDICES (Simplified Disposition-Based)
# =============================================================================

# Smoothing parameter Δ_s for nearly linear behavior between SoC 0.4 and 0.8
# With Δ_s = 0.20: tanh(-1) ≈ -0.76 at SoC=0.4, tanh(1) ≈ 0.76 at SoC=0.8
DELTA_S_SMOOTHING = 0.20
SOC_TARGET = 0.60


@njit(cache=True)
def calculate_disposition_index(
    soc: float,
    soc_target: float = SOC_TARGET,
    delta_s: float = DELTA_S_SMOOTHING
) -> float:
    """
    Calculate disposition index for storage.
    
    Simple formula: d_s = tanh((SoC - SoC_target) / Δ_s)
    
    Result is normalized to [0, 1] range:
    - 1.0: Storage full (SoC=1) → wants to discharge
    - 0.5: At target (SoC=0.6) → neutral
    - 0.0: Storage empty (SoC=0) → wants to charge
    
    Args:
        soc: State of charge [0, 1]
        soc_target: Target SoC (default 0.60)
        delta_s: Smoothing parameter for linear behavior (default 0.20)
        
    Returns:
        Disposition index in [0, 1]
    """
    # Raw tanh in [-1, 1]
    raw = np.tanh((soc - soc_target) / delta_s)
    # Transform to [0, 1]: (raw + 1) / 2
    return (raw + 1.0) / 2.0


def calculate_storage_weights(
    storages: Dict[str, Any],
    for_discharge: bool
) -> Dict[str, float]:
    """
    Calculate dispatch weights for all storages based on disposition indices.
    
    Weight formula:
    - For discharge (deficit): weight_s = d_s / sum(d_s)  [prioritize high SoC]
    - For charge (surplus):    weight_s = (1-d_s) / sum(1-d_s)  [prioritize low SoC]
    
    Args:
        storages: Dictionary of objects with 'soc' attribute (StorageState or similar)
        for_discharge: True for deficit (discharge), False for surplus (charge)
        
    Returns:
        Dictionary mapping storage name to weight [0, 1], weights sum to 1
    """
    # Calculate disposition index for each storage
    dispositions = {}
    for name, storage in storages.items():
        d = calculate_disposition_index(storage.soc)
        dispositions[name] = d
    
    # Calculate weights
    weights = {}
    
    if for_discharge:
        # For discharge: use d directly (higher SoC = higher priority)
        total = sum(dispositions.values())
        if total > 1e-9:
            for name, d in dispositions.items():
                weights[name] = d / total
        else:
            # All at zero → equal weights
            n = len(storages)
            for name in storages:
                weights[name] = 1.0 / n if n > 0 else 0.0
    else:
        # For charge: use (1 - d) (lower SoC = higher priority)
        inverted = {name: 1.0 - d for name, d in dispositions.items()}
        total = sum(inverted.values())
        if total > 1e-9:
            for name, inv_d in inverted.items():
                weights[name] = inv_d / total
        else:
            # All at one → equal weights
            n = len(storages)
            for name in storages:
                weights[name] = 1.0 / n if n > 0 else 0.0
    
    return weights


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
# STORAGE-BASED SYSTEM ONLY
# =============================================================================
# ALL PPUs must extract from storage - no direct fuel generation
# Energy flow: Incidence/Import → Storage → Extract → Grid
# 
# Examples:
#   - BIO_WOOD: Wood → Wood Storage → Pyrolysis → ... → Grid
#   - IMP_BIOG: Biogas → Biogas Storage → Purification → ... → Grid
#   - THERM: Fuel Tank storage → Transport → CCPP → Grid
#
# If storage is empty and demand exceeds renewable production:
#   → Buy electricity from spot market (LAST RESORT ONLY)
# =============================================================================

# Mapping from storage name to the PPU(s) that extract from it
# Used to:
# 1. Attribute storage discharge production to the correct PPU for RoT calculation
# 2. Look up the extracting PPU's LCOE for discharge cost
STORAGE_TO_EXTRACTING_PPU = {
    'Lake': 'HYD_S',
    'Solar salt': 'SOL_SALT',  # Could also be SOL_STEAM, using primary
    'Biogas': 'IMP_BIOG',
    'H2 UG 200bar': 'H2P_G',
    'CH4 200bar': 'THERM_CH4',
    'Liquid H2': 'H2P_L',
    'Fuel Tank': 'THERM',  # Also provides aviation fuel
    'Ammonia': 'NH3_P',
    'Palm oil': 'PALM_ICE',  # Only imported bio-fuel
}


def get_storage_discharge_cost(
    storage_name: str,
    ppu_definitions: Optional[Dict] = None,
    default_cost: float = 50.0
) -> float:
    """
    Get discharge cost for a storage based on its extracting PPU's LCOE.
    
    The discharge cost = LCOE of the extracting PPU (in CHF/MWh).
    This ensures costs are consistent with PPU definitions.
    
    Default LCOE values (from ppu_efficiency_lcoe_analysis.csv):
    - HYD_S: ~4 CHF/MWh (very cheap hydro)
    - THERM: ~82 CHF/MWh (CCPP)
    - H2P_G/H2P_L: ~150 CHF/MWh (fuel cells)
    - THERM_CH4: ~95 CHF/MWh (gas turbine)
    - NH3_P: ~160 CHF/MWh (ammonia power)
    - IMP_BIOG: ~70 CHF/MWh (biogas)
    - PALM_ICE: ~180 CHF/MWh (palm oil ICE)
    
    Args:
        storage_name: Name of the storage
        ppu_definitions: Dictionary of PPUDefinition objects
        default_cost: Default cost if PPU not found (CHF/MWh)
        
    Returns:
        Discharge cost in CHF/MWh
    """
    # Default LCOE values based on extracting PPU (from ppu_efficiency_lcoe_analysis.csv)
    # These are used for ordering storages - cheapest dispatched first
    DEFAULT_LCOE_BY_STORAGE = {
        'Lake': 4,           # HYD_S - very cheap hydro turbine
        'Solar salt': 90,    # SOL_SALT - thermal extraction
        'Biogas': 70,        # IMP_BIOG - biogas CHP
        'H2 UG 200bar': 150, # H2P_G - hydrogen fuel cell
        'CH4 200bar': 95,    # THERM_CH4 - methane gas turbine
        'Liquid H2': 150,    # H2P_L - hydrogen fuel cell
        'Fuel Tank': 82,     # THERM - CCPP synthetic fuel
        'Ammonia': 160,      # NH3_P - ammonia to power
        'Palm oil': 180,     # PALM_ICE - palm oil ICE generator
    }
    
    # First try to get from default lookup
    if storage_name in DEFAULT_LCOE_BY_STORAGE:
        base_cost = DEFAULT_LCOE_BY_STORAGE[storage_name]
    else:
        base_cost = default_cost
    
    # Override with PPU definitions if available
    extracting_ppu = STORAGE_TO_EXTRACTING_PPU.get(storage_name)
    
    if extracting_ppu is not None and ppu_definitions is not None:
        ppu_def = ppu_definitions.get(extracting_ppu)
        if ppu_def is not None:
            # cost_per_mwh is actually in CHF/kWh (historical naming), convert to CHF/MWh
            if hasattr(ppu_def, 'cost_per_mwh'):
                base_cost = ppu_def.cost_per_mwh * 1000  # CHF/kWh → CHF/MWh
            elif isinstance(ppu_def, dict) and 'cost_per_mwh' in ppu_def:
                base_cost = ppu_def['cost_per_mwh'] * 1000
    
    return base_cost


# =============================================================================
# REMOVED: Dispatchable generators (THERM_G, THERM_M, BIO_WOOD as direct generators)
# =============================================================================
# These were removed because:
# 1. THERM_G and THERM_M don't exist in ppu_constructs_components.csv
# 2. BIO_WOOD requires Wood Storage (not direct input)
# 3. All PPUs must follow storage-based energy flow
# 4. Shortfalls are covered by spot market, not direct fuel import
# =============================================================================


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
    
    ACTIVE REGULATION: All storages need charge power to participate in
    SoC regulation (buy from spot when below target). If no input PPUs,
    use discharge power or max_power_MW as fallback.
    
    Capacity is scaled by input PPUs (minimum 1 for base capacity).
    
    Args:
        portfolio_counts: PPU counts from portfolio
        config: Configuration
        
    Returns:
        Dictionary of StorageState objects
    """
    storage_defs = config.storage.STORAGE_DEFINITIONS
    initial_soc = config.storage.INITIAL_SOC_FRACTION
    mw_per_unit = config.ppu.MW_PER_UNIT  # 10 MW per PPU unit
    
    # Storages that use ghost PPU mechanism (sell electricity → buy fuel)
    GHOST_PPU_STORAGES = {'Palm oil'}
    
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
        
        # =====================================================================
        # PPU-ONLY CHARGING: No fallback - only INPUT PPUs can charge storage
        # =====================================================================
        # If no INPUT PPUs in portfolio, charge_power_mw stays 0
        # This ensures storages can ONLY be charged by their designated PPUs
        # No artificial regulation via spot market
        #
        # EXCEPTION: Ghost PPU storages (Palm oil) use spot market to buy fuel
        # They need charge power > 0 for the ghost mechanism to work
        # =====================================================================
        if storage_name in GHOST_PPU_STORAGES and charge_power_mw == 0:
            # Ghost PPU storages: allow unlimited charging via spot market
            # Use the storage's max_power_MW as the charge limit
            charge_power_mw = storage_def.get('max_power_MW', 5000)
        
        # Apply physical power cap if defined (e.g., Lake = 2 GW)
        physical_cap = storage_def.get('physical_power_cap_MW')
        if physical_cap is not None:
            charge_power_mw = min(charge_power_mw, physical_cap)
            discharge_power_mw = min(discharge_power_mw, physical_cap)
        
        # Scale capacity by input PPU count (minimum 1 for base capacity)
        # Special handling: Lake and ghost PPU storages don't scale capacity
        if storage_name == 'Lake' or storage_name in GHOST_PPU_STORAGES:
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
    Run dispatch for a single timestep (simplified disposition-based).
    
    Uses disposition index d_s = (tanh((SoC - 0.6) / Δ_s) + 1) / 2
    with weight-based energy distribution across storages.
    
    Special cases:
    - Empty storages are excluded from discharge
    - Full storages are excluded from charge
    - Spot market is last resort (only when all storages are empty/full)
    
    Args:
        t: Timestep index
        demand_mw: Demand (MW)
        spot_price: Spot price (CHF/MWh)
        renewable_production_mw: Total renewable production (MW)
        state: Current dispatch state
        ppu_dictionary: PPU dictionary DataFrame
        config: Configuration
        phi_smoothed: Smoothed shortfall (unused, kept for API compatibility)
        
    Returns:
        Tuple of (updated_state, phi_smoothed)
    """
    timestep_h = 0.25  # 15 minutes
    epsilon = 1e-6  # Tolerance
    EMPTY_THRESHOLD = 0.0001  # Consider empty if SoC <= 0.01%
    FULL_THRESHOLD = 0.9999   # Consider full if SoC >= 99.99%
    
    # Net system balance
    overflow_mw = demand_mw - renewable_production_mw
    state.overflow_series.append(overflow_mw)
    
    if overflow_mw > epsilon:
        # DEFICIT: Need more energy - discharge storages
        remaining_deficit = overflow_mw * timestep_h  # MWh
        state.total_deficit_mwh += remaining_deficit
        
        # Filter out EMPTY storages
        available_storages = {
            name: s for name, s in state.storages.items()
            if s.soc > EMPTY_THRESHOLD and s.current_mwh > 0
        }
        
        if not available_storages:
            # All storages empty → buy from spot
            state.spot_bought.append((t, remaining_deficit / timestep_h))
            state.total_spot_buy_mwh += remaining_deficit
            state.total_spot_buy_cost += remaining_deficit * spot_price
        else:
            # Calculate weights for non-empty storages (prioritize high SoC)
            weights = calculate_storage_weights(available_storages, for_discharge=True)
            
            for storage_name, weight in weights.items():
                if remaining_deficit <= 0:
                    break
                if weight <= 0:
                    continue
                
                storage = state.storages[storage_name]
                
                # Energy this storage should handle
                storage_energy = remaining_deficit * weight
                
                # Discharge (limited by storage)
                discharged = storage.discharge(storage_energy)
                remaining_deficit -= discharged
                
                # Track production
                ppu_name = f"DISPATCH_{storage_name}"
                if ppu_name not in state.production_by_ppu:
                    state.production_by_ppu[ppu_name] = []
                state.production_by_ppu[ppu_name].append((t, discharged / timestep_h))
            
            # Buy remaining from spot market (last resort - power limits hit)
            if remaining_deficit > 0:
                state.spot_bought.append((t, remaining_deficit / timestep_h))
                state.total_spot_buy_mwh += remaining_deficit
                state.total_spot_buy_cost += remaining_deficit * spot_price
    
    elif overflow_mw < -epsilon:
        # SURPLUS: Excess energy - charge storages
        remaining_surplus = abs(overflow_mw) * timestep_h  # MWh
        state.total_surplus_mwh += remaining_surplus
        
        # Filter out FULL storages
        available_storages = {
            name: s for name, s in state.storages.items()
            if s.soc < FULL_THRESHOLD and (s.capacity_mwh - s.current_mwh) > 0
        }
        
        if not available_storages:
            # All storages full → sell to spot
            state.spot_sold.append((t, remaining_surplus / timestep_h))
            state.total_spot_sell_mwh += remaining_surplus
            state.total_spot_sell_revenue += remaining_surplus * spot_price
        else:
            # Calculate weights for non-full storages (prioritize low SoC)
            weights = calculate_storage_weights(available_storages, for_discharge=False)
            
            for storage_name, weight in weights.items():
                if remaining_surplus <= 0:
                    break
                if weight <= 0:
                    continue
                
                storage = state.storages[storage_name]
                
                # Energy this storage should handle
                storage_energy = remaining_surplus * weight
                
                # Charge (limited by storage)
                charged = storage.charge(storage_energy)
                remaining_surplus -= charged
            
            # Sell remaining to spot market (power limits hit)
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
    
    # Store initial SoC for cyclic constraint checking
    initial_storage_soc = {name: s.soc for name, s in storages.items()}
    
    # Pure storage-based system - no direct fuel generators
    # All power comes from: renewable incidence → storage → extraction → grid
    # Shortfalls are covered by spot market purchases
    
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
    # Mandatory synthetic fuel from Fuel Tank for aviation: 23 TWh/year = ~2625.57 MWh/hour
    # NOTE: Changed from Biooil to Fuel Tank - synthetic fuel for aviation
    aviation_fuel_required_mwh = config.energy_system.AVIATION_FUEL_HOURLY_MWH * timestep_h
    aviation_fuel_consumed_series = np.zeros(n_timesteps)  # Track actual consumption
    aviation_fuel_shortfall_series = np.zeros(n_timesteps)  # Track any shortfall
    aviation_fuel_import_cost_series = np.zeros(n_timesteps)  # Track import costs
    
    # Synthetic fuel cost (from Fuel Tank, produced by SYN_FT/SYN_CRACK)
    # Use LCOE of SYN_FT as the reference cost for aviation fuel
    synthetic_fuel_cost_per_mwh = 60.0  # Default, can be loaded from LCOE data
    
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
    
    if verbose:
        print(f"Running dispatch simulation for {n_timesteps} timesteps...")
        print(f"  Storage-based system: All power from renewables + storage")
    
    # Load water inflow data for Lake (if available)
    from data_loader import load_all_data
    cached_data = load_all_data(config)
    water_inflow_data = cached_data.get_water_inflow(copy=False)  # MWh per hour
    
    for i, t in enumerate(scenario_indices):
        # Get data for this timestep
        demand_mw = demand_data[t] if t < len(demand_data) else 0.0
        
        # CRITICAL: No fallback spot prices allowed
        if t >= len(spot_data):
            raise IndexError(
                f"CRITICAL: Spot price index {t} out of bounds (data length: {len(spot_data)}). "
                f"Cannot use fallback price as it would falsify costs. "
                f"Ensure spot price data covers all simulation timesteps."
            )
        spot_price = spot_data[t]
        
        # Configuration parameters
        target_soc = config.storage.TARGET_SOC_FRACTION  # 0.60
        soc_deadband = 0.05  # ±5% deadband around target
        
        # ===== WATER INFLOW TO LAKE =====
        if 'Lake' in state.storages:
            lake = state.storages['Lake']
            water_inflow_mwh = water_inflow_data[t] if t < len(water_inflow_data) else 0.0
            if water_inflow_mwh > 0:
                space_available = lake.capacity_mwh - lake.current_mwh
                actual_inflow = min(water_inflow_mwh, space_available)
                lake.current_mwh += actual_inflow
        
        # ===== STEP 1: Calculate renewable production =====
        renewable_mw, renewable_breakdown = calculate_renewable_production_fast(
            t, precomputed_renewables, solar_data, wind_data
        )
        renewable_production_series[i] = renewable_mw
        
        for name, prod in renewable_breakdown.items():
            if name in ppu_production_hourly:
                ppu_production_hourly[name][i] = prod
        
        # Calculate initial balance
        balance_mw = renewable_mw - demand_mw  # Positive = surplus, Negative = deficit
        
        storage_discharged_mw = 0.0
        storage_charged_mw = 0.0
        dispatchable_mw = 0.0
        
        # =========================================================================
        # STEP 2: SIMPLIFIED DISPOSITION-BASED DISPATCH
        # =========================================================================
        # Uses disposition index d_s = (tanh((SoC - 0.6) / Δ_s) + 1) / 2
        # 
        # Weight calculation:
        # - Discharge: weight_s = d_s / sum(d_s)  [prioritize high SoC]
        # - Charge:    weight_s = (1-d_s) / sum(1-d_s)  [prioritize low SoC]
        #
        # PPU distribution within storage:
        # - PPU_energy = storage_energy × (PPU_count / total_PPUs_for_storage)
        #
        # Palm oil special case: sell electricity → buy palm oil
        # Last resort: spot market
        # =========================================================================
        
        mw_per_unit = config.ppu.MW_PER_UNIT  # 10 MW per PPU unit
        
        # Build storage → PPU mappings
        STORAGE_INPUT_PPUS = {}   # storage_name → list of {ppu_name, count, max_power}
        STORAGE_EXTRACT_PPUS = {} # storage_name → list of {ppu_name, count, max_power}
        
        for storage_name, storage_def in config.storage.STORAGE_DEFINITIONS.items():
            if storage_name not in state.storages:
                continue
            
            # INPUT PPUs for this storage
            input_list = []
            for ppu_name in storage_def.get('input_by', []):
                ppu_count = portfolio_counts.get(ppu_name, 0)
                if ppu_count > 0:
                    input_list.append({
                        'ppu_name': ppu_name,
                        'count': ppu_count,
                        'max_power_mw': ppu_count * mw_per_unit,
                        'efficiency': storage_def['efficiency_charge'],
                    })
            STORAGE_INPUT_PPUS[storage_name] = input_list
            
            # EXTRACT PPUs for this storage
            extract_list = []
            for ppu_name in storage_def.get('extracted_by', []):
                ppu_count = portfolio_counts.get(ppu_name, 0)
                if ppu_count > 0:
                    extract_list.append({
                        'ppu_name': ppu_name,
                        'count': ppu_count,
                        'max_power_mw': ppu_count * mw_per_unit,
                        'efficiency': storage_def['efficiency_discharge'],
                    })
            STORAGE_EXTRACT_PPUS[storage_name] = extract_list
        
        # =========================================================================
        # HANDLE DEFICIT (demand > renewable): Discharge storages
        # =========================================================================
        if balance_mw < 0:
            remaining_deficit_mwh = abs(balance_mw) * timestep_h
            
            # SPECIAL CASE: Filter out EMPTY storages before calculating weights
            # Only storages with available energy participate in discharging
            EMPTY_THRESHOLD = 0.0001  # Consider empty if SoC <= 0.01%
            
            available_storages = {}
            for name, storage in state.storages.items():
                if storage.soc > EMPTY_THRESHOLD and storage.current_mwh > 0:
                    available_storages[name] = storage
            
            # If ALL storages are empty → buy everything from spot market
            if not available_storages:
                buy_mwh = remaining_deficit_mwh
                state.spot_bought.append((i, buy_mwh / timestep_h))
                state.total_spot_buy_mwh += buy_mwh
                state.total_spot_buy_cost += buy_mwh * spot_price
                state.total_deficit_mwh += buy_mwh
                deficit_series[i] = buy_mwh / timestep_h
            else:
                # Calculate weights only for non-empty storages (prioritize high SoC)
                weights = calculate_storage_weights(available_storages, for_discharge=True)
                
                # Distribute energy across available storages by weight
                for storage_name, weight in weights.items():
                    if remaining_deficit_mwh <= 0:
                        break
                    if weight <= 0:
                        continue
                    
                    storage = state.storages[storage_name]
                    extract_ppus = STORAGE_EXTRACT_PPUS.get(storage_name, [])
                    
                    # Skip if no extracting PPUs
                    if not extract_ppus:
                        continue
                    
                    # Energy this storage should handle = total × weight
                    storage_energy_mwh = remaining_deficit_mwh * weight
                    
                    # Check constraints: storage available and power capacity
                    total_extract_power = sum(p['max_power_mw'] for p in extract_ppus)
                    max_from_power = total_extract_power * timestep_h
                    max_from_storage = storage.current_mwh * storage.efficiency_discharge
                    
                    # Actual energy this storage can provide
                    actual_discharge_mwh = min(storage_energy_mwh, max_from_power, max_from_storage)
                    
                    if actual_discharge_mwh <= 0:
                        continue
                    
                    # Discharge from storage
                    fuel_withdrawn = actual_discharge_mwh / storage.efficiency_discharge
                    storage.current_mwh = max(0, storage.current_mwh - fuel_withdrawn)
                    
                    remaining_deficit_mwh -= actual_discharge_mwh
                    dis_mw = actual_discharge_mwh / timestep_h
                    storage_discharged_mw += dis_mw
                    
                    # Distribute among EXTRACT PPUs proportionally
                    total_ppu_count = sum(p['count'] for p in extract_ppus)
                    for ppu_info in extract_ppus:
                        ppu_share = ppu_info['count'] / total_ppu_count if total_ppu_count > 0 else 0
                        ppu_energy_mw = dis_mw * ppu_share
                        
                        if ppu_info['ppu_name'] not in ppu_production_hourly:
                            ppu_production_hourly[ppu_info['ppu_name']] = np.zeros(n_timesteps)
                        ppu_production_hourly[ppu_info['ppu_name']][i] += ppu_energy_mw
            
                # SPOT MARKET: Last resort if still deficit (power limits hit)
            if remaining_deficit_mwh > 0:
                buy_mwh = remaining_deficit_mwh
                state.spot_bought.append((i, buy_mwh / timestep_h))
                state.total_spot_buy_mwh += buy_mwh
                state.total_spot_buy_cost += buy_mwh * spot_price
                state.total_deficit_mwh += buy_mwh
                deficit_series[i] = buy_mwh / timestep_h
        
        # =========================================================================
        # HANDLE SURPLUS (renewable > demand): Charge storages
        # =========================================================================
        else:
            surplus_mwh = balance_mw * timestep_h
            state.total_surplus_mwh += surplus_mwh
            remaining_surplus_mwh = surplus_mwh
            
            # SPECIAL CASE: Filter out FULL storages before calculating weights
            # Only storages with available capacity participate in charging
            FULL_THRESHOLD = 0.9999  # Consider full if SoC >= 99.99%
            
            available_storages = {}
            for name, storage in state.storages.items():
                available_capacity = storage.capacity_mwh - storage.current_mwh
                if storage.soc < FULL_THRESHOLD and available_capacity > 0:
                    available_storages[name] = storage
            
            # If ALL storages are full → sell everything to spot market
            if not available_storages:
                surplus_series[i] = remaining_surplus_mwh / timestep_h
                state.spot_sold.append((i, remaining_surplus_mwh / timestep_h))
                state.total_spot_sell_mwh += remaining_surplus_mwh
                state.total_spot_sell_revenue += remaining_surplus_mwh * spot_price
            else:
                # Calculate weights only for non-full storages (prioritize low SoC)
                weights = calculate_storage_weights(available_storages, for_discharge=False)
                
                # Distribute energy across available storages by weight
                for storage_name, weight in weights.items():
                    if remaining_surplus_mwh <= 0:
                        break
                    if weight <= 0:
                        continue
                    
                    storage = state.storages[storage_name]
                    input_ppus = STORAGE_INPUT_PPUS.get(storage_name, [])
                    
                    # Skip if no input PPUs (except Palm oil which uses ghost mechanism)
                    if not input_ppus and storage_name != 'Palm oil':
                        continue
                    
                    # Energy this storage should handle = total × weight
                    storage_energy_mwh = remaining_surplus_mwh * weight
                    
                    # PALM OIL SPECIAL CASE: Sell electricity → buy palm oil
                    if storage_name == 'Palm oil':
                        available_capacity = storage.capacity_mwh - storage.current_mwh
                        if available_capacity <= 0:
                            continue
                        
                        # Sell electricity on spot market
                        elec_to_sell = min(storage_energy_mwh, available_capacity)
                        spot_revenue = elec_to_sell * spot_price
                        
                        # Buy palm oil with revenue
                        day_of_year = t // 24
                        import_price = cached_data.get_palm_oil_price(day_of_year)
                        
                        if import_price > 0:
                            fuel_mwh = spot_revenue / import_price
                            fuel_stored = min(fuel_mwh, available_capacity)
                            storage.current_mwh += fuel_stored
                            
                            if fuel_stored > 0:
                                actual_elec_sold = fuel_stored * import_price / spot_price
                                remaining_surplus_mwh -= actual_elec_sold
                                storage_charged_mw += actual_elec_sold / timestep_h
                                state.total_spot_sell_mwh += actual_elec_sold
                                state.total_spot_sell_revenue += actual_elec_sold * spot_price
                        continue
                    
                    # Regular storage charging via INPUT PPUs
                    total_input_power = sum(p['max_power_mw'] for p in input_ppus)
                    max_from_power = total_input_power * timestep_h
                    available_capacity = storage.capacity_mwh - storage.current_mwh
                    
                    # Actual energy we can charge
                    actual_charge_mwh = min(storage_energy_mwh, max_from_power, available_capacity)
                    
                    if actual_charge_mwh <= 0:
                        continue
                    
                    # Charge storage (apply efficiency)
                    fuel_stored = actual_charge_mwh * storage.efficiency_charge
                    storage.current_mwh = min(storage.capacity_mwh, storage.current_mwh + fuel_stored)
                    
                    remaining_surplus_mwh -= actual_charge_mwh
                    storage_charged_mw += actual_charge_mwh / timestep_h
                
                # Sell remaining surplus to spot market ONLY if couldn't fit in any storage
                # (due to power limits, not capacity - capacity was filtered above)
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
        # MANDATORY AVIATION FUEL CONSUMPTION (from Fuel Tank)
        # =====================================================================
        # Aviation fuel (23 TWh/year synthetic fuel) is a HARD constraint
        # Supplied from Fuel Tank (produced by SYN_FT, SYN_CRACK PPUs)
        # We ALWAYS meet it by:
        # 1. First, use synthetic fuel from Fuel Tank (if available)
        # 2. Then, track shortfall (cannot import synthetic fuel directly)
        
        from_storage_mwh = 0.0
        shortfall_mwh = 0.0
        
        if 'Fuel Tank' in state.storages:
            fuel_tank = state.storages['Fuel Tank']
            
            # Try to discharge from storage first
            available_fuel_mwh = fuel_tank.current_mwh
            from_storage_mwh = min(aviation_fuel_required_mwh, available_fuel_mwh)
            
            # Withdraw from storage (no efficiency loss - direct fuel to planes)
            fuel_tank.current_mwh -= from_storage_mwh
        
        # Track shortfall (synthetic fuel cannot be directly imported)
        shortfall_mwh = aviation_fuel_required_mwh - from_storage_mwh
        
        # Total consumed = what we could get from storage
        total_consumed_mwh = from_storage_mwh
        
        # Track consumption
        aviation_fuel_consumed_series[i] = total_consumed_mwh
        total_aviation_fuel_consumed_mwh += total_consumed_mwh
        
        # Track shortfall (this is a constraint violation if > 0)
        aviation_fuel_shortfall_series[i] = shortfall_mwh
        total_aviation_fuel_shortfall_mwh += shortfall_mwh
        
        # Calculate production cost for fuel consumed (based on Fuel Tank LCOE)
        # Cost is embedded in the synthetic fuel production, not import
        production_cost = total_consumed_mwh * synthetic_fuel_cost_per_mwh
        aviation_fuel_import_cost_series[i] = production_cost
        total_aviation_fuel_import_cost_chf += production_cost
    
    # Calculate total renewable production
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
        'initial_storage_soc': initial_storage_soc,  # For cyclic constraint checking
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
            # CRITICAL: No fallback prices allowed - must find actual cost
            cost_per_mwh = None
            
            # Try to get accurate cost from ppu_definitions first
            if ppu_definitions is not None:
                ppu_def = ppu_definitions.get(ppu_name)
                if ppu_def is not None:
                    if hasattr(ppu_def, 'cost_per_mwh'):
                        # cost_per_mwh is actually in CHF/kWh, convert to CHF/MWh
                        cost_per_mwh = ppu_def.cost_per_mwh * CHF_KWH_TO_CHF_MWH
                    elif isinstance(ppu_def, dict) and 'cost_per_mwh' in ppu_def:
                        cost_per_mwh = ppu_def['cost_per_mwh'] * CHF_KWH_TO_CHF_MWH
            
            # Try to get from ppu_dictionary DataFrame if not found yet
            if cost_per_mwh is None and not ppu_dictionary.empty:
                ppu_rows = ppu_dictionary[ppu_dictionary['PPU_Name'] == ppu_name]
                if not ppu_rows.empty and 'Cost_CHF_per_MWh' in ppu_rows.columns:
                    cost_per_mwh = ppu_rows['Cost_CHF_per_MWh'].values[0]
            
            # CRITICAL: Raise error if cost not found - no fallback allowed
            if cost_per_mwh is None:
                raise ValueError(
                    f"CRITICAL: Cannot find cost_per_mwh for PPU '{ppu_name}'. "
                    f"Cannot use fallback price as it would falsify costs. "
                    f"Please ensure PPU cost is defined in ppu_definitions or ppu_dictionary."
                )
        
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
    print("\n" + "="*60)
    print("SIMPLIFIED DISPOSITION INDEX TEST")
    print("="*60)
    print(f"SoC_target = {SOC_TARGET}, Δ_s = {DELTA_S_SMOOTHING}")
    print("\nDisposition index d_s in [0, 1]:")
    print("  - d_s = 1.0 → full storage, wants to discharge")
    print("  - d_s = 0.5 → at target, neutral")
    print("  - d_s = 0.0 → empty storage, wants to charge")
    print()
    
    # Test disposition index across SoC range
    print(f"{'SoC':>6} | {'d_s':>6} | Interpretation")
    print("-" * 40)
    for soc in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        d = calculate_disposition_index(soc)
        if d > 0.6:
            interp = "wants to discharge"
        elif d < 0.4:
            interp = "wants to charge"
        else:
            interp = "neutral"
        print(f"{soc:>6.2f} | {d:>6.3f} | {interp}")
    
    # Test weight calculation
    print("\n" + "="*60)
    print("WEIGHT CALCULATION TEST")
    print("="*60)
    
    # Create mock storages with different SoC
    class MockStorage:
        def __init__(self, name, soc):
            self.name = name
            self.soc = soc
    
    mock_storages = {
        'Battery': MockStorage('Battery', 0.3),  # Low - wants to charge
        'H2': MockStorage('H2', 0.7),            # High - wants to discharge
        'Lake': MockStorage('Lake', 0.6),         # At target - neutral
    }
    
    print("\nMock storages: Battery(SoC=0.3), H2(SoC=0.7), Lake(SoC=0.6)")
    
    # Test discharge weights
    dis_weights = calculate_storage_weights(mock_storages, for_discharge=True)
    print("\nDischarge weights (prioritize high SoC):")
    for name, w in sorted(dis_weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")
    
    # Test charge weights
    chg_weights = calculate_storage_weights(mock_storages, for_discharge=False)
    print("\nCharge weights (prioritize low SoC):")
    for name, w in sorted(chg_weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")

