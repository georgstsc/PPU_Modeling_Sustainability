"""
================================================================================
PPU FRAMEWORK - Power Production Unit Definitions and Cost Calculations
================================================================================

This module handles PPU (Power Production Unit) definitions, cost calculations,
and portfolio management for the Swiss Energy Storage Optimization project.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import ast
from pathlib import Path

from config import Config, DEFAULT_CONFIG


# =============================================================================
# PPU DATA STRUCTURES
# =============================================================================

@dataclass
class PPUCostData:
    """Cost and efficiency data for a PPU component chain."""
    
    efficiency: float  # Overall chain efficiency [0, 1]
    cost_per_mwh: float  # NOTE: Actually stored in CHF/kWh (misleading field name for historical reasons)
    cost_per_hour: float  # CHF per MW-hour of capacity
    components: List[str]  # Component chain
    
    # Breakdown by component
    component_efficiencies: List[float] = field(default_factory=list)
    component_costs: List[float] = field(default_factory=list)


@dataclass
class PPUDefinition:
    """Complete PPU type definition."""
    
    name: str  # PPU short name (e.g., 'PV', 'HYD_S')
    category: str  # 'Production' or 'Storage'
    extract_type: str  # 'Incidence', 'Flex', 'Store'
    components: List[str]  # Component chain
    
    # Computed cost data
    efficiency: float = 1.0
    cost_per_mwh: float = 0.0  # NOTE: Actually stored in CHF/kWh (misleading field name)
    
    # Storage relationships
    can_extract_from: List[str] = field(default_factory=list)
    can_input_to: List[str] = field(default_factory=list)
    
    # Location rank for renewables (assigned during optimization)
    location_rank: int = 0


# =============================================================================
# COST CALCULATION FUNCTIONS
# =============================================================================

def load_cost_table(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the component cost table.
    
    Args:
        filepath: Path to cost_table_tidy.csv
        
    Returns:
        DataFrame indexed by component name
    """
    df = pd.read_csv(filepath)
    
    # Convert numeric columns
    numeric_cols = ['efficiency', 'w', 'cost', 'investment_chf_per_kw', 
                    'capex', 'opex', 'lifetime', 'cycle_no', 'power', 'capacity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    if 'w' in df.columns:
        df['w'] = df['w'].fillna(0)
    if 'efficiency' in df.columns:
        df['efficiency'] = df['efficiency'].fillna(1.0)
    if 'cost' in df.columns:
        df['cost'] = df['cost'].fillna(0.0)
    
    # Index by item name for fast lookup
    if 'item' in df.columns:
        df = df.set_index('item', drop=False)
    
    return df


def load_ppu_constructs(filepath: str) -> pd.DataFrame:
    """
    Load PPU constructs (component chains).
    
    Args:
        filepath: Path to ppu_constructs_components.csv
        
    Returns:
        DataFrame with PPU definitions
    """
    df = pd.read_csv(filepath)
    
    # Parse component lists from string
    if 'Components' in df.columns:
        df['Components'] = df['Components'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    return df


def get_component_data(component: str, cost_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get efficiency and cost data for a single component.
    
    Args:
        component: Component name
        cost_df: Cost table DataFrame
        
    Returns:
        Dict with efficiency, cost, auxiliary energy (w)
    """
    if component in cost_df.index:
        row = cost_df.loc[component]
        return {
            'efficiency': row['efficiency'] if not pd.isna(row.get('efficiency', np.nan)) else 1.0,
            'cost': row['cost'] if not pd.isna(row.get('cost', np.nan)) else 0.0,
            'w': row['w'] if not pd.isna(row.get('w', np.nan)) else 0.0,
            'type': row.get('component', 'unknown'),
        }
    return {'efficiency': 1.0, 'cost': 0.0, 'w': 0.0, 'type': 'unknown'}


def calculate_chain_efficiency(components: List[str], cost_df: pd.DataFrame) -> float:
    """
    Calculate overall efficiency of a component chain.
    
    The chain efficiency is the product of all component efficiencies.
    
    Args:
        components: List of component names
        cost_df: Cost table DataFrame
        
    Returns:
        Overall efficiency [0, 1]
    """
    efficiency = 1.0
    for component in components:
        comp_data = get_component_data(component, cost_df)
        efficiency *= comp_data['efficiency']
    return efficiency


def calculate_chain_cost(
    components: List[str], 
    cost_df: pd.DataFrame
) -> PPUCostData:
    """
    Calculate the overall cost of a component chain.
    
    Accounts for:
    - Direct component costs
    - Cumulative efficiency losses
    
    Note: Auxiliary energy requirements (w) are ignored as negligible.
    
    Args:
        components: List of component names
        cost_df: Cost table DataFrame
        
    Returns:
        PPUCostData with complete cost breakdown
    """
    total_cost = 0.0
    cumulative_efficiency = 1.0
    component_efficiencies = []
    component_costs = []
    
    for component in components:
        comp_data = get_component_data(component, cost_df)
        
        # Direct cost
        component_cost = comp_data['cost']
        total_cost += component_cost
        component_costs.append(component_cost)
        
        # Track efficiency
        component_efficiencies.append(comp_data['efficiency'])
        cumulative_efficiency *= comp_data['efficiency']
        
        # Auxiliary energy cost is ignored (negligible)
        # Previously: aux_cost = w / cumulative_efficiency
    
    return PPUCostData(
        efficiency=cumulative_efficiency,
        cost_per_mwh=total_cost,  # NOTE: Actually stored in CHF/kWh (misleading field name)
        cost_per_hour=total_cost * 0.25,  # CHF per 15-min timestep per MW
        components=components,
        component_efficiencies=component_efficiencies,
        component_costs=component_costs,
    )


# =============================================================================
# PPU DICTIONARY MANAGEMENT
# =============================================================================

def build_ppu_definitions(
    ppu_constructs_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    storage_definitions: Dict[str, Dict],
) -> Dict[str, PPUDefinition]:
    """
    Build complete PPU definitions from construct data.
    
    Args:
        ppu_constructs_df: PPU constructs DataFrame
        cost_df: Cost table DataFrame
        storage_definitions: Storage configuration from config
        
    Returns:
        Dictionary mapping PPU name to PPUDefinition
    """
    ppu_defs = {}
    
    for _, row in ppu_constructs_df.iterrows():
        ppu_name = row['PPU']
        components = row['Components']
        category = row.get('Category', 'Production')
        extract_type = row.get('Extract', 'Flex')
        
        # Calculate costs
        cost_data = calculate_chain_cost(components, cost_df)
        
        # Determine storage relationships
        can_extract_from = []
        can_input_to = []
        
        for storage_name, storage_def in storage_definitions.items():
            if ppu_name in storage_def.get('extracted_by', []):
                can_extract_from.append(storage_name)
            if ppu_name in storage_def.get('input_by', []):
                can_input_to.append(storage_name)
        
        ppu_defs[ppu_name] = PPUDefinition(
            name=ppu_name,
            category=category,
            extract_type=extract_type,
            components=components,
            efficiency=cost_data.efficiency,
            cost_per_mwh=cost_data.cost_per_mwh,
            can_extract_from=can_extract_from,
            can_input_to=can_input_to,
        )
    
    return ppu_defs


# =============================================================================
# PORTFOLIO REPRESENTATION
# =============================================================================

@dataclass
class Portfolio:
    """
    Represents a PPU portfolio (mix of power production units).
    
    The portfolio is represented as counts of each PPU type.
    """
    
    # PPU counts: {ppu_name: count}
    ppu_counts: Dict[str, int] = field(default_factory=dict)
    
    # Computed properties (cached)
    _total_capacity_gw: Optional[float] = None
    _annual_production_twh: Optional[float] = None
    
    def __post_init__(self):
        """Ensure all counts are non-negative integers."""
        self.ppu_counts = {
            k: max(0, int(v)) for k, v in self.ppu_counts.items()
        }
    
    def get_count(self, ppu_name: str) -> int:
        """Get count for a specific PPU type."""
        return self.ppu_counts.get(ppu_name, 0)
    
    def set_count(self, ppu_name: str, count: int) -> None:
        """Set count for a specific PPU type."""
        self.ppu_counts[ppu_name] = max(0, int(count))
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._total_capacity_gw = None
        self._annual_production_twh = None
    
    def to_array(self, ppu_order: List[str]) -> np.ndarray:
        """Convert portfolio to numpy array in specified order."""
        return np.array([self.get_count(ppu) for ppu in ppu_order], dtype=np.int32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, ppu_order: List[str]) -> 'Portfolio':
        """Create portfolio from numpy array."""
        counts = {ppu: int(arr[i]) for i, ppu in enumerate(ppu_order)}
        return cls(ppu_counts=counts)
    
    def copy(self) -> 'Portfolio':
        """Create a deep copy of this portfolio."""
        return Portfolio(ppu_counts=self.ppu_counts.copy())
    
    def total_units(self) -> int:
        """Total number of PPU units."""
        return sum(self.ppu_counts.values())
    
    def __repr__(self) -> str:
        active = {k: v for k, v in self.ppu_counts.items() if v > 0}
        return f"Portfolio({active})"


def estimate_annual_production(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config = DEFAULT_CONFIG,
) -> float:
    """
    Estimate annual production capacity of a portfolio (TWh/year).
    
    This is a rough estimate based on capacity factors:
    - Solar: ~12% capacity factor
    - Wind onshore: ~25% capacity factor
    - Wind offshore: ~35% capacity factor
    - Hydro: ~40% capacity factor
    - Thermal/other: ~80% capacity factor (dispatchable)
    
    Args:
        portfolio: Portfolio to estimate
        ppu_definitions: PPU definitions
        config: Configuration
        
    Returns:
        Estimated annual production in TWh
    """
    CAPACITY_FACTORS = {
        'PV': 0.12,
        'WD_ON': 0.25,
        'WD_OFF': 0.35,
        'HYD_R': 0.40,
        'HYD_S': 0.40,
        'BIO_WOOD': 0.70,
        'THERM': 0.80,
        'H2P_G': 0.80,
        'H2P_L': 0.80,
        'SOL_SALT': 0.60,
        'SOL_STEAM': 0.60,
        'BIO_OIL_ICE': 0.80,
        'PALM_ICE': 0.80,
        'IMP_BIOG': 0.80,
        'THERM_CH4': 0.80,
        'NH3_P': 0.80,
    }
    
    # Storage PPUs don't produce, they store
    STORAGE_PPUS = {'PHS', 'H2_G', 'H2_GL', 'H2_L', 'SYN_FT', 'SYN_METH', 
                    'NH3_FULL', 'SYN_CRACK', 'CH4_BIO', 'SOL_SALT_STORE',
                    'BIOOIL_IMPORT', 'PALM_IMPORT'}
    
    total_twh = 0.0
    mw_per_unit = config.ppu.MW_PER_UNIT
    hours_per_year = 8760
    
    for ppu_name, count in portfolio.ppu_counts.items():
        if count <= 0 or ppu_name in STORAGE_PPUS:
            continue
        
        # Get capacity factor
        cf = CAPACITY_FACTORS.get(ppu_name, 0.50)  # Default 50%
        
        # Get efficiency
        ppu_def = ppu_definitions.get(ppu_name)
        efficiency = ppu_def.efficiency if ppu_def else 0.8
        
        # Annual production = count * MW_per_unit * CF * efficiency * hours / 1e6
        annual_mwh = count * mw_per_unit * cf * efficiency * hours_per_year
        annual_twh = annual_mwh / 1e6
        total_twh += annual_twh
    
    return total_twh


def check_energy_sovereignty(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config = DEFAULT_CONFIG,
) -> Tuple[bool, float, float]:
    """
    Check if portfolio meets energy sovereignty requirement.
    
    Args:
        portfolio: Portfolio to check
        ppu_definitions: PPU definitions
        config: Configuration
        
    Returns:
        Tuple of (is_sovereign, annual_production_twh, target_twh)
    """
    annual_production = estimate_annual_production(portfolio, ppu_definitions, config)
    target = config.energy_system.TARGET_ANNUAL_DEMAND_TWH
    is_sovereign = annual_production >= target
    
    return is_sovereign, annual_production, target


def check_aviation_fuel_capacity(
    portfolio: Portfolio,
    config: Config = DEFAULT_CONFIG,
) -> Tuple[bool, int, int, float]:
    """
    Check if portfolio has sufficient BIO_OIL_ICE capacity for aviation fuel.
    
    Aviation fuel requirement: 23 TWh/year = ~2625.57 MWh/hour
    Each BIO_OIL_ICE unit provides 10 MW = 10 MWh/hour
    Minimum units needed: ceil(2625.57 / 10) = 263 units
    
    Note: This is a capacity check only. Actual biooil storage replenishment
    (via domestic production or imports) must be sufficient to meet hourly demand.
    
    Args:
        portfolio: Portfolio to check
        config: Configuration
        
    Returns:
        Tuple of (has_capacity, current_units, required_units, hourly_capacity_mwh)
    """
    bio_oil_ice_count = portfolio.ppu_counts.get('BIO_OIL_ICE', 0)
    required_hourly_mwh = config.energy_system.AVIATION_FUEL_HOURLY_MWH
    
    # Each unit = 10 MW = 10 MWh/hour discharge capacity
    mw_per_unit = config.ppu.MW_PER_UNIT
    hourly_capacity_mwh = bio_oil_ice_count * mw_per_unit
    
    # Calculate minimum required units
    min_required_units = int(np.ceil(required_hourly_mwh / mw_per_unit))
    
    has_capacity = bio_oil_ice_count >= min_required_units
    
    return has_capacity, bio_oil_ice_count, min_required_units, hourly_capacity_mwh


# =============================================================================
# CUMULATIVE ENERGY BALANCE CHECK (Using Real Incidence Data)
# =============================================================================

def calculate_cumulative_renewable_production(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    solar_ranking: np.ndarray,
    wind_ranking: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    precomputed_sums: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate ACTUAL cumulative renewable production using DISTRIBUTED deployment model.
    
    DISTRIBUTED MODEL:
    - Solar: Each PPU adds 1000 m² to EVERY location (distributed rooftop solar)
    - Wind: Each PPU adds 1 turbine to EVERY location (distributed wind farms)
    - Production summed across ALL locations for full year
    
    Args:
        portfolio: Portfolio with PPU counts
        ppu_definitions: PPU definitions
        solar_data: Solar irradiance array (n_hours, n_locations) [kWh/m²/h]
        wind_data: Wind speed array (n_hours, n_locations) [m/s]
        solar_ranking: Location indices (not used in distributed model)
        wind_ranking: Location indices (not used in distributed model)
        config: Configuration
        precomputed_sums: Optional dict with pre-computed sums for speed
        
    Returns:
        Tuple of (total_production_mwh, breakdown_by_type)
    """
    mw_per_unit = config.ppu.MW_PER_UNIT
    n_hours = len(solar_data)
    n_locations = solar_data.shape[1] if solar_data.ndim > 1 else 1
    
    # Production breakdown
    production_by_type = {
        'solar': 0.0,
        'wind_onshore': 0.0,
        'wind_offshore': 0.0,
        'hydro_ror': 0.0,
    }
    
    # =================================================================
    # SOLAR PV - DISTRIBUTED across ALL locations
    # Each PPU adds 1000 m² to every location
    # =================================================================
    pv_count = portfolio.get_count('PV')
    if pv_count > 0:
        pv_def = ppu_definitions.get('PV')
        pv_efficiency = pv_def.efficiency if pv_def else 0.84
        area_per_location_m2 = 1000.0  # 1000 m² per PPU per location
        efficiency_pv = 0.20  # 20% panel efficiency
        
        # Total area at each location = n_ppu × 1000 m²
        total_area_per_loc = pv_count * area_per_location_m2
        
        if precomputed_sums and 'solar_location_sums' in precomputed_sums:
            # Use pre-computed annual sums per location (FAST)
            solar_sums = precomputed_sums['solar_location_sums']
            # Sum across ALL locations (distributed model)
            total_irradiance_kwh_m2 = np.sum(solar_sums)
        else:
            # Sum all irradiance across all hours and locations
            total_irradiance_kwh_m2 = np.nansum(solar_data)
        
        # Production = total_irradiance × area × PV_eff × chain_eff
        # Result in MWh (irradiance is kWh/m², area is m², /1000 for MW, × hours implicit)
        production_by_type['solar'] = (
            total_irradiance_kwh_m2 * total_area_per_loc * efficiency_pv * pv_efficiency / 1000
        )
    
    # =================================================================
    # WIND ONSHORE - DISTRIBUTED across ALL locations
    # Each PPU adds 1 turbine (3 MW) to every location
    # =================================================================
    wind_on_count = portfolio.get_count('WD_ON')
    if wind_on_count > 0:
        wind_def = ppu_definitions.get('WD_ON')
        wind_efficiency = wind_def.efficiency if wind_def else 0.84
        turbines_per_location = 1  # 1 turbine per PPU per location
        rated_power_mw = 3.0
        
        # Total turbines at each location = n_ppu × 1
        total_turbines_per_loc = wind_on_count * turbines_per_location
        
        # Calculate wind power at ALL locations for all hours
        hourly_power = _wind_power_vectorized_distributed(
            wind_data, total_turbines_per_loc, rated_power_mw
        )
        production_by_type['wind_onshore'] = np.sum(hourly_power) * wind_efficiency
    
    # =================================================================
    # WIND OFFSHORE - DISTRIBUTED across ALL locations
    # Each PPU adds 1 turbine (5 MW) to every location
    # =================================================================
    wind_off_count = portfolio.get_count('WD_OFF')
    if wind_off_count > 0:
        wind_def = ppu_definitions.get('WD_OFF')
        wind_efficiency = wind_def.efficiency if wind_def else 0.84
        turbines_per_location = 1
        rated_power_mw = 5.0  # Larger offshore turbines
        
        total_turbines_per_loc = wind_off_count * turbines_per_location
        
        hourly_power = _wind_power_vectorized_distributed(
            wind_data, total_turbines_per_loc, rated_power_mw
        )
        production_by_type['wind_offshore'] = np.sum(hourly_power) * wind_efficiency
    
    # =================================================================
    # HYDRO RUN-OF-RIVER (unchanged - capacity factor based)
    # =================================================================
    hyd_r_count = portfolio.get_count('HYD_R')
    if hyd_r_count > 0:
        hyd_def = ppu_definitions.get('HYD_R')
        hyd_efficiency = hyd_def.efficiency if hyd_def else 0.88
        cf_ror = 0.45
        production_by_type['hydro_ror'] = (
            hyd_r_count * mw_per_unit * n_hours * cf_ror * hyd_efficiency
        )
    
    total_production = sum(production_by_type.values())
    
    return total_production, production_by_type


def _wind_power_vectorized_distributed(
    wind_speeds: np.ndarray,
    turbines_per_location: int,
    rated_power_mw: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
) -> np.ndarray:
    """
    Vectorized wind power for DISTRIBUTED model - sum across ALL locations.
    
    Args:
        wind_speeds: (n_hours, n_locations) or (n_hours,) array
        turbines_per_location: Total turbines at each location
        rated_power_mw: Rated power per turbine
        cut_in, rated_speed, cut_out: Wind turbine parameters
        
    Returns:
        Hourly power production summed across all locations (MW)
    """
    ws = np.nan_to_num(wind_speeds, nan=0.0)
    
    # Initialize power array
    power = np.zeros_like(ws)
    
    # Cubic region: cut_in <= ws < rated_speed
    cubic_mask = (ws >= cut_in) & (ws < rated_speed)
    ratio = np.where(cubic_mask, (ws - cut_in) / (rated_speed - cut_in), 0)
    power = np.where(cubic_mask, rated_power_mw * turbines_per_location * (ratio ** 3), power)
    
    # Rated region: rated_speed <= ws <= cut_out  
    rated_mask = (ws >= rated_speed) & (ws <= cut_out)
    power = np.where(rated_mask, rated_power_mw * turbines_per_location, power)
    
    # Sum across locations if 2D, otherwise return 1D
    if power.ndim > 1:
        return np.sum(power, axis=1)  # Sum across locations for each hour
    return power


def _wind_power_vectorized(
    wind_speeds: np.ndarray,
    num_turbines: int,
    rated_power_mw: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0
) -> np.ndarray:
    """
    Fully vectorized wind power calculation for 2D array (n_hours, n_locations).
    """
    power = np.zeros_like(wind_speeds)
    
    # Cubic region
    cubic_mask = (wind_speeds >= cut_in) & (wind_speeds < rated_speed)
    ratio = np.where(cubic_mask, (wind_speeds - cut_in) / (rated_speed - cut_in), 0)
    power = np.where(cubic_mask, rated_power_mw * num_turbines * (ratio ** 3), power)
    
    # Rated region
    rated_mask = (wind_speeds >= rated_speed) & (wind_speeds <= cut_out)
    power = np.where(rated_mask, rated_power_mw * num_turbines, power)
    
    return power


def _wind_power_array(
    wind_speeds: np.ndarray,
    num_turbines: int,
    rated_power_mw: float = 3.0,
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0
) -> np.ndarray:
    """
    Vectorized wind power calculation for an array of wind speeds.
    
    Returns array of power in MW for each timestep.
    """
    power = np.zeros_like(wind_speeds)
    
    # Cubic region (between cut-in and rated)
    cubic_mask = (wind_speeds >= cut_in) & (wind_speeds < rated_speed)
    ratio = (wind_speeds[cubic_mask] - cut_in) / (rated_speed - cut_in)
    power[cubic_mask] = rated_power_mw * num_turbines * (ratio ** 3)
    
    # Rated region (between rated and cut-out)
    rated_mask = (wind_speeds >= rated_speed) & (wind_speeds <= cut_out)
    power[rated_mask] = rated_power_mw * num_turbines
    
    return power


def check_cumulative_energy_balance(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    demand_data: np.ndarray,
    solar_ranking: np.ndarray,
    wind_ranking: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    storage_efficiency: float = 0.75,
    precomputed_sums: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[bool, float, float, float, Dict[str, float]]:
    """
    Check if renewable production can cover total demand over the period.
    
    This is a NECESSARY condition for energy sovereignty:
    - Storage can shift energy in time but cannot create it
    - Total production must be >= Total demand / storage_efficiency
    
    Args:
        portfolio: Portfolio to check
        ppu_definitions: PPU definitions
        solar_data: Solar irradiance data
        wind_data: Wind speed data
        demand_data: Demand time series (MW)
        solar_ranking: Solar location rankings
        wind_ranking: Wind location rankings
        config: Configuration
        storage_efficiency: Round-trip storage efficiency (default 75%)
        
    Returns:
        Tuple of:
        - is_balanced: True if production >= demand (accounting for storage losses)
        - total_production_mwh: Total renewable production
        - total_demand_mwh: Total demand
        - balance_ratio: production / required_production (>= 1.0 means OK)
        - production_breakdown: Dict with production by source type
    """
    # Calculate cumulative renewable production (using precomputed sums if available)
    total_production, breakdown = calculate_cumulative_renewable_production(
        portfolio, ppu_definitions, solar_data, wind_data,
        solar_ranking, wind_ranking, config, precomputed_sums
    )
    
    # Calculate total demand (sum of MW over hours = MWh)
    total_demand = np.sum(demand_data)
    
    # Account for storage round-trip losses
    # We need to produce MORE than demand because some energy is lost in storage
    required_production = total_demand / storage_efficiency
    
    # Check balance
    balance_ratio = total_production / required_production if required_production > 0 else float('inf')
    is_balanced = balance_ratio >= 1.0
    
    return is_balanced, total_production, total_demand, balance_ratio, breakdown


def find_minimum_renewable_portfolio(
    ppu_definitions: Dict[str, PPUDefinition],
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    demand_data: np.ndarray,
    solar_ranking: np.ndarray,
    wind_ranking: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    storage_efficiency: float = 0.75,
    pv_fraction: float = 0.6,
    wind_fraction: float = 0.3,
    hydro_fraction: float = 0.1,
) -> Tuple[Portfolio, float]:
    """
    Find the minimum portfolio that can cover total demand.
    
    Uses binary search to find the smallest portfolio where
    cumulative production >= cumulative demand / storage_efficiency.
    
    Args:
        ppu_definitions: PPU definitions
        solar_data: Solar irradiance data
        wind_data: Wind speed data
        demand_data: Demand time series
        solar_ranking: Solar location rankings
        wind_ranking: Wind location rankings
        config: Configuration
        storage_efficiency: Round-trip storage efficiency
        pv_fraction: Fraction of capacity from PV (default 60%)
        wind_fraction: Fraction of capacity from wind (default 30%)
        hydro_fraction: Fraction of capacity from hydro RoR (default 10%)
        
    Returns:
        Tuple of (minimum_portfolio, balance_ratio)
    """
    # Binary search for minimum scale factor
    min_scale = 1
    max_scale = 1000  # Maximum units per type
    
    def test_scale(scale: int) -> float:
        """Test a portfolio at given scale and return balance ratio."""
        portfolio = Portfolio(ppu_counts={
            'PV': int(scale * pv_fraction),
            'WD_ON': int(scale * wind_fraction),
            'HYD_R': int(scale * hydro_fraction),
        })
        
        is_balanced, _, _, ratio, _ = check_cumulative_energy_balance(
            portfolio, ppu_definitions, solar_data, wind_data,
            demand_data, solar_ranking, wind_ranking, config, storage_efficiency
        )
        return ratio
    
    # Binary search
    while max_scale - min_scale > 1:
        mid_scale = (min_scale + max_scale) // 2
        ratio = test_scale(mid_scale)
        
        if ratio >= 1.0:
            max_scale = mid_scale
        else:
            min_scale = mid_scale
    
    # Use max_scale to ensure we meet the requirement
    final_scale = max_scale
    final_portfolio = Portfolio(ppu_counts={
        'PV': int(final_scale * pv_fraction),
        'WD_ON': int(final_scale * wind_fraction),
        'HYD_R': int(final_scale * hydro_fraction),
    })
    
    # Get final balance ratio
    _, _, _, final_ratio, _ = check_cumulative_energy_balance(
        final_portfolio, ppu_definitions, solar_data, wind_data,
        demand_data, solar_ranking, wind_ranking, config, storage_efficiency
    )
    
    return final_portfolio, final_ratio


# =============================================================================
# PPU DICTIONARY (RUNTIME STATE)
# =============================================================================

def create_ppu_dictionary(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Create a PPU dictionary DataFrame from a portfolio.
    
    Each PPU instance gets its own row with computed costs and storage relationships.
    
    Args:
        portfolio: Portfolio with PPU counts
        ppu_definitions: PPU definitions
        config: Configuration
        
    Returns:
        DataFrame with one row per PPU instance
    """
    rows = []
    ppu_id = 0
    
    for ppu_name, count in portfolio.ppu_counts.items():
        if count <= 0:
            continue
        
        ppu_def = ppu_definitions.get(ppu_name)
        if ppu_def is None:
            continue
        
        for i in range(count):
            ppu_id += 1
            
            # Cost escalation using config's PROGRESSIVE_COST_CAPS
            # Only apply escalation after soft_cap units
            if ppu_name in config.ppu.PROGRESSIVE_COST_CAPS:
                cap_info = config.ppu.PROGRESSIVE_COST_CAPS[ppu_name]
                soft_cap = cap_info.get('soft_cap', 9999)
                factor = cap_info.get('factor', 0.0)
                # Escalation only starts after soft_cap
                units_over_cap = max(0, i - soft_cap)
                escalation_factor = 1.0 + factor * units_over_cap
            else:
                # No escalation for PPUs not in the config
                escalation_factor = 1.0
            
            cost_per_mwh = ppu_def.cost_per_mwh * escalation_factor
            
            rows.append({
                'PPU_ID': ppu_id,
                'PPU_Name': ppu_name,
                'PPU_Category': ppu_def.category,
                'PPU_Extract': ppu_def.extract_type,
                'can_extract_from': ppu_def.can_extract_from.copy(),
                'can_input_to': ppu_def.can_input_to.copy(),
                'Chain_Efficiency': ppu_def.efficiency,
                'Cost_CHF_per_kWh': cost_per_mwh,  # cost_per_mwh is actually in CHF/kWh
                'Cost_CHF_per_MWh': cost_per_mwh * 1000,  # Convert CHF/kWh to CHF/MWh
                'Cost_CHF_per_Quarter_Hour': cost_per_mwh * 0.25,
                'Components': ppu_def.components.copy(),
                'Location_Rank': np.nan,  # Set for renewables
                'Unit_Index': i,
            })
    
    return pd.DataFrame(rows)


def assign_renewable_locations(
    ppu_dictionary: pd.DataFrame,
    solar_ranking: np.ndarray,
    wind_ranking: np.ndarray,
) -> pd.DataFrame:
    """
    Assign optimal locations to renewable PPUs based on rankings.
    
    Solar and wind PPUs are assigned locations in order of best to worst.
    
    Args:
        ppu_dictionary: PPU dictionary DataFrame
        solar_ranking: Array of solar location indices sorted by quality
        wind_ranking: Array of wind location indices sorted by quality
        
    Returns:
        Updated PPU dictionary with Location_Rank filled in
    """
    df = ppu_dictionary.copy()
    
    # Track assigned locations
    solar_idx = 0
    wind_idx = 0
    
    for idx, row in df.iterrows():
        ppu_name = row['PPU_Name']
        components = row['Components']
        
        # Check if solar
        if 'PV' in components:
            if solar_idx < len(solar_ranking):
                df.loc[idx, 'Location_Rank'] = solar_idx + 1  # 1-based ranking
                solar_idx += 1
        
        # Check if wind (onshore or offshore)
        elif 'Wind (onshore)' in components or 'Wind (offshore)' in components:
            if wind_idx < len(wind_ranking):
                df.loc[idx, 'Location_Rank'] = wind_idx + 1
                wind_idx += 1
    
    return df


# =============================================================================
# CONVENIENCE LOADERS
# =============================================================================

def load_all_ppu_data(
    config: Config = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, PPUDefinition]]:
    """
    Load all PPU-related data.
    
    Args:
        config: Configuration
        
    Returns:
        Tuple of (cost_df, constructs_df, ppu_definitions)
    """
    data_dir = Path(config.paths.DATA_DIR)
    
    # Load cost table
    cost_df = load_cost_table(data_dir / config.paths.COST_TABLE)
    
    # Load constructs
    constructs_df = load_ppu_constructs(data_dir / config.paths.PPU_CONSTRUCTS)
    
    # Build definitions
    ppu_definitions = build_ppu_definitions(
        constructs_df, 
        cost_df, 
        config.storage.STORAGE_DEFINITIONS
    )
    
    return cost_df, constructs_df, ppu_definitions


if __name__ == "__main__":
    # Test PPU loading
    cost_df, constructs_df, ppu_defs = load_all_ppu_data()
    
    print(f"Loaded {len(cost_df)} cost items")
    print(f"Loaded {len(constructs_df)} PPU constructs")
    print(f"Built {len(ppu_defs)} PPU definitions")
    
    print("\nPPU Definitions:")
    for name, ppu_def in ppu_defs.items():
        # cost_per_mwh is actually stored in CHF/kWh, convert for display
        cost_chf_per_mwh = ppu_def.cost_per_mwh * 1000
        print(f"  {name}: {ppu_def.category}, η={ppu_def.efficiency:.3f}, "
              f"cost={cost_chf_per_mwh:.2f} CHF/MWh")

