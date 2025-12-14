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
    cost_per_mwh: float  # CHF per MWh output
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
    cost_per_mwh: float = 0.0
    
    # Storage relationships
    can_extract_from: List[str] = field(default_factory=list)
    can_input_to: List[str] = field(default_factory=list)


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
    - Auxiliary energy requirements (w)
    - Cumulative efficiency losses
    
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
        
        # Auxiliary energy cost (scaled by position in chain)
        w = comp_data['w']
        if w > 0 and cumulative_efficiency > 0:
            aux_cost = w / cumulative_efficiency
            total_cost += aux_cost
    
    return PPUCostData(
        efficiency=cumulative_efficiency,
        cost_per_mwh=total_cost,  # CHF per MWh
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
            
            # Cost escalation: 10% increase per existing unit
            escalation_factor = 1.0 + 0.10 * i
            cost_per_mwh = ppu_def.cost_per_mwh * escalation_factor
            
            rows.append({
                'PPU_ID': ppu_id,
                'PPU_Name': ppu_name,
                'PPU_Category': ppu_def.category,
                'PPU_Extract': ppu_def.extract_type,
                'can_extract_from': ppu_def.can_extract_from.copy(),
                'can_input_to': ppu_def.can_input_to.copy(),
                'Chain_Efficiency': ppu_def.efficiency,
                'Cost_CHF_per_kWh': cost_per_mwh / 1000,  # Convert to kWh
                'Cost_CHF_per_MWh': cost_per_mwh,
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
        print(f"  {name}: {ppu_def.category}, η={ppu_def.efficiency:.3f}, "
              f"cost={ppu_def.cost_per_mwh:.2f} CHF/MWh")

