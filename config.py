"""
================================================================================
CONFIGURATION FILE - All Hyperparameters and Arbitrary Choices
================================================================================

This file centralizes all tunable parameters for the Swiss Energy Storage
Optimization project. Modify values here to experiment with different settings.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# RUN MODE CONFIGURATION
# =============================================================================

@dataclass
class RunModeConfig:
    """Configuration for optimization run duration."""
    
    # Preset modes
    MODE: str = "quick"  # Options: "quick" (1 hour), "overnight" (8 hours), "custom"
    
    # Quick mode settings (~1 hour)
    QUICK_POP_SIZE: int = 25
    QUICK_N_GENERATIONS: int = 5
    QUICK_PLATEAU_GENERATIONS: int = 3
    
    # Overnight mode settings (~8 hours)
    OVERNIGHT_POP_SIZE: int = 200
    OVERNIGHT_N_GENERATIONS: int = 150
    OVERNIGHT_PLATEAU_GENERATIONS: int = 10
    
    # Custom mode settings (modify as needed)
    CUSTOM_POP_SIZE: int = 100
    CUSTOM_N_GENERATIONS: int = 60
    CUSTOM_PLATEAU_GENERATIONS: int = 7
    
    def get_settings(self) -> Tuple[int, int, int]:
        """Return (pop_size, n_generations, plateau_generations) for current mode."""
        if self.MODE == "quick":
            return self.QUICK_POP_SIZE, self.QUICK_N_GENERATIONS, self.QUICK_PLATEAU_GENERATIONS
        elif self.MODE == "overnight":
            return self.OVERNIGHT_POP_SIZE, self.OVERNIGHT_N_GENERATIONS, self.OVERNIGHT_PLATEAU_GENERATIONS
        else:
            return self.CUSTOM_POP_SIZE, self.CUSTOM_N_GENERATIONS, self.CUSTOM_PLATEAU_GENERATIONS


# =============================================================================
# ENERGY SYSTEM PARAMETERS
# =============================================================================

@dataclass
class EnergySystemConfig:
    """Physical parameters of the Swiss energy system."""
    
    # Target annual energy demand (TWh/year)
    # Switzerland's 2050 target is ~113 TWh
    TARGET_ANNUAL_DEMAND_TWH: float = 113.0
    
    # Conversion to MWh (used internally)
    @property
    def TARGET_ANNUAL_DEMAND_MWH(self) -> float:
        return self.TARGET_ANNUAL_DEMAND_TWH * 1e6
    
    # Current annual demand (~55 TWh for reference)
    CURRENT_ANNUAL_DEMAND_TWH: float = 55.0

    # ==========================================================================
    # AVIATION FUEL REQUIREMENT
    # ==========================================================================
    # Aviation requires 23 TWh/year of biooil fuel (CO2-neutral synthetic fuel)
    # This is a HARD constraint: biooil must be discharged every hour
    # Reference: Züttel et al. (2024) - Swiss energy transition pathway
    
    AVIATION_FUEL_DEMAND_TWH_YEAR: float = 23.0
    
    # Hourly biooil discharge requirement (MWh/hour)
    # 23 TWh/year = 23,000,000 MWh/year ÷ 8760 hours ≈ 2625.57 MWh/hour
    @property
    def AVIATION_FUEL_HOURLY_MWH(self) -> float:
        return self.AVIATION_FUEL_DEMAND_TWH_YEAR * 1e6 / 8760
    
    # Minimum THERM units needed to extract aviation fuel from Fuel Tank
    # Aviation fuel comes from Fuel Tank (synthetic fuel) via THERM PPU
    # Each unit = 10 MW, hourly discharge = 2625.57 MWh = 2625.57 MW for 1 hour
    # Need ~263 units minimum (but THERM also serves electricity demand)
    @property
    def MIN_THERM_UNITS_FOR_AVIATION(self) -> int:
        return int(np.ceil(self.AVIATION_FUEL_HOURLY_MWH / 10.0))


# =============================================================================
# GENETIC ALGORITHM PARAMETERS
# =============================================================================

@dataclass
class GAConfig:
    """Genetic Algorithm hyperparameters."""
    
    # Population and generations (overridden by RunModeConfig)
    DEFAULT_POP_SIZE: int = 100
    DEFAULT_N_GENERATIONS: int = 50
    
    # Selection and reproduction
    TOURNAMENT_SIZE: int = 3
    CROSSOVER_RATE: float = 0.8
    MUTATION_RATE: float = 0.2
    MUTATION_SIGMA: float = 0.1  # Gaussian mutation std dev (relative)
    
    # Elitism
    ELITE_FRACTION: float = 0.1  # Top 10% survive unchanged
    
    # Stopping criteria
    PLATEAU_GENERATIONS: int = 5  # Stop if no improvement for N generations
    CONVERGENCE_THRESHOLD: float = 0.001  # Relative improvement threshold
    
    # Random seed (None for random, int for reproducible)
    RANDOM_SEED: Optional[int] = 42


# =============================================================================
# SCENARIO GENERATION PARAMETERS
# =============================================================================

@dataclass
class ScenarioConfig:
    """Parameters for scenario sampling from 2024 data."""
    
    # Number of days to sample per scenario
    DAYS_PER_SCENARIO: int = 30
    
    # Timesteps per hour (96 = 15-min resolution)
    TIMESTEPS_PER_HOUR: int = 1  # Hourly data
    
    # Total timesteps per scenario
    @property
    def TIMESTEPS_PER_SCENARIO(self) -> int:
        return self.DAYS_PER_SCENARIO * 24 * self.TIMESTEPS_PER_HOUR
    
    # Number of scenarios per portfolio evaluation
    SCENARIOS_PER_EVALUATION: int = 3
    
    # Data year
    DATA_YEAR: int = 2024


# =============================================================================
# STORAGE PARAMETERS
# =============================================================================

@dataclass
class StorageConfig:
    """Configuration for energy storage systems."""
    
    # Initial State of Charge (fraction of capacity)
    INITIAL_SOC_FRACTION: float = 0.60  # 60% full at start
    
    # Target State of Charge (for end-of-period)
    TARGET_SOC_FRACTION: float = 0.60
    
    # Safety margins (avoid fully empty/full)
    MIN_SOC_FRACTION: float = 0.05  # 5% minimum
    MAX_SOC_FRACTION: float = 0.95  # 95% maximum
    
    # ==========================================================================
    # FINAL STORAGE CONSTRAINT
    # ==========================================================================
    # System must end with storage levels within tolerance of initial levels
    # This prevents "cheating" by depleting storage over the simulation year
    FINAL_SOC_TOLERANCE: float = 0.05  # ±5% of initial SoC allowed
    FINAL_SOC_PENALTY_MULTIPLIER: float = 1e8  # Heavy penalty for violation
    
    # Storage definitions: (capacity_MWh, max_power_MW)
    # These are initial/default values, scaled by portfolio
    STORAGE_DEFINITIONS: Dict[str, Dict] = field(default_factory=lambda: {
        'Lake': {
            'capacity_MWh': 8_870_000,  # Swiss hydro reservoir ~8.87 TWh (max capacity)
            'max_power_MW': 2_000,      # Physical limit: 2 GW (regardless of PPU count)
            'physical_power_cap_MW': 2_000,  # Hard cap on Lake power
            'efficiency_charge': 0.88,  # PHS pump efficiency
            'efficiency_discharge': 0.88,  # Turbine efficiency
            'extracted_by': ['HYD_S'],
            'input_by': ['PHS'],
            # Water inflow from precipitation (from Swiss_Water_Hourly_2024.csv)
            # Conversion: 0.9 kWh per m³ (from hydrodaten.admin.ch: 36.5 TWh / 40.5 km³)
            'water_energy_kwh_per_m3': 0.9,  # kWh per cubic meter
            'receives_water_inflow': True,   # Receives hourly water inflow
        },
        'Fuel Tank': {
            'capacity_MWh': 20_000_000,
            'max_capacity_cap_MWh': 25_000_000,  # 25 TWh max
            'max_power_MW': 50_000,
            'efficiency_charge': 0.70,
            'efficiency_discharge': 0.50,
            'extracted_by': ['THERM'],
            'input_by': ['SYN_FT', 'SYN_CRACK'],
        },
        'H2 UG 200bar': {
            'capacity_MWh': 700_000,
            'max_capacity_cap_MWh': 50_000_000,  # 50 TWh max
            'max_power_MW': 5_000,
            'efficiency_charge': 0.60,
            'efficiency_discharge': 0.50,
            'extracted_by': ['H2P_G'],
            'input_by': ['H2_G', 'H2_GL'],
        },
        'Liquid H2': {
            'capacity_MWh': 100_000,
            'max_capacity_cap_MWh': 50_000_000,  # 50 TWh max
            'max_power_MW': 2_000,
            'efficiency_charge': 0.70,
            'efficiency_discharge': 0.50,
            'extracted_by': ['H2P_L'],
            'input_by': ['H2_L'],
        },
        'Solar salt': {
            'capacity_MWh': 50_000,
            'max_power_MW': 1_000,
            'efficiency_charge': 0.90,
            'efficiency_discharge': 0.90,
            'extracted_by': ['SOL_STEAM', 'SOL_SALT'],
            'input_by': ['SOL_SALT_STORE'],
        },
        # NOTE: Biooil storage removed - Palm oil is the only imported bio-fuel
        # Aviation fuel requirement (23 TWh/year) now comes from Fuel Tank (synthetic fuel)
        'Palm oil': {
            'capacity_MWh': 500_000,
            'max_power_MW': 5_000,
            'efficiency_charge': 1.0,  # Purchase at market price
            'efficiency_discharge': 0.30,  # ICE efficiency
            'extracted_by': ['PALM_ICE'],
            'input_by': ['PALM_IMPORT'],
            # Price is DYNAMIC: loaded from rea_holdings_share_prices.csv (USD/metric ton)
            # Converted to CHF/MWh using USD/CHF exchange rate and 44 MJ/kg energy density
            # 1 metric ton = 1000 kg × 44 MJ/kg = 44,000 MJ = 12.22 MWh
            'energy_density_mwh_per_ton': 12.22,  # 44 MJ/kg average
        },
        'Biogas': {
            'capacity_MWh': 200_000,
            'max_capacity_cap_MWh': 25_000_000,  # 25 TWh max
            'max_power_MW': 2_000,
            'efficiency_charge': 0.98,
            'efficiency_discharge': 0.50,
            'extracted_by': ['IMP_BIOG'],
            'input_by': ['CH4_BIO'],
        },
        'CH4 200bar': {
            'capacity_MWh': 700_000,
            'max_capacity_cap_MWh': 50_000_000,  # 50 TWh max
            'max_power_MW': 5_000,
            'efficiency_charge': 0.78,
            'efficiency_discharge': 0.50,
            'extracted_by': ['THERM_CH4'],
            'input_by': ['SYN_METH'],
        },
        'Ammonia': {
            'capacity_MWh': 300_000,
            'max_capacity_cap_MWh': 50_000_000,  # 50 TWh max
            'max_power_MW': 3_000,
            'efficiency_charge': 0.78,
            'efficiency_discharge': 0.375,  # 0.75 * 0.50
            'extracted_by': ['NH3_P'],
            'input_by': ['NH3_FULL'],
        },
    })


# =============================================================================
# PPU (Power Production Unit) PARAMETERS
# =============================================================================

@dataclass
class PPUConfig:
    """Configuration for Power Production Units."""
    
    # Incidence-based PPUs (depend on weather)
    INCIDENCE_PPUS: List[str] = field(default_factory=lambda: [
        'PV',       # Solar photovoltaic
        'WD_ON',    # Wind onshore
        'WD_OFF',   # Wind offshore
        'HYD_R',    # Run-of-river hydro
        # Note: BIO_WOOD moved to DISPATCHABLE_GENERATORS (direct biomass input, not incidence-based)
    ])
    
    # Flexible/dispatchable PPUs (depend on storage)
    FLEX_PPUS: List[str] = field(default_factory=lambda: [
        'HYD_S',      # Hydro storage
        'THERM',      # Thermal (fuel tank) - also supplies aviation fuel
        'H2P_G',      # H2 gas power
        'H2P_L',      # H2 liquid power
        'SOL_SALT',   # Solar salt thermal
        'SOL_STEAM',  # Solar steam
        'PALM_ICE',   # Palm oil ICE (only imported bio-fuel)
        'IMP_BIOG',   # Imported biogas
        'THERM_CH4',  # Methane thermal
        'NH3_P',      # Ammonia power
        # NOTE: BIO_OIL_ICE removed - redundant with PALM_ICE
    ])
    
    # Storage input PPUs
    STORAGE_PPUS: List[str] = field(default_factory=lambda: [
        'PHS',         # Pumped hydro storage
        'H2_G',        # H2 production (gaseous)
        'H2_GL',       # H2 production (gas from liquid)
        'H2_L',        # H2 production (liquid)
        'SYN_FT',      # Fischer-Tropsch synthesis
        'SYN_METH',    # Methane synthesis
        'NH3_FULL',    # Ammonia synthesis
        'SYN_CRACK',   # Cracking synthesis
        'CH4_BIO',     # Biogas to methane
        'SOL_SALT_STORE',  # Solar salt storage
    ])
    
    # Portfolio bounds: (min_units, max_units) for each PPU type
    # A "unit" represents 10 MW capacity - gives granularity for ~7 GW average demand
    # 20 GW = 2,000 units (2,000 × 10 MW = 20,000 MW = 20 GW)
    PORTFOLIO_BOUNDS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        # Incidence PPUs (hard cap: 2000)
        'PV': (0, 2000),          # Up to 20 GW (2,000 × 10 MW)
        'WD_ON': (0, 2000),       # Up to 20 GW
        'WD_OFF': (0, 2000),      # Up to 20 GW
        'HYD_R': (0, 300),        # Up to 3 GW (300 × 10 MW) - physical hard cap (limited sites)
        'BIO_WOOD': (0, 2000),    # Up to 20 GW
        
        # Flex PPUs (extraction from storage) - hard cap: 2000
        'HYD_S': (0, 300),        # Up to 3 GW (300 × 10 MW) - physical hard cap (Lake 2 GW)
        'THERM': (0, 2000),       # Up to 20 GW
        'H2P_G': (0, 2000),       # Up to 20 GW
        'H2P_L': (0, 2000),       # Up to 20 GW
        'SOL_SALT': (0, 2000),    # Up to 20 GW
        'SOL_STEAM': (0, 2000),   # Up to 20 GW
        'PALM_ICE': (0, 2000),    # Up to 20 GW (only imported bio-fuel)
        'IMP_BIOG': (0, 2000),    # Up to 20 GW
        'THERM_CH4': (0, 2000),   # Up to 20 GW
        'NH3_P': (0, 2000),       # Up to 20 GW
        
        # Storage PPUs (input to storage) - hard cap: 2000
        'PHS': (0, 2000),         # Up to 20 GW (Lake will cap at 2 GW physical)
        'H2_G': (0, 2000),        # Up to 20 GW
        'H2_GL': (0, 2000),       # Up to 20 GW
        'H2_L': (0, 2000),        # Up to 20 GW
        'SYN_FT': (0, 2000),      # Up to 20 GW
        'SYN_METH': (0, 2000),    # Up to 20 GW
        'NH3_FULL': (0, 2000),    # Up to 20 GW
        'SYN_CRACK': (0, 2000),   # Up to 20 GW
        'CH4_BIO': (0, 2000),     # Up to 20 GW
        'SOL_SALT_STORE': (0, 2000),  # Up to 20 GW
    })
    
    # Capacity per unit (MW per portfolio unit)
    # Each PPU unit = 10 MW power flow capacity
    MW_PER_UNIT: float = 10.0  # 10 MW per unit (100 units = 1 GW)
    
    # ==========================================================================
    # PROGRESSIVE COST CAPS
    # ==========================================================================
    # After 'soft_cap' units, each additional unit increases cost by 'factor'
    # Formula: cost_multiplier = 1 + factor * max(0, units - soft_cap)
    # Example: factor=0.1, soft_cap=100, units=150 → multiplier = 1 + 0.1*50 = 6x
    #
    # INCIDENCE PPUs: soft_cap = 100% of hard cap → NO price escalation
    # NON-INCIDENCE PPUs: soft_cap = 50% of hard cap → price escalation after 50%
    #
    # Set factor=0 to disable progressive cost for a PPU type
    
    PROGRESSIVE_COST_CAPS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # =======================================================================
        # INCIDENCE PPUs: soft_cap = hard_cap (NO progressive cost penalty)
        # =======================================================================
        'PV': {'soft_cap': 2000, 'factor': 0.0},        # No escalation (incidence)
        'WD_ON': {'soft_cap': 2000, 'factor': 0.0},     # No escalation (incidence)
        'WD_OFF': {'soft_cap': 2000, 'factor': 0.0},    # No escalation (incidence)
        'HYD_R': {'soft_cap': 300, 'factor': 0.0},      # No escalation (incidence, physical cap)
        'BIO_WOOD': {'soft_cap': 2000, 'factor': 0.0},  # No escalation (incidence-like)
        
        # =======================================================================
        # FLEX PPUs: soft_cap = 50% of hard_cap (1000 units = 10 GW)
        # =======================================================================
        'HYD_S': {'soft_cap': 150, 'factor': 0.001},     # Physical limit (Lake 2 GW cap)
        'THERM': {'soft_cap': 1000, 'factor': 0.0002},   # After 10 GW, escalation
        'H2P_G': {'soft_cap': 1000, 'factor': 0.0003},   # After 10 GW, H2 infrastructure
        'H2P_L': {'soft_cap': 1000, 'factor': 0.0004},   # After 10 GW, liquid H2 expensive
        'SOL_SALT': {'soft_cap': 1000, 'factor': 0.0005},  # After 10 GW, CSP limited
        'SOL_STEAM': {'soft_cap': 1000, 'factor': 0.0005}, # After 10 GW
        'PALM_ICE': {'soft_cap': 1000, 'factor': 0.0004},  # After 10 GW, import dependency
        'IMP_BIOG': {'soft_cap': 1000, 'factor': 0.0005},  # After 10 GW, import capacity
        'THERM_CH4': {'soft_cap': 1000, 'factor': 0.0002}, # After 10 GW
        'NH3_P': {'soft_cap': 1000, 'factor': 0.0004},     # After 10 GW, ammonia infra
        
        # =======================================================================
        # STORAGE PPUs: soft_cap = 50% of hard_cap (1000 units = 10 GW)
        # =======================================================================
        'PHS': {'soft_cap': 1000, 'factor': 0.001},       # After 10 GW (Lake 2 GW physical)
        'H2_G': {'soft_cap': 1000, 'factor': 0.0003},     # After 10 GW, underground storage
        'H2_GL': {'soft_cap': 1000, 'factor': 0.0004},    # After 10 GW
        'H2_L': {'soft_cap': 1000, 'factor': 0.0005},     # After 10 GW, cryogenic expensive
        'SYN_FT': {'soft_cap': 1000, 'factor': 0.0004},   # After 10 GW
        'SYN_METH': {'soft_cap': 1000, 'factor': 0.0004}, # After 10 GW
        'NH3_FULL': {'soft_cap': 1000, 'factor': 0.0004}, # After 10 GW
        'SYN_CRACK': {'soft_cap': 1000, 'factor': 0.0005},# After 10 GW
        'CH4_BIO': {'soft_cap': 1000, 'factor': 0.0003},  # After 10 GW
        'SOL_SALT_STORE': {'soft_cap': 1000, 'factor': 0.0005}, # After 10 GW
    })


# =============================================================================
# FITNESS FUNCTION PARAMETERS
# =============================================================================

@dataclass
class FitnessConfig:
    """Parameters for the fitness/objective function."""
    
    # Energy sovereignty penalty
    # Applied when portfolio doesn't meet demand
    SOVEREIGNTY_PENALTY_MULTIPLIER: float = 1e9  # Heavy penalty
    
    # Cost aggregation method
    # Options: "harmonic_mean", "arithmetic_mean", "worst_case"
    COST_AGGREGATION: str = "harmonic_mean"
    
    # CVaR (Conditional Value at Risk) confidence level
    CVAR_ALPHA: float = 0.95  # 95th percentile
    
    # Weight between mean cost and CVaR
    CVAR_WEIGHT: float = 0.3  # 30% weight on tail risk
    
    # Cumulative energy balance check
    # If True, verify that total renewable production >= total demand
    # (accounting for storage round-trip losses)
    # WARNING: This is computationally expensive! Disable for faster optimization.
    CHECK_CUMULATIVE_BALANCE: bool = False  # Disabled by default for performance
    
    # Round-trip storage efficiency for cumulative balance check
    # Production must exceed demand / this value to account for losses
    STORAGE_ROUND_TRIP_EFFICIENCY: float = 0.75


# =============================================================================
# DISPATCH PARAMETERS
# =============================================================================

@dataclass
class DispatchConfig:
    """Parameters for the dispatch simulation."""
    
    # Priority order for storage discharge (when demand > supply)
    DISCHARGE_PRIORITY: List[str] = field(default_factory=lambda: [
        'Lake',        # Cheapest, most efficient
        'Solar salt',  # High efficiency
        'Biogas',      # Medium efficiency
        'H2 UG 200bar', # Lower efficiency but available
        'CH4 200bar',
        'Liquid H2',
        'Fuel Tank',   # Flexible but lower efficiency (also supplies aviation fuel)
        'Ammonia',
        'Palm oil',    # Only imported bio-fuel (Biooil removed)
    ])
    
    # Priority order for storage charging (when supply > demand)
    CHARGE_PRIORITY: List[str] = field(default_factory=lambda: [
        'Lake',        # PHS - most efficient
        'Solar salt',  # Direct storage
        'H2 UG 200bar', # Electrolysis
        'CH4 200bar',  # Methanation
        'Biogas',
        'Liquid H2',
        'Fuel Tank',   # FT synthesis (also supplies aviation fuel)
        'Ammonia',
        'Palm oil',    # Ghost PPU import (only bio-fuel)
    ])
    
    # Spot price thresholds for storage decisions (CHF/MWh)
    SPOT_PRICE_LOW_THRESHOLD: float = 30.0   # Below this, charge storage
    SPOT_PRICE_HIGH_THRESHOLD: float = 80.0  # Above this, discharge storage
    
    # Future price expectation window (hours)
    PRICE_EXPECTATION_WINDOW: int = 24
    
    # EMA smoothing factor for price expectations
    PRICE_EMA_ALPHA: float = 0.1


# =============================================================================
# DATA PATHS
# =============================================================================

@dataclass
class DataPathConfig:
    """File paths for data loading."""
    
    DATA_DIR: str = 'data'
    
    # Core data files
    COST_TABLE: str = 'cost_table_tidy.csv'
    PPU_CONSTRUCTS: str = 'ppu_constructs_components.csv'
    
    # Time series data
    SOLAR_INCIDENCE: str = 'solar_incidence_hourly_2024.csv'
    WIND_INCIDENCE: str = 'wind_incidence_hourly_2024.csv'
    SPOT_PRICES: str = 'spot_price_hourly.csv'
    DEMAND: str = 'monthly_hourly_load_values_2024.csv'
    
    # Hydro data
    RESERVOIR_LEVELS: str = 'water_monthly_reservoir_2024.csv'
    ROR_MONTHLY: str = 'water_monthly_ror_2024.csv'
    
    # Rankings
    SOLAR_RANKING: str = 'ranking_incidence/solar_incidence_ranking.csv'
    WIND_RANKING: str = 'ranking_incidence/wind_incidence_ranking.csv'


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Master configuration object containing all settings."""
    
    run_mode: RunModeConfig = field(default_factory=RunModeConfig)
    energy_system: EnergySystemConfig = field(default_factory=EnergySystemConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ppu: PPUConfig = field(default_factory=PPUConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    paths: DataPathConfig = field(default_factory=DataPathConfig)
    
    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        pop, gen, plateau = self.run_mode.get_settings()
        return f"""
================================================================================
CONFIGURATION SUMMARY
================================================================================
Run Mode: {self.run_mode.MODE}
  - Population Size: {pop}
  - Generations: {gen}
  - Plateau Detection: {plateau} generations

Energy Target: {self.energy_system.TARGET_ANNUAL_DEMAND_TWH} TWh/year

Scenario Settings:
  - Days per scenario: {self.scenario.DAYS_PER_SCENARIO}
  - Scenarios per evaluation: {self.scenario.SCENARIOS_PER_EVALUATION}
  - Total timesteps: {self.scenario.TIMESTEPS_PER_SCENARIO}

GA Parameters:
  - Crossover Rate: {self.ga.CROSSOVER_RATE}
  - Mutation Rate: {self.ga.MUTATION_RATE}
  - Elite Fraction: {self.ga.ELITE_FRACTION}

Storage Initial SoC: {self.storage.INITIAL_SOC_FRACTION * 100}%
================================================================================
"""


# Default configuration instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    # Print configuration summary when run directly
    print(DEFAULT_CONFIG.summary())

