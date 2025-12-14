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
    QUICK_POP_SIZE: int = 10
    QUICK_N_GENERATIONS: int = 3
    QUICK_PLATEAU_GENERATIONS: int = 2
    
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
    
    # Storage definitions: (capacity_MWh, max_power_MW)
    # These are initial/default values, scaled by portfolio
    STORAGE_DEFINITIONS: Dict[str, Dict] = field(default_factory=lambda: {
        'Lake': {
            'capacity_MWh': 8_870_000,  # Swiss hydro reservoir ~8.87 TWh
            'max_power_MW': 10_000,
            'efficiency_charge': 0.88,  # PHS pump efficiency
            'efficiency_discharge': 0.88,  # Turbine efficiency
            'extracted_by': ['HYD_S'],
            'input_by': ['PHS'],
        },
        'Fuel Tank': {
            'capacity_MWh': 20_000_000,
            'max_power_MW': 50_000,
            'efficiency_charge': 0.70,
            'efficiency_discharge': 0.50,
            'extracted_by': ['THERM'],
            'input_by': ['SYN_FT', 'SYN_CRACK'],
        },
        'H2 UG 200bar': {
            'capacity_MWh': 700_000,
            'max_power_MW': 5_000,
            'efficiency_charge': 0.60,
            'efficiency_discharge': 0.50,
            'extracted_by': ['H2P_G'],
            'input_by': ['H2_G', 'H2_GL'],
        },
        'Liquid H2': {
            'capacity_MWh': 100_000,
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
        'Biooil': {
            'capacity_MWh': 500_000,
            'max_power_MW': 5_000,
            'efficiency_charge': 1.0,  # Purchase at market price
            'efficiency_discharge': 0.30,  # ICE efficiency
            'extracted_by': ['BIO_OIL_ICE'],
            'input_by': ['BIOOIL_IMPORT'],  # Buy at 67 CHF/MWh
            'import_price_chf_per_mwh': 67.0,
        },
        'Palm oil': {
            'capacity_MWh': 500_000,
            'max_power_MW': 5_000,
            'efficiency_charge': 1.0,
            'efficiency_discharge': 0.30,
            'extracted_by': ['PALM_ICE'],
            'input_by': ['PALM_IMPORT'],  # Buy at 87 CHF/MWh
            'import_price_chf_per_mwh': 87.0,
        },
        'Biogas': {
            'capacity_MWh': 200_000,
            'max_power_MW': 2_000,
            'efficiency_charge': 0.98,
            'efficiency_discharge': 0.50,
            'extracted_by': ['IMP_BIOG'],
            'input_by': ['CH4_BIO'],
        },
        'CH4 200bar': {
            'capacity_MWh': 700_000,
            'max_power_MW': 5_000,
            'efficiency_charge': 0.78,
            'efficiency_discharge': 0.50,
            'extracted_by': ['THERM_CH4'],
            'input_by': ['SYN_METH'],
        },
        'Ammonia': {
            'capacity_MWh': 300_000,
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
        'BIO_WOOD', # Biomass (wood)
    ])
    
    # Flexible/dispatchable PPUs (depend on storage)
    FLEX_PPUS: List[str] = field(default_factory=lambda: [
        'HYD_S',      # Hydro storage
        'THERM',      # Thermal (fuel tank)
        'H2P_G',      # H2 gas power
        'H2P_L',      # H2 liquid power
        'SOL_SALT',   # Solar salt thermal
        'SOL_STEAM',  # Solar steam
        'BIO_OIL_ICE', # Biooil ICE
        'PALM_ICE',   # Palm oil ICE
        'IMP_BIOG',   # Imported biogas
        'THERM_CH4',  # Methane thermal
        'NH3_P',      # Ammonia power
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
    # A "unit" represents a standardized capacity block
    PORTFOLIO_BOUNDS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        # Incidence PPUs
        'PV': (0, 500),          # Up to 500 GW
        'WD_ON': (0, 100),       # Up to 100 GW
        'WD_OFF': (0, 50),       # Up to 50 GW
        'HYD_R': (0, 20),        # Limited by geography
        'BIO_WOOD': (0, 50),     # Limited by biomass availability
        
        # Flex PPUs
        'HYD_S': (0, 50),        # Limited by lake capacity
        'THERM': (0, 100),
        'H2P_G': (0, 100),
        'H2P_L': (0, 50),
        'SOL_SALT': (0, 20),
        'SOL_STEAM': (0, 20),
        'BIO_OIL_ICE': (0, 50),
        'PALM_ICE': (0, 50),
        'IMP_BIOG': (0, 50),
        'THERM_CH4': (0, 100),
        'NH3_P': (0, 50),
        
        # Storage PPUs
        'PHS': (0, 50),
        'H2_G': (0, 100),
        'H2_GL': (0, 50),
        'H2_L': (0, 50),
        'SYN_FT': (0, 50),
        'SYN_METH': (0, 50),
        'NH3_FULL': (0, 50),
        'SYN_CRACK': (0, 30),
        'CH4_BIO': (0, 50),
        'SOL_SALT_STORE': (0, 20),
    })
    
    # Capacity per unit (MW per portfolio unit)
    MW_PER_UNIT: float = 1000.0  # 1 GW per unit


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
        'Fuel Tank',   # Flexible but lower efficiency
        'Ammonia',
        'Biooil',      # Low efficiency but unlimited import
        'Palm oil',
    ])
    
    # Priority order for storage charging (when supply > demand)
    CHARGE_PRIORITY: List[str] = field(default_factory=lambda: [
        'Lake',        # PHS - most efficient
        'Solar salt',  # Direct storage
        'H2 UG 200bar', # Electrolysis
        'CH4 200bar',  # Methanation
        'Biogas',
        'Liquid H2',
        'Fuel Tank',   # FT synthesis
        'Ammonia',
        'Biooil',      # Only when very cheap
        'Palm oil',
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

