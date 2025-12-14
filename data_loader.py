"""
================================================================================
DATA LOADER - Immutable Data Loading with Caching
================================================================================

This module handles all data loading operations for the Swiss Energy Storage
Optimization project. Key features:
- Caching to avoid repeated disk reads
- Immutable data structures (always returns copies)
- Consistent preprocessing and validation

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from config import Config, DEFAULT_CONFIG


# =============================================================================
# DATA CACHE CONTAINER
# =============================================================================

@dataclass
class CachedData:
    """Container for all cached data. Ensures immutability via deep copies."""
    
    # Time series data (hourly for 2024)
    solar_incidence: np.ndarray  # Shape: (8760, n_locations)
    wind_incidence: np.ndarray   # Shape: (8760, n_locations)
    spot_prices: np.ndarray      # Shape: (8760,)
    demand: np.ndarray           # Shape: (8760,)
    
    # Location rankings (for building in optimal locations)
    solar_ranking: np.ndarray    # Indices sorted by average incidence
    wind_ranking: np.ndarray     # Indices sorted by average incidence
    
    # Hydro data
    reservoir_levels: np.ndarray  # Monthly reservoir levels
    ror_production: np.ndarray    # Run-of-river monthly production
    
    # Cost data
    cost_table: pd.DataFrame
    ppu_components: pd.DataFrame
    
    # Metadata
    n_hours: int = 8760
    n_solar_locations: int = 0
    n_wind_locations: int = 0
    
    def get_solar_incidence(self) -> np.ndarray:
        """Return a copy of solar incidence data."""
        return self.solar_incidence.copy()
    
    def get_wind_incidence(self) -> np.ndarray:
        """Return a copy of wind incidence data."""
        return self.wind_incidence.copy()
    
    def get_spot_prices(self) -> np.ndarray:
        """Return a copy of spot prices."""
        return self.spot_prices.copy()
    
    def get_demand(self) -> np.ndarray:
        """Return a copy of demand data."""
        return self.demand.copy()
    
    def get_cost_table(self) -> pd.DataFrame:
        """Return a copy of cost table."""
        return self.cost_table.copy()
    
    def get_ppu_components(self) -> pd.DataFrame:
        """Return a copy of PPU components."""
        return self.ppu_components.copy()


# =============================================================================
# GLOBAL CACHE
# =============================================================================

_DATA_CACHE: Optional[CachedData] = None


def clear_cache() -> None:
    """Clear the data cache."""
    global _DATA_CACHE
    _DATA_CACHE = None


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def _load_solar_incidence(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load solar incidence data and ranking.
    
    The CSV has a special format:
    - Row 0: latitude values
    - Row 1: longitude values
    - Row 2: empty/time header
    - Row 3+: timestamp, data values
    
    Returns:
        Tuple of (incidence_array, ranking_indices)
    """
    solar_path = data_dir / 'solar_incidence_hourly_2024.csv'
    
    # Read without header to handle special format
    df = pd.read_csv(solar_path, header=None)
    
    # Extract lat/lon from first two rows
    latitudes = df.iloc[0, 1:].astype(float).values
    longitudes = df.iloc[1, 1:].astype(float).values
    
    # Data starts from row 3 (after lat, lon, time header rows)
    # First column is timestamp, rest is data
    incidence = df.iloc[3:, 1:].astype(np.float32).values
    
    # Compute ranking: sort by mean incidence descending
    mean_incidence = incidence.mean(axis=0)
    ranking = np.argsort(mean_incidence)[::-1].astype(np.int32)
    
    # Try to load existing ranking file for consistency
    ranking_path = data_dir / 'ranking_incidence' / 'solar_incidence_ranking.csv'
    if ranking_path.exists():
        try:
            ranking_df = pd.read_csv(ranking_path)
            if 'rank' in ranking_df.columns:
                # Use the pre-computed ranking
                ranking = ranking_df['rank'].values.astype(np.int32) - 1  # Convert 1-based to 0-based
        except Exception:
            pass  # Fall back to computed ranking
    
    return incidence, ranking


def _load_wind_incidence(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wind incidence data and ranking.
    
    The CSV has multi-index header with (lat, lon) pairs.
    
    Returns:
        Tuple of (incidence_array, ranking_indices)
    """
    wind_path = data_dir / 'wind_incidence_hourly_2024.csv'
    
    try:
        # Try reading with multi-index header
        df = pd.read_csv(wind_path, header=[0, 1], index_col=0, parse_dates=True)
        incidence = df.values.astype(np.float32)
    except Exception:
        # Fallback: read as simple CSV
        df = pd.read_csv(wind_path)
        incidence_cols = [c for c in df.columns if c not in ['timestamp', 'hour', 'Unnamed: 0', 'time']]
        incidence = df[incidence_cols].values.astype(np.float32)
    
    # Compute ranking: sort by mean wind speed descending
    mean_incidence = incidence.mean(axis=0)
    ranking = np.argsort(mean_incidence)[::-1].astype(np.int32)
    
    # Try to load existing ranking
    ranking_path = data_dir / 'ranking_incidence' / 'wind_incidence_ranking.csv'
    if ranking_path.exists():
        try:
            ranking_df = pd.read_csv(ranking_path)
            if 'rank' in ranking_df.columns:
                ranking = ranking_df['rank'].values.astype(np.int32) - 1
        except Exception:
            pass
    
    return incidence, ranking


def _load_spot_prices(data_dir: Path) -> np.ndarray:
    """Load spot electricity prices."""
    spot_path = data_dir / 'spot_price_hourly.csv'
    df = pd.read_csv(spot_path)
    
    # Find price column
    price_col = None
    for col in df.columns:
        if 'price' in col.lower() or 'chf' in col.lower() or 'eur' in col.lower():
            price_col = col
            break
    
    if price_col is None:
        # Use second column (first is usually timestamp)
        price_col = df.columns[1]
    
    prices = df[price_col].values.astype(np.float32)
    
    # Handle NaN values
    if np.isnan(prices).any():
        warnings.warn(f"Found {np.isnan(prices).sum()} NaN values in spot prices, filling with mean")
        prices = np.nan_to_num(prices, nan=np.nanmean(prices))
    
    return prices


def _load_demand(data_dir: Path) -> np.ndarray:
    """Load electricity demand data for Switzerland."""
    demand_path = data_dir / 'monthly_hourly_load_values_2024.csv'
    df = pd.read_csv(demand_path, sep='\t' if '\t' in open(demand_path).read(1000) else ',')
    
    # Filter for Switzerland if CountryCode column exists
    if 'CountryCode' in df.columns:
        df_ch = df[df['CountryCode'] == 'CH'].copy()
        if len(df_ch) == 0:
            # Fallback to all data if CH not found
            warnings.warn("No CH data found, using all countries")
            df_ch = df
    else:
        df_ch = df
    
    # Find the value column
    demand_col = None
    for col in ['Value', 'Value_ScaleTo100', 'load', 'demand', 'MW']:
        if col in df_ch.columns:
            demand_col = col
            break
    
    if demand_col is None:
        # Find any numeric column
        for col in df_ch.columns:
            if df_ch[col].dtype in [np.float64, np.int64, float, int]:
                demand_col = col
                break
    
    if demand_col is None:
        raise ValueError(f"Could not find demand column in {demand_path}")
    
    demand = df_ch[demand_col].values.astype(np.float32)
    
    # Handle NaN
    if np.isnan(demand).any():
        warnings.warn(f"Found {np.isnan(demand).sum()} NaN values in demand, filling with mean")
        demand = np.nan_to_num(demand, nan=np.nanmean(demand))
    
    return demand


def _load_hydro_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load hydro reservoir and run-of-river data."""
    # Reservoir levels
    reservoir_path = data_dir / 'water_monthly_reservoir_2024.csv'
    if reservoir_path.exists():
        res_df = pd.read_csv(reservoir_path)
        reservoir = res_df.iloc[:, 1].values.astype(np.float32)
    else:
        reservoir = np.zeros(12)
    
    # Run-of-river
    ror_path = data_dir / 'water_monthly_ror_2024.csv'
    if ror_path.exists():
        ror_df = pd.read_csv(ror_path)
        ror = ror_df.iloc[:, 1].values.astype(np.float32)
    else:
        ror = np.zeros(12)
    
    return reservoir, ror


def _load_cost_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cost table and PPU components."""
    cost_path = data_dir / 'cost_table_tidy.csv'
    cost_df = pd.read_csv(cost_path)
    
    ppu_path = data_dir / 'ppu_constructs_components.csv'
    ppu_df = pd.read_csv(ppu_path)
    
    return cost_df, ppu_df


# =============================================================================
# MAIN LOADING FUNCTION
# =============================================================================

def load_all_data(config: Config = DEFAULT_CONFIG, force_reload: bool = False) -> CachedData:
    """
    Load all data files and cache them.
    
    Args:
        config: Configuration object with file paths
        force_reload: If True, reload even if cached
        
    Returns:
        CachedData object with all loaded data
    """
    global _DATA_CACHE
    
    if _DATA_CACHE is not None and not force_reload:
        return _DATA_CACHE
    
    data_dir = Path(config.paths.DATA_DIR)
    
    print("Loading data files...")
    
    # Load all data
    solar_incidence, solar_ranking = _load_solar_incidence(data_dir)
    print(f"  - Solar incidence: {solar_incidence.shape}")
    
    wind_incidence, wind_ranking = _load_wind_incidence(data_dir)
    print(f"  - Wind incidence: {wind_incidence.shape}")
    
    spot_prices = _load_spot_prices(data_dir)
    print(f"  - Spot prices: {spot_prices.shape}")
    
    demand = _load_demand(data_dir)
    print(f"  - Demand: {demand.shape}")
    
    reservoir, ror = _load_hydro_data(data_dir)
    print(f"  - Hydro data loaded")
    
    cost_table, ppu_components = _load_cost_data(data_dir)
    print(f"  - Cost data loaded")
    
    # Ensure consistent lengths across all time series
    # Use the minimum length (typically demand/spot which is 8784 for leap year 2024)
    n_hours = min(
        len(demand),
        len(spot_prices),
        len(solar_incidence),
        len(wind_incidence)
    )
    
    # Trim all arrays to consistent length
    solar_incidence = solar_incidence[:n_hours]
    wind_incidence = wind_incidence[:n_hours]
    spot_prices = spot_prices[:n_hours]
    demand = demand[:n_hours]
    
    print(f"  - Aligned all data to {n_hours} timesteps ({n_hours/24:.0f} days)")
    
    # Create cached data object
    _DATA_CACHE = CachedData(
        solar_incidence=solar_incidence,
        wind_incidence=wind_incidence,
        spot_prices=spot_prices,
        demand=demand,
        solar_ranking=solar_ranking,
        wind_ranking=wind_ranking,
        reservoir_levels=reservoir,
        ror_production=ror,
        cost_table=cost_table,
        ppu_components=ppu_components,
        n_hours=n_hours,
        n_solar_locations=solar_incidence.shape[1] if solar_incidence.ndim > 1 else 1,
        n_wind_locations=wind_incidence.shape[1] if wind_incidence.ndim > 1 else 1,
    )
    
    print("Data loading complete!")
    return _DATA_CACHE


# =============================================================================
# SCENARIO SAMPLING
# =============================================================================

def sample_scenario_indices(
    n_hours: int = 8760,
    days_per_scenario: int = 30,
    n_scenarios: int = 3,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample random day indices for scenario generation.
    
    Args:
        n_hours: Total hours in dataset (typically 8760)
        days_per_scenario: Number of days to sample per scenario
        n_scenarios: Number of scenarios to generate
        rng: Random number generator (for reproducibility)
        
    Returns:
        Array of shape (n_scenarios, days_per_scenario * 24) with hour indices
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_days = n_hours // 24
    
    scenarios = []
    for _ in range(n_scenarios):
        # Sample random days
        sampled_days = rng.choice(n_days, size=days_per_scenario, replace=False)
        sampled_days = np.sort(sampled_days)  # Sort for temporal consistency
        
        # Convert to hour indices
        hour_indices = []
        for day in sampled_days:
            start_hour = day * 24
            hour_indices.extend(range(start_hour, start_hour + 24))
        
        scenarios.append(hour_indices)
    
    return np.array(scenarios)


def get_scenario_data(
    data: CachedData,
    hour_indices: np.ndarray,
    n_solar_units: int = 0,
    n_wind_units: int = 0
) -> Dict[str, np.ndarray]:
    """
    Extract scenario-specific data slices.
    
    Uses ranking to select the best locations first.
    
    Args:
        data: CachedData object
        hour_indices: Array of hour indices to extract
        n_solar_units: Number of solar units (locations to use)
        n_wind_units: Number of wind units (locations to use)
        
    Returns:
        Dictionary with scenario data arrays (COPIES, safe to mutate)
    """
    # Get copies of relevant data slices
    spot = data.spot_prices[hour_indices].copy()
    demand = data.demand[hour_indices].copy()
    
    # Solar: use best locations based on ranking
    if n_solar_units > 0 and data.solar_incidence.ndim > 1:
        best_solar_locs = data.solar_ranking[:n_solar_units]
        solar = data.solar_incidence[hour_indices][:, best_solar_locs].copy()
    else:
        solar = data.solar_incidence[hour_indices].copy()
    
    # Wind: use best locations based on ranking
    if n_wind_units > 0 and data.wind_incidence.ndim > 1:
        best_wind_locs = data.wind_ranking[:n_wind_units]
        wind = data.wind_incidence[hour_indices][:, best_wind_locs].copy()
    else:
        wind = data.wind_incidence[hour_indices].copy()
    
    return {
        'spot_prices': spot,
        'demand': demand,
        'solar_incidence': solar,
        'wind_incidence': wind,
        'n_hours': len(hour_indices),
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(data: CachedData) -> bool:
    """
    Validate loaded data for consistency.
    
    Returns:
        True if all checks pass, raises ValueError otherwise
    """
    errors = []
    
    # Check array lengths
    expected_hours = data.n_hours
    
    if len(data.spot_prices) != expected_hours:
        errors.append(f"Spot prices length mismatch: {len(data.spot_prices)} vs {expected_hours}")
    
    if len(data.demand) != expected_hours:
        errors.append(f"Demand length mismatch: {len(data.demand)} vs {expected_hours}")
    
    if len(data.solar_incidence) != expected_hours:
        errors.append(f"Solar incidence length mismatch: {len(data.solar_incidence)} vs {expected_hours}")
    
    if len(data.wind_incidence) != expected_hours:
        errors.append(f"Wind incidence length mismatch: {len(data.wind_incidence)} vs {expected_hours}")
    
    # Check for NaN values
    if np.isnan(data.spot_prices).any():
        errors.append("NaN values in spot prices")
    
    if np.isnan(data.demand).any():
        errors.append("NaN values in demand")
    
    # Check for negative values where inappropriate
    if (data.demand < 0).any():
        errors.append("Negative values in demand")
    
    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))
    
    print("Data validation passed!")
    return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_annual_statistics(data: CachedData) -> Dict[str, float]:
    """Calculate annual statistics from the data."""
    return {
        'total_demand_mwh': float(data.demand.sum()),
        'total_demand_twh': float(data.demand.sum() / 1e6),
        'peak_demand_mw': float(data.demand.max()),
        'avg_demand_mw': float(data.demand.mean()),
        'avg_spot_price': float(data.spot_prices.mean()),
        'max_spot_price': float(data.spot_prices.max()),
        'min_spot_price': float(data.spot_prices.min()),
    }


if __name__ == "__main__":
    # Test data loading
    data = load_all_data()
    validate_data(data)
    
    stats = get_annual_statistics(data)
    print("\nAnnual Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,.2f}")

