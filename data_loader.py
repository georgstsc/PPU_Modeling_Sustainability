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
from dataclasses import dataclass, field
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
    spot_prices: np.ndarray      # Shape: (8760,) - in CHF/MWh (converted from EUR)
    demand: np.ndarray           # Shape: (8760,)
    
    # Location rankings (for building in optimal locations)
    solar_ranking: np.ndarray    # Indices sorted by average incidence
    wind_ranking: np.ndarray     # Indices sorted by average incidence
    
    # Hydro data
    reservoir_levels: np.ndarray  # Monthly reservoir levels
    ror_production: np.ndarray    # Run-of-river monthly production
    water_inflow_mwh: np.ndarray  # Hourly water inflow to lakes (MWh) - from precipitation
    
    # Palm oil and exchange rates (for dynamic pricing)
    palm_oil_chf_mwh: np.ndarray  # Daily palm oil prices (CHF/MWh) - indexed by day of year
    usd_chf_daily: np.ndarray     # Daily USD/CHF rates - for palm oil conversion
    eur_chf_daily: np.ndarray     # Daily EUR/CHF rates - for spot price conversion
    
    # Cost data
    cost_table: pd.DataFrame
    ppu_components: pd.DataFrame
    ppu_lcoe: pd.DataFrame        # PPU efficiency and LCOE (from ppu_efficiency_lcoe_analysis.csv)
    
    # Metadata
    n_hours: int = 8760
    n_solar_locations: int = 0
    n_wind_locations: int = 0
    
    # Demand scenario multiplier (1.0 for 2024, 1.6 for 2050)
    demand_multiplier: float = 1.0
    demand_scenario: str = "2024"
    
    # PRE-COMPUTED SUMS for fast cumulative calculations (computed once on load)
    _solar_location_sums: Optional[np.ndarray] = None  # Sum over hours per location
    _wind_location_production: Optional[np.ndarray] = None  # Wind production per location
    
    def __post_init__(self):
        """Pre-compute location sums for fast cumulative calculations."""
        if self._solar_location_sums is None and self.solar_incidence is not None:
            self._solar_location_sums = np.sum(self.solar_incidence, axis=0)
        if self._wind_location_production is None and self.wind_incidence is not None:
            # Pre-compute wind production using vectorized power curve
            self._wind_location_production = self._compute_wind_production()
    
    def _compute_wind_production(self) -> np.ndarray:
        """Pre-compute annual wind production per location (MWh per unit)."""
        wind = self.wind_incidence
        num_turbines = 333  # Default: 1000 MW / 3 MW per turbine
        rated_power = 3.0
        cut_in, rated_speed, cut_out = 3.0, 12.0, 25.0
        
        # Vectorized power calculation
        power = np.zeros_like(wind)
        cubic_mask = (wind >= cut_in) & (wind < rated_speed)
        ratio = np.where(cubic_mask, (wind - cut_in) / (rated_speed - cut_in), 0)
        power = np.where(cubic_mask, rated_power * num_turbines * (ratio ** 3), power)
        rated_mask = (wind >= rated_speed) & (wind <= cut_out)
        power = np.where(rated_mask, rated_power * num_turbines, power)
        
        # Sum over hours for each location
        return np.sum(power, axis=0)
    
    def get_precomputed_sums(self) -> Dict[str, np.ndarray]:
        """Return pre-computed sums for fast cumulative calculations."""
        return {
            'solar_location_sums': self._solar_location_sums,
            'wind_location_production': self._wind_location_production,
        }
    
    def get_solar_incidence(self, copy: bool = True) -> np.ndarray:
        """Return solar incidence data. Set copy=False for read-only operations."""
        return self.solar_incidence.copy() if copy else self.solar_incidence
    
    def get_wind_incidence(self, copy: bool = True) -> np.ndarray:
        """Return wind incidence data. Set copy=False for read-only operations."""
        return self.wind_incidence.copy() if copy else self.wind_incidence
    
    def get_spot_prices(self, copy: bool = True) -> np.ndarray:
        """Return spot prices. Set copy=False for read-only operations."""
        return self.spot_prices.copy() if copy else self.spot_prices
    
    def get_demand(self, copy: bool = True) -> np.ndarray:
        """Return demand data with scenario multiplier applied.
        
        The stored demand is the 2024 base curve. This method applies
        the demand_multiplier (1.0 for 2024, 1.6 for 2050).
        
        Set copy=False for read-only operations (but note that multiplier is always applied).
        """
        scaled_demand = self.demand * self.demand_multiplier
        return scaled_demand.copy() if copy else scaled_demand
    
    def get_raw_demand(self, copy: bool = True) -> np.ndarray:
        """Return raw 2024 demand data without scenario multiplier."""
        return self.demand.copy() if copy else self.demand
    
    def get_cost_table(self) -> pd.DataFrame:
        """Return a copy of cost table."""
        return self.cost_table.copy()
    
    def get_ppu_components(self) -> pd.DataFrame:
        """Return a copy of PPU components."""
        return self.ppu_components.copy()
    
    def get_ppu_lcoe(self) -> pd.DataFrame:
        """Return a copy of PPU LCOE data."""
        return self.ppu_lcoe.copy()
    
    def get_water_inflow(self, copy: bool = True) -> np.ndarray:
        """Return water inflow to lakes (MWh). Set copy=False for read-only operations."""
        return self.water_inflow_mwh.copy() if copy else self.water_inflow_mwh
    
    def get_palm_oil_price(self, day_of_year: int) -> float:
        """Return palm oil price (CHF/MWh) for a specific day of year (0-365)."""
        day_idx = min(day_of_year, len(self.palm_oil_chf_mwh) - 1)
        return float(self.palm_oil_chf_mwh[day_idx])
    
    def get_palm_oil_prices_daily(self, copy: bool = True) -> np.ndarray:
        """Return daily palm oil prices (CHF/MWh)."""
        return self.palm_oil_chf_mwh.copy() if copy else self.palm_oil_chf_mwh
    
    def get_eur_chf_rate(self, day_of_year: int) -> float:
        """Return EUR/CHF rate for a specific day of year (0-365)."""
        day_idx = min(day_of_year, len(self.eur_chf_daily) - 1)
        return float(self.eur_chf_daily[day_idx])


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
    
    NOTE: Wind data appears to be scaled incorrectly in the source file.
    Typical Swiss wind speeds: mean ~6-7 m/s, peaks 20-25 m/s
    Current data: mean ~1.3 m/s, max ~10 m/s
    We apply a scaling factor to correct this.
    
    Returns:
        Tuple of (incidence_array, ranking_indices)
    """
    wind_path = data_dir / 'wind_incidence_hourly_2024.csv'
    
    # NO WIND SPEED SCALING - using raw data as-is
    # With distributed model (1 turbine per location × 1150 locations per PPU),
    # even low wind speeds produce meaningful aggregate output
    
    try:
        # Try reading with multi-index header
        df = pd.read_csv(wind_path, header=[0, 1], index_col=0, parse_dates=True)
        incidence = df.values.astype(np.float32)
    except Exception:
        # Fallback: read as simple CSV
        df = pd.read_csv(wind_path)
        incidence_cols = [c for c in df.columns if c not in ['timestamp', 'hour', 'Unnamed: 0', 'time']]
        incidence = df[incidence_cols].values.astype(np.float32)
    
    print(f"  - Wind data loaded (mean: {np.nanmean(incidence):.1f} m/s, no scaling)")
    
    # Compute ranking: sort by mean wind speed descending
    mean_incidence = np.nanmean(incidence, axis=0)
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
    
    # CRITICAL: No fallback prices allowed - raise error on missing data
    if np.isnan(prices).any():
        nan_count = np.isnan(prices).sum()
        raise ValueError(
            f"CRITICAL: Found {nan_count} NaN values in spot prices from {spot_path}. "
            f"Missing price data cannot be replaced with fallback values. "
            f"Please fix the data source or handle missing values explicitly."
        )
    
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


def _load_cost_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cost table, PPU components, and PPU LCOE."""
    cost_path = data_dir / 'cost_table_tidy.csv'
    cost_df = pd.read_csv(cost_path)
    
    ppu_path = data_dir / 'ppu_constructs_components.csv'
    ppu_df = pd.read_csv(ppu_path)
    
    # Load PPU efficiency and LCOE data (with parallelism)
    lcoe_path = data_dir / 'ppu_efficiency_lcoe_analysis.csv'
    if lcoe_path.exists():
        lcoe_df = pd.read_csv(lcoe_path)
    else:
        lcoe_df = pd.DataFrame()
        warnings.warn("PPU LCOE file not found - using empty DataFrame")
    
    return cost_df, ppu_df, lcoe_df


def _load_palm_oil_prices(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load palm oil prices and convert to CHF/MWh.
    
    Source: rea_holdings_share_prices.csv (USD/metric ton)
    Conversion: USD/metric ton → CHF/MWh
        - 1 metric ton = 12.22 MWh (at 44 MJ/kg energy density)
        - USD → CHF via DAT_ASCII_USDCHF_M1_2024.csv
    
    Returns:
        Tuple of (palm_oil_chf_mwh, usd_chf_daily) arrays indexed by day of year
    """
    # Load palm oil prices (USD/metric ton)
    palm_path = data_dir / 'rea_holdings_share_prices.csv'
    if palm_path.exists():
        palm_df = pd.read_csv(palm_path)
        # Clean up the DataFrame - the file has irregular structure
        # First column should be date, use the second numeric column as price
        palm_df.columns = ['date', 'open', 'high', 'low', 'close']
        palm_df = palm_df.iloc[1:]  # Skip header row
        palm_df['date'] = pd.to_datetime(palm_df['date'], errors='coerce')
        palm_df['close'] = pd.to_numeric(palm_df['close'], errors='coerce')
        palm_df = palm_df.dropna(subset=['date', 'close'])
        palm_df = palm_df.set_index('date').sort_index()
        
        # Filter to 2024 and resample to daily (forward fill missing days)
        palm_2024 = palm_df['2024-01-01':'2024-12-31']['close']
        palm_daily = palm_2024.resample('D').ffill().bfill()
    else:
        # CRITICAL: No fallback prices allowed
        raise FileNotFoundError(
            f"CRITICAL: Palm oil price file not found at {palm_path}. "
            f"Cannot use fallback prices as they would falsify costs. "
            f"Please ensure the data file exists and is properly formatted."
        )
    
    # Load USD/CHF exchange rates
    usd_chf_path = data_dir / 'DAT_ASCII_USDCHF_M1_2024.csv'
    if usd_chf_path.exists():
        # This is minute data with format: YYYYMMDD HHMMSS;open;high;low;close;volume
        usd_df = pd.read_csv(usd_chf_path, sep=';', header=None,
                             names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        usd_df['date'] = pd.to_datetime(usd_df['datetime'].str[:8], format='%Y%m%d', errors='coerce')
        # Aggregate to daily average - the rate is USD per CHF
        usd_daily = usd_df.groupby('date')['close'].mean()
        # Convert USD per CHF to CHF per USD (invert)
        chf_per_usd = 1.0 / usd_daily
        chf_per_usd = chf_per_usd.resample('D').ffill().bfill()
    else:
        # CRITICAL: No fallback exchange rates allowed
        raise FileNotFoundError(
            f"CRITICAL: USD/CHF exchange rate file not found at {usd_chf_path}. "
            f"Cannot use fallback rates as they would falsify costs. "
            f"Please ensure the data file exists and is properly formatted."
        )
    
    # Align indices
    date_range = pd.date_range('2024-01-01', periods=366, freq='D')
    # Forward-fill from previous values, then backfill any remaining gaps (e.g., first day)
    # This uses actual data from the time series, not arbitrary fallback values
    palm_aligned = palm_daily.reindex(date_range, method='ffill').bfill()
    chf_aligned = chf_per_usd.reindex(date_range, method='ffill').bfill()
    
    # CRITICAL: Check for missing values after alignment - no fallbacks allowed
    # After forward-fill and backfill, any remaining NaNs indicate data gaps that cannot be filled
    if palm_aligned.isna().any():
        missing_days = palm_aligned.isna().sum()
        missing_dates = palm_aligned[palm_aligned.isna()].index.strftime('%Y-%m-%d').tolist()
        raise ValueError(
            f"CRITICAL: {missing_days} missing palm oil price values after alignment. "
            f"Cannot use fallback prices as they would falsify costs. "
            f"Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}. "
            f"Please ensure complete price data coverage for 2024."
        )
    
    if chf_aligned.isna().any():
        missing_days = chf_aligned.isna().sum()
        missing_dates = chf_aligned[chf_aligned.isna()].index.strftime('%Y-%m-%d').tolist()
        raise ValueError(
            f"CRITICAL: {missing_days} missing USD/CHF exchange rate values after alignment. "
            f"Cannot use fallback rates as they would falsify costs. "
            f"Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}. "
            f"Please ensure complete exchange rate data coverage for 2024."
        )
    
    # Convert USD/metric ton to CHF/MWh
    # price_chf_mwh = price_usd_ton × chf_per_usd / mwh_per_ton
    MWH_PER_TON = 12.22  # 44 MJ/kg × 1000 kg/ton ÷ 3600 MJ/MWh
    palm_chf_mwh = (palm_aligned.values * chf_aligned.values) / MWH_PER_TON
    
    return palm_chf_mwh.astype(np.float32), chf_aligned.values.astype(np.float32)


def _load_eur_chf_rates(data_dir: Path) -> np.ndarray:
    """
    Load EUR/CHF exchange rates for spot price conversion.
    
    Source: chf_to_eur_2024.csv
    - Column "CHF to EUR" = X means: 1 CHF = X EUR (CHF is stronger when X > 1.0)
    - We need EUR/CHF: 1 EUR = (1/X) CHF
    - Historical EUR/CHF in 2024: ~0.93-0.99 (CHF stronger, so EUR/CHF < 1.0)
    
    Returns:
        Array of EUR/CHF rates indexed by day of year (366 days for 2024)
        Values should be < 1.0 (typically 0.93-0.99) since CHF is stronger than EUR
    """
    eur_path = data_dir / 'chf_to_eur_2024.csv'
    if eur_path.exists():
        eur_df = pd.read_csv(eur_path)
        eur_df['Date'] = pd.to_datetime(eur_df['Date'], errors='coerce')
        eur_df = eur_df.dropna(subset=['Date'])
        eur_df = eur_df.set_index('Date').sort_index()
        
        # The file has "CHF to EUR" which means 1 CHF = X EUR
        # Since CHF is stronger than EUR, X > 1.0 (typically 1.05-1.08 in 2024)
        # We need EUR/CHF: 1 EUR = (1/X) CHF
        # This should give values < 1.0 (typically 0.93-0.99)
        chf_to_eur = eur_df['CHF to EUR']
        
        # Verify interpretation: if CHF is stronger, chf_to_eur should be > 1.0
        if chf_to_eur.mean() < 1.0:
            raise ValueError(
                f"CRITICAL: Exchange rate interpretation error. "
                f"'CHF to EUR' values average {chf_to_eur.mean():.4f} (< 1.0), "
                f"but CHF should be stronger than EUR. "
                f"Please verify the data file format and interpretation."
            )
        
        eur_to_chf = 1.0 / chf_to_eur
        
        # Verify result: EUR/CHF should be < 1.0 if CHF is stronger
        if eur_to_chf.mean() > 1.0:
            warnings.warn(
                f"WARNING: Calculated EUR/CHF = {eur_to_chf.mean():.4f} (> 1.0). "
                f"Expected < 1.0 since CHF is stronger. "
                f"Historical range: 0.93-0.99. Please verify conversion logic."
            )
        
        # Resample to daily
        eur_to_chf = eur_to_chf.resample('D').ffill().bfill()
    else:
        # CRITICAL: No fallback exchange rates allowed
        raise FileNotFoundError(
            f"CRITICAL: EUR/CHF exchange rate file not found at {eur_path}. "
            f"Cannot use fallback rates as they would falsify costs. "
            f"Please ensure the data file exists and is properly formatted."
        )
    
    # Align to full year
    date_range = pd.date_range('2024-01-01', periods=366, freq='D')
    # Forward-fill from previous values, then backfill any remaining gaps (e.g., first day)
    # This uses actual data from the time series, not arbitrary fallback values
    eur_aligned = eur_to_chf.reindex(date_range, method='ffill').bfill()
    
    # CRITICAL: Check for missing values after alignment - no fallbacks allowed
    # After forward-fill and backfill, any remaining NaNs indicate data gaps that cannot be filled
    if eur_aligned.isna().any():
        missing_days = eur_aligned.isna().sum()
        missing_dates = eur_aligned[eur_aligned.isna()].index.strftime('%Y-%m-%d').tolist()
        raise ValueError(
            f"CRITICAL: {missing_days} missing EUR/CHF exchange rate values after alignment. "
            f"Cannot use fallback rates as they would falsify costs. "
            f"Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}. "
            f"Please ensure complete exchange rate data coverage for 2024."
        )
    
    return eur_aligned.values.astype(np.float32)


def _load_water_inflow(data_dir: Path, kwh_per_m3: float = 0.9) -> np.ndarray:
    """
    Load water inflow to lakes from precipitation data.
    
    Source: Swiss_Water_Hourly_2024.csv (Water_Reaching_Lakes_m3 column)
    Conversion: m³ × 0.9 kWh/m³ = kWh, then /1000 = MWh
    
    Based on Swiss hydropower data from hydrodaten.admin.ch:
    - 36.5 TWh annual production from 40.5 km³ runoff
    - 36,500 GWh / 40,500 million m³ ≈ 0.90 kWh/m³
    
    Returns:
        Array of hourly water inflow (MWh) - shape (8784,) for 2024
    """
    water_path = data_dir / 'Swiss_Water_Hourly_2024.csv'
    if water_path.exists():
        water_df = pd.read_csv(water_path)
        if 'Water_Reaching_Lakes_m3' in water_df.columns:
            water_m3 = water_df['Water_Reaching_Lakes_m3'].values
            # Convert m³ to MWh: m³ × kWh/m³ ÷ 1000
            water_mwh = water_m3 * kwh_per_m3 / 1000.0
        else:
            water_mwh = np.zeros(8784)
            warnings.warn("Water inflow column not found - using zeros")
    else:
        water_mwh = np.zeros(8784)
        warnings.warn("Water inflow file not found - using zeros")
    
    return water_mwh.astype(np.float32)


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
    
    spot_prices_eur = _load_spot_prices(data_dir)
    print(f"  - Spot prices (EUR): {spot_prices_eur.shape}")
    
    demand = _load_demand(data_dir)
    print(f"  - Demand: {demand.shape}")
    
    reservoir, ror = _load_hydro_data(data_dir)
    print(f"  - Hydro data loaded")
    
    cost_table, ppu_components, ppu_lcoe = _load_cost_data(data_dir)
    print(f"  - Cost data loaded (LCOE: {len(ppu_lcoe)} PPUs)")
    
    # Load exchange rates for currency conversion
    eur_chf_daily = _load_eur_chf_rates(data_dir)
    print(f"  - EUR/CHF rates: {len(eur_chf_daily)} days")
    
    # Load palm oil prices and USD/CHF rates
    palm_oil_chf_mwh, usd_chf_daily = _load_palm_oil_prices(data_dir)
    print(f"  - Palm oil prices (CHF/MWh): {len(palm_oil_chf_mwh)} days, avg {palm_oil_chf_mwh.mean():.2f}")
    
    # Load water inflow to lakes
    water_inflow_mwh = _load_water_inflow(data_dir)
    print(f"  - Water inflow: {len(water_inflow_mwh)} hours, total {water_inflow_mwh.sum()/1e6:.2f} TWh")
    
    # Convert spot prices from EUR to CHF (hourly)
    # Map each hour to its day of year to get the right exchange rate
    n_hours_spot = len(spot_prices_eur)
    day_indices = np.arange(n_hours_spot) // 24
    day_indices = np.clip(day_indices, 0, len(eur_chf_daily) - 1)
    spot_prices = spot_prices_eur * eur_chf_daily[day_indices]
    print(f"  - Spot prices converted to CHF: avg {spot_prices.mean():.2f} CHF/MWh")
    
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
    
    # Align water inflow to same length
    water_inflow_mwh = water_inflow_mwh[:n_hours]
    
    # Get demand scenario multiplier from config
    demand_multiplier = config.energy_system.DEMAND_MULTIPLIER
    demand_scenario = config.energy_system.DEMAND_SCENARIO
    
    # Create cached data object
    _DATA_CACHE = CachedData(
        solar_incidence=solar_incidence,
        wind_incidence=wind_incidence,
        spot_prices=spot_prices,
        demand=demand,  # Raw 2024 demand - multiplier applied in get_demand()
        solar_ranking=solar_ranking,
        wind_ranking=wind_ranking,
        reservoir_levels=reservoir,
        ror_production=ror,
        water_inflow_mwh=water_inflow_mwh,
        palm_oil_chf_mwh=palm_oil_chf_mwh,
        usd_chf_daily=usd_chf_daily,
        eur_chf_daily=eur_chf_daily,
        cost_table=cost_table,
        ppu_components=ppu_components,
        ppu_lcoe=ppu_lcoe,
        n_hours=n_hours,
        n_solar_locations=solar_incidence.shape[1] if solar_incidence.ndim > 1 else 1,
        n_wind_locations=wind_incidence.shape[1] if wind_incidence.ndim > 1 else 1,
        demand_multiplier=demand_multiplier,
        demand_scenario=demand_scenario,
    )
    
    print(f"Data loading complete! (Demand scenario: {demand_scenario}, multiplier: {demand_multiplier}x)")
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

