# ============================================================================
# ENERGY DISPATCH PIPELINE FRAMEWORK
# ============================================================================

import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class MockTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def update(self, n=1):
            pass  # No-op for update
        def close(self):
            pass  # No-op for close
    def tqdm(iterable, **kwargs):
        return MockTqdm(iterable, **kwargs)
try:
    from numba import njit, prange  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    # Fallback no-op decorator if numba isn't available
    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func
        return wrapper
    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)

# Import existing frameworks
from calculationsCostFramework import (
    calculate_disposition_index,
    calculate_utility_indices,
    calculate_monetary_index,
    exponential_moving_average
)
from calculationsPpuFramework import (
    load_cost_data,
    get_incidence_data,
    initialize_ppu_dictionary,
    add_ppu_to_dictionary,
    load_location_rankings,
    load_ppu_data,
    calculate_chain_cost
)

# ============================================================================
# STATIC DATA CACHING - Load once and reuse across multiple pipeline calls
# ============================================================================
_STATIC_DATA_CACHE: Optional[Dict[str, Any]] = None
_ANNUAL_DATA_CACHE: Optional[Dict[str, Any]] = None



def load_static_data(data_dir: str = 'data') -> Dict:
    """
    Load all static configuration data once and cache it.
    
    Static data includes:
    - PPU constructs (ppu_constructs_components.csv)
    - Cost table (cost_table_tidy.csv)
    - Solar location rankings
    - Wind location rankings
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with cached static data
    """
    global _STATIC_DATA_CACHE
    
    if _STATIC_DATA_CACHE is None:
        print("[CACHE] Loading static data (first time, will be cached)...")
        _STATIC_DATA_CACHE = {
            'ppu_constructs_df': load_ppu_data(f'{data_dir}/ppu_constructs_components.csv'),
            'cost_df': load_cost_data(f'{data_dir}/cost_table_tidy.csv'),
            'solar_locations_df': load_location_rankings('solar'),
            'wind_locations_df': load_location_rankings('wind'),
        }
    return _STATIC_DATA_CACHE

def load_annual_data(data_dir: str = 'data') -> Optional[Dict[str, Any]]:
    """
    Load annual time-series data once and cache it.

    The returned dictionary contains:
      - 'demand_15min'
      - 'spot_15min'
      - 'ror_df'
      - 'solar_15min'
      - 'wind_15min'
      - 'timestamp_index' (canonical DatetimeIndex for 15-min timesteps)

    Args:
        data_dir: Directory containing the data files.

    Returns:
        The cached annual data dictionary or None if loading failed.
    """
    global _ANNUAL_DATA_CACHE

    if _ANNUAL_DATA_CACHE is None:
        print("[CACHE] Loading annual data (first time, will be cached)...")
        # Reuse the existing loaders in this module so behavior stays consistent
        demand_15min, spot_15min, ror_df = load_energy_data(data_dir)
        solar_15min, wind_15min = load_incidence_data(data_dir)

        # Use demand index as the canonical timestamp index when available
        timestamp_index = getattr(demand_15min, 'index', None)

        _ANNUAL_DATA_CACHE = {
            'demand_15min': demand_15min,
            'spot_15min': spot_15min,
            'ror_df': ror_df,
            'solar_15min': solar_15min,
            'wind_15min': wind_15min,
            'timestamp_index': timestamp_index,
        }

    return _ANNUAL_DATA_CACHE

def load_cache_data(data_dir: str = 'data') -> None:
    """
    Load both static and annual data caches.
    """
    load_static_data(data_dir=data_dir)
    load_annual_data(data_dir=data_dir)

def get_annual_data_cache(data_dir: str = 'data') -> Optional[Dict[str, Any]]:
    """
    Return the cached annual dataset, ensuring it is loaded first.
    """
    if _ANNUAL_DATA_CACHE is None:
        load_cache_data(data_dir=data_dir)
    return _ANNUAL_DATA_CACHE


def get_static_data_cache(data_dir: str = 'data') -> Optional[Dict[str, Any]]:
    """
    Return the cached static dataset, ensuring it is loaded first.
    """
    if _STATIC_DATA_CACHE is None:
        load_cache_data(data_dir=data_dir)
    return _STATIC_DATA_CACHE


def generate_random_scenario(
    annual_data: Optional[Dict[str, Any]] = None,
    num_days: int = 30,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Sample a sequence of day indices WITH replacement and return only that series.

    If `annual_data` is not provided, this function will use the module-level
    annual cache (loaded/validated via `get_annual_data_cache`). Callers that
    prefer to remain cache-free may pass `annual_data` explicitly.

    Parameters
    ----------
    annual_data : dict, optional
        Annual data (only `timestamp_index` used here). If None, the global
        cache is used.
    num_days : int
        Number of day indices to sample
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    scenario_dict : dict
        - 'selected_days': Ordered list (length = num_days; duplicates allowed)
        - 'num_days': Number of sampled days
        - 'sampled_dates': List of datetime objects representing start of each sampled day
    """
    # Resolve annual data (use validated cache if not supplied)
    if annual_data is None:
        annual_data = get_annual_data_cache(data_dir='data')

    if seed is not None:
        np.random.seed(seed)

    n_days_in_year = 365
    day_indices = np.random.randint(0, n_days_in_year, size=num_days).tolist()

    # Build list of datetime objects (start of each sampled day) if available
    sampled_dates = []
    if isinstance(annual_data, dict) and 'timestamp_index' in annual_data:
        for day_idx in day_indices:
            start_pos = day_idx * 96
            try:
                if start_pos < len(annual_data['timestamp_index']):
                    sampled_dates.append(annual_data['timestamp_index'][start_pos])
                else:
                    sampled_dates.append(None)
            except Exception:
                sampled_dates.append(None)

    return {
        'selected_days': day_indices,
        'num_days': num_days,
        'sampled_dates': sampled_dates
    }


def scale_storage_capacities_by_unit_counts(
    ppu_dictionary: pd.DataFrame,
    raw_energy_storage: List[Dict],
    do_not_scale: Optional[List[str]] = None
) -> None:
    """
    Scale each storage's max capacity ('value') by the number of Store-PPUs that input to it.

    Example: If there are 3 NH3_FULL units (each feeding 'Ammonia storage'), and base per-unit
    capacity is 100,000, the storage 'value' becomes 300,000.

    Mutates raw_energy_storage in place. Persists 'base_value' to avoid compounding when the
    function is called multiple times in one process.

    Args:
        ppu_dictionary: Configured PPUs with 'can_input_to'
        raw_energy_storage: Storage dicts with 'storage', 'value', 'current_value'
        do_not_scale: Optional list of storages to exclude from scaling (e.g., ['Lake'])
    """
    try:
        # Count how many PPUs can INPUT to each storage
        storage_unit_counts: Dict[str, int] = {}
        for _, row in ppu_dictionary.iterrows():
            can_input_to = row.get('can_input_to', []) or []
            if not isinstance(can_input_to, list):
                continue
            for sto in can_input_to:
                storage_unit_counts[sto] = storage_unit_counts.get(sto, 0) + 1

        do_not_scale_set = set(do_not_scale or ['Lake'])

        for storage in raw_energy_storage:
            name = storage.get('storage')
            if not name or name in do_not_scale_set:
                continue

            # Save base per-unit capacity the first time
            if 'base_value' not in storage:
                storage['base_value'] = storage.get('value', 0.0)

            unit_count = storage_unit_counts.get(name, 0)
            if unit_count > 0:
                new_max = float(storage['base_value']) * float(unit_count)
                storage['value'] = new_max
                if storage.get('current_value', 0.0) > new_max:
                    storage['current_value'] = new_max
    except Exception as e:
        print(f"[WARN] Failed to scale storage capacities by unit counts: {e}")


# ------------------------------------------------------------------
# Ultra-fast solar production mapper (build once, reuse)
# ------------------------------------------------------------------
class SolarMapper:
    """
    Builds once: (lat, lon) -> closest column index, with 15-min irradiance matrix.
    """
    __slots__ = ("lats", "lons", "cols", "data", "rank_order")

    def __init__(self, csv_path: str):
        # The CSV is shaped like solar_incidence_hourly_2024.csv
        df = pd.read_csv(csv_path, header=None)
        self.lats = df.iloc[0, 1:].astype(np.float32).values
        self.lons = df.iloc[1, 1:].astype(np.float32).values
        self.cols = np.arange(self.lats.size, dtype=np.int32)

        # Interpolate hourly to 15-min using numpy interpolation
        times = pd.to_datetime(df.iloc[3:, 0].values)
        times_dt = times.to_numpy()
        times_seconds = times_dt.astype('datetime64[s]').astype(np.int64)
        data_hourly = df.iloc[3:, 1:].astype(np.float32).values

        target_dt = pd.date_range(start=times_dt[0], end=times_dt[-1], freq='15min').to_numpy()
        target_seconds = target_dt.astype('datetime64[s]').astype(np.int64)

        nt = target_seconds.size
        nc = data_hourly.shape[1]
        data_15 = np.empty((nt, nc), dtype=np.float32)
        for j in range(nc):
            data_15[:, j] = np.interp(target_seconds, times_seconds, data_hourly[:, j])

        self.data = data_15

        # Precompute ranking order (highest mean irradiance first)
        means = self.data.mean(axis=0)
        self.rank_order = np.argsort(means)[::-1].astype(np.int32)

    def closest_column(self, lat: float, lon: float) -> int:
        # Manhattan distance in lat/lon grid (consistent with previous logic)
        dist = np.abs(self.lats - lat) + np.abs(self.lons - lon)
        return int(self.cols[np.argmin(dist)])

    def col_for_rank(self, rank_1based: int) -> int:
        # Cycle if rank exceeds number of locations
        idx = (int(rank_1based) - 1) % self.rank_order.size
        return int(self.rank_order[idx])

    def irradiance(self, t_idx: int, col_idx: int) -> float:
        return float(self.data[t_idx, col_idx])

    def precompute_productions(self, ranks: np.ndarray, areas: np.ndarray, num_timesteps: int) -> np.ndarray:
        """
        Precompute productions for all ranks and timesteps.
        Args:
            ranks: 1D array of 1-based location ranks
            areas: 1D array of total areas (m²) for each rank
            num_timesteps: Number of timesteps to compute
        Returns:
            (num_timesteps, len(ranks)) array of MW productions
        """
        # Map ranks to column indices
        col_idxs = np.array([self.col_for_rank(int(r)) for r in ranks], dtype=np.int32)
        # Extract irradiance matrix slice
        irr_matrix = self.data[:num_timesteps, col_idxs]
        # Batch compute productions
        return _solar_prod_batch(irr_matrix, areas)


_SOLAR_MAPPER: Optional[SolarMapper] = None


def _init_solar_mapper(csv_path: str = 'data/solar_incidence_hourly_2024.csv') -> None:
    global _SOLAR_MAPPER
    if _SOLAR_MAPPER is None:
        try:
            _SOLAR_MAPPER = SolarMapper(csv_path)
        except Exception:
            # Leave as None: fallback path will be used
            _SOLAR_MAPPER = None


@njit(cache=True)
def _solar_prod_fast(irradiance: float, area_m2: float) -> float:
    """Convert kWh/m²/hour -> MW for a 15-min slice."""
    return irradiance * area_m2 * 0.25 / 1000.0


@njit(parallel=NUMBA_AVAILABLE, cache=True)
def _solar_prod_batch(irr_matrix: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """
    Vectorized batch solar production calculation.
    Args:
        irr_matrix: (num_timesteps, num_ranks) irradiance values
        areas: (num_ranks,) area in m² for each rank
    Returns:
        (num_timesteps, num_ranks) production in MW
    """
    nt, nr = irr_matrix.shape
    prods = np.zeros((nt, nr), dtype=np.float32)
    for t in prange(nt):
        for r in range(nr):
            prods[t, r] = irr_matrix[t, r] * areas[r] * 0.25 / 1000.0
    return prods


# ------------------------------------------------------------------
# Ultra-fast wind production mapper (build once, reuse)
# ------------------------------------------------------------------
class WindMapper:
    __slots__ = ("lats", "lons", "cols", "data", "rank_order")

    def __init__(self, csv_path: str):
        # Expect multi-index header (lat, lon), datetime index
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
        # Extract lat/lon from multi-index columns
        self.lats = np.array([float(c[0]) for c in df.columns], dtype=np.float32)
        self.lons = np.array([float(c[1]) for c in df.columns], dtype=np.float32)
        self.cols = np.arange(self.lats.size, dtype=np.int32)
        # Resample to 15-min using numpy interpolation to avoid pandas interpolation overhead
        times_dt = df.index.to_numpy()
        times_seconds = times_dt.astype('datetime64[s]').astype(np.int64)
        data_hourly = df.to_numpy(dtype=np.float32)

        target_dt = pd.date_range(start=times_dt[0], end=times_dt[-1], freq='15min').to_numpy()
        target_seconds = target_dt.astype('datetime64[s]').astype(np.int64)

        nt = target_seconds.size
        nc = data_hourly.shape[1]
        data_15 = np.empty((nt, nc), dtype=np.float32)
        for j in range(nc):
            data_15[:, j] = np.interp(target_seconds, times_seconds, data_hourly[:, j])

        self.data = data_15
        # Precompute ranking by mean wind speed
        means = self.data.mean(axis=0)
        self.rank_order = np.argsort(means)[::-1].astype(np.int32)

    def closest_column(self, lat: float, lon: float) -> int:
        dist = np.abs(self.lats - lat) + np.abs(self.lons - lon)
        return int(self.cols[np.argmin(dist)])

    def col_for_rank(self, rank_1based: int) -> int:
        idx = (int(rank_1based) - 1) % self.rank_order.size
        return int(self.rank_order[idx])

    def speed(self, t_idx: int, col_idx: int) -> float:
        return float(self.data[int(t_idx), int(col_idx)])

    def precompute_productions(self, ranks: np.ndarray, num_turbines: np.ndarray, num_timesteps: int) -> np.ndarray:
        """
        Precompute productions for all ranks and timesteps.
        Args:
            ranks: 1D array of 1-based location ranks
            num_turbines: 1D array of total turbine counts for each rank
            num_timesteps: Number of timesteps to compute
        Returns:
            (num_timesteps, len(ranks)) array of MW productions
        """
        # Map ranks to column indices
        col_idxs = np.array([self.col_for_rank(int(r)) for r in ranks], dtype=np.int32)
        # Extract wind speed matrix slice
        wspd_matrix = self.data[:num_timesteps, col_idxs]
        # Batch compute productions
        return _wind_power_batch(wspd_matrix, num_turbines)


_WIND_MAPPER: Optional[WindMapper] = None


def _init_wind_mapper(csv_path: str = 'data/wind_incidence_hourly_2024.csv') -> None:
    global _WIND_MAPPER
    if _WIND_MAPPER is None:
        try:
            _WIND_MAPPER = WindMapper(csv_path)
        except Exception:
            _WIND_MAPPER = None


@njit(cache=True)
def _wind_power_fast(wspd: float, num_turbines: int) -> float:
    # Generic 120 m rotor, with cut-in/out limits
    rotor_d = 120.0
    rho = 1.225
    A = np.pi * (rotor_d / 2.0) ** 2
    cp = 0.45
    # Apply cut-in/out
    if wspd < 3.0 or wspd > 25.0:
        return 0.0
    power_W = 0.5 * rho * A * (wspd ** 3) * cp
    return power_W * num_turbines / 1e6  # MW


@njit(parallel=NUMBA_AVAILABLE, cache=True)
def _wind_power_batch(wspd_matrix: np.ndarray, num_turbines: np.ndarray) -> np.ndarray:
    """
    Vectorized batch wind production calculation.
    Args:
        wspd_matrix: (num_timesteps, num_ranks) wind speeds
        num_turbines: (num_ranks,) turbine count for each rank
    Returns:
        (num_timesteps, num_ranks) production in MW
    """
    nt, nr = wspd_matrix.shape
    prods = np.zeros((nt, nr), dtype=np.float32)
    rotor_d = 120.0
    rho = 1.225
    A = np.pi * (rotor_d / 2.0) ** 2
    cp = 0.45
    
    for t in prange(nt):
        for r in range(nr):
            wspd = wspd_matrix[t, r]
            if wspd < 3.0 or wspd > 25.0:
                prods[t, r] = 0.0
            else:
                power_W = 0.5 * rho * A * (wspd ** 3) * cp
                prods[t, r] = power_W * num_turbines[r] / 1e6
    return prods


def load_energy_data(data_dir: str = "data") -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Load all energy data sources for the dispatch pipeline.

    Parameters:
        data_dir: Directory containing data files

    Returns:
        Tuple of (demand_15min, spot_15min, ror_df)
    """
    # Load demand data (tab-delimited, filter for Switzerland only)
    demand_df = pd.read_csv(f'{data_dir}/monthly_hourly_load_values_2024.csv', sep='\t')
    demand_df = demand_df[demand_df['CountryCode'] == 'CH'].copy()
    demand_df['datetime'] = pd.to_datetime(demand_df['DateUTC'], dayfirst=True)
    demand_df = demand_df.set_index('datetime').sort_index()
    demand_df = demand_df[~demand_df.index.duplicated(keep='first')]
    demand_15min = demand_df['Value'].resample('15min').interpolate(method='linear')

    # Load spot prices
    spot_df = pd.read_csv(f'{data_dir}/spot_price_hourly.csv', parse_dates=['time'])
    spot_df = spot_df.set_index('time').sort_index()
    spot_15min = spot_df['price'].resample('15min').interpolate(method='linear')

    # Load RoR data (already 15-min)
    ror_df = pd.read_csv(f'{data_dir}/water_quarterly_ror_2024.csv', parse_dates=['timestamp'])
    ror_df = ror_df.set_index('timestamp').sort_index()
    ror_df.head()
    return demand_15min, spot_15min, ror_df


def load_incidence_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load solar and wind incidence data.

    Returns:
        Tuple of (solar_15min, wind_15min)
    """
    # Load solar ranking to get top locations
    solar_ranking_df = pd.read_csv(f'{data_dir}/ranking_incidence/solar_incidence_ranking.csv').head(10)

    # Load hourly solar data
    solar_hourly_df = pd.read_csv(f'{data_dir}/solar_incidence_hourly_2024.csv', header=None)
    latitudes_solar = solar_hourly_df.iloc[0, 1:].astype(float).values
    longitudes_solar = solar_hourly_df.iloc[1, 1:].astype(float).values
    times_solar = pd.to_datetime(solar_hourly_df.iloc[3:, 0].values)
    solar_data = solar_hourly_df.iloc[3:, 1:].astype(float).values

    # Create MultiIndex for solar
    solar_columns = pd.MultiIndex.from_arrays([latitudes_solar, longitudes_solar], names=['lat', 'lon'])
    solar_df = pd.DataFrame(solar_data, index=times_solar, columns=solar_columns)
    # Fast numpy-based interpolation to 15-min (avoid heavy pandas resample/interpolate)
    times_solar_dt = times_solar.to_numpy()
    times_solar_seconds = times_solar_dt.astype('datetime64[s]').astype(np.int64)
    target_dt = pd.date_range(start=times_solar_dt[0], end=times_solar_dt[-1], freq='15min')
    target_seconds = target_dt.to_numpy().astype('datetime64[s]').astype(np.int64)
    solar_hourly = solar_data.astype(np.float32)
    nt = target_seconds.size
    nc = solar_hourly.shape[1]
    solar_interp = np.empty((nt, nc), dtype=np.float32)
    for j in range(nc):
        solar_interp[:, j] = np.interp(target_seconds, times_solar_seconds, solar_hourly[:, j])
    solar_15min = pd.DataFrame(solar_interp, index=target_dt, columns=solar_columns)

    # Load hourly wind data
    wind_hourly_df = pd.read_csv(f'{data_dir}/wind_incidence_hourly_2024.csv', index_col=0, header=[0, 1], parse_dates=True)
    # Fast interpolation for wind as well
    wind_times = wind_hourly_df.index.to_numpy()
    wind_seconds = wind_times.astype('datetime64[s]').astype(np.int64)
    target_dt_w = pd.date_range(start=wind_times[0], end=wind_times[-1], freq='15min')
    target_seconds_w = target_dt_w.to_numpy().astype('datetime64[s]').astype(np.int64)
    wind_hourly_values = wind_hourly_df.to_numpy(dtype=np.float32)
    ntw = target_seconds_w.size
    ncw = wind_hourly_values.shape[1]
    wind_interp = np.empty((ntw, ncw), dtype=np.float32)
    for j in range(ncw):
        wind_interp[:, j] = np.interp(target_seconds_w, wind_seconds, wind_hourly_values[:, j])
    wind_15min = pd.DataFrame(wind_interp, index=target_dt_w, columns=wind_hourly_df.columns)

    return solar_15min, wind_15min


def initialize_technology_tracking(ppu_dictionary: pd.DataFrame) -> Dict[str, Dict]:
    """
    Initialize technology volume tracking structure.

    Parameters:
        ppu_dictionary: PPU configuration DataFrame

    Returns:
        Technology volume tracking dictionary
    """
    Technology_volume = {}
    for _, ppu_row in ppu_dictionary.iterrows():
        tech_type = ppu_row['PPU_Name']
        if tech_type not in Technology_volume:
            Technology_volume[tech_type] = {
                'production': [],
                'spot_bought': [],
                'spot_sold': [],
                'cost_indices': []
            }

    return Technology_volume


def precompute_renewable_productions(ppu_dictionary: pd.DataFrame, selected_days: List[int]) -> Dict:
    """
    Precompute renewable productions for the given selection of days.

    Parameters
    ----------
    ppu_dictionary : pd.DataFrame
        PPU configuration DataFrame (must include 'PPU_Name' and 'Location_Rank',
        and 'can_extract_from').
    selected_days : List[int]
        List of day indices (0-based, 0..364) to include in the precomputation.

    Returns
    -------
    dict
        Same output structure as before:
          - 'solar_prod_matrix': (num_timesteps, num_solar_ranks) or None
          - 'wind_prod_matrix': (num_timesteps, num_wind_ranks) or None
          - 'solar_rank_to_ppus': {rank: [(ppu_name, area_fraction), ...]}
          - 'wind_rank_to_ppus': {rank: [(ppu_name, turbine_fraction), ...]}
          - 'solar_ranks': sorted array of ranks
          - 'wind_ranks': sorted array of ranks
    """
    # Validate input
    if not isinstance(selected_days, (list, tuple, np.ndarray)):
        raise TypeError("selected_days must be a list/tuple/ndarray of day indices")

    # Build grouping maps from PPUs (same logic as before)
    solar_groups = {}
    solar_ppu_map = {}
    wind_groups = {}
    wind_ppu_map = {}

    for _, row in ppu_dictionary.iterrows():
        can_extract_from = row.get("can_extract_from", []) or []
        ppu_name = row['PPU_Name']
        location_rank = row.get('Location_Rank', np.nan)
        if pd.notna(location_rank):
            rank = int(location_rank)
            if 'Solar' in can_extract_from:
                area_m2 = row.get('area_m2', 100000)
                solar_groups[rank] = solar_groups.get(rank, 0.0) + float(area_m2)
                solar_ppu_map.setdefault(rank, []).append((ppu_name, float(area_m2)))
            if 'Wind' in can_extract_from:
                num_turbines = row.get('num_turbines', 5)
                wind_groups[rank] = wind_groups.get(rank, 0.0) + float(num_turbines)
                wind_ppu_map.setdefault(rank, []).append((ppu_name, float(num_turbines)))

    # Convert totals to fraction maps
    solar_rank_to_ppus = {r: [(n, a / solar_groups[r]) for n, a in solar_ppu_map[r]]
                          for r in solar_ppu_map}
    wind_rank_to_ppus = {r: [(n, t / wind_groups[r]) for n, t in wind_ppu_map[r]]
                         for r in wind_ppu_map}

    result = {
        'solar_prod_matrix': None,
        'wind_prod_matrix': None,
        'solar_rank_to_ppus': solar_rank_to_ppus,
        'wind_rank_to_ppus': wind_rank_to_ppus,
        'solar_ranks': np.array([]),
        'wind_ranks': np.array([])
    }

    # Compute number of timesteps we will produce (96 timesteps per day)
    DAY_STEPS = 96
    sel_days = list(selected_days)
    if len(sel_days) == 0:
        return result
    num_timesteps = len(sel_days) * DAY_STEPS

    # Get static data (may contain location dfs or other metadata)
    static_data = get_static_data_cache()

    # Precompute solar productions if any solar PPUs present
    if solar_groups:
        _init_solar_mapper('data/solar_incidence_hourly_2024.csv')
        if _SOLAR_MAPPER is not None:
            try:
                solar_ranks = np.array(sorted(solar_groups.keys()), dtype=np.int32)
                solar_areas = np.array([solar_groups[r] for r in solar_ranks], dtype=np.float32)
                # Map ranks -> column indices
                col_idxs = np.array([_SOLAR_MAPPER.col_for_rank(int(r)) for r in solar_ranks], dtype=np.int32)
                parts = []
                for d in sel_days:
                    s = int(d) * DAY_STEPS
                    e = s + DAY_STEPS
                    parts.append(_SOLAR_MAPPER.data[s:e, col_idxs])
                irr_matrix = np.vstack(parts)
                result['solar_prod_matrix'] = _solar_prod_batch(irr_matrix, solar_areas)
                result['solar_ranks'] = solar_ranks
                print(f"  Precomputed solar productions: {len(solar_ranks)} ranks, {num_timesteps} timesteps")
            except Exception as e:
                print(f"  Solar precomputation failed: {e}, will use fallback")

    # Precompute wind productions if any wind PPUs present
    if wind_groups:
        _init_wind_mapper('data/wind_incidence_hourly_2024.csv')
        if _WIND_MAPPER is not None:
            try:
                wind_ranks = np.array(sorted(wind_groups.keys()), dtype=np.int32)
                wind_nums = np.array([wind_groups[r] for r in wind_ranks], dtype=np.float32)
                col_idxs = np.array([_WIND_MAPPER.col_for_rank(int(r)) for r in wind_ranks], dtype=np.int32)
                parts = []
                for d in sel_days:
                    s = int(d) * DAY_STEPS
                    e = s + DAY_STEPS
                    parts.append(_WIND_MAPPER.data[s:e, col_idxs])
                wspd_matrix = np.vstack(parts)
                result['wind_prod_matrix'] = _wind_power_batch(wspd_matrix, wind_nums)
                result['wind_ranks'] = wind_ranks
                print(f"  Precomputed wind productions: {len(wind_ranks)} ranks, {num_timesteps} timesteps")
            except Exception as e:
                print(f"  Wind precomputation failed: {e}, will use fallback")

    return result


def calculate_solar_production(location_rank: int, area_m2: float, t: int) -> float:
    """
    Calculate solar production for a location at timestep t.

    Fast path: use singleton SolarMapper + Numba for direct numpy lookup.
    Fallback: original pandas-based nearest column search on solar_15min.

    Parameters:
        location_rank: Ranking position (1-based)
        area_m2: Solar panel area in square meters
        t: Timestep index
        solar_15min: Solar incidence data
        solar_ranking_df: Location rankings

    Returns:
        Production in MW
    """
    # Fast path using SolarMapper
    _init_solar_mapper('data/solar_incidence_hourly_2024.csv')
    static_data = get_static_data_cache('data')
    if not isinstance(static_data, dict):
        raise TypeError("There is a problem with your static variable")
    ranking_solar = static_data['solar_locations_df']
    if _SOLAR_MAPPER is not None:
        try: 
            lat = lon = None
            if len(ranking_solar) > 0:
                location_idx = (location_rank - 1) % len(ranking_solar)
                loc_row = ranking_solar.iloc[location_idx]
                lat = float(loc_row['latitude'])
                lon = float(loc_row['longitude'])
            if lat is not None and lon is not None:
                col_idx = _SOLAR_MAPPER.closest_column(lat, lon)
            else:
                col_idx = _SOLAR_MAPPER.col_for_rank(location_rank)
            irr = _SOLAR_MAPPER.irradiance(int(t), int(col_idx))
            return float(_solar_prod_fast(float(irr), float(area_m2)))
        except Exception:
            pass
    raise RuntimeError(f"SolarMapper is broken please fix")


    # # Fallback: original implementation using provided solar_15min
    # num_locations = len(solar_ranking_df)
    # if num_locations == 0:
    #     return 0.0
    # location_idx = (location_rank - 1) % num_locations
    # loc = solar_ranking_df.iloc[location_idx]
    # lat, lon = float(loc['latitude']), float(loc['longitude'])

    # closest_col = None
    # min_dist = float('inf')
    # for col in solar_15min.columns:
    #     dist = abs(float(col[0]) - lat) + abs(float(col[1]) - lon)
    #     if dist < min_dist:
    #         min_dist = dist
    #         closest_col = col

    # if closest_col is not None:
    #     incidence = solar_15min.iloc[t][closest_col]  # type: ignore
    #     incidence = incidence.item() if hasattr(incidence, 'item') else float(incidence)  # type: ignore
    # else:
    #     incidence = 0.0

    # production_MW = float(incidence) * float(area_m2) * 0.25 / 1000.0
    # return production_MW


def calculate_wind_production(location_rank: int, num_turbines: int, t: int) -> float:
    """
    Calculate wind production for a location at timestep t.

    Fast path: use singleton WindMapper + Numba power curve.
    Fallback: original nearest-column lookup on wind_15min and physics formula.

    Parameters:
        location_rank: Ranking position (1-based)
        num_turbines: Number of wind turbines
        t: Timestep index
        wind_15min: Wind incidence data
        wind_ranking_df: Location rankings

    Returns:
        Production in MW
    """
    # Fast path using WindMapper
    _init_wind_mapper('data/wind_incidence_hourly_2024.csv')
    static_data = get_static_data_cache('data')
    if not isinstance(static_data, dict):
        raise TypeError("There is a problem with your static variable")
    ranking_wind = static_data['wind_locations_df']
    if _WIND_MAPPER is not None:
        try:
            lat = lon = None
            if len(ranking_wind) > 0:
                location_idx = (location_rank - 1) % len(ranking_wind)
                loc_row = ranking_wind.iloc[location_idx]
                lat = float(loc_row['latitude'])
                lon = float(loc_row['longitude'])
            if lat is not None and lon is not None:
                col_idx = _WIND_MAPPER.closest_column(lat, lon)
            else:
                col_idx = _WIND_MAPPER.col_for_rank(location_rank)
            wspd = _WIND_MAPPER.speed(int(t), int(col_idx))
            return float(_wind_power_fast(float(wspd), int(num_turbines)))
        except Exception:
            pass
    raise RuntimeError(f"WindMapper is broken please fix")

    # # Fallback: original implementation using provided wind_15min
    # # Get location coordinates with bounds checking
    # num_locations = len(wind_ranking_df)
    # if num_locations == 0:
    #     return 0.0
    # location_idx = (location_rank - 1) % num_locations
    # loc = wind_ranking_df.iloc[location_idx]
    # lat, lon = float(loc['latitude']), float(loc['longitude'])

    # wind_speed = 0.0
    # if len(wind_15min.columns) > 0:
    #     lats = np.array([float(col[0]) for col in wind_15min.columns])
    #     lons = np.array([float(col[1]) for col in wind_15min.columns])
    #     distances = np.abs(lats - lat) + np.abs(lons - lon)
    #     closest_idx = int(np.argmin(distances))
    #     wind_speed = float(wind_15min.iloc[t, closest_idx])  # type: ignore

    # # Physics-based power with simple cut-in/out
    # rotor_diameter_m = 120.0
    # air_density = 1.225
    # swept_area = np.pi * (rotor_diameter_m / 2.0) ** 2
    # power_coefficient = 0.45
    # if wind_speed < 3.0 or wind_speed > 25.0:
    #     return 0.0
    # power_per_turbine_W = 0.5 * air_density * swept_area * (wind_speed ** 3) * power_coefficient
    # return (power_per_turbine_W / 1e6) * float(num_turbines)


def run_dispatch_simulation(scenario: Dict[str, Any], ppu_dictionary: pd.DataFrame,
                          raw_energy_storage: List[Dict], raw_energy_incidence: List[Dict],
                          hyperparams: Dict) -> Tuple[Dict, float]:
    """
    Run the main dispatch simulation loop with the new structure.

    Parameters:
        demand_15min: Demand time series
        spot_15min: Spot price time series
        ror_df: Run-of-river data
        solar_15min: Solar incidence data
        wind_15min: Wind incidence data
        ppu_dictionary: PPU configuration
        solar_ranking_df: Solar location rankings
        wind_ranking_df: Wind location rankings
        raw_energy_storage: Controllable storage systems
        raw_energy_incidence: Uncontrollable energy sources
        hyperparams: Simulation hyperparameters

    Returns:
        Tuple of (Technology_volume, phi_smoothed)
    """
    
    # Precompute renewable productions (MAJOR SPEEDUP)
    print("Precomputing renewable productions...")
    precomputed = precompute_renewable_productions(
        ppu_dictionary, scenario['selected_days']
    )
    
    # Initialize technology tracking
    Technology_volume = initialize_technology_tracking(ppu_dictionary)
    phi_smoothed = 0.0
    
    # Track how often the system is in deficit (overflow>+epsilon), surplus (overflow<-epsilon), or balanced
    overflow_count = 0
    surplus_count = 0
    balanced_count = 0
    
    # Iterate over the scenario-selected days and their 96 timesteps.
    sel_days = list(scenario.get('selected_days', []))
    DAY_STEPS = 96
    total_timesteps = len(sel_days) * DAY_STEPS

    # Load caches used inside the timestep loop (safely)
    cached_annual = get_annual_data_cache('data') or {}
    demand_15min = cached_annual.get('demand_15min') if isinstance(cached_annual, dict) else None

    # Create progress bar across all selected timesteps
    progress = tqdm(iterable=total_timesteps, desc="⚡ Dispatch Simulation",
                    unit="timestep", ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for day_idx, day in enumerate(sel_days):
        for sub in range(DAY_STEPS):
            # Global timestep index in the full-year series
            t_global = int(day) * DAY_STEPS + int(sub)
            progress.update(1)
            # Provide the precomputed row index for lookups (concatenated days)
            precomputed_row = day_idx * DAY_STEPS + sub
            raw_energy_incidence = update_raw_energy_incidence(
                ppu_dictionary,
                raw_energy_incidence,
                Technology_volume,
                t_global
            )

            # Get energy produced by incidence PPUs (stored in "Grid")
            grid_energy = 0.0
            for incidence_item in raw_energy_incidence:
                if incidence_item['storage'] == 'Grid':
                    # Get current value directly (already updated in update_raw_energy_incidence)
                    grid_energy = incidence_item.get('current_value', 0.0)
                    break

            # 4) Compute overflow = demand - energy in "Grid"
            if demand_15min is not None and 0 <= t_global < len(demand_15min):
                demand_MW = float(demand_15min.iloc[t_global])
            else:
                demand_MW = 0.0
            overflow_MW = demand_MW - grid_energy

            # Count state occurrence for this timestep
            if overflow_MW > hyperparams['epsilon']:
                overflow_count += 1
            elif overflow_MW < -hyperparams['epsilon']:
                surplus_count += 1
            else:
                balanced_count += 1
            # First compute the d_stor, u_dis, u_chg and m for each PPU
            ppu_indices = calculate_ppu_indices(
                ppu_dictionary, 
                raw_energy_storage, 
                overflow_MW, 
                phi_smoothed,
                t_global, 
                hyperparams
            )
            # Update EMA of shortfall for utility index
            phi_smoothed = exponential_moving_average(overflow_MW, phi_smoothed, hyperparams['ema_beta'])

            # Store cost indices in Technology_volume
            for ppu_name, indices in ppu_indices.items():
                if ppu_name not in Technology_volume:
                    Technology_volume[ppu_name] = {
                        'production': [],
                        'spot_bought': [],
                        'spot_sold': [],
                        'cost_indices': []
                    }
                Technology_volume[ppu_name]['cost_indices'].append((
                    t_global,
                    indices['d_stor'],
                    indices['u_dis'],
                    indices['u_chg'],
                    indices['m'],
                    indices.get('kappa_dis', 1.0),
                    indices.get('kappa_chg', 1.0)
                ))

            if t_global % 5000 == 0:
                print("[DEBUG] Storage state BEFORE handling:")
                for storage in raw_energy_storage:
                    print(f"  {storage['storage']}: {storage['current_value']:.2f} MW")

            # 4.1) If overflow > 0 (we need more energy): use 'Flex' PPUs
            if overflow_MW > hyperparams['epsilon']:
                raw_energy_storage = handle_energy_deficit(
                    overflow_MW, 
                    ppu_dictionary, 
                    raw_energy_storage, 
                    Technology_volume,
                    ppu_indices, 
                    t_global, 
                    hyperparams
                )
            # 4.2) If overflow < 0: use 'Store' PPUs
            elif overflow_MW < -hyperparams['epsilon']:
                surplus_MW = abs(overflow_MW)
                raw_energy_storage = handle_energy_surplus(
                    surplus_MW, 
                    ppu_dictionary, 
                    raw_energy_storage, 
                    Technology_volume,
                    ppu_indices, 
                    t_global, 
                    hyperparams
                )
    # Attach pipeline state statistics for later reporting
    Technology_volume['__pipeline_stats__'] = {
        'overflow_count': overflow_count,
        'surplus_count': surplus_count,
        'balanced_count': balanced_count,
        'total_timesteps': total_timesteps
    }

    print(f"  ✓ Simulation complete: {total_timesteps:,} timesteps")
    print(f"    - Overflow (deficit) steps: {overflow_count:,}")
    print(f"    - Surplus steps:            {surplus_count:,}")
    print(f"    - Balanced steps:           {balanced_count:,}")
    return Technology_volume, phi_smoothed


def compute_cost_breakdown(ppu_dictionnary: pd.DataFrame, 
                        technology_volume: Dict[str, Dict], 
                        hyperparams: Dict) -> Dict[str, Dict]:
    """
    Compute retroactive cost breakdown by technology type.

    Parameters:
        technology_volume: Production and transaction tracking
        spot_15min: Spot price time series
        hyperparams: Simulation parameters
        data_dir: Data directory path

    Returns:
        Cost summary dictionary
    """
    # Derive technology-specific costs from the provided PPU dictionary.
    # Prefer `Cost_CHF_per_kWh` (CHF/kWh). If only `Cost_CHF_per_Quarter_Hour`
    # is available, convert to CHF/kWh by dividing by the timestep hours.
    tech_costs: Dict[str, float] = {}
    try:
        if isinstance(ppu_dictionnary, pd.DataFrame) and 'PPU_Name' in ppu_dictionnary.columns:
            grouped = ppu_dictionnary.groupby('PPU_Name')
            for name, group in grouped:
                val = None
                if 'Cost_CHF_per_kWh' in group.columns and group['Cost_CHF_per_kWh'].notna().any():
                    val = float(group['Cost_CHF_per_kWh'].astype(float).mean())
                elif 'Cost_CHF_per_Quarter_Hour' in group.columns and group['Cost_CHF_per_Quarter_Hour'].notna().any():
                    # convert CHF per quarter-hour -> CHF per kWh using timestep hours
                    timestep_h = float(hyperparams.get('timestep_hours', 0.25)) if isinstance(hyperparams, dict) else 0.25
                    # Cost_CHF_per_Quarter_Hour was stored as cost_per_kwh * delta_t in add_ppu_to_dictionary
                    val = float(group['Cost_CHF_per_Quarter_Hour'].astype(float).mean()) / max(timestep_h, 1e-9)
                else:
                    val = None

                if val is not None:
                    tech_costs[str(name)] = float(val)
    except Exception:
        # If anything goes wrong, leave tech_costs empty and fall back to defaults below
        tech_costs = {}

    # Load spot price series from the annual cache for accurate per-timestep pricing
    annual_cache = None
    try:
        annual_cache = get_annual_data_cache()
    except Exception:
        annual_cache = None

    spot_series = None
    if isinstance(annual_cache, dict):
        spot_series = annual_cache.get('spot_15min')

    Cost_summary = {}
    for tech_type, metrics in technology_volume.items():
        # Skip non-PPU entries (e.g., attached pipeline stats) or malformed entries
        if isinstance(tech_type, str) and tech_type.startswith('__'):
            continue
        if not isinstance(metrics, dict):
            continue
        if 'production' not in metrics or 'spot_bought' not in metrics or 'spot_sold' not in metrics:
            continue
        production_cost = 0.0
        spot_buy_cost = 0.0
        spot_sell_revenue = 0.0
        # Detailed per-timestep breakdowns (list of dicts)
        spot_buy_breakdown = []
        spot_sell_breakdown = []

        # Production cost: prefer cost from PPU dictionary, else sensible fallback
        cost_per_kwh = tech_costs.get(tech_type)
        if cost_per_kwh is None:
            # Try to lookup single-row match in the DataFrame (robust fallback)
            try:
                if isinstance(ppu_dictionnary, pd.DataFrame) and 'PPU_Name' in ppu_dictionnary.columns:
                    match = ppu_dictionnary[ppu_dictionnary['PPU_Name'] == tech_type]
                    if not match.empty:
                        if 'Cost_CHF_per_kWh' in match.columns and match['Cost_CHF_per_kWh'].notna().any():
                            cost_per_kwh = float(match['Cost_CHF_per_kWh'].astype(float).mean())
                        elif 'Cost_CHF_per_Quarter_Hour' in match.columns and match['Cost_CHF_per_Quarter_Hour'].notna().any():
                            timestep_h = float(hyperparams.get('timestep_hours', 0.25)) if isinstance(hyperparams, dict) else 0.25
                            cost_per_kwh = float(match['Cost_CHF_per_Quarter_Hour'].astype(float).mean()) / max(timestep_h, 1e-9)
            except Exception:
                cost_per_kwh = None

        if cost_per_kwh is None:
            # Final fallback: use a conservative default from hyperparams or constant
            cost_per_kwh = float(hyperparams.get('wood_price_chf_per_kwh', 0.095))
        for (t, vol_MW) in metrics['production']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            production_cost += energy_MWh * cost_per_kwh * 1000  # Convert to kWh

        # Spot market transactions
        for (t, vol_MW) in metrics['spot_bought']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            spot_price = 0.0
            try:
                if spot_series is not None and 0 <= int(t) < len(spot_series):
                    spot_price = float(spot_series.iloc[int(t)])
            except Exception:
                spot_price = 0.0
            cost_chf = energy_MWh * spot_price
            spot_buy_cost += cost_chf
            spot_buy_breakdown.append({'t': int(t), 'energy_MWh': float(energy_MWh), 'spot_price': float(spot_price), 'cost_CHF': float(cost_chf)})

        for (t, vol_MW) in metrics['spot_sold']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            spot_price = 0.0
            try:
                if spot_series is not None and 0 <= int(t) < len(spot_series):
                    spot_price = float(spot_series.iloc[int(t)])
            except Exception:
                spot_price = 0.0
            revenue_chf = energy_MWh * spot_price
            spot_sell_revenue += revenue_chf
            spot_sell_breakdown.append({'t': int(t), 'energy_MWh': float(energy_MWh), 'spot_price': float(spot_price), 'revenue_CHF': float(revenue_chf)})

        Cost_summary[tech_type] = {
            'production_cost_CHF': production_cost,
            'spot_buy_cost_CHF': spot_buy_cost,
            'spot_sell_revenue_CHF': spot_sell_revenue,
            'net_cost_CHF': production_cost + spot_buy_cost - spot_sell_revenue,
            'spot_buy_breakdown': spot_buy_breakdown,
            'spot_sell_breakdown': spot_sell_breakdown
        }

    return Cost_summary


def compute_portfolio_metrics(cost_summary: Dict[str, Dict], 
                            hyperparams: Dict) -> Dict[str, float]:
    """
    Compute portfolio performance metrics.

    Parameters:
        cost_summary: Cost breakdown by technology
        spot_15min: Spot price time series
        demand_15min: Demand time series
        hyperparams: Simulation parameters

    Returns:
        Portfolio metrics dictionary
    """
    
    total_net_cost = sum(cost_summary[tech]['net_cost_CHF'] for tech in cost_summary)

    # Load demand and spot from the global annual cache if available
    annual_cache = None
    try:
        annual_cache = get_annual_data_cache()
    except Exception:
        annual_cache = None

    demand_15min = None
    spot_15min = None
    if isinstance(annual_cache, dict):
        demand_15min = annual_cache.get('demand_15min')
        spot_15min = annual_cache.get('spot_15min')

    # Compute spot-only baseline safely
    spot_only_cost = 0.0
    num_timesteps = 0
    if demand_15min is not None and spot_15min is not None:
        try:
            num_timesteps = min(len(demand_15min), len(spot_15min))
        except Exception:
            num_timesteps = 0

    for t in range(num_timesteps):
        try:
            demand_MWh = float(demand_15min.iloc[t]) * float(hyperparams.get('timestep_hours', 0.25))
            spot_only_cost += demand_MWh * float(spot_15min.iloc[t])
        except Exception:
            continue

    savings = spot_only_cost - total_net_cost
    margin_pct = (savings / spot_only_cost) * 100 if spot_only_cost != 0 else 0.0

    # Calculate price volatility (coefficient of variation) safely
    price_volatility_pct = 0.0
    try:
        if spot_15min is not None and getattr(spot_15min, 'mean', None) is not None and spot_15min.mean() != 0:
            price_volatility_pct = (float(spot_15min.std()) / float(spot_15min.mean())) * 100
    except Exception:
        price_volatility_pct = 0.0

    return {
        'total_cost_CHF': total_net_cost,
        'spot_only_cost_CHF': spot_only_cost,
        'savings_CHF': savings,
        'margin_pct': margin_pct,
        'volatility_pct': float(price_volatility_pct),
        'num_timesteps': num_timesteps
    }


def run_complete_pipeline(scenario : Dict[str, Any], ppu_counts: Dict[str, int], raw_energy_storage: List[Dict], 
                        raw_energy_incidence: List[Dict], 
                        hyperparams: Optional[Dict] = None) -> Dict:
    """
    Run the complete energy dispatch pipeline.

    Parameters:
        ppu_counts: Dictionary mapping PPU type names to number of instances (e.g., {'PV': 5, 'HYD_ROR': 3})
        raw_energy_storage: List of storage dictionaries with capacity and tracking info
        raw_energy_incidence: List of incidence dictionaries with availability tracking
        data_dir: Directory containing data files
        hyperparams: Optional hyperparameters override
        static_data: Optional pre-loaded static data (PPU constructs, costs, locations). If None, loads from disk with caching.

    Returns:
        Complete pipeline results dictionary
    """
    # Initialize history tracking for raw_energy_storage and raw_energy_incidence
    for storage in raw_energy_storage:
        if 'history' not in storage:
            storage['history'] = []
    
    for incidence in raw_energy_incidence:
        if 'history' not in incidence:
            incidence['history'] = []

    hyperparams = provide_hyper_parameters(hyperparams=hyperparams)

    ppu_dictionary = build_ppu_dictionary(
        ppu_counts, raw_energy_storage, raw_energy_incidence
    )
    print("We build the PPU dictionnary for this run")
    technology_volume, phi_smoothed = run_dispatch_simulation(
        scenario = scenario, ppu_dictionary = ppu_dictionary, 
        raw_energy_storage = raw_energy_storage, 
        raw_energy_incidence = raw_energy_incidence, 
        hyperparams = hyperparams
    )
    print("We ran the dispatch simulation")
    cost_summary = compute_cost_breakdown(
                                        ppu_dictionary, 
                                        technology_volume,
                                        hyperparams
                                        )
    portfolio_metrics = compute_portfolio_metrics(
                                            cost_summary, 
                                            hyperparams
                                            )

    print("\n" + "=" * 80)
    print("PORTFOLIO RESULT")
    print("=" * 80)
    print(f"  X-axis (Price Volatility): {portfolio_metrics['volatility_pct']:.2f}%")
    print(f"  Y-axis (Savings Margin): {portfolio_metrics['margin_pct']:.2f}%")
    print("=" * 80)
    
    # Print pipeline state counts if available
    pipeline_stats = technology_volume.get('__pipeline_stats__', {})
    if pipeline_stats:
        print("PIPELINE STATE COUNTS")
        print("-" * 80)
        print(f"  Overflow (deficit) steps: {pipeline_stats.get('overflow_count', 0):,}")
        print(f"  Surplus steps:            {pipeline_stats.get('surplus_count', 0):,}")
        print(f"  Balanced steps:           {pipeline_stats.get('balanced_count', 0):,}")
        print("=" * 80)

    # Compile complete results
    results = {
        'portfolio_result': {
            'x_volatility_pct': portfolio_metrics['volatility_pct'],
            'y_margin_pct': portfolio_metrics['margin_pct'],
            'total_cost_CHF': portfolio_metrics['total_cost_CHF'],
            'spot_only_cost_CHF': portfolio_metrics['spot_only_cost_CHF'],
            'savings_CHF': portfolio_metrics['savings_CHF'],
            'num_timesteps': portfolio_metrics['num_timesteps']
        },
        'cost_summary': cost_summary,
        'technology_volume': technology_volume,
        'pipeline_stats': technology_volume.get('__pipeline_stats__', {}),
        'hyperparams': hyperparams,
        'data_shapes': {
            'demand': len(demand_15min),
            'spot': len(spot_15min),
            'ror': len(ror_df),
            'solar': len(solar_15min),
            'wind': len(wind_15min)
        }
    }

    print("\n✓ PIPELINE EXECUTION COMPLETE")
    return results

def provide_hyper_parameters(hyperparams: Optional[Dict] = None) -> Dict:
    defaults = {
        'alpha_d': 0.5,
        'alpha_u': 5000.0,
        'alpha_m': 5.0,
        'weight_spread': 1.0,
        'weight_volatility': 1.0,
        'volatility_scale': 30.0,
        'epsilon': 1e-6,
        'timestep_hours': 0.25,
        'ema_beta': 0.2,
        'horizons': ['1d', '3d', '7d', '30d'],
        'wood_price_chf_per_kwh': 0.095,
        'palm_oil_price_chf_per_kwh': 0.070,
        'spot_unit': 'CHF_per_MWh',
        'ppu_cost_unit': 'CHF_per_KWh',
        'cost_column': 'cost',
        'use_efficiency': True,
        'beta_c': 0.12,
        'auto_beta_c': True,
        'beta_c_pctl': 90,
        'beta_c_min': 0.03,
        'beta_c_max': 0.30,
        'stor_deadband': 0.05,
        'default_target_soc': 0.6
    }

    if not hyperparams:
        return dict(defaults)

    merged = dict(defaults)
    for k in defaults:
        v = hyperparams.get(k)
        # treat missing, None, empty string, and empty container as "not provided"
        if v is None or v == "" or (hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == 0):
            continue
        merged[k] = v

    return merged
    


def build_ppu_dictionary(ppu_counts, raw_energy_storage, raw_energy_incidence):
    static_data = get_static_data_cache()
    if static_data is None:
        # Try to load static data from disk as a fallback
        try:
            static_data = load_static_data()
        except Exception as e:
            raise RuntimeError(f"Failed to load static data: {e}, make sure it is loaded appropriately.")
    if static_data is None:
        raise RuntimeError("Static data could not be loaded; ensure the data directory contains the required files and call load_cache_data() if needed.")

    ppu_constructs_df = static_data['ppu_constructs_df']
    cost_df = static_data['cost_df']
    solar_locations_df = static_data['solar_locations_df']
    wind_locations_df = static_data['wind_locations_df']

    ppu_dict = initialize_ppu_dictionary()
    for ppu_type, count in ppu_counts.items():
        for _ in range(count):
            ppu_dict = add_ppu_to_dictionary(
                ppu_dict,
                ppu_type,
                ppu_constructs_df,
                cost_df,
                solar_locations_df,
                wind_locations_df,
                raw_energy_storage=raw_energy_storage,
                raw_energy_incidence=raw_energy_incidence
            )
    scale_storage_capacities_by_unit_counts(ppu_dict, raw_energy_storage, do_not_scale=['Lake'])
    return ppu_dict


def update_raw_energy_incidence(ppu_dictionary : pd.DataFrame,
                                raw_energy_incidence: List[Dict], 
                                Technology_volume: Dict, 
                                t: int
                                ) -> List[Dict]:
    """
    Update raw_energy_incidence by simulating PPU extractions from each storage.
    
    For each storage:
    1. Find PPUs that can extract from this storage
    2. Calculate extraction for each PPU until max limit per PPU or all energy extracted
    3. Set current_value to total extracted energy
    4. Record extractions in Technology_volume
    
    Parameters:
        raw_energy_incidence: List of incidence dictionaries
        Technology_volume: Technology tracking dictionary
        t: Timestep index

    Returns:
        Updated raw_energy_incidence list
    """
    # Load annual/static caches for incidence data and rankings
    annual = get_annual_data_cache('data') or {}
    static = get_static_data_cache('data') or {}

    solar_15min = annual.get('solar_15min') if isinstance(annual, dict) else None
    wind_15min = annual.get('wind_15min') if isinstance(annual, dict) else None
    ror_df = annual.get('ror_df') if isinstance(annual, dict) else None

    solar_ranking_df = static.get('solar_locations_df') if isinstance(static, dict) else None
    wind_ranking_df = static.get('wind_locations_df') if isinstance(static, dict) else None

    # Prepare accumulators
    total_solar = 0.0
    total_wind = 0.0
    total_river = 0.0
    total_wood = 0.0

    # Helper to ensure Technology_volume entries exist
    def ensure_tech(ppu_name: str) -> None:
        if ppu_name not in Technology_volume:
            Technology_volume[ppu_name] = {
                'production': [],
                'spot_bought': [],
                'spot_sold': [],
                'cost_indices': []
            }

    # Process each incidence storage independently (switch/case)
    for incidence_item in raw_energy_incidence:
        storage = incidence_item.get('storage')
        if storage == 'Solar':
            # Sum production from all PPUs that extract from Solar
            acc = 0.0
            _init_solar_mapper('data/solar_incidence_hourly_2024.csv')
            for _, ppu in ppu_dictionary.iterrows():
                can_extract = ppu.get('can_extract_from', []) or []
                if 'Solar' not in can_extract:
                    continue
                ppu_name = ppu['PPU_Name']
                ensure_tech(ppu_name)
                loc_rank = ppu.get('Location_Rank', np.nan)
                area_m2 = float(ppu.get('area_m2', 100000))
                prod = 0.0
                if pd.notna(loc_rank):
                    try:
                        if _SOLAR_MAPPER is not None:
                            col = _SOLAR_MAPPER.col_for_rank(int(loc_rank))
                            irr = _SOLAR_MAPPER.irradiance(int(t), int(col))
                            prod = float(_solar_prod_fast(float(irr), float(area_m2)))
                        else:
                            prod = calculate_solar_production(int(loc_rank), area_m2, t)
                    except Exception:
                        prod = 0.0
                Technology_volume[ppu_name]['production'].append((t, prod))
                acc += prod
            incidence_item['current_value'] = acc
            incidence_item.setdefault('history', []).append((t, acc))
            total_solar = acc

        elif storage == 'Wind':
            acc = 0.0
            _init_wind_mapper('data/wind_incidence_hourly_2024.csv')
            for _, ppu in ppu_dictionary.iterrows():
                can_extract = ppu.get('can_extract_from', []) or []
                if 'Wind' not in can_extract:
                    continue
                ppu_name = ppu['PPU_Name']
                ensure_tech(ppu_name)
                loc_rank = ppu.get('Location_Rank', np.nan)
                num_turbs = int(ppu.get('num_turbines', 5))
                prod = 0.0
                if pd.notna(loc_rank):
                    try:
                        if _WIND_MAPPER is not None:
                            col = _WIND_MAPPER.col_for_rank(int(loc_rank))
                            wspd = _WIND_MAPPER.speed(int(t), int(col))
                            prod = float(_wind_power_fast(float(wspd), int(num_turbs)))
                        else:
                            prod = calculate_wind_production(int(loc_rank), num_turbs, t)
                    except Exception:
                        prod = 0.0
                Technology_volume[ppu_name]['production'].append((t, prod))
                acc += prod
            incidence_item['current_value'] = acc
            incidence_item.setdefault('history', []).append((t, acc))
            total_wind = acc

        elif storage == 'River':
            # Sequential extraction like previous logic
            available = 0.0
            try:
                if ror_df is not None:
                    available = float(ror_df.iloc[t].get('RoR_MW', 12.0))
            except Exception:
                available = 0.0
            acc = 0.0
            for _, ppu in ppu_dictionary.iterrows():
                can_extract = ppu.get('can_extract_from', []) or []
                if 'River' not in can_extract:
                    continue
                ppu_name = ppu['PPU_Name']
                ensure_tech(ppu_name)
                if available > 1000:
                    extraction = 1000.0
                    available -= 1000.0
                else:
                    extraction = float(available)
                    available = 0.0
                Technology_volume[ppu_name]['production'].append((t, extraction))
                acc += extraction
            incidence_item['current_value'] = acc
            incidence_item.setdefault('history', []).append((t, acc))
            total_river = acc

        elif storage == 'Wood':
            available = 500.0
            acc = 0.0
            for _, ppu in ppu_dictionary.iterrows():
                can_extract = ppu.get('can_extract_from', []) or []
                if 'Wood' not in can_extract:
                    continue
                ppu_name = ppu['PPU_Name']
                ensure_tech(ppu_name)
                if available > 500:
                    extraction = 500.0
                    available -= 500.0
                else:
                    extraction = float(available)
                    available = 0.0
                Technology_volume[ppu_name]['production'].append((t, extraction))
                acc += extraction
            incidence_item['current_value'] = acc
            incidence_item.setdefault('history', []).append((t, acc))
            total_wood = acc

        else:
            # For Grid or unknown storages, leave as-is (Grid will be set later)
            continue

    # After all storages processed, set Grid to total extracted energy
    total_energy = total_solar + total_wind + total_river + total_wood
    for incidence_item in raw_energy_incidence:
        if incidence_item.get('storage') == 'Grid':
            incidence_item['current_value'] = total_energy
            incidence_item.setdefault('history', []).append((t, total_energy))
    return raw_energy_incidence


def calculate_incidence_production(ppu_dictionary: pd.DataFrame, raw_energy_incidence: List[Dict],
                                 Technology_volume: Dict, t: int, solar_ranking_df: pd.DataFrame,
                                 wind_ranking_df: pd.DataFrame, hyperparams: Dict) -> List[Dict]:
    """
    Calculate total incidence production from extracted energy in raw_energy_incidence.
    
    Since update_raw_energy_incidence now handles extraction simulation and recording,
    this function just sums the extracted energy and puts it in Grid.
    """
    total_production_MW = 0.0
    
    # Sum extracted energy from all incidence storages
    for incidence_item in raw_energy_incidence:
        if incidence_item['storage'] != 'Grid':  # Grid is the output, not input
            total_production_MW += incidence_item['current_value']
    
    # Update raw_energy_incidence: set Grid to total production, others remain as extracted amounts
    for incidence_item in raw_energy_incidence:
        if incidence_item['storage'] == 'Grid':
            incidence_item['current_value'] = total_production_MW
        # Other storages keep their extracted amounts
        
        # Update history
        incidence_item['history'].append((t, incidence_item['current_value']))
    
    return raw_energy_incidence


def calculate_ppu_indices(ppu_dictionary: pd.DataFrame, raw_energy_storage: List[Dict],
                        overflow_MW: float, phi_smoothed: float, 
                        t: int, hyperparams: Dict) -> Dict[str, Dict]:
    """
    Calculate d_stor, u_dis, u_chg and m (monetary volatility-aware) indices for each PPU.
    Uses RELATIVE VALUES throughout to avoid dimension issues and stuck costs.

    Parameters:
        ppu_dictionary: PPU configuration
        raw_energy_storage: Storage systems
        overflow_MW: Current overflow (demand - supply)
        phi_smoothed: Smoothed overflow
        t: Current timestep
                         overflow_MW: float, phi_smoothed: float, 
                         t: int, hyperparams: Dict) -> Dict[str, Dict]:
    Returns:
        Dictionary mapping PPU names to their indices and costs
    """
    # concise, robust implementation
    ppu_indices: Dict[str, Dict] = {}
    alpha_u = float(hyperparams.get('alpha_u', 5000.0))
    alpha_m = float(hyperparams.get('alpha_m', 5.0))
    deadband = float(hyperparams.get('stor_deadband', 0.05))

    horizon_steps = {'1d': 96, '3d': 288, '7d': 672, '30d': 2880}
    horizons_30d = ['1d', '3d', '7d', '30d']

    # Load spot series from annual cache (if available)
    spot_series = None
    try:
        spot_series = get_annual_data_cache().get('spot_15min')
    except Exception:
        spot_series = None

    spot_now = 0.0
    if spot_series is not None and len(spot_series) > 0 and t < len(spot_series):
        try:
            spot_now = float(spot_series.iloc[t])
        except Exception:
            spot_now = float(spot_series.iloc[min(t, len(spot_series)-1)])

    # helper for future-window average safely
    def future_avg(start: int, steps: int) -> float:
        if spot_series is None or len(spot_series) == 0:
            return 0.0
        end = min(start + steps, len(spot_series))
        if start >= end:
            return float(spot_series.iloc[-1]) if len(spot_series) > 0 else 0.0
        try:
            return float(spot_series.iloc[start:end].mean())
        except Exception:
            return 0.0

    for _, ppu_row in ppu_dictionary.iterrows():
        ppu_name = str(ppu_row['PPU_Name'])
        ppu_category = str(ppu_row.get('PPU_Category', ''))

        # defaults
        d_stor = 0.0
        u_dis = 0.0
        u_chg = 0.0
        m_idx = 0.0

        # disposition for storage: normalized deviation from target SoC
        if ppu_category == 'Storage':
            can_extract = ppu_row.get('can_extract_from', []) or []
            if can_extract:
                storage_name = can_extract[0]
                storage_item = next((s for s in raw_energy_storage if s.get('storage') == storage_name), None)
                if storage_item:
                    max_v = storage_item.get('value', 1.0) or 1.0
                    cur = storage_item.get('current_value', 0.0)
                    target = storage_item.get('target_SoC', 0.6)
                    soc = (cur / max_v) if max_v > 0 else 0.0
                    d_stor = float(np.tanh((soc - target) / max(deadband, 1e-6)))

        # utilities from overflow (phi_smoothed)
        rel_over = (phi_smoothed / alpha_u) if alpha_u != 0 else 0.0
        u_dis = float(np.tanh(rel_over))
        u_chg = float(np.tanh(abs(rel_over))) if phi_smoothed < 0 else 0.0

        # monetary index for storage PPUs based on future price spreads and variance
        if ppu_category == 'Storage':
            efficiency = float(ppu_row.get('Efficiency', 1.0) or 1.0)
            spreads = []
            future_vals = []
            for h in horizons_30d:
                steps = horizon_steps[h]
                favg = future_avg(t, steps)
                future_vals.append((favg - spot_now) / max(abs(spot_now), 1e-6))
                # spread = spot_now - efficiency * future_price (relative)
                spreads.append((spot_now - efficiency * favg) / max(abs(spot_now), 1e-6))

            S_rel = float(np.mean(spreads)) if len(spreads) > 0 else 0.0
            V_rel = float(np.var(future_vals)) if len(future_vals) > 1 else 0.0
            weight_spread = float(hyperparams.get('weight_spread', 1.0))
            weight_vol = float(hyperparams.get('weight_volatility', 1.0))
            X = weight_spread * S_rel - weight_vol * V_rel
            m_idx = float(np.tanh(X / max(alpha_m, 1e-6)))

        # benefits and costs (relative in [-1,1])
        B_dis = (d_stor + u_dis + m_idx) / 3.0
        B_chg = (-d_stor + u_chg - m_idx) / 3.0
        kappa_dis = 1.0 - B_dis
        kappa_chg = 1.0 - B_chg

        ppu_indices[ppu_name] = {
            'd_stor': float(d_stor),
            'u_dis': float(u_dis),
            'u_chg': float(u_chg),
            'm': float(m_idx),
            'B_dis': float(B_dis),
            'B_chg': float(B_chg),
            'kappa_dis': float(kappa_dis),
            'kappa_chg': float(kappa_chg)
        }
    return ppu_indices


def handle_energy_deficit(deficit_MW: float, ppu_dictionary: pd.DataFrame,
                         raw_energy_storage: List[Dict], Technology_volume: Dict,
                         ppu_indices: Dict, t: int,
                         hyperparams: Dict) -> List[Dict]:
    """
    Handle energy deficit using 'Flex' PPUs that extract from raw_energy_storage.

    Parameters:
        deficit_MW: Energy deficit to cover
        ppu_dictionary: PPU configuration
        raw_energy_storage: Storage systems
        Technology_volume: Technology tracking
        ppu_indices: PPU indices for cost calculation
        spot_price: Current spot price
        t: Timestep index
        hyperparams: Simulation hyperparameters

    Returns:
        Updated raw_energy_storage
    """
    
    # Find Flex PPUs
    flex_ppus = []
    for _, ppu_row in ppu_dictionary.iterrows():
        if str(ppu_row['PPU_Extract']) == 'Flex':
            flex_ppus.append(ppu_row)

    if not flex_ppus:
        print(f"Why are there no PPU left to produce energy?")
        # No Flex PPUs available, buy from spot market - but doesn't seem to be the case 
        tech_count = len(Technology_volume)
        spot_buy_per_tech = deficit_MW / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_bought'].append((t, spot_buy_per_tech))
        return raw_energy_storage

    # Calculate discharge benefits and priorities
    discharge_benefits = []
    B_dis_total = 0.0

    for ppu_row in flex_ppus:
        ppu_name = str(ppu_row['PPU_Name'])
        indices = ppu_indices.get(ppu_name, {
            'd_stor': 0, 'u_dis': 0, 'u_chg': 0, 'm': 0, 
            'B_dis': 0.0, 'B_chg': 0.0,
            'kappa_dis': 1.0, 'kappa_chg': 1.0
        })
        
        # Use pre-calculated discharge benefit from calculate_ppu_indices
        # IMPORTANT: Only use POSITIVE discharge benefits (PPUs that want to discharge)
        B_dis = max(0.0, indices['B_dis'])
        
        discharge_benefits.append({
            'ppu_name': ppu_name,
            'B_dis': B_dis,
            'ppu_row': ppu_row
        })

        B_dis_total += B_dis

    for each_flex_ppu in discharge_benefits:
        allocation = each_flex_ppu['B_dis'] / max(B_dis_total, hyperparams['epsilon'])
        efficiency = each_flex_ppu['ppu_row'].get('Chain_Efficiency', 1.0)
        target_discharge_MW = deficit_MW * allocation / efficiency  # Divide by efficiency for extraction
        spot_buy = 0.0
        
        # Find available capacity in storage linked to each_flex_ppu 
        can_extract_from = each_flex_ppu['ppu_row'].get('can_extract_from', [])
        
        # Find capacity stored in can_extract_from storages
        available_capacity_MW = sum(
            storage_item['current_value'] 
            for storage_item in raw_energy_storage 
            if storage_item['storage'] in can_extract_from
        )
        # Limit by the smallest: 1 GW PPU capacity, available storage, and target
        actual_discharge_MW = min(target_discharge_MW, available_capacity_MW, 1000 / efficiency)
        
        if actual_discharge_MW == target_discharge_MW:
            for storage_name in can_extract_from:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item and storage_item['current_value'] > 0:
                    # Calculate proportion of this storage's available capacity
                    if available_capacity_MW > 0:
                        fraction = storage_item['current_value'] / available_capacity_MW
                        discharge_amount = actual_discharge_MW * fraction
                        old_value = storage_item['current_value']
                        storage_item['current_value'] -= discharge_amount
                        storage_item['history'].append((t, -discharge_amount))  # Negative for discharge
                        
        elif actual_discharge_MW == available_capacity_MW:
            # Storage capacity is the limiting factor - empty storage completely, buy rest
            spot_buy = target_discharge_MW - actual_discharge_MW
            for storage_name in can_extract_from:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item:
                    discharged = storage_item['current_value']
                    storage_item['current_value'] = 0
                    storage_item['history'].append((t, -discharged))  # Negative for discharge
                    
        else:
            # PPU capacity is the limiting factor - partial discharge
            spot_buy = target_discharge_MW - actual_discharge_MW
            for storage_name in can_extract_from:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item and storage_item['current_value'] > 0:
                    if available_capacity_MW > 0:
                        fraction = storage_item['current_value'] / available_capacity_MW
                        discharge_amount = actual_discharge_MW * fraction
                        storage_item['current_value'] -= discharge_amount
                        storage_item['history'].append((t, -discharge_amount))  # Negative for discharge
        
        # Record production
        Technology_volume[each_flex_ppu['ppu_name']]['production'].append((t, actual_discharge_MW))
        # Record spot market buy if needed
        Technology_volume[each_flex_ppu['ppu_name']]['spot_bought'].append((t, spot_buy))

    return raw_energy_storage


def handle_energy_surplus(surplus_MW: float, ppu_dictionary: pd.DataFrame,
                        raw_energy_storage: List[Dict], Technology_volume: Dict,
                        ppu_indices: Dict, t: int,
                        hyperparams: Dict) -> List[Dict]:
    """
    Handle energy surplus using 'Store' PPUs that store in raw_energy_storage.
    Takes energy from grid and pushes it into storage systems.

    Parameters:
        surplus_MW: Energy surplus to store
        ppu_dictionary: PPU configuration
        raw_energy_storage: Storage systems
        Technology_volume: Technology tracking
        ppu_indices: PPU indices for cost calculation
        spot_price: Current spot price
        t: Timestep index
        hyperparams: Simulation hyperparameters

    Returns:
        Updated raw_energy_storage
    """
    
    # Find Store PPUs
    store_ppus = []
    for _, ppu_row in ppu_dictionary.iterrows():
        if str(ppu_row['PPU_Extract']) == 'Store':
            store_ppus.append(ppu_row)

    if not store_ppus:
        print(f"Warning: No Store PPUs available at t={t}")
        # No Store PPUs available, sell to spot market
        tech_count = len(Technology_volume)
        spot_sell_per_tech = surplus_MW / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_sold'].append((t, spot_sell_per_tech))
        return raw_energy_storage

    # Calculate charge benefits and priorities
    charge_benefits = []
    B_chg_total = 0.0

    for ppu_row in store_ppus:
        ppu_name = str(ppu_row['PPU_Name'])
        indices = ppu_indices.get(ppu_name, {
            'd_stor': 0, 'u_dis': 0, 'u_chg': 0, 'm': 0, 
            'B_dis': 0.0, 'B_chg': 0.0,
            'kappa_dis': 1.0, 'kappa_chg': 1.0
        })
        
        # Use pre-calculated charge benefit from calculate_ppu_indices
        # IMPORTANT: Only use POSITIVE charge benefits (PPUs that want to charge)
        B_chg = max(0.0, indices['B_chg'])
        
        charge_benefits.append({
            'ppu_name': ppu_name,
            'B_chg': B_chg,
            'ppu_row': ppu_row
        })

        B_chg_total += B_chg

    # Handle case where all benefits are 0 (equal allocation)
    if B_chg_total == 0:
        weight_per_ppu = 1.0 / len(charge_benefits)
        for benefit_info in charge_benefits:
            benefit_info['weight'] = weight_per_ppu
    else:
        # Proportional allocation based on benefits
        for benefit_info in charge_benefits:
            benefit_info['weight'] = benefit_info['B_chg'] / B_chg_total

    # Special PPUs that convert electricity → commodity → biooil (unlimited flow)
    special_biooil_ppus = ['BIO_OIL_FROM_WOOD', 'BIO_OIL_FROM_PALM', 'PALM_STORE_IMPORT']
    
    for each_store_ppu in charge_benefits:
        ppu_name = each_store_ppu['ppu_name']
        allocation = each_store_ppu['B_chg'] / max(B_chg_total, hyperparams['epsilon'])
        efficiency = each_store_ppu['ppu_row'].get('Chain_Efficiency', 1.0)
        
        # Check if this is a special biooil conversion PPU
        if ppu_name in special_biooil_ppus:
            # SPECIAL LOGIC: Sell electricity on spot, buy commodity, convert to biooil
            # 1. Allocated energy (MW) to this PPU
            allocated_energy_MW = surplus_MW * allocation
            
            # 2. Convert to energy (MWh for this 15-min timestep)
            energy_MWh = allocated_energy_MW * hyperparams['timestep_hours']
            
            # 3. Sell on spot market to get CHF
            revenue_CHF = energy_MWh * spot_price
            
            # 4. Determine commodity type and price
            can_extract_from = each_store_ppu['ppu_row'].get('can_extract_from', [])
            if 'Wood' in can_extract_from:
                commodity_price_chf_per_kwh = hyperparams.get('wood_price_chf_per_kwh', 0.095)
                commodity_name = 'Wood'
            elif 'Palm oil' in can_extract_from:
                commodity_price_chf_per_kwh = hyperparams.get('palm_oil_price_chf_per_kwh', 0.070)
                commodity_name = 'Palm oil'
            else:
                print(f"Warning: Special PPU {ppu_name} has unknown commodity source.")
                # Unknown commodity, skip special logic
                commodity_price_chf_per_kwh = 0.0
                commodity_name = None
            
            if commodity_name:
                # 5. Buy commodity with revenue (MWh of commodity)
                commodity_energy_MWh = revenue_CHF / (commodity_price_chf_per_kwh * 1000.0)  # Convert kWh to MWh
                
                # 6. Convert commodity → biooil via chain efficiency
                biooil_energy_MWh = commodity_energy_MWh * efficiency
                biooil_energy_MW = biooil_energy_MWh / hyperparams['timestep_hours']  # Convert back to MW
                
                # 7. Store in Biooil storage (check capacity)
                can_input_to = each_store_ppu['ppu_row'].get('can_input_to', [])
                for storage_name in can_input_to:
                    if storage_name == 'Biooil':
                        storage_item = next((item for item in raw_energy_storage if item['storage'] == 'Biooil'), None)
                        if storage_item:
                            storage_available = storage_item['value'] - storage_item['current_value']
                            actual_stored_MW = min(biooil_energy_MW, storage_available)
                            
                            # Update storage
                            storage_item['current_value'] += actual_stored_MW
                            storage_item['history'].append((t, actual_stored_MW))
                            
                            # Record production (energy converted and stored)
                            Technology_volume[ppu_name]['production'].append((t, -actual_stored_MW))
                            
                            # If storage was full, the excess is lost (net loss recorded via production)
                            # Note: We don't mark spot_sold because we already used that money to buy commodity
                            break
                
                # Continue to next PPU
                continue
        
        # NORMAL STORE PPU LOGIC (for non-special PPUs)
        target_charge_MW = surplus_MW * allocation * efficiency  # Apply efficiency for charging
        spot_sell = 0.0
        
        # Find available capacity in storage linked to each_store_ppu 
        can_input_to = each_store_ppu['ppu_row'].get('can_input_to', [])
        
        # Find remaining capacity in can_input_to storages
        available_capacity_MW = sum(
            storage_item['value'] - storage_item['current_value']
            for storage_item in raw_energy_storage 
            if storage_item['storage'] in can_input_to
        )
        
        # Limit by: available storage capacity, 1 GW PPU capacity, and target charge
        # Note: Special biooil PPUs bypass the 1 GW limit, but they're handled above
        actual_charge_MW = min(target_charge_MW, available_capacity_MW, 1000 * efficiency)

        if actual_charge_MW == target_charge_MW:
            # Can charge exactly what we want - update storage proportionally
            new_available = available_capacity_MW - actual_charge_MW
            for storage_name in can_input_to:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item:
                    # Calculate proportion of this storage's available capacity
                    storage_available = storage_item['value'] - storage_item['current_value']
                    if available_capacity_MW > 0:
                        fraction = storage_available / available_capacity_MW
                        charge_amount = actual_charge_MW * fraction
                        old_value = storage_item['current_value']
                        storage_item['current_value'] += charge_amount
                        storage_item['history'].append((t, charge_amount))  # Positive for charge
                        
        elif actual_charge_MW == available_capacity_MW:
            # Storage capacity is the limiting factor - fill storage completely
            spot_sell = target_charge_MW - actual_charge_MW
            for storage_name in can_input_to:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item:
                    storage_available = storage_item['value'] - storage_item['current_value']
                    storage_item['current_value'] = storage_item['value']  # Fill to max
                    storage_item['history'].append((t, storage_available))  # Positive for charge
                    
        else:
            # PPU capacity is the limiting factor - partial charge
            spot_sell = target_charge_MW - actual_charge_MW
            for storage_name in can_input_to:
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                if storage_item:
                    storage_available = storage_item['value'] - storage_item['current_value']
                    if available_capacity_MW > 0:
                        fraction = storage_available / available_capacity_MW
                        charge_amount = actual_charge_MW * fraction
                        storage_item['current_value'] += charge_amount
                        storage_item['history'].append((t, charge_amount))  # Positive for charge
        
        # Record consumption (negative production for storage charging)
        Technology_volume[each_store_ppu['ppu_name']]['production'].append((t, -actual_charge_MW))
        # Record spot market sell if needed
        Technology_volume[each_store_ppu['ppu_name']]['spot_sold'].append((t, spot_sell))

    return raw_energy_storage
