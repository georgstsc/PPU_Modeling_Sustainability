# ============================================================================
# ENERGY DISPATCH PIPELINE FRAMEWORK
# ============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
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
    load_ppu_data
)


# ------------------------------------------------------------------
# Ultra-fast solar production mapper (build once, reuse)
# ------------------------------------------------------------------
class SolarMapper:
    """
    Builds once: (lat, lon) -> closest column index, with 15-min irradiance matrix.
    """
    __slots__ = ("lats", "lons", "cols", "data", "rank_order")

    def __init__(self, csv_path: str):
        # The CSV is shaped like the provided solar_incidence_hourly_2024.csv
        df = pd.read_csv(csv_path, header=None)
        self.lats = df.iloc[0, 1:].astype(np.float32).values
        self.lons = df.iloc[1, 1:].astype(np.float32).values
        self.cols = np.arange(self.lats.size, dtype=np.int32)

        # Extract hourly incidence and resample to 15-min
        times = pd.to_datetime(df.iloc[3:, 0])
        data_hourly = df.iloc[3:, 1:].astype(np.float32).values
        data_15 = pd.DataFrame(data_hourly, index=times).resample("15min").interpolate().to_numpy()
        self.data = np.asarray(data_15, dtype=np.float32)

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
        # Resample to 15-min and convert to numpy
        data_15 = df.resample('15min').interpolate().to_numpy(dtype=np.float32)
        self.data = np.asarray(data_15, dtype=np.float32)
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
    solar_15min = solar_df.resample('15min').interpolate(method='linear')

    # Load hourly wind data
    wind_hourly_df = pd.read_csv(f'{data_dir}/wind_incidence_hourly_2024.csv', index_col=0, header=[0, 1], parse_dates=True)
    wind_15min = wind_hourly_df.resample('15min').interpolate(method='linear')

    return solar_15min, wind_15min


def configure_ppu_portfolio() -> Dict[str, Dict]:
    """
    Configure the PPU portfolio for the dispatch simulation.

    Returns:
        Dictionary of PPU configurations
    """
    PPU_dict = {
        'PV_1': {'type': 'PV', 'classification': 'Incidence', 'storage_used': None,
                 'location_rank': 1, 'area_m2': 100000},
        'PV_2': {'type': 'PV', 'classification': 'Incidence', 'storage_used': None,
                 'location_rank': 2, 'area_m2': 100000},
        'WIND_1': {'type': 'WIND_ONSHORE', 'classification': 'Incidence', 'storage_used': None,
                   'location_rank': 1, 'num_turbines': 5},
        'WIND_2': {'type': 'WIND_ONSHORE', 'classification': 'Incidence', 'storage_used': None,
                   'location_rank': 2, 'num_turbines': 5},
        'HYD_ROR_1': {'type': 'HYD_ROR', 'classification': 'Incidence', 'storage_used': None},
        'HYD_ROR_2': {'type': 'HYD_ROR', 'classification': 'Incidence', 'storage_used': None},
    }

    return PPU_dict


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


def precompute_renewable_productions(ppu_dictionary: pd.DataFrame, num_timesteps: int,
                                     solar_ranking_df: pd.DataFrame, wind_ranking_df: pd.DataFrame) -> Dict:
    """
    Precompute all renewable productions by grouping PPUs by rank.
    
    Args:
        ppu_dictionary: PPU configuration
        num_timesteps: Number of timesteps to precompute
        solar_ranking_df: Solar location rankings
        wind_ranking_df: Wind location rankings
    
    Returns:
        Dictionary with:
            'solar_prod_matrix': (num_timesteps, num_solar_ranks) or None
            'wind_prod_matrix': (num_timesteps, num_wind_ranks) or None
            'solar_rank_to_ppus': {rank: [(ppu_name, area_fraction), ...]}
            'wind_rank_to_ppus': {rank: [(ppu_name, turbine_fraction), ...]}
            'solar_ranks': sorted array of ranks
            'wind_ranks': sorted array of ranks
    """
    # Group solar PPUs by rank
    solar_groups = {}  # rank: total_area
    solar_ppu_map = {}  # rank: [(ppu_name, area), ...]
    
    wind_groups = {}  # rank: total_turbines
    wind_ppu_map = {}  # rank: [(ppu_name, num_turbines), ...]
    
    for idx, row in ppu_dictionary.iterrows():
        can_extract_from = row.get("can_extract_from", [])
        ppu_name = row['PPU_Name']
        location_rank = row.get('Location_Rank', np.nan)
        
        if pd.notna(location_rank):
            rank = int(location_rank)
            
            if 'Solar' in can_extract_from:
                area_m2 = 100000  # Default area (could be from config)
                solar_groups[rank] = solar_groups.get(rank, 0.0) + area_m2
                if rank not in solar_ppu_map:
                    solar_ppu_map[rank] = []
                solar_ppu_map[rank].append((ppu_name, area_m2))
                
            elif 'Wind' in can_extract_from:
                num_turbines = 5  # Default (could be from config)
                wind_groups[rank] = wind_groups.get(rank, 0.0) + num_turbines
                if rank not in wind_ppu_map:
                    wind_ppu_map[rank] = []
                wind_ppu_map[rank].append((ppu_name, num_turbines))
    
    # Convert to fraction maps for distribution
    solar_rank_to_ppus = {}
    for rank, ppus in solar_ppu_map.items():
        total = solar_groups[rank]
        solar_rank_to_ppus[rank] = [(name, area/total) for name, area in ppus]
    
    wind_rank_to_ppus = {}
    for rank, ppus in wind_ppu_map.items():
        total = wind_groups[rank]
        wind_rank_to_ppus[rank] = [(name, turb/total) for name, turb in ppus]
    
    result = {
        'solar_prod_matrix': None,
        'wind_prod_matrix': None,
        'solar_rank_to_ppus': solar_rank_to_ppus,
        'wind_rank_to_ppus': wind_rank_to_ppus,
        'solar_ranks': np.array([]),
        'wind_ranks': np.array([])
    }
    
    # Precompute solar productions
    if solar_groups:
        _init_solar_mapper('data/solar_incidence_hourly_2024.csv')
        if _SOLAR_MAPPER is not None:
            try:
                solar_ranks = np.array(sorted(solar_groups.keys()), dtype=np.int32)
                solar_areas = np.array([solar_groups[r] for r in solar_ranks], dtype=np.float32)
                result['solar_prod_matrix'] = _SOLAR_MAPPER.precompute_productions(
                    solar_ranks, solar_areas, num_timesteps
                )
                result['solar_ranks'] = solar_ranks
                print(f"  Precomputed solar productions: {len(solar_ranks)} ranks, {num_timesteps} timesteps")
            except Exception as e:
                print(f"  Solar precomputation failed: {e}, will use fallback")
    
    # Precompute wind productions
    if wind_groups:
        _init_wind_mapper('data/wind_incidence_hourly_2024.csv')
        if _WIND_MAPPER is not None:
            try:
                wind_ranks = np.array(sorted(wind_groups.keys()), dtype=np.int32)
                wind_nums = np.array([wind_groups[r] for r in wind_ranks], dtype=np.float32)
                result['wind_prod_matrix'] = _WIND_MAPPER.precompute_productions(
                    wind_ranks, wind_nums, num_timesteps
                )
                result['wind_ranks'] = wind_ranks
                print(f"  Precomputed wind productions: {len(wind_ranks)} ranks, {num_timesteps} timesteps")
            except Exception as e:
                print(f"  Wind precomputation failed: {e}, will use fallback")
    
    return result


def calculate_solar_production(location_rank: int, area_m2: float, t: int,
                              solar_15min: pd.DataFrame, solar_ranking_df: pd.DataFrame) -> float:
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
    if _SOLAR_MAPPER is not None:
        try:
            lat = lon = None
            if len(solar_ranking_df) > 0:
                location_idx = (location_rank - 1) % len(solar_ranking_df)
                loc_row = solar_ranking_df.iloc[location_idx]
                lat = float(loc_row['latitude'])
                lon = float(loc_row['longitude'])
            if lat is not None and lon is not None:
                col_idx = _SOLAR_MAPPER.closest_column(lat, lon)
            else:
                col_idx = _SOLAR_MAPPER.col_for_rank(location_rank)
            irr = _SOLAR_MAPPER.irradiance(int(t), int(col_idx))
            return float(_solar_prod_fast(float(irr), float(area_m2)))
        except Exception:
            # Fall back to original path below
            pass

    # Fallback: original implementation using provided solar_15min
    num_locations = len(solar_ranking_df)
    if num_locations == 0:
        return 0.0
    location_idx = (location_rank - 1) % num_locations
    loc = solar_ranking_df.iloc[location_idx]
    lat, lon = float(loc['latitude']), float(loc['longitude'])

    closest_col = None
    min_dist = float('inf')
    for col in solar_15min.columns:
        dist = abs(float(col[0]) - lat) + abs(float(col[1]) - lon)
        if dist < min_dist:
            min_dist = dist
            closest_col = col

    if closest_col is not None:
        incidence = solar_15min.iloc[t][closest_col]  # type: ignore
        incidence = incidence.item() if hasattr(incidence, 'item') else float(incidence)  # type: ignore
    else:
        incidence = 0.0

    production_MW = float(incidence) * float(area_m2) * 0.25 / 1000.0
    return production_MW


def calculate_wind_production(location_rank: int, num_turbines: int, t: int,
                             wind_15min: pd.DataFrame, wind_ranking_df: pd.DataFrame) -> float:
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
    if _WIND_MAPPER is not None:
        try:
            lat = lon = None
            if len(wind_ranking_df) > 0:
                location_idx = (location_rank - 1) % len(wind_ranking_df)
                loc_row = wind_ranking_df.iloc[location_idx]
                lat = float(loc_row['latitude'])
                lon = float(loc_row['longitude'])
            if lat is not None and lon is not None:
                col_idx = _WIND_MAPPER.closest_column(lat, lon)
            else:
                col_idx = _WIND_MAPPER.col_for_rank(location_rank)
            wspd = _WIND_MAPPER.speed(int(t), int(col_idx))
            return float(_wind_power_fast(float(wspd), int(num_turbines)))
        except Exception:
            # Fall back to original path below
            pass

    # Fallback: original implementation using provided wind_15min
    # Get location coordinates with bounds checking
    num_locations = len(wind_ranking_df)
    if num_locations == 0:
        return 0.0
    location_idx = (location_rank - 1) % num_locations
    loc = wind_ranking_df.iloc[location_idx]
    lat, lon = float(loc['latitude']), float(loc['longitude'])

    wind_speed = 0.0
    if len(wind_15min.columns) > 0:
        lats = np.array([float(col[0]) for col in wind_15min.columns])
        lons = np.array([float(col[1]) for col in wind_15min.columns])
        distances = np.abs(lats - lat) + np.abs(lons - lon)
        closest_idx = int(np.argmin(distances))
        wind_speed = float(wind_15min.iloc[t, closest_idx])  # type: ignore

    # Physics-based power with simple cut-in/out
    rotor_diameter_m = 120.0
    air_density = 1.225
    swept_area = np.pi * (rotor_diameter_m / 2.0) ** 2
    power_coefficient = 0.45
    if wind_speed < 3.0 or wind_speed > 25.0:
        return 0.0
    power_per_turbine_W = 0.5 * air_density * swept_area * (wind_speed ** 3) * power_coefficient
    return (power_per_turbine_W / 1e6) * float(num_turbines)


def run_dispatch_simulation(demand_15min: pd.Series, spot_15min: pd.Series,
                          ror_df: pd.DataFrame, solar_15min: pd.DataFrame,
                          wind_15min: pd.DataFrame, ppu_dictionary: pd.DataFrame,
                          solar_ranking_df: pd.DataFrame, wind_ranking_df: pd.DataFrame,
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
    num_timesteps = min(len(demand_15min), len(spot_15min), len(ror_df),
                       len(solar_15min), len(wind_15min))
    # num_timesteps = 1000
    
    # Precompute renewable productions (MAJOR SPEEDUP)
    print("Precomputing renewable productions...")
    precomputed = precompute_renewable_productions(
        ppu_dictionary, num_timesteps, solar_ranking_df, wind_ranking_df
    )
    
    # Initialize technology tracking
    Technology_volume = initialize_technology_tracking(ppu_dictionary)
    phi_smoothed = 0.0
    
    # Track how often the system is in deficit (overflow>+epsilon), surplus (overflow<-epsilon), or balanced
    overflow_count = 0
    surplus_count = 0
    balanced_count = 0
    for t in range(num_timesteps):
        if t % 5000 == 0:
            print(f"  Progress: {t:,}/{num_timesteps:,} ({t/num_timesteps*100:.1f}%)")
            # show the state of each storage 
            for storage in raw_energy_storage:
                print(f"    {storage['storage']}: {storage['current_value']} MW")
            for storage in raw_energy_incidence:
                print(f"    {storage['storage']}: {storage['current_value']} MW")
        # 1) Import the demand
        demand_MW = demand_15min.iloc[t]
        spot_price = spot_15min.iloc[t]

        # 2) Update raw_energy_incidence with respective values for each energy source/storage
        raw_energy_incidence = update_raw_energy_incidence(
            raw_energy_incidence, Technology_volume, t, solar_ranking_df, wind_ranking_df,
            ppu_dictionary, ror_df, solar_15min, wind_15min, precomputed
        )

        # Get energy produced by incidence PPUs (stored in "Grid")
        grid_energy = 0.0
        for incidence_item in raw_energy_incidence:
            if incidence_item['storage'] == 'Grid':
                # Get current value directly (already updated in update_raw_energy_incidence)
                grid_energy = incidence_item['current_value']
                break
        # 4) Compute overflow = demand - energy in "Grid"
        overflow_MW = demand_MW - grid_energy
        # print(f"The overflow is of : {overflow_MW} MW at timestep {t}")
        # Count state occurrence for this timestep
        if overflow_MW > hyperparams['epsilon']:
            overflow_count += 1
        elif overflow_MW < -hyperparams['epsilon']:
            surplus_count += 1
        else:
            balanced_count += 1

        # First compute the d_stor, u_dis, u_chg and m for each PPU
        ppu_indices = calculate_ppu_indices(
            ppu_dictionary, raw_energy_storage, overflow_MW, phi_smoothed,
            spot_price, spot_15min, t, hyperparams
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
                t, 
                indices['d_stor'], 
                indices['u_dis'], 
                indices['u_chg'], 
                indices['m'],
                indices['kappa_dis'],
                indices['kappa_chg']
            ))
        if t % 5000 == 0:
            print("[DEBUG] Storage state BEFORE handling:")
            for storage in raw_energy_storage:
                print(f"  {storage['storage']}: {storage['current_value']:.2f} MW")
        # 4.1) If overflow > 0 (we need more energy): use 'Flex' PPUs
        if overflow_MW > hyperparams['epsilon']:
            raw_energy_storage = handle_energy_deficit(
                overflow_MW, ppu_dictionary, raw_energy_storage, Technology_volume,
                ppu_indices, spot_price, t, hyperparams
            )
        # 4.2) If overflow < 0: use 'Store' PPUs
        elif overflow_MW < -hyperparams['epsilon']:
            surplus_MW = abs(overflow_MW)
            raw_energy_storage = handle_energy_surplus(
                surplus_MW, ppu_dictionary, raw_energy_storage, Technology_volume,
                ppu_indices, spot_price, t, hyperparams
            )
        # DEBUG: Show storage state before handling
        if t % 5000 == 0:
            print("[DEBUG] Storage state AFTER handling:")
            for storage in raw_energy_storage:
                print(f"  {storage['storage']}: {storage['current_value']:.2f} MW")

    # Attach pipeline state statistics for later reporting
    Technology_volume['__pipeline_stats__'] = {
        'overflow_count': overflow_count,
        'surplus_count': surplus_count,
        'balanced_count': balanced_count,
        'total_timesteps': num_timesteps
    }

    print(f"  ✓ Simulation complete: {num_timesteps:,} timesteps")
    print(f"    - Overflow (deficit) steps: {overflow_count:,}")
    print(f"    - Surplus steps:            {surplus_count:,}")
    print(f"    - Balanced steps:           {balanced_count:,}")
    return Technology_volume, phi_smoothed


def compute_cost_breakdown(technology_volume: Dict[str, Dict], spot_15min: pd.Series,
                          hyperparams: Dict, data_dir: str = "data") -> Dict[str, Dict]:
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
    # Load cost data
    cost_df = load_cost_data(f'{data_dir}/cost_table_tidy.csv')

    # Technology-specific costs (simplified, using PPU names from dictionary)
    tech_costs = {
        'PV': 0.10,      # CHF/kWh - Solar PV
        'WD_OFF': 0.08,  # CHF/kWh - Offshore Wind
        'WD_ON': 0.08,   # CHF/kWh - Onshore Wind
        'HYD_ROR': 0.05, # CHF/kWh - Run-of-river Hydro
        'HYD_S': 0.05,   # CHF/kWh - Storage Hydro
        'H2_G': 0.15,    # CHF/kWh - Hydrogen Storage
        'THERM': 0.12,   # CHF/kWh - Thermal
        'SOL_SALT': 0.11 # CHF/kWh - Solar Salt Storage
    }

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

        # Production cost
        cost_per_kwh = tech_costs.get(tech_type, 0.10)
        for (t, vol_MW) in metrics['production']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            production_cost += energy_MWh * cost_per_kwh * 1000  # Convert to kWh

        # Spot market transactions
        for (t, vol_MW) in metrics['spot_bought']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            spot_buy_cost += energy_MWh * spot_15min.iloc[t]

        for (t, vol_MW) in metrics['spot_sold']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            spot_sell_revenue += energy_MWh * spot_15min.iloc[t]

        Cost_summary[tech_type] = {
            'production_cost_CHF': production_cost,
            'spot_buy_cost_CHF': spot_buy_cost,
            'spot_sell_revenue_CHF': spot_sell_revenue,
            'net_cost_CHF': production_cost + spot_buy_cost - spot_sell_revenue
        }

    return Cost_summary


def compute_portfolio_metrics(cost_summary: Dict[str, Dict], spot_15min: pd.Series,
                             demand_15min: pd.Series, hyperparams: Dict) -> Dict[str, float]:
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

    # Compute spot-only baseline
    spot_only_cost = 0.0
    num_timesteps = min(len(demand_15min), len(spot_15min))
    for t in range(num_timesteps):
        demand_MWh = demand_15min.iloc[t] * hyperparams['timestep_hours']
        spot_only_cost += demand_MWh * spot_15min.iloc[t]

    savings = spot_only_cost - total_net_cost
    margin_pct = (savings / spot_only_cost) * 100

    # Calculate price volatility (coefficient of variation)
    price_volatility_pct = (spot_15min.std() / spot_15min.mean()) * 100
    price_volatility_pct = float(price_volatility_pct)  # type: ignore

    return {
        'total_cost_CHF': total_net_cost,
        'spot_only_cost_CHF': spot_only_cost,
        'savings_CHF': savings,
        'margin_pct': margin_pct,
        'volatility_pct': price_volatility_pct,
        'num_timesteps': num_timesteps
    }


def run_complete_pipeline(ppu_counts: Dict[str, int], raw_energy_storage: List[Dict], 
                        raw_energy_incidence: List[Dict], data_dir: str = "data", 
                        hyperparams: Optional[Dict] = None) -> Dict:
    """
    Run the complete energy dispatch pipeline.

    Parameters:
        ppu_counts: Dictionary mapping PPU type names to number of instances (e.g., {'PV': 5, 'HYD_ROR': 3})
        raw_energy_storage: List of storage dictionaries with capacity and tracking info
        raw_energy_incidence: List of incidence dictionaries with availability tracking
        data_dir: Directory containing data files
        hyperparams: Optional hyperparameters override

    Returns:
        Complete pipeline results dictionary
    """
    print("=" * 80)
    print("ENERGY DISPATCH PIPELINE - FULL YEAR 2024 SIMULATION")
    print("=" * 80)

    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'alpha_d': 0.5,
            'alpha_u': 1000.0,
            'alpha_m': 5.0,
            'weight_spread': 1.0,
            'weight_volatility': 1.0,
            'volatility_scale': 30.0,
            'epsilon': 1e-6,
            'timestep_hours': 0.25,
            'ema_beta': 0.2,
            'horizons': ['1d', '3d', '7d', '30d']
        }
    hyperparams.update({
    'spot_unit': 'CHF_per_MWh',       # your spot is ~90–100 CHF/MWh
    'ppu_cost_unit': 'CHF_per_KWh',   # your per-PPU 'cost' column is per kWh
    'cost_column': 'cost',            # USE VARIABLE COST (not LCOE)
    'use_efficiency': True,           # divide cost_in by efficiency to get cost_out
    'beta_c': 0.12,                   # sensitivity in CHF/kWh (~120 CHF/MWh) to avoid pinning
    'auto_beta_c': True,              # optional: auto-scale each step
    'beta_c_pctl': 90,                # use 90th percentile |spread|
    'beta_c_min': 0.03,               # clamp in CHF/kWh
    'beta_c_max': 0.30,
    'alpha_u': 5000.0,                # MW scale for utility to avoid u_dis/u_chg=1.0
    'stor_deadband': 0.05,
    'default_target_soc': 0.6
    })

    print("\n[STEP 1] Loading data...")

    # Load all data 

    demand_15min, spot_15min, ror_df = load_energy_data(data_dir)
    solar_15min, wind_15min = load_incidence_data(data_dir)
    solar_ranking_df = pd.read_csv(f'{data_dir}/ranking_incidence/solar_incidence_ranking.csv').head(10)
    wind_ranking_df = pd.read_csv(f'{data_dir}/ranking_incidence/wind_incidence_ranking.csv').head(10)
    ppu_constructs_df = load_ppu_data(f'{data_dir}/ppu_constructs_components.csv')
    cost_df = load_cost_data(f'{data_dir}/cost_table_tidy.csv')
    solar_locations_df = load_location_rankings('solar')
    wind_locations_df = load_location_rankings('wind')

    # Initialize history tracking for raw_energy_storage and raw_energy_incidence
    for storage in raw_energy_storage:
        if 'history' not in storage:
            storage['history'] = []
    
    for incidence in raw_energy_incidence:
        if 'history' not in incidence:
            incidence['history'] = []

    # Configure PPUs using dictionary approach
    ppu_dictionary = initialize_ppu_dictionary()

    # Add PPUs based on counts
    ppu_types = list(ppu_counts.keys())
    for ppu_type in ppu_types:
        count = ppu_counts[ppu_type]
        for i in range(count):
            ppu_dictionary = add_ppu_to_dictionary(
                ppu_dictionary,
                ppu_type,
                ppu_constructs_df,
                cost_df,
                solar_locations_df,
                wind_locations_df,
                raw_energy_storage=raw_energy_storage,
                raw_energy_incidence=raw_energy_incidence
            )
    # Print entire PPU dictionnary 
    print(f"  ✓ Loaded demand: {len(demand_15min)} timesteps")
    print(f"  ✓ Loaded spot prices: {len(spot_15min)} timesteps")
    print(f"  ✓ Loaded solar/wind data: {len(solar_15min)}/{len(wind_15min)} timesteps")
    print(f"  ✓ Configured {len(ppu_dictionary)} PPUs from {len(ppu_types)} types")

    # Print entire PPU dictionary with comprehensive details
    print("\nPPU CONFIGURATION:")
    print("=" * 120)
    for each_ppu in ppu_dictionary.itertuples():
        ppu_name = each_ppu.PPU_Name
        ppu_type = getattr(each_ppu, 'Type', 'N/A')
        classification = getattr(each_ppu, 'PPU_Extract', 'N/A')
        
        # Get storage-related information
        can_extract = getattr(each_ppu, 'can_extract_from', [])
        can_input = getattr(each_ppu, 'can_input_to', [])
        
        # Get efficiency and cost
        efficiency = getattr(each_ppu, 'Chain_Efficiency', 'N/A')
        cost_kwh = getattr(each_ppu, 'Cost_CHF_per_kWh', 'N/A')
        
        # Get location rank if applicable
        location = getattr(each_ppu, 'Location_Rank', 'N/A')
        
        print(f"  {ppu_name:15} | Classification: {classification:10} | Type: {ppu_type:15}")
        
        if can_extract:
            print(f"      → Extracts from: {', '.join(can_extract)}")
        if can_input:
            print(f"      → Inputs to: {', '.join(can_input)}")
        
        print(f"      → Efficiency: {efficiency if isinstance(efficiency, str) else f'{efficiency:.2%}'} | Cost: {cost_kwh if isinstance(cost_kwh, str) else f'{cost_kwh:.4f}'} CHF/kWh", end="")
        
        if location != 'N/A' and pd.notna(location):
            print(f" | Location Rank: {int(location)}")
        else:
            print()
        print()
    print("=" * 120)

    print("\n[STEP 2] Running dispatch simulation...")
    technology_volume, phi_smoothed = run_dispatch_simulation(
        demand_15min, spot_15min, ror_df, solar_15min, wind_15min,
        ppu_dictionary, solar_ranking_df, wind_ranking_df, 
        raw_energy_storage, raw_energy_incidence, hyperparams
    )

    print("\n[STEP 3] Computing costs...")
    cost_summary = compute_cost_breakdown(technology_volume, spot_15min, hyperparams, data_dir)

    print("\n[STEP 4] Computing portfolio metrics...")
    portfolio_metrics = compute_portfolio_metrics(cost_summary, spot_15min, demand_15min, hyperparams)

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


def update_raw_energy_incidence(raw_energy_incidence: List[Dict], Technology_volume: Dict, t: int,
                               solar_ranking_df: pd.DataFrame, wind_ranking_df: pd.DataFrame,
                               ppu_dictionary: pd.DataFrame, ror_df: pd.DataFrame, 
                               solar_15min: pd.DataFrame, wind_15min: pd.DataFrame,
                               precomputed: Optional[Dict] = None) -> List[Dict]:
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
        solar_ranking_df: Solar location rankings
        wind_ranking_df: Wind location rankings
        ppu_dictionary: PPU configuration
        ror_df: Run-of-river DataFrame (timestamp-indexed) loaded earlier
        solar_15min: Solar generation DataFrame (timestamp-indexed) loaded earlier
        wind_15min: Wind generation DataFrame (timestamp-indexed) loaded earlier

    Returns:
        Updated raw_energy_incidence list
    """
    available_energy_ror, available_energy_wood, available_energy_solar, available_energy_wind = 0.0, 0.0, 0.0, 0.0
    for incidence_item in raw_energy_incidence:
        if incidence_item['storage'] == 'River':
            # Extract available energy from the provided ror_df (safe access with fallback)
            try:
                # Prefer .iloc by timestep index if ror_df is time-indexed and t is positional
                available_energy_ror = float(ror_df.iloc[t].get('RoR_MW', 12.0))
            except Exception:
                available_energy_ror = 0.0
        elif incidence_item['storage'] == 'Wood':
            available_energy_wood = 500

    # We track ror and wood in a descending way
    available_energy_river_track = 0
    available_energy_wood_track = 0
    for idx, each_ppu in ppu_dictionary.iterrows():
        can_extract_from = each_ppu.get("can_extract_from", [])
        ppu_name = each_ppu['PPU_Name']
        
        # Initialize Technology_volume for this PPU if not exists
        if ppu_name not in Technology_volume:
            Technology_volume[ppu_name] = {
                'production': [],
                'spot_bought': [],
                'spot_sold': [],
                'cost_indices': []
            }
        
        if 'Solar' in can_extract_from:
            location_rank = each_ppu.get('Location_Rank', np.nan)
            if pd.notna(location_rank):
                # Fast path: use precomputed matrix if available
                current_solar_prod = 0.0
                if (precomputed and precomputed.get('solar_prod_matrix') is not None 
                    and int(location_rank) in precomputed['solar_rank_to_ppus']):
                    # Look up from precomputed matrix
                    rank = int(location_rank)
                    rank_idx = np.where(precomputed['solar_ranks'] == rank)[0]
                    if len(rank_idx) > 0:
                        total_prod = float(precomputed['solar_prod_matrix'][t, rank_idx[0]])
                        # Get this PPU's fraction
                        for ppu, frac in precomputed['solar_rank_to_ppus'][rank]:
                            if ppu == ppu_name:
                                current_solar_prod = total_prod * frac
                                break
                else:
                    # Fallback to original calculation
                    area_m2 = 100000
                    current_solar_prod = calculate_solar_production(int(location_rank), area_m2, t, solar_15min, solar_ranking_df)
                
                Technology_volume[ppu_name]['production'].append((t, current_solar_prod))
                available_energy_solar += current_solar_prod    
        elif 'Wind' in can_extract_from:
            location_rank = each_ppu.get('Location_Rank', np.nan)
            if pd.notna(location_rank):
                # Fast path: use precomputed matrix if available
                current_wind_prod = 0.0
                if (precomputed and precomputed.get('wind_prod_matrix') is not None 
                    and int(location_rank) in precomputed['wind_rank_to_ppus']):
                    # Look up from precomputed matrix
                    rank = int(location_rank)
                    rank_idx = np.where(precomputed['wind_ranks'] == rank)[0]
                    if len(rank_idx) > 0:
                        total_prod = float(precomputed['wind_prod_matrix'][t, rank_idx[0]])
                        # Get this PPU's fraction
                        for ppu, frac in precomputed['wind_rank_to_ppus'][rank]:
                            if ppu == ppu_name:
                                current_wind_prod = total_prod * frac
                                break
                else:
                    # Fallback to original calculation
                    num_turbines = 5
                    current_wind_prod = calculate_wind_production(int(location_rank), num_turbines, t, wind_15min, wind_ranking_df)
                
                Technology_volume[ppu_name]['production'].append((t, current_wind_prod))
                available_energy_wind += current_wind_prod
        elif 'River' in can_extract_from:
            if available_energy_ror > 1000: 
                extraction = 1000
                available_energy_river_track += 1000
                available_energy_ror -= 1000
            else:
                extraction = available_energy_ror
                available_energy_river_track += available_energy_ror
                available_energy_ror = 0
            Technology_volume[ppu_name]['production'].append((t, extraction))
        elif 'Wood' in can_extract_from:
            if available_energy_wood > 500:
                extraction = 500
                available_energy_wood -= 500
                available_energy_wood_track += 500
            else:
                extraction = available_energy_wood
                available_energy_wood_track += available_energy_wood
                available_energy_wood = 0
            Technology_volume[ppu_name]['production'].append((t, extraction))
    
    total_energy = available_energy_river_track + available_energy_wood_track + available_energy_solar + available_energy_wind
    # in raw_energy_incidence get "Grid" and set current_value to total_energy
    for incidence_item in raw_energy_incidence:
        if incidence_item['storage'] == 'Grid':
            incidence_item['current_value'] = total_energy
            incidence_item['history'].append((t, total_energy))
        elif incidence_item['storage'] == 'Solar':
            incidence_item['current_value'] = available_energy_solar
            incidence_item['history'].append((t, available_energy_solar))
        elif incidence_item['storage'] == 'Wind':
            incidence_item['current_value'] = available_energy_wind
            incidence_item['history'].append((t, available_energy_wind))
        elif incidence_item['storage'] == 'River':
            incidence_item['current_value'] = available_energy_river_track
            incidence_item['history'].append((t, available_energy_river_track))
        elif incidence_item['storage'] == 'Wood':
            incidence_item['current_value'] = available_energy_wood_track
            incidence_item['history'].append((t, available_energy_wood_track))
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
                         overflow_MW: float, phi_smoothed: float, spot_price: float,
                         spot_15min: pd.Series, t: int, hyperparams: Dict) -> Dict[str, Dict]:
    """
    Calculate d_stor, u_dis, u_chg and m (monetary volatility-aware) indices for each PPU.
    Uses RELATIVE VALUES throughout to avoid dimension issues and stuck costs.

    Parameters:
        ppu_dictionary: PPU configuration
        raw_energy_storage: Storage systems
        overflow_MW: Current overflow (demand - supply)
        phi_smoothed: Smoothed overflow
        spot_price: Current spot price
        spot_15min: Full spot price time series
        t: Current timestep
        hyperparams: Simulation hyperparameters

    Returns:
        Dictionary mapping PPU names to their indices and costs
    """
    ppu_indices = {}
    
    # Get hyperparameters
    alpha_u = hyperparams.get('alpha_u', 5000.0)
    alpha_m = hyperparams.get('alpha_m', 5.0)
    deadband = hyperparams.get('stor_deadband', 0.05)
    
    # Define horizons in number of timesteps (15-min intervals)
    # 1d=96, 3d=288, 7d=672, 30d=2880
    horizon_steps = {'1d': 96, '3d': 288, '7d': 672, '30d': 2880}
    horizons_30d = ['1d', '3d', '7d', '30d']
    
    for _, ppu_row in ppu_dictionary.iterrows():
        ppu_name = str(ppu_row['PPU_Name'])
        ppu_category = str(ppu_row['PPU_Category'])

        # Initialize indices
        d_stor = 0.0
        u_dis = 0.0
        u_chg = 0.0
        m_idx = 0.0

        # ===== 1. Calculate disposition index (d_stor) for storage PPUs =====
        # Use RELATIVE values: normalize by deadband and max excursion
        if ppu_category == 'Storage':
            # Get storage name from can_extract_from
            can_extract_from = ppu_row.get('can_extract_from', [])
            if can_extract_from:
                storage_name = can_extract_from[0]  # Use first storage
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)

                if storage_item:
                    current_value = storage_item['current_value']
                    max_value = storage_item['value']
                    current_soc = current_value / max_value
                    target_soc = storage_item.get('target_SoC', 0.6)

                    # Calculate RELATIVE deviation from target (normalized by deadband)
                    soc_deviation = (current_soc - target_soc) / deadband
                    
                    # Apply tanh to keep in [-1, 1] range
                    d_stor = np.tanh(soc_deviation)

        # ===== 2. Calculate utility indices (u_dis, u_chg) =====
        # Use RELATIVE values: normalize overflow by system scale
        if phi_smoothed != 0:
            # Relative overflow: normalize by alpha_u (system scale)
            relative_overflow = phi_smoothed / alpha_u
            
            # Apply tanh for smooth [-1, 1] range
            u_dis = np.tanh(relative_overflow)
            
            # Charge utility only non-zero during surplus (negative overflow)
            if phi_smoothed < 0:
                u_chg = np.tanh(abs(relative_overflow))
            else:
                u_chg = 0.0
        else:
            # No overflow - neutral utility
            u_dis = 0.0
            u_chg = 0.0

        # ===== 3. Calculate monetary volatility-aware index (m) =====
        # Use RELATIVE values throughout monetary calculation
        if ppu_category == 'Storage':
            ppu_cost_per_kWh = ppu_row.get('Cost_CHF_per_kWh', 0.1)
            ppu_cost_per_MWh = ppu_cost_per_kWh * 1000.0  # Convert to CHF/MWh
            efficiency = ppu_row.get('Efficiency', 1.0)
            
            # Compute RELATIVE future prices (normalized by current price)
            future_prices_relative = {}
            for horizon_name, steps in horizon_steps.items():
                future_idx = min(t + steps, len(spot_15min) - 1)
                future_window = spot_15min.iloc[t:min(t + steps, len(spot_15min))]
                if len(future_window) > 0:
                    future_price_avg = future_window.mean()
                    # RELATIVE price: (future - current) / current
                    future_prices_relative[horizon_name] = (future_price_avg - spot_price) / max(spot_price, 0.01)
                else:
                    future_prices_relative[horizon_name] = 0.0
            
            # Approximate RELATIVE co-state values
            spreads_relative = []
            for horizon_name in horizons_30d:
                # Co-state relative to current price
                lambda_relative = (efficiency * spot_15min.iloc[min(t + horizon_steps[horizon_name], len(spot_15min) - 1)] - spot_price) / max(spot_price, 0.01)
                spread_relative = (spot_price - efficiency * spot_15min.iloc[min(t + horizon_steps[horizon_name], len(spot_15min) - 1)]) / max(spot_price, 0.01)
                spreads_relative.append(spread_relative)
            
            # Average RELATIVE spread across horizons
            S_relative = np.mean(spreads_relative)
            
            # RELATIVE volatility impact
            future_price_values = [future_prices_relative[h] for h in horizons_30d]
            price_variance_relative = np.var(future_price_values) if len(future_price_values) > 1 else 0.0
            
            # Normalize RELATIVE variance by volatility scale
            volatility_scale = hyperparams.get('volatility_scale', 30.0)
            V_relative = price_variance_relative / (volatility_scale ** 2) if volatility_scale > 0 else 0.0
            
            # Combine RELATIVE spread and volatility
            weight_spread = hyperparams.get('weight_spread', 1.0)
            weight_volatility = hyperparams.get('weight_volatility', 1.0)
            X_relative = weight_spread * S_relative - weight_volatility * V_relative
            
            # Squash RELATIVE score to [-1, 1]
            m_idx = np.tanh(X_relative / alpha_m)
        else:
            # Non-storage PPUs: neutral monetary index
            m_idx = 0.0

        # ===== 4. Calculate composite benefits and costs =====
        # All components now use RELATIVE values in [-1, 1] range
        
        # Discharge benefit: B_dis = (d_stor + u_dis + m) / 3
        B_dis = (d_stor + u_dis + m_idx) / 3.0
        
        # Charge benefit: B_chg = (-d_stor + u_chg + (-m)) / 3
        B_chg = (-d_stor + u_chg - m_idx) / 3.0
        
        # Convert benefits to costs (lower is better)
        kappa_dis = 1.0 - B_dis  # Range: [0, 2]
        kappa_chg = 1.0 - B_chg  # Range: [0, 2]
        
        ppu_indices[ppu_name] = {
            'd_stor': d_stor,
            'u_dis': u_dis,
            'u_chg': u_chg,
            'm': m_idx,
            'B_dis': B_dis,
            'B_chg': B_chg,
            'kappa_dis': kappa_dis,
            'kappa_chg': kappa_chg
        }
    
    return ppu_indices


def handle_energy_deficit(deficit_MW: float, ppu_dictionary: pd.DataFrame,
                         raw_energy_storage: List[Dict], Technology_volume: Dict,
                         ppu_indices: Dict, spot_price: float, t: int,
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
                        ppu_indices: Dict, spot_price: float, t: int,
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

    for each_store_ppu in charge_benefits:
        allocation = each_store_ppu['B_chg'] / max(B_chg_total, hyperparams['epsilon'])
        efficiency = each_store_ppu['ppu_row'].get('Chain_Efficiency', 1.0)
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
        
        # Limit by the smallest: 1 GW PPU capacity, available storage capacity, and target charge
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

