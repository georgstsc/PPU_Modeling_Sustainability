# ============================================================================
# ENERGY DISPATCH PIPELINE FRAMEWORK
# ============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline

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


def calculate_solar_production(location_rank: int, area_m2: float, t: int,
                              solar_15min: pd.DataFrame, solar_ranking_df: pd.DataFrame) -> float:
    """
    Calculate solar production for a location at timestep t.

    Parameters:
        location_rank: Ranking position (1-based)
        area_m2: Solar panel area in square meters
        t: Timestep index
        solar_15min: Solar incidence data
        solar_ranking_df: Location rankings

    Returns:
        Production in MW
    """
    # Get location coordinates from ranking
    loc = solar_ranking_df.iloc[location_rank - 1]
    lat, lon = loc['latitude'], loc['longitude']

    # Find closest column in solar_15min
    closest_col = None
    min_dist = float('inf')
    for col in solar_15min.columns:
        dist = abs(col[0] - lat) + abs(col[1] - lon)
        if dist < min_dist:
            min_dist = dist
            closest_col = col

    # Get incidence (kWh/m²/hour)
    if closest_col is not None:
        incidence = solar_15min.iloc[t][closest_col]  # type: ignore
        # Extract scalar value from pandas object
        incidence = incidence.item() if hasattr(incidence, 'item') else float(incidence)  # type: ignore
    else:
        incidence = 0.0

    # Convert to MW for 15-min: (kWh/m²/hour) * area_m2 * (15min/60min) / 1000
    production_MW = float(incidence) * area_m2 * 0.25 / 1000
    return production_MW


def calculate_wind_production(location_rank: int, num_turbines: int, t: int,
                             wind_15min: pd.DataFrame, wind_ranking_df: pd.DataFrame) -> float:
    """
    Calculate wind production for a location at timestep t.

    Parameters:
        location_rank: Ranking position (1-based)
        num_turbines: Number of wind turbines
        t: Timestep index
        wind_15min: Wind incidence data
        wind_ranking_df: Location rankings

    Returns:
        Production in MW
    """
    # Get location coordinates from ranking
    loc = wind_ranking_df.iloc[location_rank - 1]
    lat, lon = loc['latitude'], loc['longitude']

    # Find closest column in wind_15min using numpy for efficiency
    if len(wind_15min.columns) > 0:
        # Extract lat/lon from column tuples
        lats = np.array([float(col[0]) for col in wind_15min.columns])
        lons = np.array([float(col[1]) for col in wind_15min.columns])

        # Calculate distances
        distances = np.abs(lats - lat) + np.abs(lons - lon)
        closest_idx = np.argmin(distances)
        closest_col = wind_15min.columns[closest_idx]

        # Get wind speed (m/s)
        wind_speed = wind_15min.iloc[t, closest_idx]
        wind_speed = float(wind_speed)  # type: ignore
    else:
        wind_speed = 0.0

    # Wind turbine parameters
    rotor_diameter_m = 120
    air_density = 1.225  # kg/m³
    swept_area = np.pi * (rotor_diameter_m / 2) ** 2  # m²
    power_coefficient = 0.45  # turbine efficiency

    # Power per turbine: 0.5 * ρ * A * v³ * Cp
    power_per_turbine_W = 0.5 * air_density * swept_area * (wind_speed ** 3) * power_coefficient
    power_per_turbine_MW = power_per_turbine_W / 1e6

    # Total production for all turbines
    production_MW = power_per_turbine_MW * num_turbines
    return production_MW


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

    # Initialize technology tracking
    Technology_volume = initialize_technology_tracking(ppu_dictionary)
    phi_smoothed = 0.0

    for t in range(num_timesteps):
        if t % 5000 == 0:
            print(f"  Progress: {t:,}/{num_timesteps:,} ({t/num_timesteps*100:.1f}%)")

        # 1) Import the demand
        demand_MW = demand_15min.iloc[t]
        spot_price = spot_15min.iloc[t]

        # 2) Update raw_energy_incidence with respective values for each energy source/storage
        raw_energy_incidence = update_raw_energy_incidence(
            raw_energy_incidence, t, solar_ranking_df, wind_ranking_df, ppu_dictionary
        )

        # 3) Use 'Incidence' PPUs to calculate energy production and cost
        raw_energy_incidence = calculate_incidence_production(
            ppu_dictionary, raw_energy_incidence, Technology_volume, t,
            solar_ranking_df, wind_ranking_df, hyperparams
        )

        # Get energy produced by incidence PPUs (stored in "Grid")
        grid_energy = 0.0
        for incidence_item in raw_energy_incidence:
            if incidence_item['storage'] == 'Grid':
                grid_energy = incidence_item['current_value']
                break

        # 4) Compute overflow = demand - energy in "Grid"
        overflow_MW = demand_MW - grid_energy

        # First compute the d_stor, u_dis, u_chg and c for each PPU
        ppu_indices = calculate_ppu_indices(
            ppu_dictionary, raw_energy_storage, overflow_MW, phi_smoothed,
            spot_price, hyperparams
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
            Technology_volume[ppu_name]['cost_indices'].append((t, indices['d_stor'], indices['u_dis'], indices['u_chg'], indices['c']))

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

        # 5) The supply of energy should be satisfied (already handled above)
        # 6) Return the volume transactions for the year (done at end of function)

    print(f"  ✓ Simulation complete: {num_timesteps:,} timesteps")
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
    for tech_type in technology_volume:
        production_cost = 0.0
        spot_buy_cost = 0.0
        spot_sell_revenue = 0.0

        # Production cost
        cost_per_kwh = tech_costs.get(tech_type, 0.10)
        for (t, vol_MW) in technology_volume[tech_type]['production']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            production_cost += energy_MWh * cost_per_kwh * 1000  # Convert to kWh

        # Spot market transactions
        for (t, vol_MW) in technology_volume[tech_type]['spot_bought']:
            energy_MWh = vol_MW * hyperparams['timestep_hours']
            spot_buy_cost += energy_MWh * spot_15min.iloc[t]

        for (t, vol_MW) in technology_volume[tech_type]['spot_sold']:
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

    print(f"  ✓ Loaded demand: {len(demand_15min)} timesteps")
    print(f"  ✓ Loaded spot prices: {len(spot_15min)} timesteps")
    print(f"  ✓ Loaded solar/wind data: {len(solar_15min)}/{len(wind_15min)} timesteps")
    print(f"  ✓ Configured {len(ppu_dictionary)} PPUs from {len(ppu_types)} types")

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


def update_raw_energy_incidence(raw_energy_incidence: List[Dict], t: int,
                               solar_ranking_df: pd.DataFrame, wind_ranking_df: pd.DataFrame,
                               ppu_dictionary: pd.DataFrame) -> List[Dict]:
    """
    Update raw_energy_incidence with respective values for each energy source/storage.

    Parameters:
        raw_energy_incidence: List of incidence dictionaries
        t: Timestep index
        solar_ranking_df: Solar location rankings
        wind_ranking_df: Wind location rankings
        ppu_dictionary: PPU configuration

    Returns:
        Updated raw_energy_incidence list
    """
    for incidence_item in raw_energy_incidence:
        storage_name = incidence_item['storage']

        if storage_name == "River":
            # Update with current ROR incidence
            incidence_item["current_value"] = get_incidence_data('ror', t=t)
        elif storage_name == "Solar":
            # Update with current solar incidence (use first PV location as reference)
            pv_ppus = ppu_dictionary[ppu_dictionary['PPU_Name'] == 'PV']
            if not pv_ppus.empty:
                first_pv = pv_ppus.iloc[0]
                if not pd.isna(first_pv['Location_Rank']):
                    lat = solar_ranking_df.iloc[int(first_pv['Location_Rank']) - 1]['latitude']
                    lon = solar_ranking_df.iloc[int(first_pv['Location_Rank']) - 1]['longitude']
                    incidence_item["current_value"] = get_incidence_data('solar', lat=lat, lon=lon, t=t)
        elif storage_name == "Wind":
            # Update with current wind incidence (use first wind location as reference)
            wind_ppus = ppu_dictionary[ppu_dictionary['PPU_Name'].isin(['WD_OFF', 'WD_ON'])]
            if not wind_ppus.empty:
                first_wind = wind_ppus.iloc[0]
                if not pd.isna(first_wind['Location_Rank']):
                    lat = wind_ranking_df.iloc[int(first_wind['Location_Rank']) - 1]['latitude']
                    lon = wind_ranking_df.iloc[int(first_wind['Location_Rank']) - 1]['longitude']
                    incidence_item["current_value"] = get_incidence_data('wind', lat=lat, lon=lon, t=t)
        elif storage_name == "Wood":
            # Wood is a constant resource, set to maximum available
            incidence_item["current_value"] = 1000.0  # 1 GW constant availability
        # Grid and other storages remain at 0 until updated by incidence production

        # Update history for all items
        incidence_item['history'].append((t, incidence_item['current_value']))

    return raw_energy_incidence


def calculate_incidence_production(ppu_dictionary: pd.DataFrame, raw_energy_incidence: List[Dict],
                                 Technology_volume: Dict, t: int, solar_ranking_df: pd.DataFrame,
                                 wind_ranking_df: pd.DataFrame, hyperparams: Dict) -> List[Dict]:
    """
    Use 'Incidence' PPUs to calculate energy production and cost, update Technology_volume,
    and store produced energy in "Grid" in raw_energy_incidence.

    Parameters:
        ppu_dictionary: PPU configuration
        raw_energy_incidence: Incidence dictionaries
        Technology_volume: Technology tracking dictionary
        t: Timestep index
        solar_ranking_df: Solar location rankings
        wind_ranking_df: Wind location rankings
        hyperparams: Simulation hyperparameters

    Returns:
        Updated raw_energy_incidence with production stored in "Grid"
    """
    total_production_MW = 0.0

    for _, ppu_row in ppu_dictionary.iterrows():
        ppu_name = str(ppu_row['PPU_Name'])
        ppu_extract = str(ppu_row['PPU_Extract'])

        # Only process Incidence-based PPUs
        if ppu_extract != 'Incidence':
            continue

        location_rank_raw = ppu_row['Location_Rank']
        location_rank = location_rank_raw if not pd.isna(location_rank_raw) else np.nan

        production_MW = 0.0

        # Get storages this PPU can extract from
        can_extract_from = ppu_row.get('can_extract_from', [])
        if not can_extract_from:
            continue

        # Calculate production from each extractable storage
        for storage_name in can_extract_from:
            storage_production = 0.0

            if storage_name == "Solar":
                # Calculate solar production
                if pd.notna(location_rank):
                    rank_idx = int(location_rank) - 1
                    lat = solar_ranking_df.iloc[rank_idx]['latitude']
                    lon = solar_ranking_df.iloc[rank_idx]['longitude']
                    incidence = get_incidence_data('solar', lat=lat, lon=lon, t=t)
                    # Assume 100,000 m² area (this should be parameterized)
                    area_m2 = 100000
                    storage_production = float(incidence) * area_m2 * 0.25 / 1000

            elif storage_name == "Wind":
                # Calculate wind production
                if pd.notna(location_rank):
                    rank_idx = int(location_rank) - 1
                    lat = wind_ranking_df.iloc[rank_idx]['latitude']
                    lon = wind_ranking_df.iloc[rank_idx]['longitude']
                    wind_speed = get_incidence_data('wind', lat=lat, lon=lon, t=t)

                    # Wind turbine parameters
                    rotor_diameter_m = 120
                    air_density = 1.225  # kg/m³
                    swept_area = np.pi * (rotor_diameter_m / 2) ** 2  # m²
                    power_coefficient = 0.45  # turbine efficiency
                    num_turbines = 5  # Assume 5 turbines

                    power_per_turbine_W = 0.5 * air_density * swept_area * (float(wind_speed) ** 3) * power_coefficient
                    power_per_turbine_MW = power_per_turbine_W / 1e6
                    storage_production = power_per_turbine_MW * num_turbines

            elif storage_name == "River":
                # Calculate ROR production
                ror_total_MW = get_incidence_data('ror', t=t)
                num_ror_ppus = len(ppu_dictionary[(ppu_dictionary['PPU_Extract'] == 'Incidence') &
                                                 (ppu_dictionary['can_extract_from'].apply(lambda x: 'River' in x if isinstance(x, list) else False))])
                if num_ror_ppus > 0:
                    storage_production = min(float(ror_total_MW) / num_ror_ppus, 1000)  # 1 GW limit per PPU

            elif storage_name == "Wood":
                # Wood is a constant resource, assume 1 GW available per PPU
                storage_production = 1000.0

            production_MW += storage_production

        # Each PPU can produce at most 1 GW
        production_MW = min(production_MW, 1000)

        # Record production in Technology_volume
        Technology_volume[ppu_name]['production'].append((t, production_MW))

        # Add to total production
        total_production_MW += production_MW

    # Update raw_energy_incidence: set all to 0 except "Grid" which gets total production
    for incidence_item in raw_energy_incidence:
        if incidence_item['storage'] == 'Grid':
            incidence_item['current_value'] = total_production_MW
        else:
            incidence_item['current_value'] = 0.0

        # Update history
        incidence_item['history'].append((t, incidence_item['current_value']))

    return raw_energy_incidence


def calculate_ppu_indices(ppu_dictionary: pd.DataFrame, raw_energy_storage: List[Dict],
                         overflow_MW: float, phi_smoothed: float, spot_price: float,
                         hyperparams: Dict) -> Dict[str, Dict]:
    """
    Calculate d_stor, u_dis, u_chg and c indices for each PPU.

    Parameters:
        ppu_dictionary: PPU configuration
        raw_energy_storage: Storage systems
        overflow_MW: Current overflow (demand - supply)
        phi_smoothed: Smoothed overflow
        spot_price: Current spot price
        hyperparams: Simulation hyperparameters

    Returns:
        Dictionary mapping PPU names to their indices
    """
    ppu_indices = {}

    for _, ppu_row in ppu_dictionary.iterrows():
        ppu_name = str(ppu_row['PPU_Name'])
        ppu_category = str(ppu_row['PPU_Category'])

        # Initialize indices
        d_stor = 0.0
        u_dis = 0.0
        u_chg = 0.0
        c_idx = 0.0

        # Calculate disposition index for storage PPUs
        if ppu_category == 'Storage':
            # Get storage name from can_extract_from
            can_extract_from = ppu_row.get('can_extract_from', [])
            if can_extract_from:
                storage_name = can_extract_from[0]  # Use first storage
                storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)

                if storage_item:
                    current_soc = storage_item['current_value'] / storage_item['value']
                    target_soc = storage_item.get('target_SoC', 0.6)
                    deadband = 0.05

                    # Calculate disposition index
                    if current_soc > target_soc + deadband:
                        d_stor = min((current_soc - (target_soc + deadband)) / (1 - target_soc - deadband), 1.0)
                    elif current_soc < target_soc - deadband:
                        d_stor = max((current_soc - (target_soc - deadband)) / (target_soc - deadband), -1.0)
                    else:
                        d_stor = 0.0

        # Calculate utility indices based on overflow
        if overflow_MW > 0:
            u_dis = min(overflow_MW / hyperparams.get('alpha_u', 1000.0), 1.0)
        else:
            u_chg = min(abs(overflow_MW) / hyperparams.get('alpha_u', 1000.0), 1.0)

        # Calculate cost index (simplified)
        ppu_cost = ppu_row.get('Cost_CHF_per_kWh', 0.1)
        c_idx = (spot_price - ppu_cost) / max(ppu_cost, 0.01)
        c_idx = max(min(c_idx, 1.0), -1.0)  # Clamp to [-1, 1]

        ppu_indices[ppu_name] = {
            'd_stor': d_stor,
            'u_dis': u_dis,
            'u_chg': u_chg,
            'c': c_idx
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
        # No Flex PPUs available, buy from spot market
        tech_count = len(Technology_volume)
        spot_buy_per_tech = deficit_MW / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_bought'].append((t, spot_buy_per_tech))
        return raw_energy_storage

    # Calculate discharge costs and priorities
    discharge_costs = []
    total_inverse_cost = 0.0

    for ppu_row in flex_ppus:
        ppu_name = str(ppu_row['PPU_Name'])
        indices = ppu_indices.get(ppu_name, {'d_stor': 0, 'u_dis': 0, 'u_chg': 0, 'c': 0})

        # Calculate composite discharge benefit/cost
        B_dis = (indices['d_stor'] + indices['u_dis'] + indices['c']) / 3.0
        kappa_dis = 1.0 - B_dis

        discharge_costs.append({
            'ppu_name': ppu_name,
            'kappa_dis': kappa_dis,
            'ppu_row': ppu_row
        })

        total_inverse_cost += 1.0 / max(kappa_dis, hyperparams['epsilon'])

    # Allocate deficit inversely to cost
    deficit_handled = 0.0

    for cost_info in discharge_costs:
        ppu_name = cost_info['ppu_name']
        kappa_dis = cost_info['kappa_dis']
        ppu_row = cost_info['ppu_row']

        # Calculate allocation
        weight = (1.0 / max(kappa_dis, hyperparams['epsilon'])) / total_inverse_cost
        target_discharge_MW = deficit_MW * weight

        # Check available energy from linked storages
        can_extract_from = ppu_row.get('can_extract_from', [])
        available_energy_MW = 0.0

        for storage_name in can_extract_from:
            storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
            if storage_item:
                available_energy_MW += storage_item['current_value']

        # Limit by available energy and 1 GW PPU capacity
        actual_discharge_MW = min(target_discharge_MW, available_energy_MW, 1000)

        if actual_discharge_MW > 0:
            # Distribute discharge across storages proportionally
            total_available = available_energy_MW
            if total_available > 0:
                for storage_name in can_extract_from:
                    storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                    if storage_item and storage_item['current_value'] > 0:
                        storage_fraction = storage_item['current_value'] / total_available
                        storage_discharge = actual_discharge_MW * storage_fraction
                        storage_item['current_value'] -= storage_discharge
                        storage_item['history'].append((t, -storage_discharge))  # Negative for discharge

            # Record production
            Technology_volume[ppu_name]['production'].append((t, actual_discharge_MW))
            deficit_handled += actual_discharge_MW

    # Handle remaining deficit with spot market
    remaining_deficit = deficit_MW - deficit_handled
    if remaining_deficit > hyperparams['epsilon']:
        tech_count = len(Technology_volume)
        spot_buy_per_tech = remaining_deficit / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_bought'].append((t, spot_buy_per_tech))

    return raw_energy_storage


def handle_energy_surplus(surplus_MW: float, ppu_dictionary: pd.DataFrame,
                        raw_energy_storage: List[Dict], Technology_volume: Dict,
                        ppu_indices: Dict, spot_price: float, t: int,
                        hyperparams: Dict) -> List[Dict]:
    """
    Handle energy surplus using 'Store' PPUs that store in raw_energy_storage.

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
        # No Store PPUs available, sell to spot market
        tech_count = len(Technology_volume)
        spot_sell_per_tech = surplus_MW / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_sold'].append((t, spot_sell_per_tech))
        return raw_energy_storage

    # Calculate storage costs and priorities
    storage_costs = []
    total_inverse_cost = 0.0

    for ppu_row in store_ppus:
        ppu_name = str(ppu_row['PPU_Name'])
        indices = ppu_indices.get(ppu_name, {'d_stor': 0, 'u_dis': 0, 'u_chg': 0, 'c': 0})

        # Calculate composite storage benefit/cost (flip disposition and monetary terms)
        B_chg = (-indices['d_stor'] + indices['u_chg'] - indices['c']) / 3.0
        kappa_chg = 1.0 - B_chg

        storage_costs.append({
            'ppu_name': ppu_name,
            'kappa_chg': kappa_chg,
            'ppu_row': ppu_row
        })

        total_inverse_cost += 1.0 / max(kappa_chg, hyperparams['epsilon'])

    # Allocate surplus inversely to cost
    surplus_handled = 0.0

    for cost_info in storage_costs:
        ppu_name = cost_info['ppu_name']
        kappa_chg = cost_info['kappa_chg']
        ppu_row = cost_info['ppu_row']

        # Calculate allocation
        weight = (1.0 / max(kappa_chg, hyperparams['epsilon'])) / total_inverse_cost
        target_charge_MW = surplus_MW * weight

        # Check available capacity in linked storages
        can_input_to = ppu_row.get('can_input_to', [])
        available_capacity_MW = 0.0

        for storage_name in can_input_to:
            storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
            if storage_item:
                available_capacity_MW += storage_item['value'] - storage_item['current_value']

        # Limit by available capacity and 1 GW PPU capacity
        actual_charge_MW = min(target_charge_MW, available_capacity_MW, 1000)

        if actual_charge_MW > 0:
            # Distribute charge across storages proportionally
            total_available = available_capacity_MW
            if total_available > 0:
                for storage_name in can_input_to:
                    storage_item = next((item for item in raw_energy_storage if item['storage'] == storage_name), None)
                    if storage_item:
                        storage_capacity = storage_item['value'] - storage_item['current_value']
                        if storage_capacity > 0:
                            storage_fraction = storage_capacity / total_available
                            storage_charge = actual_charge_MW * storage_fraction
                            storage_item['current_value'] += storage_charge
                            storage_item['history'].append((t, storage_charge))  # Positive for charge

            # Record consumption (negative production for storage)
            Technology_volume[ppu_name]['production'].append((t, -actual_charge_MW))
            surplus_handled += actual_charge_MW

    # Handle remaining surplus with spot market
    remaining_surplus = surplus_MW - surplus_handled
    if remaining_surplus > hyperparams['epsilon']:
        tech_count = len(Technology_volume)
        spot_sell_per_tech = remaining_surplus / tech_count
        for tech_type in Technology_volume:
            Technology_volume[tech_type]['spot_sold'].append((t, spot_sell_per_tech))

    return raw_energy_storage

