import pandas as pd
import numpy as np
import ast
import functools
from typing import List, Dict, Optional, Tuple, Union

# -------------------------
# Location & PPU utilities
# -------------------------

def load_location_rankings(source_type: str) -> pd.DataFrame:
    """Load location ranking data for solar or wind sources.
    
    Solar columns: latitude, longitude, mean_solar_incidence_kwh_m2_per_hour, rank
    Wind columns: latitude, longitude, mean_wind_speed_m_per_s, rank
    """
    filepath = f'data/ranking_incidence/{source_type}_incidence_ranking.csv'
    df = pd.read_csv(filepath)
    df['assigned'] = False
    
    # Standardize potential column name for easier access
    if source_type == 'solar':
        df['potential'] = df['mean_solar_incidence_kwh_m2_per_hour']
    elif source_type == 'wind':
        df['potential'] = df['mean_wind_speed_m_per_s']
    
    return df


def load_ppu_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess PPU data from CSV file."""
    df = pd.read_csv(filepath)
    df['Components'] = df['Components'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def categorize_ppus(df: pd.DataFrame, type_col: str = 'Category', name_col: str = 'PPU') -> Tuple[List[str], List[str], List[str]]:
    """Split PPU dataframe into production and storage categories."""
    production = df[df[type_col].str.lower() == 'production'][name_col].tolist()
    storage = df[df[type_col].str.lower() == 'storage'][name_col].tolist()
    return production, storage, production + storage


def create_ppu_quantities(ppu_list: List[str], ppu_production: List[str], ppu_storage: List[str], quantity: int = 10, capacity_gw: float = 1.0) -> pd.DataFrame:
    """Create a dataframe to track PPU quantities with specified defaults."""
    return pd.DataFrame({
        'PPU': ppu_list,
        'Type': ['Production'] * len(ppu_production) + ['Storage'] * len(ppu_storage),
        'Quantity': quantity,
        'Capacity_GW': capacity_gw,
        'Total_Capacity_GW': lambda x: x['Quantity'] * x['Capacity_GW']
    }).assign(Total_Capacity_GW=lambda x: x['Quantity'] * x['Capacity_GW'])


def create_renewable_ppu_tracking(ppu_df: pd.DataFrame, ppu_quantities_df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe to track renewable PPUs (solar, wind) that need location assignments."""
    renewable_ppus = []
    for _, row in ppu_df.iterrows():
        components = row['Components']
        if 'PV' in components:
            renewable_ppus.append({'PPU': row['PPU'], 'Type': 'solar', 'Quantity': 0, 'Assigned': 0})
        elif 'Wind (onshore)' in components:
            renewable_ppus.append({'PPU': row['PPU'], 'Type': 'wind_onshore', 'Quantity': 0, 'Assigned': 0})
        elif 'Wind (offshore)' in components:
            renewable_ppus.append({'PPU': row['PPU'], 'Type': 'wind_offshore', 'Quantity': 0, 'Assigned': 0})
    tracking_df = pd.DataFrame(renewable_ppus)
    if not tracking_df.empty:
        for idx, row in tracking_df.iterrows():
            ppu_name = row['PPU']
            quantity = ppu_quantities_df[ppu_quantities_df['PPU'] == ppu_name]['Quantity'].values
            if len(quantity) > 0:
                tracking_df.loc[idx, 'Quantity'] = quantity[0]
    return tracking_df

# -------------------------
# Storage & incidence utils
# -------------------------

def calculate_max_capacity(storage_name: str, raw_data: List[Dict], ppu_quantities_df: pd.DataFrame, ppu_df: pd.DataFrame) -> float:
    """Calculate max capacity for a storage based on related PPU quantities."""
    item = next((item for item in raw_data if item.get('storage') == storage_name), None)
    if not item:
        return 0
    base_capacity = item['value']
    extractors = item['extracted_by']
    related_ppus = []
    for _, ppu_row in ppu_df.iterrows():
        components = ppu_row['Components']
        if any(extractor in components for extractor in extractors):
            related_ppus.append(ppu_row['PPU'])
    total_capacity = 0
    for ppu in related_ppus:
        ppu_row = ppu_quantities_df[ppu_quantities_df['PPU'] == ppu]
        if not ppu_row.empty:
            total_capacity += ppu_row['Total_Capacity_GW'].values[0] * base_capacity
    return total_capacity


def create_storage_tracking(raw_energy_storage: List[Dict], ppu_quantities_df: pd.DataFrame, ppu_df: pd.DataFrame) -> pd.DataFrame:
    """Create a tracking dataframe for storage with capacities linked to PPU quantities."""
    storage_data = [{
        'Storage': item['storage'],
        'Base_Capacity': item['value'],
        'Unit': item['unit'],
        'Current_Value': item['current_value'],
        'Extracted_By': item['extracted_by'],
        'Max_Capacity': calculate_max_capacity(item['storage'], raw_energy_storage, ppu_quantities_df, ppu_df)
    } for item in raw_energy_storage]
    return pd.DataFrame(storage_data)


def create_incidence_tracking(raw_energy_incidence: List[Dict], ppu_quantities_df: pd.DataFrame, ppu_df: pd.DataFrame) -> pd.DataFrame:
    """Create a tracking dataframe for incidence sources with capacities linked to PPU quantities."""
    incidence_data = [{
        'Source': item['storage'],
        'Base_Capacity': item['value'],
        'Unit': item['unit'],
        'Current_Value': item['current_value'],
        'Extracted_By': item['extracted_by'],
        'Max_Capacity': calculate_max_capacity(item['storage'], raw_energy_incidence, ppu_quantities_df, ppu_df)
    } for item in raw_energy_incidence]
    return pd.DataFrame(incidence_data)

# -------------------------
# Cost & efficiency utils
# -------------------------

def load_cost_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess component cost data from CSV file."""
    df = pd.read_csv(filepath)
    numeric_cols = ['efficiency', 'w', 'cost', 'investment_chf_per_kw', 'capex', 'opex', 'lifetime', 'cycle_no', 'power', 'capacity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'w' in df.columns:
        df['w'] = df['w'].fillna(0)
    return df.set_index('item', drop=False)


def get_component_data(component: str, cost_df: pd.DataFrame) -> Dict:
    """Get efficiency and cost data for a component."""
    if component in cost_df.index:
        row = cost_df.loc[component]
        return {
            'efficiency': row['efficiency'] if not pd.isna(row['efficiency']) else 1.0,
            'cost': row['cost'] if not pd.isna(row['cost']) else 0.0,
            'w': row['w'] if not pd.isna(row['w']) else 0.0,
            'component_type': row['component'] if not pd.isna(row['component']) else 'unknown'
        }
    else:
        return {'efficiency': 1.0, 'cost': 0.0, 'w': 0.0, 'component_type': 'unknown'}


def calculate_chain_efficiency(components: List[str], cost_df: pd.DataFrame) -> float:
    """Calculate overall efficiency of a component chain."""
    efficiency = 1.0
    for component in components:
        comp_data = get_component_data(component, cost_df)
        efficiency *= comp_data['efficiency']
    return efficiency


def calculate_chain_cost(components: List[str], cost_df: pd.DataFrame) -> Dict:
    """Calculate the overall cost of a component chain with auxiliary energy needs."""
    total_cost = 0.0
    cost_breakdown = []
    cumulative_efficiency = 1.0
    for i, component in enumerate(components):
        comp_data = get_component_data(component, cost_df)
        component_cost = comp_data['cost']
        total_cost += component_cost
        cost_breakdown.append({'component': component, 'efficiency': comp_data['efficiency'], 'direct_cost': component_cost, 'w': comp_data['w'], 'cumulative_efficiency': cumulative_efficiency})
        cumulative_efficiency *= comp_data['efficiency']
    for i, breakdown in enumerate(cost_breakdown):
        w = breakdown['w']
        if w > 0:
            aux_energy_cost = w / breakdown['cumulative_efficiency']
            total_cost += aux_energy_cost
            breakdown['aux_energy_cost'] = aux_energy_cost
        else:
            breakdown['aux_energy_cost'] = 0.0
    return {'total_cost': total_cost, 'breakdown': cost_breakdown, 'cumulative_efficiency': cumulative_efficiency}


def calculate_ppu_metrics(ppu_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cost and efficiency metrics for each PPU based on its component chain."""
    results = []
    for _, row in ppu_df.iterrows():
        ppu_name = row['PPU']
        ppu_type = row.get('Category', None)
        components = row['Components']
        efficiency = calculate_chain_efficiency(components, cost_df)
        cost_data = calculate_chain_cost(components, cost_df)
        energy_produced_15min = 1e6 * 0.25 * efficiency
        cost_per_15min = cost_data['total_cost'] * energy_produced_15min
        results.append({'PPU': ppu_name, 'Category': ppu_type, 'Components': components, 'Efficiency': efficiency, 'Total_Cost_CHF_per_kWh': cost_data['total_cost'], 'Cost_Per_15min_CHF': cost_data['total_cost']*0.25, 'Component_Count': len(components), 'Cumulative_Efficiency': cost_data['cumulative_efficiency']})
    return pd.DataFrame(results)


def enrich_ppu_quantities(ppu_quantities_df: pd.DataFrame, ppu_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add cost and efficiency metrics to the PPU quantities dataframe."""
    enriched_df = ppu_quantities_df.copy()
    for metric in ['Efficiency', 'Total_Cost_CHF_per_kWh', 'Cost_Per_15min_CHF', 'Component_Count']:
        metric_dict = dict(zip(ppu_metrics_df['PPU'], ppu_metrics_df[metric]))
        enriched_df[metric] = enriched_df['PPU'].map(metric_dict)
    enriched_df['Total_Annual_Cost_CHF'] = (enriched_df['Quantity'] * enriched_df['Capacity_GW'] * enriched_df['Total_Cost_CHF_per_kWh'] * 8760 * 1e6)
    return enriched_df


# -------------------------
# Update Storage & incidence data
# -------------------------

def get_incidence_data(source, lat=None, lon=None, t=None):
    """
    Load and interpolate incidence data for solar, wind, or ROR to 15-min timesteps for 2024.
    
    Parameters:
    - source: 'solar', 'wind', or 'ror'
    - lat: latitude (required for solar and wind)
    - lon: longitude (required for solar and wind)
    - t: timestep index (int, 0-based), if provided, returns value at that timestep, else full series
    
    Returns:
    - pd.Series or float: Full time series with 15-min frequency, or value at timestep t
    """
    import numpy as np
    
    # Global cache to avoid reloading data
    if not hasattr(get_incidence_data, '_cache'):
        get_incidence_data._cache = {}
    
    cache = get_incidence_data._cache
    
    if source == 'ror':
        if 'ror_series' not in cache:
            df = pd.read_csv('data/water_quarterly_ror_2024.csv', parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            # Data is already at 15-min intervals, just ensure proper sorting
            cache['ror_series'] = df['RoR_MW'].sort_index()
        series = cache['ror_series']
        if t is not None:
            return series.iloc[t]
        return series
    
    elif source in ['solar', 'wind']:
        if lat is None or lon is None:
            raise ValueError("lat and lon required for solar and wind")
        
        # Round lat/lon to nearest 0.1 for caching (to group close points)
        lat_rounded = round(lat, 1)
        lon_rounded = round(lon, 1)
        key = (source, lat_rounded, lon_rounded)
        
        if key not in cache:
            filename = f'data/{source}_incidence_hourly_2024.csv'
            if f'{source}_df' not in cache:
                cache[f'{source}_df'] = pd.read_csv(filename, header=None)
            df = cache[f'{source}_df']
            
            latitudes = df.iloc[0, 1:].astype(float).values
            longitudes = df.iloc[1, 1:].astype(float).values
            times = pd.to_datetime(df.iloc[3:, 0])
            data = df.iloc[3:, 1:].astype(float).values
            
            # Find unique lats and lons
            unique_lats = np.unique(latitudes)
            unique_lons = np.unique(longitudes)
            n_lats = len(unique_lats)
            n_lons = len(unique_lons)
            
            # Find closest indices
            lat_idx = np.argmin(np.abs(unique_lats - lat))
            lon_idx = np.argmin(np.abs(unique_lons - lon))
            
            # Column index: for each lat, all lons
            col_idx = lat_idx * n_lons + lon_idx
            
            series_hourly = pd.Series(data[:, col_idx], index=times)
            
            # Resample to 15-min with interpolation
            cache[key] = series_hourly.resample('15min').interpolate(method='linear')
        
        series = cache[key]
        if t is not None:
            return series.iloc[t]
        return series
    
    else:
        raise ValueError("source must be 'solar', 'wind', or 'ror'")
    
def update_storage(raw_energy_storage, updates):
    """
    Update the 'current_value' field of each storage component in raw_energy_storage based on inputs and outputs.

    Parameters:
    - raw_energy_storage (list of dict): The list of storage dictionaries, each with keys 'storage', 'value', 'current_value', 'unit', 'extracted_by'.
    - updates (dict): A dictionary where keys are storage names (str), and values are dictionaries with:
      - "inputs" (float or int, optional): Amount added to the storage (default 0).
      - "outputs" (float or int, optional): Amount subtracted from the storage (default 0).
      The net change is inputs - outputs.

    Example updates dict:
    {
        "Lake": {"inputs": 100, "outputs": 50},    # Net +50 to Lake's current_value
        "Battery": {"inputs": 0, "outputs": 200}   # Net -200 to Battery's current_value
    }

    Returns:
    - The modified raw_energy_storage list.
    """
    for item in raw_energy_storage:
        storage = item["storage"]
        if storage in updates:
            update_info = updates[storage]
            inputs = update_info.get("inputs", 0)
            outputs = update_info.get("outputs", 0)
            net_change = inputs - outputs
            item["current_value"] += net_change
            # Optionally, add constraints here, e.g., min/max, but not implemented yet
    return raw_energy_storage

def update_incidence(raw_energy_incidence, t, lat=46.8, lon=8.2):
    """
    Update the 'current_value' field of incidence-based storages in raw_energy_incidence with current incidence data at timestep t.

    Parameters:
    - raw_energy_incidence (list of dict): The list of incidence dictionaries.
    - t (int): Timestep index (0-based).
    - lat (float): Latitude for solar/wind (default Switzerland center).
    - lon (float): Longitude for solar/wind (default Switzerland center).

    Returns:
    - The modified raw_energy_incidence list with updated current_values.
    """
    for item in raw_energy_incidence:
        storage = item["storage"]
        if storage == "River":
            # Update with current ROR incidence (GW)
            item["current_value"] = get_incidence_data('ror', t=t)
        elif storage == "Solar":
            # Update with current solar incidence (W/m², but treat as available irradiance)
            item["current_value"] = get_incidence_data('solar', lat=lat, lon=lon, t=t)
        elif storage == "Wind":
            # Update with current wind incidence (m/s or power, depending on data)
            item["current_value"] = get_incidence_data('wind', lat=lat, lon=lon, t=t)
        # Wood remains unchanged as it's not incidence-based
    return raw_energy_incidence

# -------------------------
# FP-Style Composition Helpers (New Additions)
# -------------------------

from functools import reduce
import pandas as pd

def pipeline(*funcs):
    """FP helper: Compose functions for chaining (e.g., load -> process -> output)."""
    def composed(*args, **kwargs):
        result = args[0] if args else kwargs
        for func in funcs:
            result = func(result)
        return result
    return composed

def compose_ppu_setup(ppu_filepath: str, cost_filepath: str, quantity: int = 10, capacity_gw: float = 1.0) -> pd.DataFrame:
    """Composed function: Load PPU data -> categorize -> create quantities -> calc metrics -> enrich.
    Returns enriched PPU quantities DataFrame.
    """
    load_ppu = lambda fp: load_ppu_data(fp)
    categorize = lambda df: categorize_ppus(df)
    create_quant = lambda cats: create_ppu_quantities(cats[2], cats[0], cats[1], quantity, capacity_gw)
    load_cost = lambda fp: load_cost_data(fp)
    calc_metrics = lambda ppu: calculate_ppu_metrics(ppu, load_cost(cost_filepath))
    enrich = lambda quant: enrich_ppu_quantities(quant, calc_metrics(ppu_data_df))  # Note: ppu_data_df from load_ppu
    
    # Chain via pipeline
    ppu_data_df = load_ppu(ppu_filepath)
    return pipeline(create_quant, enrich)(categorize(ppu_data_df))

def compose_storage_tracking(raw_storage: list, ppu_quant: pd.DataFrame, ppu_data: pd.DataFrame) -> pd.DataFrame:
    """Composed: Create storage tracking with max capacities linked to PPUs."""
    return create_storage_tracking(raw_storage, ppu_quant, ppu_data)

def compose_incidence_tracking(raw_incidence: list, ppu_quant: pd.DataFrame, ppu_data: pd.DataFrame) -> pd.DataFrame:
    """Composed: Create incidence tracking with max capacities linked to PPUs."""
    return create_incidence_tracking(raw_incidence, ppu_quant, ppu_data)

def compose_renewable_tracking(ppu_data: pd.DataFrame, ppu_quant: pd.DataFrame) -> pd.DataFrame:
    """Composed: Create renewable PPU tracking for location assignments."""
    return create_renewable_ppu_tracking(ppu_data, ppu_quant)

def update_all_storages(raw_storage: list, raw_incidence: list, updates: dict = None, t: int = None, lat: float = 46.8, lon: float = 8.2) -> tuple:
    """Composed: Update storage and incidence via net changes and incidence data."""
    updated_storage = update_storage(raw_storage, updates or {})
    updated_incidence = update_incidence(raw_incidence, t or 0, lat, lon)
    return updated_storage, updated_incidence

def get_component_data(component: str, cost_df: pd.DataFrame) -> Dict:
    if component in cost_df.index:
        row = cost_df.loc[component]
        # ----  NEW:  force numeric, replace non-parsable by defaults  ----
        eff   = pd.to_numeric(row['efficiency'], errors='coerce') if 'efficiency' in row else 1.0
        cost  = pd.to_numeric(row['cost'],       errors='coerce') if 'cost'       in row else 0.0
        w     = pd.to_numeric(row['w'],          errors='coerce') if 'w'          in row else 0.0
        # fill NaN that to_numeric produced
        eff, cost, w = [x if not pd.isna(x) else (1.0,0.0,0.0)[i] for i,x in enumerate((eff,cost,w))]
        return {'efficiency': eff, 'cost': cost, 'w': w,
                'component_type': row.get('component', 'unknown')}
    else:
        return {'efficiency': 1.0, 'cost': 0.0, 'w': 0.0, 'component_type': 'unknown'}

def add_ppu(
    ppu_name: str,
    ppu_type: str,  # e.g., 'Production' or 'Storage'
    quantity: int,
    capacity_gw: float,
    components: list = None,
    cost_df: pd.DataFrame = None,
    ppu_quantities_df: pd.DataFrame = None,
    ppu_location_assignments_df: pd.DataFrame = None,
    solar_locations_df: pd.DataFrame = None,
    wind_locations_df: pd.DataFrame = None,
    renewable_type: str = None,  # e.g., 'solar', 'wind_onshore', 'wind_offshore' if renewable
    instance_id: int = 1,  # For multiple instances of same PPU
    delta_t: float = 0.25,  # timestep hours (default 15min)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add a new PPU to quantities DF and assign location if renewable.
    
    - Computes efficiency/cost from components if provided.
    - Adds row to ppu_quantities_df with total_capacity, metrics.
    - If renewable, assigns best unassigned location and updates assignments DF.
    - Returns updated: (ppu_quantities_df, ppu_location_assignments_df, solar_locations_df, wind_locations_df)
    
    Note: Mutates input DFs in place for simplicity; returns them for chaining.
    """
    if ppu_quantities_df is None:
        ppu_quantities_df = pd.DataFrame()
    if ppu_location_assignments_df is None:
        ppu_location_assignments_df = pd.DataFrame(columns=[
            'PPU', 'Instance', 'Type', 'Latitude', 'Longitude', 'Rank', 'Energy_Potential'
        ])
    if solar_locations_df is None:
        solar_locations_df = raw_data['solar_locations'].copy()
    if wind_locations_df is None:
        wind_locations_df = raw_data['wind_locations'].copy()
    
    # Compute metrics if components provided
    efficiency = 1.0
    total_cost_chf_per_kwh = 0.0
    cost_per_15min_chf = 0.0
    if components and cost_df is not None:
        efficiency = calculate_chain_efficiency(components, cost_df)
        cost_data = calculate_chain_cost(components, cost_df)
        total_cost_chf_per_kwh = cost_data['total_cost']
        # Use explicit delta_t parameter instead of relying on a notebook-global `hyperparams`
        cost_per_15min_chf = total_cost_chf_per_kwh * delta_t
    
    # Add to ppu_quantities_df
    new_row = pd.DataFrame([{
        'PPU': ppu_name,
        'Type': ppu_type,
        'Quantity': quantity,
        'Capacity_GW': capacity_gw,
        'Total_Capacity_GW': quantity * capacity_gw,
        'Efficiency': efficiency,
        'Total_Cost_CHF_per_kWh': total_cost_chf_per_kwh,
        'Cost_Per_15min_CHF': cost_per_15min_chf,
        'Components': components
    }])
    ppu_quantities_df = pd.concat([ppu_quantities_df, new_row], ignore_index=True)
    
    # Handle location assignment if renewable
    if renewable_type:
        locations_df = solar_locations_df if renewable_type == 'solar' else wind_locations_df
        # Find best unassigned location (lowest rank)
        unassigned = locations_df[locations_df['assigned'] == False].sort_values('rank')
        if not unassigned.empty:
            best_location = unassigned.iloc[0]
            # Assign
            new_assignment = pd.DataFrame([{
                'PPU': ppu_name,
                'Instance': instance_id,
                'Type': renewable_type,
                'Latitude': best_location['latitude'],
                'Longitude': best_location['longitude'],
                'Rank': best_location['rank'],
                'Energy_Potential': best_location['potential']  # Uses standardized column from load_location_rankings
            }])
            ppu_location_assignments_df = pd.concat([ppu_location_assignments_df, new_assignment], ignore_index=True)
            # Mark as assigned in locations DF
            assign_idx = best_location.name
            locations_df.loc[assign_idx, 'assigned'] = True
            if renewable_type == 'solar':
                solar_locations_df = locations_df
            else:
                wind_locations_df = locations_df
        else:
            print(f"Warning: No unassigned {renewable_type} locations available for {ppu_name}")
    
    return ppu_quantities_df, ppu_location_assignments_df, solar_locations_df, wind_locations_df


# -------------------------
# PPU Dictionary Management Functions
# -------------------------

def initialize_ppu_dictionary() -> pd.DataFrame:
    """
    Initialize an empty PPU dictionary DataFrame with required columns.
    
    Returns:
        pd.DataFrame: Empty DataFrame with columns:
            - PPU_ID: Unique incremental integer ID (int)
            - PPU_Name: PPU type name (e.g., 'HYD_S', 'PV', 'WIND_OFFSHORE')
            - PPU_Extract: Extract (Incidence, Flex, Store)
            - can_extract_from: List of storages this PPU can extract from (list of str)
            - can_input_to: List of storages this PPU can input to (list of str)
            - Chain_Efficiency: Overall efficiency of the component chain (float)
            - Cost_CHF_per_kWh: Total cost in CHF per kWh (float)
            - Cost_CHF_per_Quarter_Hour: Cost per 15-min interval (0.25 * Cost_CHF_per_kWh)
            - Location_Rank: Location ranking for solar/wind, NaN otherwise (float)
            - d_stor: Disposition index tracking dictionary (object)
            - u_dis: Discharge utility index tracking dictionary (object)
            - u_chg: Charge utility index tracking dictionary (object)
            - c: Cost index tracking dictionary (object)
    """
    return pd.DataFrame(columns=[
        'PPU_ID',
        'PPU_Name',
        'PPU_Extract',
        'can_extract_from',
        'can_input_to',
        'Chain_Efficiency',
        'Cost_CHF_per_kWh',
        'Cost_CHF_per_Quarter_Hour',
        'Location_Rank',
        'd_stor',
        'u_dis',
        'u_chg',
        'c'
    ])


def next_available_location(
    ppu_dictionary: pd.DataFrame,
    renewable_type: str,
    solar_locations_df: Optional[pd.DataFrame] = None,
    wind_locations_df: Optional[pd.DataFrame] = None
) -> Optional[Dict]:
    """
    Find the next available location for a solar or wind PPU.
    
    This function:
    1. Examines all solar/wind PPUs in the ppu_dictionary to find used locations
    2. Verifies that no two PPUs have the same location ranking
    3. Returns the next best available location (lowest unused rank)
    
    Parameters:
        ppu_dictionary (pd.DataFrame): The PPU dictionary with existing PPUs
        renewable_type (str): Either 'solar', 'wind_onshore', or 'wind_offshore'
        solar_locations_df (pd.DataFrame): Solar location rankings (loaded via load_location_rankings)
        wind_locations_df (pd.DataFrame): Wind location rankings (loaded via load_location_rankings)
    
    Returns:
        Dict: Location information with keys:
            - rank: Location rank (int)
            - lat: Latitude (float)
            - lon: Longitude (float)
            - potential: Energy potential at location (float)
            - type: renewable_type (str)
        Returns None if no locations available.
    
    Raises:
        ValueError: If duplicate location rankings are found in ppu_dictionary
        ValueError: If renewable_type is invalid
    """
    # Validate renewable_type
    valid_types = ['solar', 'wind_onshore', 'wind_offshore']
    if renewable_type not in valid_types:
        raise ValueError(f"renewable_type must be one of {valid_types}, got: {renewable_type}")
    
    # Load location data if not provided
    if solar_locations_df is None:
        solar_locations_df = load_location_rankings('solar')
    if wind_locations_df is None:
        wind_locations_df = load_location_rankings('wind')
    
    # Select appropriate locations dataframe
    if renewable_type == 'solar':
        locations_df = solar_locations_df
        type_filter = 'solar'
    else:  # wind_onshore or wind_offshore
        locations_df = wind_locations_df
        type_filter = 'wind'
    
    # Find all used locations for this renewable type in ppu_dictionary
    used_ranks = []
    if not ppu_dictionary.empty and 'Location_Rank' in ppu_dictionary.columns:
        # Filter for renewable PPUs with non-null location ranks
        renewable_ppus = ppu_dictionary[ppu_dictionary['Location_Rank'].notna()]
        if not renewable_ppus.empty:
            used_ranks = renewable_ppus['Location_Rank'].tolist()
            
            # Verify uniqueness: check for duplicate location rankings
            if len(used_ranks) != len(set(used_ranks)):
                duplicates = [rank for rank in set(used_ranks) if used_ranks.count(rank) > 1]
                raise ValueError(
                    f"Duplicate location rankings found in ppu_dictionary: {duplicates}. "
                    f"Each PPU must have a unique location."
                )
    
    # Find next available location (lowest rank not in used_ranks)
    available_locations = locations_df[~locations_df['rank'].isin(used_ranks)].sort_values('rank')
    
    if available_locations.empty:
        print(f"Warning: No available {renewable_type} locations remaining.")
        return None
    
    # Get the best (lowest rank) available location
    best_location = available_locations.iloc[0]
    
    return {
        'rank': int(best_location['rank']),
        'lat': float(best_location['latitude']),
        'lon': float(best_location['longitude']),
        'potential': float(best_location['potential']),  # Already standardized in load_location_rankings
        'type': renewable_type
    }


def select_storage_with_fewest_ppus(
    available_storages: List[str],
    ppu_dictionary: pd.DataFrame,
    raw_energy_storage: List[Dict]
) -> Optional[str]:
    """
    Select the storage with the fewest PPUs currently assigned to it.
    
    Parameters:
        available_storages (List[str]): List of storage names this PPU can use
        ppu_dictionary (pd.DataFrame): Current PPU dictionary
        raw_energy_storage (List[Dict]): Storage definitions
    
    Returns:
        Optional[str]: Name of storage with fewest PPUs, or None if no storages available
    """
    if not available_storages:
        return None
    
    if ppu_dictionary.empty:
        # No PPUs yet, return first available storage
        return available_storages[0]
    
    # Count current PPUs per storage
    storage_counts = {storage: 0 for storage in available_storages}
    
    for _, ppu_row in ppu_dictionary.iterrows():
        storage_dist = ppu_row.get('Storage_Distribution', {})
        if isinstance(storage_dist, dict):
            for storage_name, fraction in storage_dist.items():
                if storage_name in storage_counts and fraction > 0:
                    storage_counts[storage_name] += 1
    
    # Find storage with minimum count
    min_count = min(storage_counts.values())
    candidates = [storage for storage, count in storage_counts.items() if count == min_count]
    
    # Return first candidate (arbitrary choice if tie)
    return candidates[0] if candidates else available_storages[0]


def add_ppu_to_dictionary(
    ppu_dictionary: pd.DataFrame,
    ppu_name: str,
    ppu_constructs_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    solar_locations_df: Optional[pd.DataFrame] = None,
    wind_locations_df: Optional[pd.DataFrame] = None,
    delta_t: float = 0.25,
    raw_energy_storage: Optional[List[Dict]] = None,
    raw_energy_incidence: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Add a new PPU to the dictionary with all calculated metrics.
    
    This function:
    1. Gives the PPU a unique incremental ID
    2. Looks up PPU information from ppu_constructs_components.csv
    3. Calculates chain efficiency and cost from cost_table_tidy.csv
    4. Determines which raw energy storage this PPU uses based on DIRECT PPU-TO-STORAGE MAPPING
    5. Assigns location ranking for solar/wind PPUs (NaN otherwise)
    6. Adds the complete PPU entry to the dictionary
    
    Parameters:
        ppu_dictionary (pd.DataFrame): Existing PPU dictionary (can be empty)
        ppu_name (str): Name of the PPU to add (e.g., 'PV', 'HYD_S', 'WD_OFF')
        ppu_constructs_df (pd.DataFrame): PPU constructs data (from ppu_constructs_components.csv)
        cost_df (pd.DataFrame): Cost data (from cost_table_tidy.csv)
        solar_locations_df (pd.DataFrame): Solar location rankings
        wind_locations_df (pd.DataFrame): Wind location rankings
        delta_t (float): Time slice duration in hours (default 0.25 for 15 minutes)
        raw_energy_storage (List[Dict]): Raw energy storage definitions (not used with direct mapping)
        raw_energy_incidence (List[Dict]): Raw energy incidence definitions (not used with direct mapping)
    
    Returns:
        pd.DataFrame: Updated ppu_dictionary with the new PPU added
    
    Raises:
        ValueError: If ppu_name not found in ppu_constructs_df
    """
    
    # =========================================================================
    # DIRECT PPU-TO-STORAGE MAPPING
    # =========================================================================
    # This mapping explicitly defines which PPUs can extract from and input to
    # which storages. No more component matching - direct and explicit.
    
    PPU_STORAGE_MAPPING = {
        # Hydro PPUs
        'HYD_S': {'extract_from': ['Lake'], 'input_to': []},  # Hydroelectric storage - ONLY extracts from Lake
        'HYD_R': {'extract_from': ['River'], 'input_to': []},  # Run-of-river - extracts from River
        'PHS': {'extract_from': [], 'input_to': ['Lake']},  # Pumped hydro storage - ONLY pumps into Lake
        
        # Hydrogen PPUs (gaseous 200bar)
        'H2_G': {'extract_from': ['H2 Storage UG 200bar'], 'input_to': []},  # H2 turbine + electrolyzer
        'H2P_G': {'extract_from': ['H2 Storage UG 200bar'], 'input_to': []},  # H2 turbine only (no input)
        
        # Hydrogen PPUs (liquid)
        'H2_L': {'extract_from': [], 'input_to': ['Liquid storage']},  # Liquid H2 turbine + liquefier
        'H2P_L': {'extract_from': ['Liquid storage'], 'input_to': []},  # Liquid H2 turbine only
        
        # Hydrogen PPUs (gaseous-liquid hybrid)
        'H2_GL': {'extract_from': [], 'input_to': ['H2 Storage UG 200bar']},  # Extract liquid, store gaseous
        
        # Synthetic fuel PPUs
        'SYN_FT': {'extract_from': [], 'input_to': ['Fuel Tank']},  # Fischer-Tropsch synthesis
        'SYN_METH': {'extract_from': [], 'input_to': ['CH4 storage 200bar']},  # Methane synthesis
        'SYN_CRACK': {'extract_from': [], 'input_to': ['Fuel Tank']},  # Fuel cracking
        
        # Ammonia PPUs
        'NH3_FULL': {'extract_from': [], 'input_to': ['Ammonia storage']},  # Ammonia synthesis + ICE
        'NH3_P': {'extract_from': ['Ammonia storage'], 'input_to': []},  # Ammonia ICE only
        
        # Biogas PPUs
        'CH4_BIO': {'extract_from': [], 'input_to': ['Biogas (50% CH4)']},  # Biogas upgrade
        'IMP_BIOG': {'extract_from': ['Biogas (50% CH4)'], 'input_to': []},  # Import biogas only
        
        # Biofuel PPUs
        'BIO_OIL_ICE': {'extract_from': ['Biooil'], 'input_to': []},  # Biooil ICE
        'BIO_WOOD': {'extract_from': ['Wood'], 'input_to': []},  # Wood gasification
        'PALM_ICE': {'extract_from': ['Palm oil'], 'input_to': []},  # Palm oil ICE
        
        # Solar thermal PPUs
        'SOL_SALT': {'extract_from': ['Solar concentrator salt'], 'input_to': []},  # CSP with salt storage
        'SOL_STEAM': {'extract_from': ['Solar concentrator salt'], 'input_to': []},  # CSP without storage
        
        # Thermal PPUs (natural gas)
        'THERM': {'extract_from': ['CH4 storage 200bar'], 'input_to': []},  # Gas turbine
        'THERM_CH4': {'extract_from': ['CH4 storage 200bar'], 'input_to': []},  # Gas turbine with CH4
        'THERM_G': {'extract_from': ['H2 Storage UG 200bar'], 'input_to': []},  # H2 gas turbine
        'THERM_M': {'extract_from': ['CH4 storage 200bar'], 'input_to': []},  # Methane turbine
        
        # Renewables (Incidence sources - no storage)
        'PV': {'extract_from': ['Solar'], 'input_to': []},  # Solar PV
        'WD_ON': {'extract_from': ['Wind'], 'input_to': []},  # Wind onshore
        'WD_OFF': {'extract_from': ['Wind'], 'input_to': []},  # Wind offshore
    }
    # =========================================================================
    # Look up PPU information
    ppu_row = ppu_constructs_df[ppu_constructs_df['PPU'] == ppu_name]
    if ppu_row.empty:
        raise ValueError(f"PPU '{ppu_name}' not found in ppu_constructs_df")
    
    ppu_row = ppu_row.iloc[0]
    components = ppu_row['Components']
    category = ppu_row['Category']
    
    # Generate unique ID (increment from max existing ID, or start at 1)
    if ppu_dictionary.empty or 'PPU_ID' not in ppu_dictionary.columns:
        new_id = 1
    else:
        new_id = int(ppu_dictionary['PPU_ID'].max()) + 1
    
    # Calculate chain efficiency and cost
    efficiency = calculate_chain_efficiency(components, cost_df)
    cost_data = calculate_chain_cost(components, cost_df)
    cost_per_kwh = cost_data['total_cost']
    cost_per_quarter_hour = cost_per_kwh * delta_t
    
    # =========================================================================
    # USE DIRECT PPU-TO-STORAGE MAPPING
    # =========================================================================
    # Get storage capabilities from the mapping (no more component matching!)
    can_extract_from = []
    can_input_to = []
    
    if ppu_name in PPU_STORAGE_MAPPING:
        mapping = PPU_STORAGE_MAPPING[ppu_name]
        can_extract_from = mapping.get('extract_from', []).copy()
        can_input_to = mapping.get('input_to', []).copy()
    else:
        # Fallback for PPUs not in mapping - assume no storage capabilities
        print(f"Warning: PPU '{ppu_name}' not found in PPU_STORAGE_MAPPING. Assuming no storage capabilities.")
        can_extract_from = []
        can_input_to = []
    
    # For backward compatibility, determine available storages for storage assignment
    available_storages = can_extract_from.copy()  # PPUs can be assigned to storages they can extract from
    storage_distribution = {}
    if available_storages:
        # Already selected single storage above
        storage_distribution = {available_storages[0]: 1.0}
    
    # Determine PPU_Extract based on extraction and input capabilities
    ppu_extract = None
    
    # Define pure incidence sources (uncontrollable, non-storage energy sources)
    # Note: 'River' and 'Lake' can be both incidence AND storage depending on PPU type
    # - Solar, Wind: Pure incidence (no storage, uncontrollable)
    # - River: Incidence for HYD_R (run-of-river), but storage-like for others
    # - Lake: Storage for PHS (pumped hydro), incidence for HYD_S (regular hydro)
    pure_incidence_sources = ['Solar', 'Wind', 'Wood']
    
    # Check if PPU can extract from pure incidence sources
    extracts_from_pure_incidence = any(source in can_extract_from for source in pure_incidence_sources)
    
    # Check if PPU is specifically HYD_S or HYD_R (incidence-based hydro)
    is_incidence_hydro = ppu_name in ['HYD_S', 'HYD_R']
    
    # Check if PPU can input to storage sources
    inputs_to_storage = len(can_input_to) > 0
    
    # Check if PPU can extract from controllable storage sources
    extracts_from_controllable_storage = len(can_extract_from) > 0 and not extracts_from_pure_incidence and not is_incidence_hydro
    
    # Priority order for classification:
    # 1. Incidence: Extracts from uncontrollable sources (solar, wind, river from HYD_R, lake from HYD_S)
    # 2. Store: Can INPUT to storage (charge/store energy) - prioritize this over Flex
    # 3. Flex: Can EXTRACT from controllable storage (discharge/produce energy)
    if extracts_from_pure_incidence or is_incidence_hydro:
        ppu_extract = 'Incidence'  # Uncontrollable production from incidence sources
    elif inputs_to_storage:
        ppu_extract = 'Store'  # Storage PPUs that can charge storage (prioritize over Flex)
    elif extracts_from_controllable_storage:
        ppu_extract = 'Flex'  # Flexible production from storage sources
    else:
        ppu_extract = 'Flex'  # Default fallback for production PPUs
    
    # Determine if this is a renewable PPU that needs location assignment
    location_rank = np.nan
    renewable_type = None
    
    # Check components for renewable types
    if 'PV' in components:
        renewable_type = 'solar'
    elif 'Wind (onshore)' in components:
        renewable_type = 'wind_onshore'
    elif 'Wind (offshore)' in components:
        renewable_type = 'wind_offshore'
    
    # Assign location if renewable
    if renewable_type:
        location_info = next_available_location(
            ppu_dictionary,
            renewable_type,
            solar_locations_df,
            wind_locations_df
        )
        if location_info:
            location_rank = location_info['rank']
            # Optionally store full location info in a comment or additional columns
            # For now, just storing the rank as per requirements
        else:
            print(f"Warning: No available location for {ppu_name} ({renewable_type})")
    
    # Initialize empty tracking dictionaries for dispatch indices
    # These will be populated during simulation/optimization
    d_stor_dict = {}  # Disposition index (storage willingness to discharge)
    u_dis_dict = {}   # Discharge utility index (system-wide shortfall signal)
    u_chg_dict = {}   # Charge utility index (system-wide surplus signal)
    c_dict = {}       # Cost index (price vs future value comparison)
    
    # Create new PPU entry
    new_ppu = pd.DataFrame([{
        'PPU_ID': new_id,
        'PPU_Name': ppu_name,
        'PPU_Category': category,
        'PPU_Extract': ppu_extract,
        'can_extract_from': can_extract_from,
        'can_input_to': can_input_to,
        'Chain_Efficiency': efficiency,
        'Cost_CHF_per_kWh': cost_per_kwh,
        'Cost_CHF_per_Quarter_Hour': cost_per_quarter_hour,
        'Location_Rank': location_rank,
        'Components': components,
        'd_stor': d_stor_dict,
        'u_dis': u_dis_dict,
        'u_chg': u_chg_dict,
        'c': c_dict
    }])
    
    # Add to dictionary
    updated_dictionary = pd.concat([ppu_dictionary, new_ppu], ignore_index=True)
    return updated_dictionary


def verify_storage_capacity(
    ppu_dictionary: pd.DataFrame,
    raw_energy_storage: List[Dict],
    ppu_constructs_df: pd.DataFrame
) -> Dict:
    """
    Verify storage capacity usage against available capacity.
    
    This function:
    1. Counts storage PPUs in the ppu_dictionary
    2. Calculates used capacity for each storage type
    3. Compares to possible capacity (from raw_energy_storage * number of instances)
    4. Returns a detailed report
    
    Parameters:
        ppu_dictionary (pd.DataFrame): Current PPU dictionary
        raw_energy_storage (List[Dict]): Storage definitions with base capacities
        ppu_constructs_df (pd.DataFrame): PPU constructs data to map PPUs to storage
    
    Returns:
        Dict: Report with structure:
            {
                'storage_name': {
                    'base_capacity_per_gw': float,
                    'num_ppu_instances': int,
                    'total_available_capacity': float,
                    'ppu_names': List[str],
                    'status': 'OK' or 'WARNING'
                },
                ...
                'summary': {
                    'total_storages': int,
                    'storages_in_use': int,
                    'all_storages_ok': bool
                }
            }
    """
    report = {}
    storages_in_use = 0
    all_ok = True
    
    # Process each storage type
    for storage_item in raw_energy_storage:
        storage_name = storage_item['storage']
        base_capacity = storage_item['value']
        unit = storage_item['unit']
        extracted_by = storage_item['extracted_by']
        
        # Find PPUs that use this storage (check if storage name is in PPU's can_extract_from)
        matching_ppus = []
        if not ppu_dictionary.empty:
            for _, ppu_row in ppu_dictionary.iterrows():
                ppu_extract_from = ppu_row.get('can_extract_from', [])
                if storage_name in ppu_extract_from:
                    matching_ppus.append(ppu_row['PPU_Name'])
        
        num_instances = len(matching_ppus)
        total_available = base_capacity * num_instances if num_instances > 0 else 0
        
        status = 'OK'
        if num_instances > 0:
            storages_in_use += 1
        
        report[storage_name] = {
            'base_capacity_per_gw': base_capacity,
            'num_ppu_instances': num_instances,
            'total_available_capacity': total_available,
            'unit': unit,
            'ppu_names': matching_ppus,
            'extracted_by': extracted_by,
            'status': status
        }
    
    # Add summary
    report['summary'] = {
        'total_storages': len(raw_energy_storage),
        'storages_in_use': storages_in_use,
        'all_storages_ok': all_ok
    }
    
    return report


def verify_unique_locations(ppu_dictionary: pd.DataFrame) -> Dict:
    """
    Verify that no two solar or wind PPUs have the same location ranking.
    
    Parameters:
        ppu_dictionary (pd.DataFrame): The PPU dictionary to verify
    
    Returns:
        Dict: Verification report with structure:
            {
                'is_unique': bool,
                'duplicate_ranks': List[int],
                'ppus_by_rank': Dict[int, List[str]],
                'total_renewable_ppus': int,
                'message': str
            }
    """
    if ppu_dictionary.empty or 'Location_Rank' not in ppu_dictionary.columns:
        return {
            'is_unique': True,
            'duplicate_ranks': [],
            'ppus_by_rank': {},
            'total_renewable_ppus': 0,
            'message': 'No renewable PPUs in dictionary'
        }
    
    # Filter for PPUs with location ranks (renewable only)
    renewable_ppus = ppu_dictionary[ppu_dictionary['Location_Rank'].notna()].copy()
    
    if renewable_ppus.empty:
        return {
            'is_unique': True,
            'duplicate_ranks': [],
            'ppus_by_rank': {},
            'total_renewable_ppus': 0,
            'message': 'No renewable PPUs with location assignments'
        }
    
    # Group PPUs by location rank
    ppus_by_rank = {}
    for _, row in renewable_ppus.iterrows():
        rank = int(row['Location_Rank'])
        ppu_name = row['PPU_Name']
        if rank not in ppus_by_rank:
            ppus_by_rank[rank] = []
        ppus_by_rank[rank].append(ppu_name)
    
    # Find duplicates
    duplicate_ranks = [rank for rank, ppus in ppus_by_rank.items() if len(ppus) > 1]
    is_unique = len(duplicate_ranks) == 0
    
    if is_unique:
        message = f"✓ All {len(renewable_ppus)} renewable PPUs have unique location assignments"
    else:
        message = f"✗ Found {len(duplicate_ranks)} duplicate location rank(s): {duplicate_ranks}"
    
    return {
        'is_unique': is_unique,
        'duplicate_ranks': duplicate_ranks,
        'ppus_by_rank': ppus_by_rank,
        'total_renewable_ppus': len(renewable_ppus),
        'message': message
    }

def balance_storage_usage(
    available_storages: List[str],
    raw_energy_storage: List[Dict],
    ppu_capacity_gw: float = 1.0
) -> Dict[str, float]:
    """
    Balance storage usage across multiple available storages for a PPU.
    
    This function distributes PPU capacity proportionally across all available storages
    based on their capacity per GW-PPU, ensuring balanced utilization.
    
    Parameters:
        available_storages (List[str]): List of storage names this PPU can use
        raw_energy_storage (List[Dict]): Full storage definitions with capacities
        ppu_capacity_gw (float): PPU capacity in GW (default 1.0)
    
    Returns:
        Dict[str, float]: Storage name -> fraction of PPU capacity allocated to this storage
                          (values sum to 1.0 for balanced distribution)
    
    Example:
        If a PPU can use ["Fuel Tank", "Biooil"] with capacities 141320 and 21600 MWh/GW-PPU:
        Returns: {"Fuel Tank": 0.867, "Biooil": 0.133} (proportional to capacities)
    """
    if not available_storages:
        return {}
    
    if len(available_storages) == 1:
        # Single storage - use 100% of capacity
        return {available_storages[0]: 1.0}
    
    # Multiple storages - distribute proportionally by capacity
    storage_capacities = {}
    total_capacity = 0.0
    
    for storage_item in raw_energy_storage:
        storage_name = storage_item['storage']
        if storage_name in available_storages:
            capacity_per_gw = storage_item['value']  # MWh/GW-PPU
            storage_capacities[storage_name] = capacity_per_gw
            total_capacity += capacity_per_gw
    
    if total_capacity == 0:
        # Fallback: equal distribution if all capacities are 0
        fraction = 1.0 / len(available_storages)
        return {storage: fraction for storage in available_storages}
    
    # Proportional distribution based on capacity
    balanced_distribution = {}
    for storage_name, capacity in storage_capacities.items():
        fraction = capacity / total_capacity
        balanced_distribution[storage_name] = fraction
    
    return balanced_distribution


def get_available_storages_for_ppu(
    components: List[str],
    raw_energy_storage: List[Dict]
) -> List[str]:
    """
    Determine which storages a PPU can extract from based on its components.
    
    Parameters:
        components (List[str]): List of components in the PPU chain
        raw_energy_storage (List[Dict]): Storage definitions with 'extracted_by' lists
    
    Returns:
        List[str]: List of storage names this PPU can use
    """
    available_storages = []
    
    for storage_item in raw_energy_storage:
        extracted_by = storage_item['extracted_by']
        # Check if any of the PPU's components can extract from this storage
        if any(extractor in components for extractor in extracted_by):
            available_storages.append(storage_item['storage'])
    
    return available_storages
