import pandas as pd
import numpy as np
import ast
import functools
from typing import Dict, List, Tuple

# -------------------------
# Location & PPU utilities
# -------------------------

def load_location_rankings(source_type: str) -> pd.DataFrame:
    """Load location ranking data for solar or wind sources."""
    filepath = f'data/ranking_incidence/{source_type}_incidence_ranking.csv'
    df = pd.read_csv(filepath)
    df['assigned'] = False
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
                tracking_df.at[idx, 'Quantity'] = quantity[0]
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
            df = pd.read_csv('data/water_monthly_ror_2024.csv', parse_dates=['Month'])
            df.set_index('Month', inplace=True)
            # Calculate average power in GW per month
            df['avg_power_GW'] = df['RoR_GWh'] / (df.index.days_in_month * 24)
            # Interpolate to daily for continuity between months
            daily_series = df['avg_power_GW'].resample('D').interpolate(method='linear')
            # Then resample to 15-min
            cache['ror_series'] = daily_series.resample('15min').interpolate(method='linear')
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