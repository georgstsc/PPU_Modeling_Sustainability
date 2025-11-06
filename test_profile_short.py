import pandas as pd
import time

import calculationPipelineFramework as cpf
import calculationsPpuFramework as ppf


def build_minimal_ppu_dictionary(ppu_constructs_df, cost_df, solar_locations_df, wind_locations_df):
    ppu_dictionary = ppf.initialize_ppu_dictionary()
    # Small selection of PPUs likely present in constructs CSV; adjust if NameErrors occur
    sample_ppus = ['PV', 'WD_ON', 'H2P_G', 'BIO_OIL_ICE']
    counts = {'PV': 5, 'WD_ON': 5, 'H2P_G': 2, 'BIO_OIL_ICE': 1}
    for ppu in sample_ppus:
        cnt = counts.get(ppu, 1)
        for i in range(cnt):
            try:
                ppu_dictionary = ppf.add_ppu_to_dictionary(
                    ppu_dictionary,
                    ppu,
                    ppu_constructs_df,
                    cost_df,
                    solar_locations_df,
                    wind_locations_df,
                    raw_energy_storage=[],
                    raw_energy_incidence=[]
                )
            except Exception as e:
                # Skip if PPU not found or mapping issues
                print(f"Skipping PPU {ppu}: {e}")
    return ppu_dictionary


def make_minimal_storages():
    raw_energy_storage = [
        {'storage': 'Biooil', 'value': 21600.0, 'current_value': 0.0, 'unit': 'MWh', 'extracted_by': ['BIO_OIL_ICE'], 'history': []},
        {'storage': 'Lake', 'value': 50000.0, 'current_value': 10000.0, 'unit': 'MWh', 'extracted_by': ['PHS','HYD_S'], 'history': []},
        {'storage': 'H2 Storage UG 200bar', 'value': 10000.0, 'current_value': 5000.0, 'unit': 'MWh', 'extracted_by': ['H2_G','H2P_G'], 'history': []}
    ]
    raw_energy_incidence = [
        {'storage': 'Grid', 'value': 0.0, 'current_value': 0.0, 'unit': 'MW', 'extracted_by': [], 'history': []},
        {'storage': 'Solar', 'value': 0.0, 'current_value': 0.0, 'unit': 'MW', 'extracted_by': ['PV'], 'history': []},
        {'storage': 'Wind', 'value': 0.0, 'current_value': 0.0, 'unit': 'MW', 'extracted_by': ['WD_ON','WD_OFF'], 'history': []},
        {'storage': 'River', 'value': 0.0, 'current_value': 0.0, 'unit': 'MW', 'extracted_by': ['HYD_R'], 'history': []},
        {'storage': 'Wood', 'value': 0.0, 'current_value': 0.0, 'unit': 'MW', 'extracted_by': ['BIO_WOOD'], 'history': []}
    ]
    return raw_energy_storage, raw_energy_incidence


if __name__ == '__main__':
    data_dir = 'data'
    print('Loading data...')
    demand_15min, spot_15min, ror_df = cpf.load_energy_data(data_dir)
    solar_15min, wind_15min = cpf.load_incidence_data(data_dir)
    ppu_constructs_df = ppf.load_ppu_data(f'{data_dir}/ppu_constructs_components.csv')
    cost_df = ppf.load_cost_data(f'{data_dir}/cost_table_tidy.csv')
    solar_locations_df = ppf.load_location_rankings('solar')
    wind_locations_df = ppf.load_location_rankings('wind')

    # Slice to a short run
    N = 1000
    demand_15min = demand_15min.iloc[:N]
    spot_15min = spot_15min.iloc[:N]
    ror_df = ror_df.iloc[:N]
    solar_15min = solar_15min.iloc[:N]
    wind_15min = wind_15min.iloc[:N]

    ppu_dictionary = build_minimal_ppu_dictionary(ppu_constructs_df, cost_df, solar_locations_df, wind_locations_df)
    raw_energy_storage, raw_energy_incidence = make_minimal_storages()

    hyperparams = {
        'epsilon': 1e-6,
        'timestep_hours': 0.25,
        'ema_beta': 0.2,
        'alpha_u': 5000.0,
        'alpha_m': 5.0,
        'stor_deadband': 0.05,
        'weight_spread': 1.0,
        'weight_volatility': 1.0,
        'volatility_scale': 30.0
    }

    print('Ready to run short dispatch: timesteps=', len(demand_15min), 'ppus=', len(ppu_dictionary))

    start = time.time()
    cpf.run_dispatch_simulation(
        demand_15min, spot_15min, ror_df, solar_15min, wind_15min,
        ppu_dictionary, solar_15min, wind_15min, raw_energy_storage, raw_energy_incidence, hyperparams
    )
    end = time.time()
    print(f'Completed short dispatch in {end-start:.2f}s')
