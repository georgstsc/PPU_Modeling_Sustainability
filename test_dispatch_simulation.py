#!/usr/bin/env python3
"""
Test script for the new dispatch simulation structure.
"""

import pandas as pd
import numpy as np
from calculationPipelineFramework import run_dispatch_simulation

def create_test_data():
    """Create minimal test data for dispatch simulation."""
    # Create 10 timesteps of test data
    timesteps = 10

    # Demand: 1000 MW constant
    demand_15min = pd.Series([1000.0] * timesteps, index=pd.date_range('2024-01-01', periods=timesteps, freq='15min'))

    # Spot prices: 50 CHF/MWh constant
    spot_15min = pd.Series([50.0] * timesteps, index=pd.date_range('2024-01-01', periods=timesteps, freq='15min'))

    # ROR data: 200 MW constant
    ror_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=timesteps, freq='15min'),
        'value': [200.0] * timesteps
    }).set_index('timestamp')

    # Solar data: 500 W/m² constant
    solar_15min = pd.DataFrame(
        [[500.0] * 5] * timesteps,  # 5 locations, timesteps rows
        index=pd.date_range('2024-01-01', periods=timesteps, freq='15min'),
        columns=pd.MultiIndex.from_tuples([(46.5, 7.5), (46.8, 8.0), (47.0, 7.8), (46.2, 6.9), (47.2, 8.5)], names=['lat', 'lon'])
    )

    # Wind data: 8 m/s constant
    wind_15min = pd.DataFrame(
        [[8.0] * 5] * timesteps,  # 5 locations, timesteps rows
        index=pd.date_range('2024-01-01', periods=timesteps, freq='15min'),
        columns=pd.MultiIndex.from_tuples([(46.5, 7.5), (46.8, 8.0), (47.0, 7.8), (46.2, 6.9), (47.2, 8.5)], names=['lat', 'lon'])
    )

    # PPU dictionary with Incidence PPUs
    ppu_dictionary = pd.DataFrame({
        'PPU_Name': ['PV', 'WD_ON', 'HYD_ROR'],
        'PPU_Extract': ['Incidence', 'Incidence', 'Incidence'],
        'PPU_Category': ['Renewable', 'Renewable', 'Renewable'],
        'Location_Rank': [1, 1, np.nan],
        'Cost_CHF_per_kWh': [0.10, 0.08, 0.05]
    })

    # Rankings
    solar_ranking_df = pd.DataFrame({
        'latitude': [46.5, 46.8, 47.0, 46.2, 47.2],
        'longitude': [7.5, 8.0, 7.8, 6.9, 8.5]
    })

    wind_ranking_df = pd.DataFrame({
        'latitude': [46.5, 46.8, 47.0, 46.2, 47.2],
        'longitude': [7.5, 8.0, 7.8, 6.9, 8.5]
    })

    # Raw energy storage (empty for this test)
    raw_energy_storage = []

    # Raw energy incidence
    raw_energy_incidence = [
        {'storage': 'River', 'current_value': 0.0, 'history': []},
        {'storage': 'Solar', 'current_value': 0.0, 'history': []},
        {'storage': 'Wind', 'current_value': 0.0, 'history': []},
        {'storage': 'Grid', 'current_value': 0.0, 'history': []}
    ]

    # Hyperparameters
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

    return (demand_15min, spot_15min, ror_df, solar_15min, wind_15min,
            ppu_dictionary, solar_ranking_df, wind_ranking_df,
            raw_energy_storage, raw_energy_incidence, hyperparams)

def test_dispatch_simulation():
    """Test the new dispatch simulation structure."""
    print("Testing dispatch simulation...")

    # Create test data
    (demand_15min, spot_15min, ror_df, solar_15min, wind_15min,
     ppu_dictionary, solar_ranking_df, wind_ranking_df,
     raw_energy_storage, raw_energy_incidence, hyperparams) = create_test_data()

    print(f"Test data created: {len(demand_15min)} timesteps")

    # Run simulation
    try:
        technology_volume, phi_smoothed = run_dispatch_simulation(
            demand_15min, spot_15min, ror_df, solar_15min, wind_15min,
            ppu_dictionary, solar_ranking_df, wind_ranking_df,
            raw_energy_storage, raw_energy_incidence, hyperparams
        )

        print("✓ Simulation completed successfully!")
        print(f"  Final phi_smoothed: {phi_smoothed:.4f}")
        print(f"  Technologies tracked: {list(technology_volume.keys())}")

        # Check results
        total_production = 0
        for tech, data in technology_volume.items():
            prod_count = len(data['production'])
            total_production += sum(vol for _, vol in data['production'])
            print(f"  {tech}: {prod_count} production records, total: {total_production:.1f} MW")

        return True

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dispatch_simulation()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")