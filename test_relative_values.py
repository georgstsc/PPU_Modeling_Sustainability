#!/usr/bin/env python3
"""
Test script to verify that calculate_ppu_indices now uses relative values
and produces varying kappa values instead of stuck values.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to path so we can import the framework
sys.path.append('/Users/gs/Library/CloudStorage/OneDrive-Personal/EPFL/MA3/PdS2')

from calculationPipelineFramework import calculate_ppu_indices

def test_relative_values():
    """Test that the function produces varying kappa values with relative scaling."""

    # Create minimal test data
    ppu_dictionary = pd.DataFrame({
        'PPU_Name': ['Test_PPU_1', 'Test_PPU_2', 'Test_PPU_3'],
        'PPU_Category': ['Storage', 'Incidence', 'Storage'],
        'can_extract_from': [['Test_Storage_1'], [], ['Test_Storage_2']],
        'Cost_CHF_per_kWh': [0.1, 0.05, 0.08],
        'Efficiency': [0.9, 1.0, 0.95]
    })

    raw_energy_storage = [
        {'storage': 'Test_Storage_1', 'current_value': 450.0, 'value': 1000.0, 'target_SoC': 0.6, 'history': []},
        {'storage': 'Test_Storage_2', 'current_value': 200.0, 'value': 1000.0, 'target_SoC': 0.5, 'history': []}
    ]

    # Create test spot price series (varying prices)
    spot_15min = pd.Series([50.0, 55.0, 45.0, 60.0, 48.0, 52.0, 58.0, 47.0] * 100)  # 800 timesteps

    hyperparams = {
        'alpha_u': 5000.0,
        'alpha_m': 5.0,
        'stor_deadband': 0.05,
        'volatility_scale': 30.0,
        'weight_spread': 1.0,
        'weight_volatility': 1.0
    }

    # Test different scenarios
    test_scenarios = [
        {'overflow_MW': 1000.0, 'phi_smoothed': 800.0, 'spot_price': 50.0, 't': 0, 'desc': 'High deficit'},
        {'overflow_MW': -500.0, 'phi_smoothed': -300.0, 'spot_price': 45.0, 't': 100, 'desc': 'Surplus'},
        {'overflow_MW': 0.0, 'phi_smoothed': 0.0, 'spot_price': 55.0, 't': 200, 'desc': 'Balance'},
        {'overflow_MW': 2000.0, 'phi_smoothed': 1500.0, 'spot_price': 60.0, 't': 300, 'desc': 'Very high deficit'}
    ]

    print("Testing calculate_ppu_indices with RELATIVE VALUES")
    print("=" * 60)

    all_kappa_dis = []
    all_kappa_chg = []

    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['desc']}")
        print(f"  overflow_MW: {scenario['overflow_MW']:.1f}, phi_smoothed: {scenario['phi_smoothed']:.1f}, spot_price: {scenario['spot_price']:.1f}")

        ppu_indices = calculate_ppu_indices(
            ppu_dictionary=ppu_dictionary,
            raw_energy_storage=raw_energy_storage,
            overflow_MW=scenario['overflow_MW'],
            phi_smoothed=scenario['phi_smoothed'],
            spot_price=scenario['spot_price'],
            spot_15min=spot_15min,
            t=scenario['t'],
            hyperparams=hyperparams
        )

        # Collect kappa values
        kappa_dis_values = [v['kappa_dis'] for v in ppu_indices.values()]
        kappa_chg_values = [v['kappa_chg'] for v in ppu_indices.values()]

        all_kappa_dis.extend(kappa_dis_values)
        all_kappa_chg.extend(kappa_chg_values)

        print(f"  κ_dis range: [{min(kappa_dis_values):.3f}, {max(kappa_dis_values):.3f}]")
        print(f"  κ_chg range: [{min(kappa_chg_values):.3f}, {max(kappa_chg_values):.3f}]")

        # Check for stuck values
        stuck_dis = sum(1 for k in kappa_dis_values if abs(k - 1.0) < 0.1)
        stuck_chg = sum(1 for k in kappa_chg_values if abs(k - 1.0) < 0.1)

        print(f"  Stuck κ_dis (≈1.0): {stuck_dis}/{len(kappa_dis_values)}")
        print(f"  Stuck κ_chg (≈1.0): {stuck_chg}/{len(kappa_chg_values)}")

    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL RESULTS:")
    print(f"Total scenarios tested: {len(test_scenarios)}")
    print(f"Total PPUs per scenario: {len(ppu_dictionary)}")
    print(f"Overall κ_dis range: [{min(all_kappa_dis):.3f}, {max(all_kappa_dis):.3f}]")
    print(f"Overall κ_chg range: [{min(all_kappa_chg):.3f}, {max(all_kappa_chg):.3f}]")

    # Check if values vary significantly
    dis_range = max(all_kappa_dis) - min(all_kappa_dis)
    chg_range = max(all_kappa_chg) - min(all_kappa_chg)

    print(f"κ_dis total range: {dis_range:.3f}")
    print(f"κ_chg total range: {chg_range:.3f}")

    if dis_range > 0.5 and chg_range > 0.5:
        print("✅ SUCCESS: Kappa values show significant variation - relative scaling is working!")
        return True
    else:
        print("❌ FAILURE: Kappa values still stuck - relative scaling may not be working properly")
        return False

if __name__ == "__main__":
    success = test_relative_values()
    sys.exit(0 if success else 1)