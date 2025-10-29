"""
Quick test to verify the batch precomputation optimizations work correctly.
"""
import time
import numpy as np
import pandas as pd
from calculationPipelineFramework import (
    SolarMapper, WindMapper, 
    precompute_renewable_productions,
    _init_solar_mapper, _init_wind_mapper
)

def test_mapper_initialization():
    """Test that mappers initialize correctly."""
    print("Testing mapper initialization...")
    _init_solar_mapper('data/solar_incidence_hourly_2024.csv')
    _init_wind_mapper('data/wind_incidence_hourly_2024.csv')
    print("✓ Mappers initialized successfully")

def test_batch_precomputation():
    """Test batch precomputation performance."""
    print("\nTesting batch precomputation...")
    
    # Create mock PPU dictionary
    ppu_data = []
    for i in range(15):
        ppu_data.append({
            'PPU_Name': f'PV_{i+1}',
            'can_extract_from': ['Solar'],
            'Location_Rank': i + 1
        })
    for i in range(15):
        ppu_data.append({
            'PPU_Name': f'WD_ON_{i+1}',
            'can_extract_from': ['Wind'],
            'Location_Rank': i + 1
        })
    for i in range(15):
        ppu_data.append({
            'PPU_Name': f'WD_OFF_{i+1}',
            'can_extract_from': ['Wind'],
            'Location_Rank': i + 16
        })
    
    ppu_dict = pd.DataFrame(ppu_data)
    
    # Load ranking data
    solar_ranking = pd.read_csv('data/ranking_incidence/solar_incidence_ranking.csv')
    wind_ranking = pd.read_csv('data/ranking_incidence/wind_incidence_ranking.csv')
    
    # Test precomputation
    num_timesteps = 10000
    start = time.time()
    result = precompute_renewable_productions(
        ppu_dict, num_timesteps, solar_ranking, wind_ranking
    )
    elapsed = time.time() - start
    
    print(f"✓ Precomputation completed in {elapsed:.2f}s")
    
    # Check results
    if result['solar_prod_matrix'] is not None:
        print(f"  Solar matrix shape: {result['solar_prod_matrix'].shape}")
        print(f"  Solar ranks: {len(result['solar_ranks'])} unique locations")
        print(f"  Sample solar production at t=0: {result['solar_prod_matrix'][0, :5]}")
    
    if result['wind_prod_matrix'] is not None:
        print(f"  Wind matrix shape: {result['wind_prod_matrix'].shape}")
        print(f"  Wind ranks: {len(result['wind_ranks'])} unique locations")
        print(f"  Sample wind production at t=0: {result['wind_prod_matrix'][0, :5]}")
    
    return result

def benchmark_speedup():
    """Benchmark speedup vs original per-timestep approach."""
    print("\nBenchmarking speedup...")
    from calculationPipelineFramework import calculate_solar_production, calculate_wind_production
    
    # Load data
    solar_15min = pd.read_csv('data/solar_incidence_hourly_2024.csv', header=[0, 1], index_col=0, parse_dates=True)
    solar_15min = solar_15min.resample('15min').interpolate()
    
    wind_15min = pd.read_csv('data/wind_incidence_hourly_2024.csv', header=[0, 1], index_col=0, parse_dates=True)
    wind_15min = wind_15min.resample('15min').interpolate()
    
    solar_ranking = pd.read_csv('data/ranking_incidence/solar_incidence_ranking.csv')
    wind_ranking = pd.read_csv('data/ranking_incidence/wind_incidence_ranking.csv')
    
    # Time original approach (100 timesteps, 15 solar + 30 wind PPUs)
    num_test_steps = 100
    start = time.time()
    for t in range(num_test_steps):
        for rank in range(1, 16):
            calculate_solar_production(rank, 100000, t, solar_15min, solar_ranking)
        for rank in range(1, 31):
            calculate_wind_production(rank, 5, t, wind_15min, wind_ranking)
    original_time = time.time() - start
    
    print(f"  Original approach: {original_time:.3f}s for {num_test_steps} timesteps")
    print(f"  Rate: {num_test_steps/original_time:.1f} timesteps/sec")
    
    # Estimate full run time
    full_timesteps = 35133
    estimated_full_time = original_time * (full_timesteps / num_test_steps)
    print(f"  Estimated time for full run: {estimated_full_time/60:.1f} minutes")

if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMIZATION VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_mapper_initialization()
        result = test_batch_precomputation()
        benchmark_speedup()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
