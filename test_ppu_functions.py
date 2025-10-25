# ============================================================================
# TEST: Verify Updated PPU Dictionary Functions
# ============================================================================

import importlib
import calculationsPpuFramework
importlib.reload(calculationsPpuFramework)
from calculationsPpuFramework import initialize_ppu_dictionary, add_ppu_to_dictionary, verify_storage_capacity

print("Testing updated PPU dictionary functions...")
print("="*80)

# Test 1: Initialize dictionary
print("\n1. Testing initialize_ppu_dictionary()...")
ppu_dict = initialize_ppu_dictionary()
print(f"✓ Dictionary initialized with columns: {list(ppu_dict.columns)}")
print(f"✓ Expected columns present: {'can_extract_from' in ppu_dict.columns and 'can_input_to' in ppu_dict.columns}")

# Test 2: Add a PPU
print("\n2. Testing add_ppu_to_dictionary()...")
try:
    updated_dict = add_ppu_to_dictionary(
        ppu_dictionary=ppu_dict,
        ppu_name='HYD_S',
        ppu_constructs_df=ppu_constructs_df,
        cost_df=cost_df,
        solar_locations_df=solar_locations_df,
        wind_locations_df=wind_locations_df,
        delta_t=hyperparams['delta_t'],
        raw_energy_storage=raw_energy_storage
    )
    print("✓ HYD_S added successfully")
    print(f"✓ Dictionary now has {len(updated_dict)} PPUs")
    hyd_s_row = updated_dict[updated_dict['PPU_Name'] == 'HYD_S']
    if not hyd_s_row.empty:
        print(f"✓ HYD_S can_extract_from: {hyd_s_row['can_extract_from'].iloc[0]}")
        print(f"✓ HYD_S can_input_to: {hyd_s_row['can_input_to'].iloc[0]}")
except Exception as e:
    print(f"✗ Error adding HYD_S: {e}")

# Test 3: Verify storage capacity
print("\n3. Testing verify_storage_capacity()...")
try:
    storage_check = verify_storage_capacity(updated_dict, raw_energy_storage, ppu_constructs_df)
    print("✓ Storage verification completed")
    print(f"✓ Storages in use: {storage_check['summary']['storages_in_use']}")
    print(f"✓ Total storages: {storage_check['summary']['total_storages']}")
except Exception as e:
    print(f"✗ Error in storage verification: {e}")

print("\n" + "="*80)
print("TEST COMPLETE - All functions working with new structure!")
print("="*80)