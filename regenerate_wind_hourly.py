#!/usr/bin/env python3
"""
Regenerate wind_incidence_hourly_2024.csv from GRIB files with correct hourly timestamps.

CRITICAL NOTES:
1. Wind GRIB files contain u10 and v10 components (U and V wind components)
2. Wind speed must be computed: speed = sqrt(u^2 + v^2)
3. Data is INSTANTANEOUS (GRIB_stepType: instant), NOT accumulated
4. Uses 'valid_time' instead of 'time' to get hourly data (not daily aggregates)

ROOT CAUSE VERIFICATION:
- u10/v10 are instantaneous wind components (m/s)
- GRIB_stepType: instant (no differencing needed, unlike solar SSRD)
- Step 0 = wind at time+1h
- Step N = wind at time+N hours
- Wind speed is computed from components: sqrt(u^2 + v^2)

DATA QUALITY:
- Wind speeds should be 0-50 m/s (typical range)
- Wind can occur at any time (unlike solar, which is zero at night)
- Values should be physically reasonable for Switzerland region
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("REGENERATING WIND INCIDENCE HOURLY DATA FROM GRIB FILES")
print("=" * 80)

wind_dir = Path("data/wind_incidence")
valid_grib_files = sorted([p for p in wind_dir.rglob("*") if p.is_file() and not p.name.endswith('.idx')])

print(f"\nFound {len(valid_grib_files)} GRIB files:")
for f in valid_grib_files:
    print(f"  {f}")

all_dfs = []
for f in valid_grib_files:
    try:
        print(f"\nProcessing {f.name}...")
        ds = xr.open_dataset(f, engine="cfgrib")
        
        # Check for u10 and v10 components
        if 'u10' not in ds.data_vars or 'v10' not in ds.data_vars:
            print(f"  ✗ ERROR: u10 or v10 not found in {f.name}")
            continue
        
        u10 = ds['u10']
        v10 = ds['v10']
        
        # Verify data type
        step_type_u = u10.attrs.get('GRIB_stepType', '').lower()
        step_type_v = v10.attrs.get('GRIB_stepType', '').lower()
        
        if step_type_u == 'instant' and step_type_v == 'instant':
            print(f"    Data is instantaneous (GRIB_stepType: instant) - no differencing needed")
        else:
            print(f"    WARNING: Unexpected stepType - u10: {step_type_u}, v10: {step_type_v}")
        
        # Compute wind speed from u and v components
        wind_speed = np.sqrt(u10**2 + v10**2)
        
        # Round coordinates to 1 decimal place
        wind_speed['latitude'] = np.round(wind_speed['latitude'].values, 1)
        wind_speed['longitude'] = np.round(wind_speed['longitude'].values, 1)
        
        # CRITICAL: Use valid_time instead of time to get hourly data
        if "valid_time" in ds:
            # Get valid_time and flatten it
            valid_times = ds.valid_time.values.flatten()
            
            # Get data values: shape (time, step, lat, lon)
            wind_data = wind_speed.values
            n_times, n_steps, n_lats, n_lons = wind_data.shape
            
            # Reshape: (time, step, lat, lon) -> (time*step, lat, lon)
            data_reshaped = wind_data.reshape(n_times * n_steps, n_lats, n_lons)
            
            # Get lat/lon coordinates
            lats = wind_speed.latitude.values
            lons = wind_speed.longitude.values
            
            # Create MultiIndex columns
            lat_lon_pairs = [(float(lat), float(lon)) for lat in lats for lon in lons]
            
            # Flatten the lat/lon dimension for DataFrame
            data_flat = data_reshaped.reshape(n_times * n_steps, n_lats * n_lons)
            
            # Create DataFrame with hourly timestamps
            df = pd.DataFrame(data_flat, index=valid_times, columns=lat_lon_pairs)
            df.index.name = "datetime"
            
            # Data quality check: ensure wind speeds are reasonable
            # Wind speeds should be 0-50 m/s (typical range, though extreme events can exceed)
            # Set any unrealistic values (>100 m/s) to NaN for review
            unrealistic_mask = df > 100.0
            num_unrealistic = unrealistic_mask.sum().sum()
            if num_unrealistic > 0:
                print(f"    WARNING: Found {num_unrealistic} values > 100 m/s (unrealistic)")
                df[unrealistic_mask] = np.nan
            
            # Ensure non-negative (wind speed cannot be negative)
            df = df.clip(lower=0.0)
            
            # Convert to MultiIndex columns
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=['latitude', 'longitude'])
            
            all_dfs.append(df)
            print(f"  ✓ Extracted {len(df)} hourly timesteps")
        else:
            print(f"  ✗ ERROR: valid_time not found in {f.name}")
            
    except Exception as e:
        print(f"  ✗ Failed to open {f.name}: {e}")
        import traceback
        traceback.print_exc()

if not all_dfs:
    raise RuntimeError("No GRIB datasets could be processed.")

print(f"\nConcatenating {len(all_dfs)} monthly datasets...")
df_all = pd.concat(all_dfs, axis=0)
df_all = df_all.sort_index()

print(f"\nFinal dataset:")
print(f"  Shape: {df_all.shape}")
print(f"  Time range: {df_all.index.min()} to {df_all.index.max()}")
print(f"  Expected hourly timesteps for 2024: {365 * 24}")
print(f"  Actual timesteps: {len(df_all)}")

# Prepare CSV format: first 3 rows are lat, lon, time headers
# Then data rows with datetime in first column
print("\nPreparing CSV format...")

# Create header rows
lat_row = ['latitude'] + [str(col[0]) for col in df_all.columns]
lon_row = ['longitude'] + [str(col[1]) for col in df_all.columns]
time_row = ['time'] + [''] * (len(df_all.columns))

# Create data rows
data_rows = []
for dt, row in df_all.iterrows():
    data_row = [dt.strftime('%Y-%m-%d %H:%M:%S')] + [str(val) for val in row.values]
    data_rows.append(data_row)

# Write to CSV
output_csv = "data/wind_incidence_hourly_2024.csv"
print(f"\nWriting to {output_csv}...")

with open(output_csv, 'w') as f:
    # Write header rows
    f.write(','.join(lat_row) + '\n')
    f.write(','.join(lon_row) + '\n')
    f.write(','.join(time_row) + '\n')
    
    # Write data rows
    for row in data_rows:
        f.write(','.join(row) + '\n')

print(f"✓ Successfully saved hourly wind data to {output_csv}")
print(f"  Total rows: {len(data_rows) + 3} (3 header + {len(data_rows)} data)")
print(f"  Total columns: {len(lat_row)}")

# Verify the output
print("\nVerifying output...")
df_check = pd.read_csv(output_csv, header=None, nrows=10)
print("First 10 rows of output:")
print(df_check)

# Final data quality check
print("\nFinal data quality check:")
df_verify = pd.read_csv(output_csv, header=None, skiprows=3, nrows=100)
df_verify.columns = ['datetime'] + [f'loc_{i}' for i in range(len(df_verify.columns) - 1)]
df_verify['datetime'] = pd.to_datetime(df_verify['datetime'])

data_cols = [col for col in df_verify.columns if col.startswith('loc_')]
all_values = df_verify[data_cols].values.flatten()
all_values = all_values[~np.isnan(all_values)]

print(f"  Wind speed range: {all_values.min():.4f} - {all_values.max():.4f} m/s")
print(f"  Mean wind speed: {all_values.mean():.4f} m/s")
print(f"  Values > 50 m/s: {(all_values > 50).sum()} (should be rare)")
print(f"  Values < 0: {(all_values < 0).sum()} (should be 0)")

print("\n" + "=" * 80)
print("REGENERATION COMPLETE")
print("=" * 80)
print("\nThe wind_incidence_hourly_2024.csv file now contains hourly wind speed data")
print("computed from u10 and v10 components using: speed = sqrt(u^2 + v^2)")
print("\nNote: Wind data is instantaneous (not accumulated), so no differencing is needed.")

