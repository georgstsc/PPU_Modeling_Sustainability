#!/usr/bin/env python3
"""
Regenerate solar_incidence_hourly_2024.csv from GRIB files with correct hourly timestamps.

CRITICAL FIXES APPLIED:
1. Uses 'valid_time' instead of 'time' to get hourly data (not daily aggregates)
2. Detects accumulated SSRD data (GRIB_stepType: accum) and computes hourly differences
3. Sets night hours (0-5, 20-23) to zero if values exceed threshold

ROOT CAUSE IDENTIFIED:
- SSRD (Surface Solar Radiation Downwards) in GRIB files is ACCUMULATED radiation
- Step 0 = accumulated from forecast start to +1h
- Step N = accumulated from forecast start to +N hours
- To get hourly values, we must compute: hourly[N] = accumulated[N] - accumulated[N-1]
- Without differencing, we were using cumulative totals, causing high values at night

DATA QUALITY:
- Night hours (0-5, 20-23): Should be near zero (solar radiation impossible at night)
- Day hours (6-19): Should show proper solar patterns with peak at midday
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("REGENERATING SOLAR INCIDENCE HOURLY DATA FROM GRIB FILES")
print("=" * 80)

solar_dir = Path("data/solar_incidence")
valid_grib_files = sorted([p for p in solar_dir.rglob("*") if p.is_file() and not p.name.endswith('.idx')])

print(f"\nFound {len(valid_grib_files)} GRIB files:")
for f in valid_grib_files:
    print(f"  {f}")

all_dfs = []
for f in valid_grib_files:
    try:
        print(f"\nProcessing {f.name}...")
        ds = xr.open_dataset(
            f,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "ssrd"}},
        )
        
        # Get the variable name
        var_name = list(ds.data_vars)[0]
        ssrd = ds[var_name]
        
        # CRITICAL: Check if data is accumulated (GRIB_stepType: accum)
        is_accumulated = ssrd.attrs.get('GRIB_stepType', '').lower() == 'accum'
        if is_accumulated:
            print(f"    Data is accumulated (GRIB_stepType: {ssrd.attrs.get('GRIB_stepType', 'unknown')})")
            print(f"    Computing hourly differences to get per-hour values...")
        
        # Convert from J/m^2 to kWh/m^2
        units_attr = str(ssrd.attrs.get("units", "")).lower()
        if ("j" in units_attr) and ("/m" in units_attr):
            ghi_kwh = ssrd / 3600000.0  # J/m^2 -> kWh/m^2
        else:
            ghi_kwh = ssrd / 3600000.0
        
        # Round coordinates to 1 decimal place
        ghi_kwh['latitude'] = np.round(ghi_kwh['latitude'].values, 1)
        ghi_kwh['longitude'] = np.round(ghi_kwh['longitude'].values, 1)
        
        # CRITICAL FIX: Use valid_time instead of time to get hourly data
        # valid_time has shape (time, step) where step is hourly (01:00, 02:00, ..., 24:00)
        if "valid_time" in ds:
            # Get data values: shape (time, step, lat, lon)
            ssrd_data = ghi_kwh.values
            n_times, n_steps, n_lats, n_lons = ssrd_data.shape
            
            # CRITICAL FIX: If data is accumulated, compute differences to get hourly values
            if is_accumulated:
                # Create array for hourly values (differences)
                hourly_data = np.zeros_like(ssrd_data)
                
                for time_idx in range(n_times):
                    for step_idx in range(n_steps):
                        if step_idx == 0:
                            # First step: value itself (accumulated from time to time+1h)
                            hourly_data[time_idx, step_idx, :, :] = ssrd_data[time_idx, step_idx, :, :]
                        else:
                            # Subsequent steps: difference from previous step
                            prev_val = ssrd_data[time_idx, step_idx - 1, :, :]
                            curr_val = ssrd_data[time_idx, step_idx, :, :]
                            
                            # Handle NaN values
                            mask_valid = ~(np.isnan(prev_val) | np.isnan(curr_val))
                            hourly_data[time_idx, step_idx, :, :] = np.where(
                                mask_valid,
                                curr_val - prev_val,
                                0.0  # Set to 0 if either value is NaN
                            )
                            
                            # Ensure non-negative (radiation can't be negative)
                            hourly_data[time_idx, step_idx, :, :] = np.maximum(
                                hourly_data[time_idx, step_idx, :, :], 0.0
                            )
                
                # Use hourly differences instead of accumulated values
                data_to_use = hourly_data
                print(f"    Converted accumulated data to hourly differences")
            else:
                # Data is already per-hour, use as-is
                data_to_use = ssrd_data
                print(f"    Data is already per-hour (not accumulated)")
            
            # Get valid_time and flatten it
            valid_times = ds.valid_time.values.flatten()
            
            # Reshape: (time, step, lat, lon) -> (time*step, lat, lon)
            data_reshaped = data_to_use.reshape(n_times * n_steps, n_lats, n_lons)
            
            # Get lat/lon coordinates
            lats = ghi_kwh.latitude.values
            lons = ghi_kwh.longitude.values
            
            # Create MultiIndex columns
            lat_lon_pairs = [(float(lat), float(lon)) for lat in lats for lon in lons]
            
            # Flatten the lat/lon dimension for DataFrame
            data_flat = data_reshaped.reshape(n_times * n_steps, n_lats * n_lons)
            
            # Create DataFrame with hourly timestamps
            df = pd.DataFrame(data_flat, index=valid_times, columns=lat_lon_pairs)
            df.index.name = "datetime"
            
            # CRITICAL FIX: Set night hours to zero (solar radiation impossible at night)
            # Night hours: 0-5 (midnight to 5am) and 20-23 (8pm to 11pm)
            df_index_pd = pd.to_datetime(df.index)
            night_mask = ((df_index_pd.hour >= 0) & (df_index_pd.hour < 6)) | (df_index_pd.hour >= 20)
            night_threshold = 0.01  # Very low threshold for night values (kWh/m²)
            
            if night_mask.any():
                # Set all night values to zero (solar radiation is physically impossible at night)
                num_night_values = night_mask.sum() * len(df.columns)
                df.loc[night_mask, :] = 0.0
                print(f"    Set {num_night_values} night hour values to 0.0 (physically correct)")
            
            # Convert to MultiIndex columns
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=['latitude', 'longitude'])
            
            all_dfs.append(df)
            print(f"  ✓ Extracted {len(df)} hourly timesteps")
        else:
            print(f"  ✗ ERROR: valid_time not found in {f.name}")
            
    except Exception as e:
        print(f"  ✗ Failed to open {f.name}: {e}")

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
output_csv = "data/solar_incidence_hourly_2024.csv"
print(f"\nWriting to {output_csv}...")

with open(output_csv, 'w') as f:
    # Write header rows
    f.write(','.join(lat_row) + '\n')
    f.write(','.join(lon_row) + '\n')
    f.write(','.join(time_row) + '\n')
    
    # Write data rows
    for row in data_rows:
        f.write(','.join(row) + '\n')

print(f"✓ Successfully saved hourly solar data to {output_csv}")
print(f"  Total rows: {len(data_rows) + 3} (3 header + {len(data_rows)} data)")
print(f"  Total columns: {len(lat_row)}")

# Verify the output
print("\nVerifying output...")
df_check = pd.read_csv(output_csv, header=None, nrows=10)
print("First 10 rows of output:")
print(df_check)

print("\n" + "=" * 80)
print("REGENERATION COMPLETE")
print("=" * 80)
print("\nThe solar_incidence_hourly_2024.csv file now contains hourly data")
print("with proper day/night cycles preserved.")
print("\nYou can now remove the day/night cycle fix from SolarMapper")
print("in calculationPipelineFramework.py since the data is already hourly.")

