#!/usr/bin/env python3
"""Test script for data loading function"""

from calculations_optimization import load_annual_data

print('Loading annual data...')
data = load_annual_data('data/')
print('✓ Data loaded successfully!')
print()

print('Data shape verification:')
print(f'  - Demand:         {len(data["demand_15min"]):,} timesteps')
print(f'  - Spot prices:    {len(data["spot_15min"]):,} timesteps')
print(f'  - Solar:          {len(data["solar_incidence"]):,} timesteps')
print(f'  - Wind:           {len(data["wind_incidence"]):,} timesteps')
print(f'  - RoR:            {len(data["ror_15min"]):,} timesteps')
print(f'  - Expected:       35,040 timesteps (15-min for full year)')
print()

print('Value ranges:')
print(f'  - Demand:         {data["demand_15min"].min():.1f} - {data["demand_15min"].max():.1f} MW')
print(f'  - Spot price:     {data["spot_15min"].min():.2f} - {data["spot_15min"].max():.2f} CHF/MWh')
print(f'  - Solar:          {data["solar_incidence"].min():.4f} - {data["solar_incidence"].max():.4f}')
print(f'  - Wind:           {data["wind_incidence"].min():.2f} - {data["wind_incidence"].max():.2f}')
print(f'  - RoR:            {data["ror_15min"].min():.1f} - {data["ror_15min"].max():.1f} MW')
