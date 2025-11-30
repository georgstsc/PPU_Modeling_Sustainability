def visualize_storage_monthly(csv_path="data/water_monthly_storage_2024.csv",
                              value_col=None,
                              start_date=None,
                              end_date=None):
    """
    Visualize monthly water storage (GWh) as a bar chart.
    Expects a CSV with at least:
      - Month (YYYY-MM)
      - Storage_GWh OR Reservoir_Content_GWh (if value_col not specified)

    Args:
        csv_path (str): Path to the CSV file.
        value_col (str or None): Column to plot. If None, will try common names.
        start_date (str or None): Start date (YYYY-MM-DD) filter.
        end_date (str or None): End date (YYYY-MM-DD) filter.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Column inference
    if "Month" not in df.columns:
        raise RuntimeError(f"Expected a 'Month' column. Found: {list(df.columns)}")

    if value_col is None:
        candidates = ["Storage_GWh", "Reservoir_Content_GWh", "HydropowerReserve_GWh", "GWh"]
        value_col = next((c for c in candidates if c in df.columns), None)
        if value_col is None:
            raise RuntimeError(f"Expected one of {candidates} for value_col. Found: {list(df.columns)}")

    # Parse/sort/filter
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["Month", value_col]).sort_values("Month")

    if start_date:
        df = df[df["Month"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Month"] <= pd.to_datetime(end_date)]
    if df.empty:
        print(f"No data in the selected range: {start_date} to {end_date}.")
        return

    # Integer x-axis to avoid datetime bar overlap
    x = np.arange(len(df))
    labels = df["Month"].dt.strftime("%b")

    plt.figure(figsize=(12, 5))
    plt.bar(x, df[value_col], width=0.8)
    plt.xticks(x, labels, rotation=45)
    plt.title("Monthly Water Storage (GWh)")
    plt.xlabel("Month")
    plt.ylabel("GWh")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def visualize_ror_monthly(csv_path="ch_2024_ror_monthly_gwh.csv", start_date=None, end_date=None):
    """
    Visualize monthly Run-of-River production (GWh) for 2024 CSV with columns: Month, RoR_GWh.
    - Avoids datetime bar overlap by using integer x-positions.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read CSV
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if "Month" not in df.columns or "RoR_GWh" not in df.columns:
        raise RuntimeError(f"Expected columns Month and RoR_GWh, found: {list(df.columns)}")

    # Parse month and sort
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["Month", "RoR_GWh"]).sort_values("Month")

    # Optional filtering
    if start_date:
        df = df[df["Month"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Month"] <= pd.to_datetime(end_date)]
    if df.empty:
        print(f"No data in the selected range: {start_date} to {end_date}.")
        return

    # Build integer x-positions to avoid overlap
    x = np.arange(len(df))
    labels = df["Month"].dt.strftime("%b")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(x, df["RoR_GWh"], width=0.8)  # categorical-style bars; no datetime overlap
    plt.xticks(x, labels)
    plt.title("Monthly Run-of-River Production (GWh)")
    plt.xlabel("Month")
    plt.ylabel("GWh")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



def analyze_exchange_values(csv_path="data/chf_to_eur_2024.csv", date_col_candidates=["Date", "DateUTC", "DateTime"], value_col_candidates=["Value", "Rate", "ExchangeRate", "CHF/EUR"]):
    """
    Visualize and analyze exchange values (CHF/EUR) over time.
    Args:
        csv_path (str): Path to the exchange rate CSV file.
        date_col_candidates (list): Possible column names for date.
        value_col_candidates (list): Possible column names for exchange value.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(csv_path)
    # Helper to find columns ignoring case
    def find_col(df, candidates):
        for cand in candidates:
            for c in df.columns:
                if c.lower() == cand.lower():
                    return c
        return None

    # Add more robust candidate matching for value column (handle spaces and variants)
    def find_col_flexible(df, candidates):
        for cand in candidates:
            for c in df.columns:
                if c.lower().replace(' ', '').replace('_', '') == cand.lower().replace(' ', '').replace('_', ''):
                    return c
        return None

    date_col = find_col_flexible(df, date_col_candidates)
    # Add 'CHF to EUR' and variants to candidates
    value_col_candidates = value_col_candidates + ["CHF to EUR", "CHFtoEUR", "CHF_EUR"]
    value_col = find_col_flexible(df, value_col_candidates)
    if date_col is None or value_col is None:
        raise RuntimeError(f"Missing expected columns. Found: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col, value_col])
    df = df.sort_values(date_col)

    # Basic statistics
    print(f"Exchange value statistics ({date_col}):")
    print(df[value_col].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    print(f"  Min: {df[value_col].min():.4f}")
    print(f"  Max: {df[value_col].max():.4f}")
    print(f"  Mean: {df[value_col].mean():.4f}")
    print(f"  Std: {df[value_col].std():.4f}")

    # Time series plot
    plt.figure(figsize=(12, 4.5))
    plt.plot(df[date_col], df[value_col], color='tab:green', lw=1.2)
    plt.title("CHF/EUR Exchange Rate Over Time")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Exchange Rate (CHF/EUR)")
    plt.tight_layout()
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df[value_col], bins=40, color='lightgreen', edgecolor='k', alpha=0.7)
    plt.title("Distribution of CHF/EUR Exchange Rate")
    plt.xlabel("Exchange Rate (CHF/EUR)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_monthly_wind_apport(csv_path="data/wind_incidence_hourly_2024.csv", start_date=None, end_date=None):
    """
    Plot a histogram of apport in wind energy per month for comparison.
    Args:
        csv_path (str): Path to the wind incidence CSV file.
        start_date (str or None): Start date in 'YYYY-MM-DD' format. If None, use earliest.
        end_date (str or None): End date in 'YYYY-MM-DD' format. If None, use latest.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import csv
    from datetime import datetime

    # Read the first three rows for latitude, longitude, and time headers
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)

    # Read the rest as data
    df = pd.read_csv(csv_path, skiprows=3)
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter by date range
    min_time = df["datetime"].min()
    max_time = df["datetime"].max()
    start = pd.to_datetime(start_date) if start_date else min_time
    end = pd.to_datetime(end_date) if end_date else max_time
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    df = df.loc[mask]
    if df.empty:
        print(f"No data in the selected range: {start} to {end}")
        return

    # Sum apport per month (sum over all locations for each hour, then group by month)
    data_cols = [col for col in df.columns if col != "datetime"]
    df["total_wind"] = df[data_cols].sum(axis=1)
    df["month"] = df["datetime"].dt.month
    monthly_apport = df.groupby("month")["total_wind"].sum()

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_apport.index, monthly_apport.values, color='skyblue', edgecolor='k', alpha=0.8)
    plt.xticks(monthly_apport.index, [datetime(2024, m, 1).strftime('%b') for m in monthly_apport.index])
    plt.title("Total Wind Energy Apport per Month (m/s)")
    plt.xlabel("Month")
    plt.ylabel("Total Wind Energy (sum of m/s -> electric generator linear relation)")
    plt.tight_layout()
    plt.show()


def plot_monthly_solar_apport(csv_path="data/solar_incidence_hourly_2024.csv", start_date=None, end_date=None):
    """
    Plot a histogram of apport in solar energy per month for comparison.
    Args:
        csv_path (str): Path to the solar incidence CSV file.
        start_date (str or None): Start date in 'YYYY-MM-DD' format. If None, use earliest.
        end_date (str or None): End date in 'YYYY-MM-DD' format. If None, use latest.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import csv

    # Read the first three rows for latitude, longitude, and time headers
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)

    # Read the rest as data
    df = pd.read_csv(csv_path, skiprows=3)
    df.rename(columns={df.columns[0]: "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])

    # Filter by date range
    min_time = df["time"].min()
    max_time = df["time"].max()
    start = pd.to_datetime(start_date) if start_date else min_time
    end = pd.to_datetime(end_date) if end_date else max_time
    mask = (df["time"] >= start) & (df["time"] <= end)
    df = df.loc[mask]
    if df.empty:
        print(f"No data in the selected range: {start} to {end}")
        return

    # Sum apport per month (sum over all locations for each hour, then group by month)
    data_cols = [col for col in df.columns if col != "time"]
    df["total_solar"] = df[data_cols].sum(axis=1)
    df["month"] = df["time"].dt.month
    monthly_apport = df.groupby("month")["total_solar"].sum()

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_apport.index, monthly_apport.values, color='gold', edgecolor='k', alpha=0.8)
    plt.xticks(monthly_apport.index, [datetime(2024, m, 1).strftime('%b') for m in monthly_apport.index])
    plt.title("Total Solar Energy Apport per Month (kWh/m²)")
    plt.xlabel("Month")
    plt.ylabel("Total Solar Energy (kWh/m²)")
    plt.tight_layout()
    plt.show()
def analyze_monthly_hourly_load(csv_path="data/monthly_hourly_load_values_2024.csv", country_code=None, start_date=None, end_date=None):
    """
    Analyze and visualize monthly hourly load values from ENTSO-E style CSV.
    Args:
        csv_path (str): Path to the CSV file.
        country_code (str or None): Filter by country code (e.g., 'CH' for Switzerland). If None, use all.
        start_date (str or None): Start date (YYYY-MM-DD) or None.
        end_date (str or None): End date (YYYY-MM-DD) or None.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import mode, skew, kurtosis

    df = pd.read_csv(csv_path, sep='\t')
    # Parse datetime from DateUTC (format: DD-MM-YYYY HH:MM)
    df['datetime'] = pd.to_datetime(df['DateUTC'], format='%d-%m-%Y %H:%M')
    if country_code:
        df = df[df['CountryCode'] == country_code]
    if start_date:
        df = df[df['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['datetime'] <= pd.to_datetime(end_date)]
    if df.empty:
        print(f"No data in the selected range: {start_date} to {end_date} for country {country_code}")
        return

    print(f"Data from {df['datetime'].min()} to {df['datetime'].max()} ({len(df)} hours)")
    print("Statistics:")
    print(df['Value'].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    values = df['Value'].values
    print(f"  Mode: {mode(values, keepdims=True).mode[0]:.4f}")
    print(f"  Skewness: {skew(values):.4f}")
    print(f"  Kurtosis: {kurtosis(values):.4f}")

    # --- Plot 1: Daily average and 7-day rolling average (inspired by user code) ---
    daily_avg = df.set_index('datetime')['Value'].resample('D').mean()
    rolling_7d = daily_avg.rolling(7, min_periods=1).mean()
    plt.figure(figsize=(12, 4.5))
    plt.plot(daily_avg.index, daily_avg.values, label="Daily average")
    plt.plot(rolling_7d.index, rolling_7d.values, label="7-day rolling avg")
    country_str = country_code if country_code else (df['CountryCode'].iloc[0] if 'CountryCode' in df.columns else '')
    plt.title(f"Electricity Load — {country_str} (2024)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Load (units as provided)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Existing plots/statistics ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    # Time series plot
    axs[0].plot(df['datetime'], df['Value'], color='tab:blue', lw=0.7)
    axs[0].set_title('Monthly Hourly Load Time Series')
    axs[0].set_ylabel('Load (MW)')

    # Histogram
    axs[1].hist(df['Value'], bins=50, color='tab:orange', alpha=0.7)
    axs[1].set_title('Load Distribution')
    axs[1].set_xlabel('Load (MW)')

    # Mean by hour of day
    df['hour'] = df['datetime'].dt.hour
    mean_by_hour = df.groupby('hour')['Value'].mean()
    axs[2].plot(mean_by_hour.index, mean_by_hour.values, marker='o')
    axs[2].set_title('Mean Load by Hour of Day')
    axs[2].set_xlabel('Hour of Day')
    axs[2].set_ylabel('Mean Load (MW)')

    plt.tight_layout()
    plt.show()


def analyze_wind_incidence(csv_path="data/wind_incidence_hourly_2024.csv", start_date=None, end_date=None, background_img_path=None):
    """
    Analyze wind incidence data: statistics and map of mean annual wind speed by region (coordinates).
    Args:
        csv_path (str): Path to the wind incidence CSV file.
        start_date (str or None): Start date in 'YYYY-MM-DD' format. If None, use earliest.
        end_date (str or None): End date in 'YYYY-MM-DD' format. If None, use latest.
        background_img_path (str or None): Optional path to Switzerland outline image for overlay.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import mode, skew, kurtosis
    import csv

    # Read the first three rows for latitude, longitude, and datetime headers
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)

    # Read the rest as data
    df = pd.read_csv(csv_path, skiprows=3)
    # The first column is the datetime index
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter by date range
    min_time = df["datetime"].min()
    max_time = df["datetime"].max()
    start = pd.to_datetime(start_date) if start_date else min_time
    end = pd.to_datetime(end_date) if end_date else max_time
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    df = df.loc[mask]
    if df.empty:
        print(f"No data in the selected range: {start} to {end}")
        return

    # Build coordinate pairs for each column (skip 'datetime')
    data_cols = [col for col in df.columns if col != "datetime"]
    coords = [(float(lat_row[i]), float(lon_row[i])) for i in range(1, len(lat_row))]
    data = df[data_cols]

    # Compute statistics across all locations and times
    values = data.values.flatten()
    values = values[~np.isnan(values)]
    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_val = mode(values, keepdims=True).mode[0]
    std_val = np.std(values)
    skew_val = skew(values)
    kurt_val = kurtosis(values)
    print(f"Wind Speed Statistics ({start.date()} to {end.date()}):")
    print(f"  Mean: {mean_val:.4f} m/s")
    print(f"  Median: {median_val:.4f} m/s")
    print(f"  Mode: {mode_val:.4f} m/s")
    print(f"  Std: {std_val:.4f} m/s")
    print(f"  Skewness: {skew_val:.4f}")
    print(f"  Kurtosis: {kurt_val:.4f}")

    # Plot distribution
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
    plt.title(f"Wind Speed Distribution (m/s)\n{start.date()} to {end.date()}")
    plt.xlabel("Hourly Wind Speed (m/s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Compute mean annual (or period) wind speed per location
    mean_by_loc = data.mean(axis=0)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    means = mean_by_loc.values

    # Plot map of mean wind speed by location, with optional Switzerland outline
    fig, ax = plt.subplots(figsize=(8, 6))
    if background_img_path is not None:
        import matplotlib.image as mpimg
        img = mpimg.imread(background_img_path)
        extent = [5.8, 10.7, 45.7, 47.9]
        ax.imshow(img, extent=extent, aspect='auto', alpha=0.5, zorder=0)
    sc = ax.scatter(lons, lats, c=means, cmap='Blues', s=80, edgecolor='k', zorder=1)
    plt.colorbar(sc, label="Mean Wind Speed (m/s)")
    ax.set_title(f"Mean Wind Speed by Location\n({start.date()} to {end.date()})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Random Day Wind Incidence Visualization
    # Plot average wind speed across all locations for a randomly selected day in 2024
    # X-axis: Hours of the day (0-23)
    # Y-axis: Average wind speed (m/s) across all coordinates
    import random
    
    # Filter for 2024 only
    df_2024 = df[df["datetime"].dt.year == 2024].copy()
    df_2024 = df_2024.dropna(subset=['datetime'])
    
    if len(df_2024) == 0:
        print("ERROR: No valid 2024 data found. Please regenerate wind_incidence_hourly_2024.csv using regenerate_wind_hourly.py")
    else:
        # Check if data is hourly or daily
        # If hourly, there should be 24 rows per day
        sample_date = df_2024['datetime'].iloc[0].date()
        rows_for_sample = df_2024[df_2024['datetime'].dt.date == sample_date]
        is_hourly = len(rows_for_sample) >= 20  # At least 20 rows suggests hourly data
        
        if not is_hourly:
            print("WARNING: Data appears to be daily, not hourly.")
            print("Please regenerate wind_incidence_hourly_2024.csv using regenerate_wind_hourly.py")
            print("to get proper hourly data.")
            print()
            print("For now, plotting daily average (single value per day):")
            
            # Get unique dates in 2024
            unique_dates = list(df_2024['datetime'].dt.date.unique())
            selected_date = random.choice(unique_dates)
            
            # Get data for selected date
            day_data = df_2024[df_2024['datetime'].dt.date == selected_date]
            
            if len(day_data) > 0:
                # Calculate average across all locations (skip datetime column)
                data_cols = [col for col in day_data.columns if col != 'datetime']
                avg_wind = day_data[data_cols].mean(axis=1).values[0]  # Single value for daily
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar([12], [avg_wind], width=12, alpha=0.7, color='lightblue', 
                       label=f'Daily Average: {avg_wind:.4f} m/s')
                ax.set_xlabel('Hour of Day', fontsize=12)
                ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
                ax.set_title(f'Average Wind Speed - {selected_date}\n(Daily Data - Single Value)', 
                            fontsize=14, fontweight='bold')
                ax.set_xlim(0, 24)
                ax.set_xticks(range(0, 25, 2))
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend()
                plt.tight_layout()
                plt.show()
        else:
            # Data is hourly - proceed with hourly plot
            # Get unique dates in 2024
            unique_dates = sorted([d for d in df_2024['datetime'].dt.date.unique() if d.year == 2024])
            
            # Randomly select a day
            selected_date = random.choice(unique_dates)
            print(f"Selected random day: {selected_date}")
            
            # Filter data for selected day
            day_data = df_2024[df_2024['datetime'].dt.date == selected_date].copy()
            day_data = day_data.sort_values('datetime')
            
            # Extract hours
            day_data['hour'] = day_data['datetime'].dt.hour
            
            # Calculate average across all locations for each hour
            data_cols = [col for col in day_data.columns if col not in ['datetime', 'hour']]
            hourly_avg = []
            hours = []
            
            for hour in range(24):
                hour_data = day_data[day_data['hour'] == hour]
                if len(hour_data) > 0:
                    # Calculate mean across all locations, ignoring NaN values
                    avg = hour_data[data_cols].mean(axis=1).mean()
                    hourly_avg.append(avg if not np.isnan(avg) else 0.0)
                    hours.append(hour)
                else:
                    # No data for this hour
                    hourly_avg.append(0.0)
                    hours.append(hour)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot line
            ax.plot(hours, hourly_avg, 'o-', linewidth=2, markersize=8, 
                   color='lightblue', label='Average Wind Speed', alpha=0.8)
            
            # Fill area under curve
            ax.fill_between(hours, hourly_avg, alpha=0.3, color='lightblue')
            
            # Formatting
            ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Average Wind Speed Across All Locations\n{selected_date} (Randomly Selected Day in 2024)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, 23.5)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper left', fontsize=10)
            
            # Add statistics text box
            max_wind = max(hourly_avg)
            max_hour = hours[hourly_avg.index(max_wind)]
            min_wind = min(hourly_avg)
            min_hour = hours[hourly_avg.index(min_wind)]
            avg_wind = np.mean(hourly_avg)
            
            stats_text = f'Max: {max_wind:.4f} m/s at {max_hour:02d}:00\n'
            stats_text += f'Min: {min_wind:.4f} m/s at {min_hour:02d}:00\n'
            stats_text += f'Daily Average: {avg_wind:.4f} m/s'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nStatistics for {selected_date}:")
            print(f"  Maximum wind speed: {max_wind:.4f} m/s at {max_hour:02d}:00")
            print(f"  Minimum wind speed: {min_wind:.4f} m/s at {min_hour:02d}:00")
            print(f"  Daily average: {avg_wind:.4f} m/s")
            print(f"  Number of locations: {len(data_cols)}")


def analyze_solar_incidence(csv_path="data/solar_incidence_hourly_2024.csv", start_date=None, end_date=None, background_img_path=None):
    """
    Analyze solar incidence data: statistics and map of mean annual solar exposure by region (coordinates).
    Args:
        csv_path (str): Path to the solar incidence CSV file.
        start_date (str or None): Start date in 'YYYY-MM-DD' format. If None, use earliest.
        end_date (str or None): End date in 'YYYY-MM-DD' format. If None, use latest.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import mode, skew, kurtosis

    # Read the first three rows for latitude, longitude, and time headers
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lat_row = next(reader)
        lon_row = next(reader)
        time_row = next(reader)
    # Now read the rest as data
    df = pd.read_csv(csv_path, skiprows=3)
    # The first column is the time index
    df.rename(columns={df.columns[0]: "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    # Filter by date range
    min_time = df["time"].min()
    max_time = df["time"].max()
    start = pd.to_datetime(start_date) if start_date else min_time
    end = pd.to_datetime(end_date) if end_date else max_time
    mask = (df["time"] >= start) & (df["time"] <= end)
    df = df.loc[mask]
    if df.empty:
        print(f"No data in the selected range: {start} to {end}")
        return

    # Build coordinate pairs for each column (skip 'time')
    data_cols = [col for col in df.columns if col != "time"]
    coords = [(float(lat_row[i]), float(lon_row[i])) for i in range(1, len(lat_row))]
    data = df[data_cols]

    # Compute statistics across all locations and times
    values = data.values.flatten()
    values = values[~np.isnan(values)]
    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_val = mode(values, keepdims=True).mode[0]
    std_val = np.std(values)
    skew_val = skew(values)
    kurt_val = kurtosis(values)
    print(f"Solar Incidence Statistics ({start.date()} to {end.date()}):")
    print(f"  Mean: {mean_val:.4f} kWh/m^2")
    print(f"  Median: {median_val:.4f} kWh/m^2")
    print(f"  Mode: {mode_val:.4f} kWh/m^2")
    print(f"  Std: {std_val:.4f} kWh/m^2")
    print(f"  Skewness: {skew_val:.4f}")
    print(f"  Kurtosis: {kurt_val:.4f}")

    # Plot distribution
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=50, color='gold', edgecolor='k', alpha=0.7)
    plt.title(f"Solar Incidence Distribution (kWh/m^2)\n{start.date()} to {end.date()}")
    plt.xlabel("Hourly Solar Incidence (kWh/m^2)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Compute mean annual (or period) solar incidence per location
    mean_by_loc = data.mean(axis=0)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    means = mean_by_loc.values

    # Plot map of mean solar incidence by location, with optional Switzerland outline
    fig, ax = plt.subplots(figsize=(8, 6))
    # If a background image is provided, plot it first
    if background_img_path is not None:
        import matplotlib.image as mpimg
        img = mpimg.imread(background_img_path)
        # Swiss bounding box (approx): lon 5.8–10.7, lat 45.7–47.8
        extent = [5.8, 10.7, 45.7, 47.8]
        ax.imshow(img, extent=extent, aspect='auto', alpha=0.5, zorder=0)
    sc = ax.scatter(lons, lats, c=means, cmap='YlOrRd', s=80, edgecolor='k', zorder=1)
    plt.colorbar(sc, label="Mean Solar Incidence (kWh/m^2 per hour)")
    ax.set_title(f"Mean Solar Incidence by Location\n({start.date()} to {end.date()})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
import numpy as np
from scipy.stats import mode, skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_spot_price(csv_path="data/spot_price_hourly.csv", start_date=None, end_date=None):
    """
    Perform analysis of the spot price curve (distribution, stats, moments, mean by hour) between two dates.
    Args:
        csv_path (str): Path to the spot price CSV file.
        start_date (str or None): Start date in 'YYYY-MM-DD' format. If None, use earliest.
        end_date (str or None): End date in 'YYYY-MM-DD' format. If None, use latest.
    """
    def _parse_date(date_str, fallback):
        if date_str is None:
            return fallback
        return pd.to_datetime(date_str)

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    # Filter by date range
    min_time = df["time"].min()
    max_time = df["time"].max()
    start = _parse_date(start_date, min_time)
    end = _parse_date(end_date, max_time)
    mask = (df["time"] >= start) & (df["time"] <= end)
    df = df.loc[mask]
    if df.empty:
        print(f"No data in the selected range: {start} to {end}")
        return
    prices = df["price"].to_numpy()
    # Basic statistics
    mean_val = np.mean(prices)
    median_val = np.median(prices)
    mode_val = mode(prices, keepdims=True).mode[0]
    std_val = np.std(prices)
    skew_val = skew(prices)
    kurt_val = kurtosis(prices)
    print(f"Spot Price Statistics ({start.date()} to {end.date()}):")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Mode: {mode_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  Skewness (3rd moment): {skew_val:.4f}")
    print(f"  Kurtosis (4th moment): {kurt_val:.4f}")

    # Plot price distribution
    plt.figure(figsize=(8, 4))
    plt.hist(prices, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
    plt.title(f"Spot Price Distribution (EUR/MWh)\n{start.date()} to {end.date()}")
    plt.xlabel("Price (EUR/MWh)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Mean price for each hour of the day (0-23) in the selected range
    df["hour"] = df["time"].dt.hour
    hourly_mean = df.groupby("hour")["price"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(hourly_mean.index.to_numpy(), hourly_mean.values.astype(float), marker='o')
    plt.title(f"Mean Spot Price by Hour of Day\n({start.date()} to {end.date()})")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Price (EUR/MWh)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spot_price(csv_path="data/spot_price_hourly.csv", start_date="2024-01-01", end_date="2024-12-31"):
    """
    Plot the electricity spot price in Switzerland over time.
    Args:
        csv_path (str): Path to the spot price CSV file.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    # Ensure the index is timezone-naive for comparison
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("time")
    # Also make start/end date timezone-naive
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df.index >= start) & (df.index <= end)
    df_filtered = df.loc[mask]
    plt.figure(figsize=(12, 5))
    plt.plot(df_filtered.index, df_filtered["price"], label="Spot Price (EUR/MWh)")
    plt.xlabel("Time")
    plt.ylabel("Spot Price (EUR/MWh)")
    plt.title(f"Electricity Spot Price in Switzerland ({start_date} to {end_date})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_consumption_hourly(csv_path="data/consumption_hourly_2024.csv", start_date=None, end_date=None):
    """
    Analyze and visualize hourly energy consumption.
    Args:
        csv_path (str): Path to the hourly consumption CSV.
        start_date (str or None): Start date (YYYY-MM-DD) or None.
        end_date (str or None): End date (YYYY-MM-DD) or None.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import mode, skew, kurtosis

    df = pd.read_csv(csv_path, parse_dates=[0])
    # Assume first column is datetime, second is consumption
    df.columns = ['datetime', 'consumption']
    df.set_index('datetime', inplace=True)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if df.empty:
        print(f"No data in the selected range: {start_date} to {end_date}")
        return

    print(f"Data from {df.index.min()} to {df.index.max()} ({len(df)} hours)")
    print("Statistics:")
    print(df['consumption'].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))

    # Additional statistics
    values = df['consumption'].values
    print(f"  Mode: {mode(values, keepdims=True).mode[0]:.4f}")
    print(f"  Skewness: {skew(values):.4f}")
    print(f"  Kurtosis: {kurtosis(values):.4f}")

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Time series plot
    axs[0].plot(df.index, df['consumption'], color='tab:blue', lw=0.7)
    axs[0].set_title('Hourly Consumption Time Series')
    axs[0].set_ylabel('Consumption (MWh)')

    # Histogram
    axs[1].hist(df['consumption'], bins=50, color='tab:orange', alpha=0.7)
    axs[1].set_title('Consumption Distribution')
    axs[1].set_xlabel('Consumption (MWh)')

    # Daily/weekly pattern (mean by hour of day)
    df['hour'] = df.index.hour
    mean_by_hour = df.groupby('hour')['consumption'].mean()
    axs[2].plot(mean_by_hour.index, mean_by_hour.values, marker='o')
    axs[2].set_title('Mean Consumption by Hour of Day')
    axs[2].set_xlabel('Hour of Day')
    axs[2].set_ylabel('Mean Consumption (MWh)')

    plt.tight_layout()
    plt.show()

