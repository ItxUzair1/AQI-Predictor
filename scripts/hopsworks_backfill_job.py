"""
hopsworks_backfill_job.py
─────────────────────────
Self-contained backfill script to run INSIDE Hopsworks (Jupyter or as a Job).
No local dependencies required — just upload this file and run it.

Why run inside Hopsworks?
  1. Kafka port 9092 is NOT blocked (internal network).
  2. Spark materialization jobs get priority on internal resources.
  3. No Windows cert workarounds needed.

Usage:
  - Hopsworks Jupyter: Copy-paste into a cell and run.
  - Hopsworks Job: Upload this file, create a Python job, and execute.
"""

# pyrefly: ignore [missing-import]
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import argparse
import sys


def generate_historical_data(city="Karachi", days=30):
    """
    Generates synthetic historical AQI data for backfilling.
    Identical logic to scripts/backfill_data.py but fully self-contained.
    """
    np.random.seed(42)

    data = []
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1

    # ── Slow drift (random walk, mean-reverting around 70 for Karachi in late spring/summer)
    drift = np.zeros(total_hours)
    drift[0] = np.random.uniform(-5, 5)
    for i in range(1, total_hours):
        drift[i] = drift[i - 1] * 0.995 + np.random.normal(0, 0.4)

    # ── Moderate pollution events: smaller spikes in summer due to sea breeze dispersion
    events = np.zeros(total_hours)
    h = 0
    while h < total_hours:
        if np.random.random() < 0.006:  # ~1 event per 7 days
            spike_mag = np.random.uniform(15, 35)
            spike_dur = np.random.randint(24, 72)
            ramp_up = min(spike_dur // 3, 12)
            for j in range(min(spike_dur, total_hours - h)):
                if j < ramp_up:
                    events[h + j] = spike_mag * (j / ramp_up)
                elif j > spike_dur - ramp_up:
                    events[h + j] = spike_mag * ((spike_dur - j) / ramp_up)
                else:
                    events[h + j] = spike_mag
            h += spike_dur
        else:
            h += 1

    # ── Rain clearing events (rare in May, but causes sudden drops)
    rain_effect = np.zeros(total_hours)
    for h in range(total_hours):
        if np.random.random() < 0.0015:
            drop = np.random.uniform(15, 30)
            recovery = np.random.randint(8, 16)
            for j in range(min(recovery, total_hours - h)):
                rain_effect[h + j] = -drop * (1 - j / recovery)

    # ── Generate row-by-row
    current_time = start_date
    for h_idx in range(total_hours):
        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Diurnal cycle: peaks around rush hours
        diurnal = 6 * np.sin((hour - 9) * np.pi / 12) + 4 * np.sin((hour - 19) * np.pi / 6)

        # Weekly cycle
        weekly = 3 if day_of_week < 5 else -4

        # Compose AQI
        base_aqi = 68 + diurnal + weekly + drift[h_idx] + events[h_idx] + rain_effect[h_idx]
        base_aqi += np.random.normal(0, 2)
        base_aqi = max(30, min(150, base_aqi))

        # Weather features calibrated to Karachi summer (shown in screenshots)
        # Wind: 15 to 26 km/h -> 4.2 to 7.2 m/s
        wind_speed = max(1.5, np.random.uniform(4.0, 7.5) + (rain_effect[h_idx] / 10))
        # Temperature: 29C to 35C
        temperature = 31.5 + 3.5 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 0.8)
        # Humidity: 63% to 89% (coastal humidity)
        humidity = 76.0 - 1.2 * (temperature - 31.5) + np.random.normal(0, 2.0)
        if rain_effect[h_idx] < -5:
            humidity += 12
        humidity = max(45, min(95, humidity))
        # Pressure: ~1008 hPa
        pressure = 1008 + np.random.normal(0, 1.5)

        # Wind dispersion effect
        wind_adjusted_aqi = base_aqi - (wind_speed - 5.5) * 2
        wind_adjusted_aqi = max(35, min(140, wind_adjusted_aqi))

        # Pollutants scaled correctly to EPA AQI formula (for AQI 35-140)
        # pm25 range: ~10 to 50 ug/m3
        pm25 = 12.0 + (wind_adjusted_aqi - 50.0) * (23.4 / 50.0) + np.random.normal(0, 1.0)
        # pm10 range: ~35 to 130 ug/m3
        pm10 = 54.0 + (wind_adjusted_aqi - 50.0) * (96.0 / 50.0) + np.random.normal(0, 2.0)
        
        o3 = np.random.uniform(10, 25) + 5 * np.sin((hour - 13) * np.pi / 8)
        no2 = 8 + 6 * (1 if day_of_week < 5 else 0) + np.random.normal(0, 2)
        so2 = np.random.uniform(1, 6) + np.random.normal(0, 0.8)
        co = np.random.uniform(0.1, 0.4) + 0.1 * (1 if 7 <= hour <= 10 else 0)

        row = {
            'city': f"{city}, Pakistan",
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'ingestion_timestamp': current_time,
            'aqi': float(round(wind_adjusted_aqi, 1)),
            'pm25': float(max(0, round(pm25, 1))),
            'pm10': float(max(0, round(pm10, 1))),
            'o3': float(max(0, round(o3, 1))),
            'no2': float(max(0, round(no2, 1))),
            'so2': float(max(0, round(so2, 1))),
            'co': float(max(0, round(co, 2))),
            'temperature': float(round(temperature, 1)),
            'humidity': float(round(humidity, 1)),
            'pressure': float(round(pressure, 1)),
            'wind_speed': float(round(wind_speed, 1)),
            'target_aqi': 0.0,
            'target_day': (current_time + timedelta(days=3)).strftime('%Y-%m-%d'),
            'hour': int(hour),
            'day': int(current_time.day),
            'month': int(current_time.month),
            'day_of_week': int(day_of_week),
        }
        data.append(row)
        current_time += timedelta(hours=1)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])

    numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                      'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    time_cols = ['hour', 'day', 'month', 'day_of_week']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    df.fillna(0, inplace=True)
    return df


def run_backfill(days=30):
    print("=" * 60)
    print(f"  HOPSWORKS BACKFILL — {days} days of Karachi AQI data")
    print("=" * 60)

    # 1. Connect (no API key needed when running inside Hopsworks)
    print("\n📡 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 2. Clean up old feature group if it exists
    print("\n  Cleaning up old feature group (if any)...")
    try:
        old_fg = fs.get_feature_group("aqi_features", version=4)
        print("   Found existing FG v4. Deleting to recreate as regular FG...")
        old_fg.delete()
        print("   Deleted existing FG v4 successfully.")
    except Exception as e:
        print(f"   No existing v4 to delete (OK): {e}")

    # 3. Generate data
    print(f"\n🏭 Generating {days} days of historical data...")
    df = generate_historical_data("Karachi", days=days)
    print(f"   Generated {len(df)} rows")
    print(f"   Range: {df['ingestion_timestamp'].min()} → {df['ingestion_timestamp'].max()}")
    print(f"   AQI: mean={df['aqi'].mean():.1f}, std={df['aqi'].std():.1f}")

    # 4. Create a fresh REGULAR feature group (NOT StreamFeatureGroup)
    print("\n📤 Creating fresh regular feature group v4...")
    aqi_fg = fs.create_feature_group(
        name="aqi_features",
        version=4,
        primary_key=["city", "ingestion_timestamp"],
        event_time="ingestion_timestamp",
        description="AQI and weather features for city"
    )

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    aqi_fg.insert(df, write_options={"wait_for_job": True})

    print(f"\n SUCCESS: Backfilled {len(df)} rows into aqi_features v3!")
    print(f"   The data is now available in the offline store.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill AQI data (run inside Hopsworks)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backfill")
    args = parser.parse_args()
    run_backfill(days=args.days)
