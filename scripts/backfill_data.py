import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
sys.path.append(os.getcwd())
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger

logger = get_logger("Backfill")


def generate_historical_data(city="Karachi", days=30):
    """
    Generates synthetic historical AQI data for backfilling the v2 feature group.

    Key improvements over the original:
      - Weekly cycle (weekday traffic → higher AQI)
      - Seasonal / multi-day drift (random walk component)
      - Pollution events (3-5 day spikes)
      - Rain clearing (sudden AQI drops)
      - Weather features that correlate with AQI changes
      - PM2.5/PM10 that are NOT just a fixed ratio of AQI
    """
    np.random.seed(42)  # reproducible but realistic

    data = []
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1

    # ── Build a realistic AQI time-series first ────────────────────────
    # 1. Slow drift (random walk, mean-reverting around 130 for Karachi)
    drift = np.zeros(total_hours)
    drift[0] = np.random.uniform(-20, 20)
    for i in range(1, total_hours):
        drift[i] = drift[i - 1] * 0.998 + np.random.normal(0, 0.5)  # mean-reverts slowly

    # 2. Pollution events: 3-5 day spikes that raise AQI by 50-120
    events = np.zeros(total_hours)
    h = 0
    while h < total_hours:
        if np.random.random() < 0.008:  # ~1 event per 5 days
            spike_mag = np.random.uniform(50, 120)
            spike_dur = np.random.randint(48, 120)  # 2-5 days
            ramp_up = min(spike_dur // 3, 24)
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

    # 3. Rain clearing events (AQI drops by 30-60 suddenly, recovers over 12-24h)
    rain_effect = np.zeros(total_hours)
    for h in range(total_hours):
        if np.random.random() < 0.003 and events[h] < 20:  # rain during non-events
            drop = np.random.uniform(30, 60)
            recovery = np.random.randint(12, 24)
            for j in range(min(recovery, total_hours - h)):
                rain_effect[h + j] = -drop * (1 - j / recovery)

    # ── Generate row-by-row ───────────────────────────────────────────
    current_time = start_date
    for h_idx in range(total_hours):
        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Diurnal cycle: peak at 8-10 AM (rush hour) and 6-8 PM
        diurnal = 15 * np.sin((hour - 9) * np.pi / 12) + 10 * np.sin((hour - 19) * np.pi / 6)

        # Weekly cycle: weekdays higher (traffic), weekends lower
        weekly = 10 if day_of_week < 5 else -15

        # Compose AQI
        base_aqi = 130 + diurnal + weekly + drift[h_idx] + events[h_idx] + rain_effect[h_idx]
        base_aqi += np.random.normal(0, 5)  # observation noise
        base_aqi = max(15, min(400, base_aqi))  # clamp to realistic range

        # Weather features — loosely correlated with AQI
        # Higher wind → lower AQI (dispersion)
        wind_speed = max(0.5, np.random.uniform(2, 8) + (rain_effect[h_idx] / 20))
        # Temperature: diurnal + seasonal
        temperature = 28 + 6 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 1.5)
        # Humidity: inversely correlated with temperature, rain spikes it
        humidity = 55 - 0.5 * temperature + np.random.normal(0, 5)
        if rain_effect[h_idx] < -10:
            humidity += 25
        humidity = max(20, min(95, humidity))
        # Pressure
        pressure = 1012 + np.random.normal(0, 3)

        # Wind inversely affects AQI slightly
        wind_adjusted_aqi = base_aqi - (wind_speed - 4) * 3
        wind_adjusted_aqi = max(15, min(400, wind_adjusted_aqi))

        # Pollutants — derive from AQI with independent noise
        pm25 = wind_adjusted_aqi * np.random.uniform(0.6, 0.95) + np.random.normal(0, 8)
        pm10 = wind_adjusted_aqi * np.random.uniform(0.35, 0.65) + np.random.normal(0, 5)
        o3 = np.random.uniform(5, 35) + 8 * np.sin((hour - 13) * np.pi / 8)  # peaks midday
        no2 = 15 + 10 * (1 if day_of_week < 5 else 0) + np.random.normal(0, 5)  # traffic
        so2 = np.random.uniform(2, 12) + np.random.normal(0, 2)
        co = np.random.uniform(0.1, 0.8) + 0.2 * (1 if 7 <= hour <= 10 else 0)

        row = {
            'city': f"{city}, Pakistan",
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'ingestion_timestamp': current_time,
            # ── pollutant features ───────────────────────────────────────────
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
            # ── targets (reference only; model_trainer creates the real target) ─
            'target_aqi': 0.0,
            'target_day': (current_time + timedelta(days=3)).strftime('%Y-%m-%d'),
        }

        # Time features
        row['hour'] = int(hour)
        row['day'] = int(current_time.day)
        row['month'] = int(current_time.month)
        row['day_of_week'] = int(day_of_week)

        data.append(row)
        current_time += timedelta(hours=1)

    df = pd.DataFrame(data)

    # Cast types to match v2 schema
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
    try:
        logger.info(f"Starting historical data backfill for Karachi ({days} days)...")

        # 1. Generate historical data
        df = generate_historical_data("Karachi", days=days)
        logger.info(f"Generated {len(df)} rows of historical data "
                    f"({df['ingestion_timestamp'].min()} to {df['ingestion_timestamp'].max()})")

        # Quick sanity check
        aqi_std = df['aqi'].std()
        aqi_mean = df['aqi'].mean()
        logger.info(f"AQI stats — mean: {aqi_mean:.1f}, std: {aqi_std:.1f}, "
                    f"min: {df['aqi'].min():.0f}, max: {df['aqi'].max():.0f}")

        # 2. Upload to Hopsworks v2 feature group
        fs_ingestion = FeatureStoreIngestion()
        fs_ingestion.save_to_feature_group(df, "aqi_features", version=3)

        logger.info("Backfill completed successfully!")
        print(f"\nSUCCESS: Backfilled {len(df)} rows for Karachi into 'aqi_features' v2.")
        print(f"  From : {df['ingestion_timestamp'].min()}")
        print(f"  To   : {df['ingestion_timestamp'].max()}")
        print(f"  AQI  : mean={aqi_mean:.1f}, std={aqi_std:.1f}")

    except Exception as e:
        logger.error(f"Backfill failed: {str(e)}")
        print(f"\nERROR: Backfill failed. {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill AQI data into Hopsworks v2 feature group")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of past days to backfill (default: 30)")
    args = parser.parse_args()
    run_backfill(days=args.days)
