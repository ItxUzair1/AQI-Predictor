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
    # 1. Slow drift (random walk, mean-reverting around 70 for Karachi in late spring/summer)
    drift = np.zeros(total_hours)
    drift[0] = np.random.uniform(-5, 5)
    for i in range(1, total_hours):
        drift[i] = drift[i - 1] * 0.995 + np.random.normal(0, 0.4)

    # 2. Moderate pollution events: smaller spikes in summer due to sea breeze dispersion
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

    # 3. Rain clearing events (rare in May, but causes sudden drops)
    rain_effect = np.zeros(total_hours)
    for h in range(total_hours):
        if np.random.random() < 0.0015:
            drop = np.random.uniform(15, 30)
            recovery = np.random.randint(8, 16)
            for j in range(min(recovery, total_hours - h)):
                rain_effect[h + j] = -drop * (1 - j / recovery)

    # ── Generate row-by-row ───────────────────────────────────────────
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


def run_backfill(days=30, batch_size_days=5, wait_for_job=False):
    try:
        logger.info(f"Starting historical data backfill for Karachi ({days} days)...")

        # 1. Generate ALL historical data at once
        df = generate_historical_data("Karachi", days=days)
        logger.info(f"Generated {len(df)} rows of historical data "
                    f"({df['ingestion_timestamp'].min()} to {df['ingestion_timestamp'].max()})")

        # Quick sanity check
        aqi_std = df['aqi'].std()
        aqi_mean = df['aqi'].mean()
        logger.info(f"AQI stats — mean: {aqi_mean:.1f}, std: {aqi_std:.1f}, "
                    f"min: {df['aqi'].min():.0f}, max: {df['aqi'].max():.0f}")

        # 2. Connect to Hopsworks feature store
        from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
        fs_ingestion = FeatureStoreIngestion()
        fs = fs_ingestion.get_feature_store()

        # Create/get feature group
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=3,
            primary_key=["city", "ingestion_timestamp"],
            event_time="ingestion_timestamp",
            description="AQI and weather features for city"
        )

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 3. Insert in small batches WITHOUT triggering Spark materialization
        #    This prevents CPU overload on the Hopsworks shared cluster.
        batch_size_hours = batch_size_days * 24
        total_rows = len(df)
        num_batches = (total_rows + batch_size_hours - 1) // batch_size_hours

        logger.info(f"Inserting {total_rows} rows in {num_batches} batches "
                    f"(~{batch_size_days} days per batch, no Spark until the end)...")

        for i in range(num_batches):
            start_idx = i * batch_size_hours
            end_idx = min((i + 1) * batch_size_hours, total_rows)
            batch_df = df.iloc[start_idx:end_idx].copy()

            batch_num = i + 1
            is_last_batch = (batch_num == num_batches)

            if is_last_batch:
                # Last batch: trigger materialization
                logger.info(f"Batch {batch_num}/{num_batches}: {len(batch_df)} rows "
                            f"(final — triggering Spark materialization)...")
                aqi_fg.insert(batch_df, write_options={"wait_for_job": wait_for_job})
            else:
                # All other batches: insert WITHOUT Spark job
                logger.info(f"Batch {batch_num}/{num_batches}: {len(batch_df)} rows "
                            f"(no Spark job)...")
                aqi_fg.insert(batch_df, write_options={
                    "start_offline_materialization": False,
                    "wait_for_job": False
                })

        # Update local cache
        local_path = os.path.join("data", "aqi_features.csv")
        os.makedirs("data", exist_ok=True)
        df_to_cache = df.copy()
        if 'ingestion_timestamp' in df_to_cache.columns:
            df_to_cache['ingestion_timestamp'] = pd.to_datetime(df_to_cache['ingestion_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_cache.to_csv(local_path, index=False)
        logger.info(f"Local cache updated with backfill data: {local_path}")

        logger.info("Backfill completed successfully!")
        print(f"\nSUCCESS: Backfilled {total_rows} rows for Karachi into 'aqi_features' v3.")
        print(f"  From : {df['ingestion_timestamp'].min()}")
        print(f"  To   : {df['ingestion_timestamp'].max()}")
        print(f"  AQI  : mean={aqi_mean:.1f}, std={aqi_std:.1f}")
        print(f"  Batches: {num_batches} (Spark materialization triggered on last batch)")
        if not wait_for_job:
            print("  Note: Spark materialization is running in the background on Hopsworks.")
            print("        You can monitor progress in the Hopsworks UI.")

    except Exception as e:
        logger.error(f"Backfill failed: {str(e)}")
        print(f"\nERROR: Backfill failed. {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill AQI data into Hopsworks v3 feature group")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of past days to backfill (default: 30)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Days per batch (default: 5)")
    parser.add_argument("--wait", action="store_true", default=False,
                        help="Wait for the final Spark materialization job to complete on Hopsworks (default: False)")
    args = parser.parse_args()
    run_backfill(days=args.days, batch_size_days=args.batch_size, wait_for_job=args.wait)

