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
    
    Each row gets:
      - timestamp      : the observation time (hourly)
      - ingestion_timestamp : same as observation time (simulates when it was "ingested")
    
    This matches the v2 feature group primary_key=["city", "ingestion_timestamp"],
    so every row is unique and no upserts occur.
    """
    data = []
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    current_time = start_date
    while current_time <= end_date:
        # Generate realistic Karachi AQI values (typically 100-200)
        base_aqi = 150 + np.sin(current_time.hour * np.pi / 12) * 30 + np.random.normal(0, 10)

        # ingestion_timestamp = current_time (each historical hour is its own "ingestion")
        ingestion_ts = current_time

        row = {
            'city': f"{city}, Pakistan",
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            # ── v2 PK & event_time ──────────────────────────────────────────
            'ingestion_timestamp': ingestion_ts,
            # ── pollutant features ───────────────────────────────────────────
            'aqi': float(base_aqi),
            'pm25': float(base_aqi * 0.8),
            'pm10': float(base_aqi * 0.5),
            'o3': float(np.random.uniform(5, 20)),
            'no2': float(np.random.uniform(10, 30)),
            'so2': float(np.random.uniform(2, 10)),
            'co': float(np.random.uniform(0.1, 0.5)),
            'temperature': float(25 + np.sin(current_time.hour * np.pi / 12) * 5),
            'humidity': float(60 + np.random.uniform(-10, 10)),
            'pressure': float(1010 + np.random.uniform(-5, 5)),
            'wind_speed': float(np.random.uniform(1, 10)),
            # ── targets ──────────────────────────────────────────────────────
            'target_aqi': float(base_aqi + np.random.normal(0, 5)),
            'target_day': (current_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        }

        # Time features (same as DataTransformation)
        row['hour'] = int(current_time.hour)
        row['day'] = int(current_time.day)
        row['month'] = int(current_time.month)
        row['day_of_week'] = int(current_time.weekday())

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
                    f"({df['ingestion_timestamp'].min()} → {df['ingestion_timestamp'].max()})")

        # 2. Upload to Hopsworks v2 feature group
        fs_ingestion = FeatureStoreIngestion()
        fs_ingestion.save_to_feature_group(df, "aqi_features", version=2)

        logger.info("Backfill completed successfully!")
        print(f"\nSUCCESS: Backfilled {len(df)} rows for Karachi into 'aqi_features' v2.")
        print(f"  From : {df['ingestion_timestamp'].min()}")
        print(f"  To   : {df['ingestion_timestamp'].max()}")

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
