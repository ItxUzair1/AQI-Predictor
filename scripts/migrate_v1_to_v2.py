"""
Migrate existing data from aqi_features v1 to v2.

The old v1 feature group used primary_key=["city", "timestamp"].
The new v2 feature group uses primary_key=["city", "ingestion_timestamp"].

This script reads all rows from v1, adds the missing 'ingestion_timestamp' column
(using the original 'timestamp' value as a fallback), and inserts them into v2.
"""

import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger

logger = get_logger("Migration v1->v2")

def migrate():
    try:
        fs_ingestion = FeatureStoreIngestion()
        fs = fs_ingestion.get_feature_store()

        # 1. Read all data from v1
        logger.info("Reading data from aqi_features v1...")
        fg_v1 = fs.get_feature_group("aqi_features", version=1)
        df = fg_v1.read()
        print(f"Found {len(df)} rows in v1")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample:\n{df.head()}")

        if df.empty:
            print("No data to migrate.")
            return

        # 2. Add ingestion_timestamp column (use original timestamp as fallback)
        if 'ingestion_timestamp' not in df.columns:
            logger.info("Adding ingestion_timestamp column from original timestamp...")
            df['ingestion_timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 3. Ensure correct dtypes
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
        
        # Ensure int64 for time columns
        for col in ['hour', 'day', 'month', 'day_of_week']:
            if col in df.columns:
                df[col] = df[col].astype('int64')
        
        # Ensure float for numerical columns
        for col in ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                     'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

        # 4. Insert into v2
        logger.info(f"Inserting {len(df)} rows into aqi_features v2...")
        fg_v2 = fs.get_or_create_feature_group(
            name="aqi_features",
            version=2,
            primary_key=["city", "ingestion_timestamp"],
            event_time="ingestion_timestamp",
            description="AQI and weather features for city"
        )
        fg_v2.insert(df)

        print(f"\nSUCCESS: Migrated {len(df)} rows from v1 to v2!")
        logger.info(f"Migration complete: {len(df)} rows migrated")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    migrate()
