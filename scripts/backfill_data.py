import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append(os.getcwd())
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger

logger = get_logger("Backfill")

def generate_historical_data(city="Karachi", days=7):
    """
    Generates synthetic historical AQI data for demonstration purposes.
    In a real scenario, this would fetch from a historical data provider.
    """
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_time = start_date
    while current_time <= end_date:
        # Generate some realistic-ish Karachi AQI values (typically between 100 and 200)
        base_aqi = 150 + np.sin(current_time.hour * np.pi / 12) * 30 + np.random.normal(0, 10)
        
        row = {
            'city': f"{city}, Pakistan",
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
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
            'target_aqi': float(base_aqi + np.random.normal(0, 5)),
            'target_day': (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
        }
        
        # Add time features
        ts = pd.to_datetime(row['timestamp'])
        row['hour'] = ts.hour
        row['day'] = ts.day
        row['month'] = ts.month
        row['day_of_week'] = ts.dayofweek
        
        data.append(row)
        current_time += timedelta(hours=1)
        
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def run_backfill():
    try:
        logger.info("Starting historical data backfill for Karachi...")
        
        # 1. Generate data
        df = generate_historical_data("Karachi", days=7)
        logger.info(f"Generated {len(df)} rows of historical data")
        
        # 2. Upload to Hopsworks
        fs_ingestion = FeatureStoreIngestion()
        fs_ingestion.save_to_feature_group(df, "aqi_features")
        
        logger.info("Backfill completed successfully!")
        print(f"\nSUCCESS: Backfilled {len(df)} rows for Karachi into 'aqi_features'.")
        
    except Exception as e:
        logger.error(f"Backfill failed: {str(e)}")
        print(f"\nERROR: Backfill failed. {str(e)}")

if __name__ == "__main__":
    run_backfill()
