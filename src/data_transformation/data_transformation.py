import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.logger import get_logger
from src.exception import CustomException
import sys

logger = get_logger("Data Transformation")

class DataTransformation:
    def __init__(self):
        pass

    def flatten_aqi_data(self, raw_data):
        """
        Flattens the raw JSON response from AQICN API into a structured format.
        """
        try:
            data = raw_data.get('data', {})
            iaqi = data.get('iaqi', {})
            time_info = data.get('time', {})
            city_info = data.get('city', {})
            
            # Extract basic features
            features = {
                'city': city_info.get('name'),
                'timestamp': time_info.get('s'),
                'aqi': data.get('aqi'),
                'pm25': iaqi.get('pm25', {}).get('v'),
                'pm10': iaqi.get('pm10', {}).get('v'),
                'o3': iaqi.get('o3', {}).get('v'),
                'no2': iaqi.get('no2', {}).get('v'),
                'so2': iaqi.get('so2', {}).get('v'),
                'co': iaqi.get('co', {}).get('v'),
                'temperature': iaqi.get('t', {}).get('v'),
                'humidity': iaqi.get('h', {}).get('v'),
                'pressure': iaqi.get('p', {}).get('v'),
                'wind_speed': iaqi.get('w', {}).get('v')
            }
            
            return features
        except Exception as e:
            logger.error(f"Error flattening data: {str(e)}")
            raise CustomException(e, sys)

    def extract_time_features(self, df):
        """
        Extracts time-based features from the timestamp.
        """
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            return df
        except Exception as e:
            logger.error(f"Error extracting time features: {str(e)}")
            raise CustomException(e, sys)

    def extract_targets(self, raw_data):
        """
        Extracts the 3-day ahead AQI forecast from the AQICN daily forecast array.
        The API returns: [today, day+1, day+2, day+3, ...]
        We pick index 3 (day+3) to store as the reference target.
        Note: The model's actual training target is created by time-shifting
        in model_trainer.py, so this column serves as a secondary reference.
        """
        try:
            forecast = raw_data.get('data', {}).get('forecast', {}).get('daily', {})
            pm25_forecast = forecast.get('pm25', [])

            # index 3 → 3 days ahead; fallback to last available
            if len(pm25_forecast) >= 4:
                entry = pm25_forecast[3]
            elif len(pm25_forecast) > 0:
                entry = pm25_forecast[-1]
            else:
                return {'target_aqi': None, 'target_day': None}

            return {
                'target_aqi': entry.get('avg'),
                'target_day': entry.get('day')
            }
        except Exception as e:
            logger.error(f"Error extracting targets: {str(e)}")
            raise CustomException(e, sys)

    def transform(self, raw_data):
        """
        Main transformation method to get a complete feature set.
        """
        try:
            # 1. Flatten main data
            features = self.flatten_aqi_data(raw_data)
            
            # 2. Extract targets
            targets = self.extract_targets(raw_data)
            
            # Combine
            combined_data = {**features, **targets}
            
            # Create DataFrame
            df = pd.DataFrame([combined_data])
            
            # 3. Add ingestion timestamp (current time when pipeline runs)
            # This ensures each hourly run creates a unique row in the feature store
            df['ingestion_timestamp'] = pd.to_datetime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            
            # 4. Add time features
            df = self.extract_time_features(df)
            
            # 5. Cast numerical columns to correct types for Hopsworks (int64 for bigint, float for double)
            numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                              'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
            time_cols = ['hour', 'day', 'month', 'day_of_week']
            
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            for col in time_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
            
            # 6. Handle missing values (NaN) by filling them with 0
            # This is necessary because not all stations monitor all pollutants
            df.fillna(0, inplace=True)
            
            logger.info("Data transformation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test with dummy data or from ingestion
    from src.data_ingestion.data_ingestion import DataIngestion
    ingestion = DataIngestion("shanghai")
    raw_data = ingestion.get_city_data()
    
    transformer = DataTransformation()
    df = transformer.transform(raw_data)
    print(df.head())
