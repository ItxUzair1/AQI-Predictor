import pandas as pd
import numpy as np
from datetime import datetime
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
        Extracts target values (future AQI) from forecast data.
        For this example, we'll take the average PM2.5 forecast for the next available day.
        """
        try:
            forecast = raw_data.get('data', {}).get('forecast', {}).get('daily', {})
            pm25_forecast = forecast.get('pm25', [])
            
            if len(pm25_forecast) > 1:
                # Next day's average PM2.5 as target
                target_aqi = pm25_forecast[1].get('avg')
                target_day = pm25_forecast[1].get('day')
                return {'target_aqi': target_aqi, 'target_day': target_day}
            
            return {'target_aqi': None, 'target_day': None}
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
            
            # 3. Add time features
            df = self.extract_time_features(df)
            
            # 4. Cast numerical columns to float to avoid schema mismatches (e.g. bigint vs double)
            numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                              'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            # 5. Handle missing values (NaN) by filling them with 0
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
