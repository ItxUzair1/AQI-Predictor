import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("Feature Ingestion Pipeline")

# Karachi coordinates (same as backfill_openmeteo.py)
LATITUDE = 24.8607
LONGITUDE = 67.0011


class FeatureIngestionPipeline:
    def __init__(self, city_name="karachi"):
        self.city_name = city_name
        self.fs_ingestion = FeatureStoreIngestion()

    def _fetch_open_meteo_current(self):
        """
        Fetches the CURRENT hour's air quality + weather from Open-Meteo.
        Returns a single-row DataFrame matching the feature group schema.

        This replaces the AQICN API which returns stale/frozen data for Karachi.
        """
        import requests

        now_utc = datetime.now(timezone.utc)
        today = now_utc.strftime("%Y-%m-%d")

        # Fetch air quality for today
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "start_date": today,
            "end_date": today,
            "timezone": "UTC",
        }
        aq_resp = requests.get(aq_url, params=aq_params, timeout=30)
        aq_resp.raise_for_status()
        aq_hourly = aq_resp.json()["hourly"]

        # Fetch weather for today
        wx_url = "https://api.open-meteo.com/v1/forecast"
        wx_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "start_date": today,
            "end_date": today,
            "timezone": "UTC",
        }
        wx_resp = requests.get(wx_url, params=wx_params, timeout=30)
        wx_resp.raise_for_status()
        wx_hourly = wx_resp.json()["hourly"]

        # Build DataFrames and merge
        aq_df = pd.DataFrame({
            "time": aq_hourly["time"],
            "aqi": aq_hourly["us_aqi"],
            "pm25": aq_hourly["pm2_5"],
            "pm10": aq_hourly["pm10"],
            "o3": aq_hourly["ozone"],
            "no2": aq_hourly["nitrogen_dioxide"],
            "so2": aq_hourly["sulphur_dioxide"],
            "co": [v / 1000.0 if v is not None else None for v in aq_hourly["carbon_monoxide"]],
        })
        wx_df = pd.DataFrame({
            "time": wx_hourly["time"],
            "temperature": wx_hourly["temperature_2m"],
            "humidity": wx_hourly["relative_humidity_2m"],
            "pressure": wx_hourly["surface_pressure"],
            "wind_speed": wx_hourly["wind_speed_10m"],
        })
        merged = pd.merge(aq_df, wx_df, on="time", how="inner")

        # Pick the row closest to the current hour
        merged["time"] = pd.to_datetime(merged["time"])
        current_hour = now_utc.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        idx = (merged["time"] - current_hour).abs().idxmin()
        row = merged.loc[[idx]].copy()

        # Build feature row matching the schema
        row["city"] = "Karachi, Pakistan"
        row["timestamp"] = row["time"]
        row["ingestion_timestamp"] = pd.to_datetime(now_utc.strftime("%Y-%m-%d %H:%M:%S"))

        # Time features
        row["hour"] = row["timestamp"].dt.hour
        row["day"] = row["timestamp"].dt.day
        row["month"] = row["timestamp"].dt.month
        row["day_of_week"] = row["timestamp"].dt.dayofweek

        # Placeholder targets
        row["target_aqi"] = 0.0
        row["target_day"] = (now_utc + timedelta(days=3)).strftime("%Y-%m-%d")

        row = row.drop(columns=["time"])

        # Cast types
        numerical_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                          "temperature", "humidity", "pressure", "wind_speed", "target_aqi"]
        time_cols = ["hour", "day", "month", "day_of_week"]

        for col in numerical_cols:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col], errors="coerce").astype(float)
        for col in time_cols:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col], errors="coerce").astype("int64")

        row.fillna(0, inplace=True)

        logger.info(f"Fetched current data from Open-Meteo: AQI={row['aqi'].values[0]:.0f}")
        return row

    def run(self):
        try:
            logger.info(f"Starting feature ingestion for city: {self.city_name}")

            # Fetch current data from Open-Meteo (reliable, real-time)
            logger.info("Fetching current data from Open-Meteo...")
            df = self._fetch_open_meteo_current()

            # Store in feature group
            logger.info("Storing features in Hopsworks...")
            self.fs_ingestion.save_to_feature_group(df, "aqi_features", version=6)

            logger.info("Feature ingestion completed successfully")
            return df

        except Exception as e:
            logger.error(f"Feature ingestion failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = FeatureIngestionPipeline("karachi")
    pipeline.run()

