"""
backfill_openmeteo.py
─────────────────────
Fetches REAL historical air quality + weather data from Open-Meteo (free, no API key)
and inserts it into the Hopsworks Feature Store.

This solves the cold-start problem: instead of synthetic data, we use actual
historical measurements for Karachi so the model has real patterns to learn from.

Usage:
    python scripts/backfill_openmeteo.py
"""
import os
import sys
sys.path.append(os.getcwd())

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("Backfill OpenMeteo")

# ──────────────────────────────────────────────────────────────────────
# Karachi coordinates
# ──────────────────────────────────────────────────────────────────────
LATITUDE = 24.8607
LONGITUDE = 67.0011
CITY_NAME = "Karachi, Pakistan"


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly air quality data from Open-Meteo Air Quality API.
    Returns a DataFrame with columns: time, pm2_5, pm10, o3, no2, so2, co, us_aqi
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }

    logger.info(f"Fetching air quality data: {start_date} → {end_date}")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": hourly["time"],
        "pm25": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "o3": hourly["ozone"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "co": [v / 1000.0 if v is not None else None for v in hourly["carbon_monoxide"]],  # μg/m³ → mg/m³
        "aqi": hourly["us_aqi"],
    })

    logger.info(f"  Air quality: {len(df)} rows fetched")
    return df


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly weather data from Open-Meteo Weather API.
    Returns a DataFrame with columns: time, temperature, humidity, pressure, wind_speed
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }

    logger.info(f"Fetching weather data: {start_date} → {end_date}")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
    })

    logger.info(f"  Weather: {len(df)} rows fetched")
    return df


def build_feature_dataframe(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Combines air quality + weather data into the same schema used by
    the feature ingestion pipeline (matching feature_store_ingestion.py).
    """
    aq_df = fetch_air_quality(start_date, end_date)
    wx_df = fetch_weather(start_date, end_date)

    # Merge on time
    df = pd.merge(aq_df, wx_df, on="time", how="inner")

    # Build the full feature row format
    df["city"] = CITY_NAME
    df["timestamp"] = pd.to_datetime(df["time"])
    df["ingestion_timestamp"] = pd.to_datetime(df["time"])

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Target columns (will be computed properly during training, placeholder here)
    df["target_aqi"] = 0.0
    df["target_day"] = (df["timestamp"] + timedelta(days=3)).dt.strftime("%Y-%m-%d")

    # Drop the raw 'time' column
    df = df.drop(columns=["time"])

    # Cast types to match the existing feature group schema
    numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                      'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
    time_cols = ['hour', 'day', 'month', 'day_of_week']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    # Fill NaN with 0 (some hours may have missing pollutant readings)
    df.fillna(0, inplace=True)

    return df


def main():
    # Fetch last 60 days of real data
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  BACKFILL FEATURE STORE WITH REAL OPEN-METEO DATA")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  City: {CITY_NAME}")
    print("=" * 60)

    # Step 1: Build the complete feature dataframe
    print("\n[STEP 1] Fetching historical data from Open-Meteo...")
    df = build_feature_dataframe(start_date, end_date)
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df['ingestion_timestamp'].min()} → {df['ingestion_timestamp'].max()}")
    print(f"   AQI stats — mean: {df['aqi'].mean():.1f}, std: {df['aqi'].std():.1f}, "
          f"min: {df['aqi'].min():.0f}, max: {df['aqi'].max():.0f}")

    # Step 2: Save local cache
    print("\n[STEP 2] Saving local cache...")
    os.makedirs("data", exist_ok=True)
    local_path = os.path.join("data", "aqi_features.csv")
    df_to_save = df.copy()
    df_to_save['ingestion_timestamp'] = df_to_save['ingestion_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_to_save.to_csv(local_path, index=False)
    print(f"   Saved to {local_path}")

    # Step 3: Upload to Hopsworks in chunks (to avoid timeout)
    print("\n[STEP 3] Uploading to Hopsworks Feature Store...")
    fs_ingestion = FeatureStoreIngestion()

    chunk_size = 500
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunk_num = (i // chunk_size) + 1
        print(f"   Uploading chunk {chunk_num}/{total_chunks} ({len(chunk)} rows)...")
        fs_ingestion.save_to_feature_group(chunk, "aqi_features", version=5)

    print("\n" + "=" * 60)
    print(f"  [SUCCESS] Backfilled {len(df)} rows of REAL historical data!")
    print(f"  Next step: python src/model_training/model_trainer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
