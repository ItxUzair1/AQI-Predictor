import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

# Add the project root directory to the Python path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_openmeteo_historical(city="Karachi", lat=24.8607, lon=67.0011, days=30):
    """
    Fetches real historical air quality and weather data from Open-Meteo.
    """
    logger.info(f"Fetching {days} days of historical data for {city} from Open-Meteo...")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # 1. Fetch Air Quality
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ["european_aqi", "pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"],
        "timezone": "auto"
    }
    
    aq_resp = requests.get(aq_url, params=aq_params)
    if aq_resp.status_code != 200:
        raise Exception(f"Failed to fetch Air Quality data: {aq_resp.text}")
    
    aq_data = aq_resp.json()["hourly"]
    df_aq = pd.DataFrame({
        "timestamp": aq_data["time"],
        "aqi": aq_data["european_aqi"],
        "pm10": aq_data["pm10"],
        "pm25": aq_data["pm2_5"],
        "o3": aq_data["ozone"],
        "no2": aq_data["nitrogen_dioxide"],
        "so2": aq_data["sulphur_dioxide"],
        "co": aq_data["carbon_monoxide"]
    })
    
    # 2. Fetch Weather
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"],
        "timezone": "auto"
    }
    
    w_resp = requests.get(weather_url, params=weather_params)
    if w_resp.status_code != 200:
        raise Exception(f"Failed to fetch Weather data: {w_resp.text}")
        
    w_data = w_resp.json()["hourly"]
    df_w = pd.DataFrame({
        "timestamp": w_data["time"],
        "temperature": w_data["temperature_2m"],
        "humidity": w_data["relative_humidity_2m"],
        "pressure": w_data["surface_pressure"],
        "wind_speed": w_data["wind_speed_10m"]
    })
    
    # 3. Merge on timestamp
    df = pd.merge(df_aq, df_w, on="timestamp")
    
    # 4. Add static/derived columns required by Hopsworks pipeline
    df["city"] = city
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set the ingestion timestamp (unique identifier for this batch)
    current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    df['ingestion_timestamp'] = pd.to_datetime(current_time_str)
    
    # Add time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Add target reference columns (not available historically, filled with 0)
    df['target_aqi'] = 0.0
    df['target_day'] = ""
    
    # Ensure correct data types
    numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                      'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
    time_cols = ['hour', 'day', 'month', 'day_of_week']
    
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
        
    # Drop rows where future hours don't have AQI yet
    df = df.dropna(subset=['aqi'])
    
    # Fill remaining NaNs (e.g. missing pollutant data) with 0 like WAQI pipeline does
    df.fillna(0, inplace=True)
    
    return df


def run_backfill(days=30, wait_for_job=False):
    try:
        # 1. Fetch Real Historical Data
        df = fetch_openmeteo_historical("Karachi", days=days)
        logger.info(f"Generated {len(df)} rows of authentic historical data "
                    f"({df['timestamp'].min()} to {df['timestamp'].max()})")

        # Quick sanity check
        aqi_std = df['aqi'].std()
        aqi_mean = df['aqi'].mean()
        logger.info(f"AQI stats — mean: {aqi_mean:.1f}, std: {aqi_std:.1f}, "
                    f"min: {df['aqi'].min():.0f}, max: {df['aqi'].max():.0f}")

        # 2. Connect to Hopsworks directly
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
        host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

        if not api_key:
            print("ERROR: HOPSWORKS_API_KEY not set in .env")
            sys.exit(1)

        if sys.platform == "win32":
            local_tmp = os.path.join(os.getcwd(), "tmp")
            os.makedirs(local_tmp, exist_ok=True)
            os.environ["TMP"] = local_tmp
            os.environ["TEMP"] = local_tmp

        project = hopsworks.login(host=host, project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()

        # 3. Create a fresh feature group v4 (DELTA format, no Kafka)
        logger.info("Checking for existing feature group v4...")
        try:
            old_fg = fs.get_feature_group("aqi_features", version=4)
            logger.info("Found existing FG v4. Deleting to recreate...")
            old_fg.delete()
            logger.info("Deleted existing FG v4 successfully.")
        except Exception as e:
            logger.info(f"No existing v4 to delete (OK): {e}")

        logger.info("Creating fresh feature group v4 (HUDI format, stream=False)...")
        aqi_fg = fs.create_feature_group(
            name="aqi_features",
            version=4,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            stream=False,
            time_travel_format="HUDI",
            description="AQI and weather features for city"
        )

        # 4. Insert all data at once via REST API
        total_rows = len(df)
        logger.info(f"Inserting {total_rows} rows into feature group v4 (via REST API)...")
        aqi_fg.insert(df, write_options={"wait_for_job": wait_for_job})
        logger.info("Data insert completed!")

        # 5. Update local cache
        local_path = os.path.join("data", "aqi_features.csv")
        os.makedirs("data", exist_ok=True)
        df_to_cache = df.copy()
        if 'ingestion_timestamp' in df_to_cache.columns:
            df_to_cache['ingestion_timestamp'] = pd.to_datetime(
                df_to_cache['ingestion_timestamp']
            ).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_cache.to_csv(local_path, index=False)
        logger.info(f"Local cache updated with backfill data: {local_path}")

        logger.info("Backfill completed successfully!")
        print(f"\nSUCCESS: Backfilled {total_rows} rows for Karachi into 'aqi_features' v4.")
        print(f"  From : {df['timestamp'].min()}")
        print(f"  To   : {df['timestamp'].max()}")
        print(f"  AQI  : mean={aqi_mean:.1f}, std={aqi_std:.1f}")
        if not wait_for_job:
            print("  Note: Spark materialization is running in the background on Hopsworks.")
            print("        You can monitor progress in the Hopsworks UI.")

    except Exception as e:
        logger.error(f"Backfill failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nERROR: Backfill failed. {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill authentic AQI data into Hopsworks v4 feature group")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of past days to backfill (default: 30)")
    parser.add_argument("--wait", action="store_true", default=False,
                        help="Wait for Spark materialization job to complete (default: False)")
    args = parser.parse_args()
    run_backfill(days=args.days, wait_for_job=args.wait)
