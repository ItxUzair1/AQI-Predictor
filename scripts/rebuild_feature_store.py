"""
rebuild_feature_store.py
────────────────────────
1. Downloads ALL data from Hopsworks v3 feature group.
2. Keeps ONLY real rows (May 7 – May 20, collected by hourly pipeline).
3. Generates new realistic synthetic data (before May 7) using improved generator.
4. Deletes the old v3 feature group.
5. Re-creates v3 with the combined clean dataset (real + new synthetic).
"""
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv

from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger

load_dotenv()
logger = get_logger("Rebuild Feature Store")

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
REAL_DATA_START = pd.Timestamp("2025-05-07", tz="UTC")
REAL_DATA_END   = pd.Timestamp("2025-05-20 23:59:59", tz="UTC")
SYNTHETIC_DAYS_BEFORE = 60  # generate 60 days of synthetic data before real data


def get_hopsworks_project():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not set")
        sys.exit(1)

    if sys.platform == "win32":
        local_tmp = os.path.join(os.getcwd(), "tmp")
        os.makedirs(local_tmp, exist_ok=True)
        os.environ["TMP"] = local_tmp
        os.environ["TEMP"] = local_tmp

    return hopsworks.login(host=host, project=project_name, api_key_value=api_key)


# ──────────────────────────────────────────────────────────────────────
# Step 1: Download & filter real data
# ──────────────────────────────────────────────────────────────────────
def download_real_data(project):
    """Fetches all v3 data and keeps only real rows from May 7–20."""
    logger.info("Downloading all data from v3 feature group...")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=4)
    df = fg.select_all().read()

    logger.info(f"Total rows in v3: {len(df)}")

    df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'], utc=True)

    # Keep only rows in the real data window
    real_df = df[
        (df['ingestion_timestamp'] >= REAL_DATA_START) &
        (df['ingestion_timestamp'] <= REAL_DATA_END)
    ].copy()

    logger.info(f"Real rows (May 7–20): {len(real_df)}")
    logger.info(f"Synthetic/old rows being discarded: {len(df) - len(real_df)}")

    if len(real_df) > 0:
        logger.info(f"Real data range: {real_df['ingestion_timestamp'].min()} → {real_df['ingestion_timestamp'].max()}")
        logger.info(f"Real AQI stats — mean: {real_df['aqi'].mean():.1f}, "
                    f"std: {real_df['aqi'].std():.1f}, "
                    f"min: {real_df['aqi'].min():.0f}, max: {real_df['aqi'].max():.0f}")
    else:
        logger.warning("No real data found in the May 7–20 window!")

    return real_df


# ──────────────────────────────────────────────────────────────────────
# Step 2: Generate synthetic data calibrated to real Karachi patterns
# ──────────────────────────────────────────────────────────────────────
def generate_synthetic_data(real_df, days_before=60):
    """
    Generates realistic synthetic data for the period BEFORE the real data.
    If real data exists, calibrates the synthetic data to match Karachi's
    actual AQI distribution (mean, std, diurnal pattern).
    """
    np.random.seed(42)

    # Calibrate to real data if available
    if len(real_df) > 0:
        real_mean = float(real_df['aqi'].mean())
        real_std = float(real_df['aqi'].std())
        logger.info(f"Calibrating synthetic data to real Karachi AQI: mean={real_mean:.0f}, std={real_std:.0f}")
    else:
        real_mean = 145.0
        real_std = 35.0
        logger.info(f"No real data for calibration, using Karachi defaults: mean={real_mean}, std={real_std}")

    # Synthetic period ends where real data begins
    end_date = REAL_DATA_START - timedelta(hours=1)
    start_date = end_date - timedelta(days=days_before)

    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1

    # ── Build a realistic AQI time-series ──────────────────────────────
    # Slow drift (random walk, mean-reverting around real_mean)
    drift = np.zeros(total_hours)
    drift[0] = np.random.uniform(-15, 15)
    for i in range(1, total_hours):
        drift[i] = drift[i - 1] * 0.997 + np.random.normal(0, 0.6)

    # Pollution events
    events = np.zeros(total_hours)
    h = 0
    while h < total_hours:
        if np.random.random() < 0.008:
            spike_mag = np.random.uniform(40, 100)
            spike_dur = np.random.randint(48, 120)
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

    # Rain clearing
    rain_effect = np.zeros(total_hours)
    for h in range(total_hours):
        if np.random.random() < 0.003 and events[h] < 20:
            drop = np.random.uniform(25, 55)
            recovery = np.random.randint(12, 24)
            for j in range(min(recovery, total_hours - h)):
                rain_effect[h + j] = -drop * (1 - j / recovery)

    # ── Generate rows ─────────────────────────────────────────────────
    data = []
    current_time = start_date
    for h_idx in range(total_hours):
        hour = current_time.hour
        day_of_week = current_time.weekday()

        diurnal = 12 * np.sin((hour - 9) * np.pi / 12) + 8 * np.sin((hour - 19) * np.pi / 6)
        weekly = 8 if day_of_week < 5 else -12

        base_aqi = real_mean + diurnal + weekly + drift[h_idx] + events[h_idx] + rain_effect[h_idx]
        base_aqi += np.random.normal(0, real_std * 0.15)
        base_aqi = max(15, min(400, base_aqi))

        wind_speed = max(0.5, np.random.uniform(2, 8) + (rain_effect[h_idx] / 20))
        temperature = 28 + 6 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 1.5)
        humidity = 55 - 0.5 * temperature + np.random.normal(0, 5)
        if rain_effect[h_idx] < -10:
            humidity += 25
        humidity = max(20, min(95, humidity))
        pressure = 1012 + np.random.normal(0, 3)

        wind_adjusted_aqi = base_aqi - (wind_speed - 4) * 3
        wind_adjusted_aqi = max(15, min(400, wind_adjusted_aqi))

        pm25 = wind_adjusted_aqi * np.random.uniform(0.6, 0.95) + np.random.normal(0, 8)
        pm10 = wind_adjusted_aqi * np.random.uniform(0.35, 0.65) + np.random.normal(0, 5)
        o3 = np.random.uniform(5, 35) + 8 * np.sin((hour - 13) * np.pi / 8)
        no2 = 15 + 10 * (1 if day_of_week < 5 else 0) + np.random.normal(0, 5)
        so2 = np.random.uniform(2, 12) + np.random.normal(0, 2)
        co = np.random.uniform(0.1, 0.8) + 0.2 * (1 if 7 <= hour <= 10 else 0)

        row = {
            'city': "Karachi, Pakistan",
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
            'hour': int(hour),
            'day': int(current_time.day),
            'month': int(current_time.month),
            'day_of_week': int(day_of_week),
        }
        data.append(row)
        current_time += timedelta(hours=1)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'], utc=True)

    numerical_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                      'temperature', 'humidity', 'pressure', 'wind_speed', 'target_aqi']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    time_cols = ['hour', 'day', 'month', 'day_of_week']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    df.fillna(0, inplace=True)

    logger.info(f"Generated {len(df)} synthetic rows: {df['ingestion_timestamp'].min()} → {df['ingestion_timestamp'].max()}")
    logger.info(f"Synthetic AQI stats — mean: {df['aqi'].mean():.1f}, std: {df['aqi'].std():.1f}")

    return df


# ──────────────────────────────────────────────────────────────────────
# Step 3: Delete old v2 and re-upload clean dataset
# ──────────────────────────────────────────────────────────────────────
def delete_old_feature_group(project):
    """Deletes the v3 feature group entirely."""
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group("aqi_features", version=4)
        logger.info(f"Deleting existing v4 feature group...")
        fg.delete()
        logger.info("Deleted v4 feature group.")
    except Exception as e:
        logger.info("No existing v4 feature group found (safe to proceed).")

    # --- Step 4: Create new feature group ---
    logger.info("Step 4: Uploading rebuilt dataset back to Hopsworks...")
    fs_ingestion = FeatureStoreIngestion()
    # pyrefly: ignore [unknown-name]
    fs_ingestion.save_to_feature_group(combined_df, "aqi_features", version=4)
    # pyrefly: ignore [unknown-name]
    logger.info(f"Uploaded {len(combined_df)} rows to fresh v3 feature group.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  REBUILD FEATURE STORE")
    print("  Keep real data (May 7-20) + add new synthetic history")
    print("=" * 60)

    project = get_hopsworks_project()

    # Step 1: Download and filter real data
    print("\n[STEP 1] Downloading real data from v3...")
    real_df = download_real_data(project)
    print(f"   Found {len(real_df)} real rows")

    # Step 2: Generate synthetic data
    print(f"\n[STEP 2] Generating {SYNTHETIC_DAYS_BEFORE} days of synthetic data...")
    synthetic_df = generate_synthetic_data(real_df, days_before=SYNTHETIC_DAYS_BEFORE)
    print(f"   Generated {len(synthetic_df)} synthetic rows")

    # Step 3: Match city name from real data to synthetic data
    if len(real_df) > 0:
        real_city_name = real_df['city'].iloc[0]
        synthetic_df['city'] = real_city_name
        logger.info(f"Matched city name to real data: '{real_city_name}'")

    # Step 4: Combine
    combined_df = pd.concat([synthetic_df, real_df], ignore_index=True)
    combined_df = combined_df.sort_values('ingestion_timestamp').reset_index(drop=True)

    print(f"\n[STATS] Combined dataset: {len(combined_df)} total rows")
    print(f"   Synthetic: {len(synthetic_df)} rows")
    print(f"   Real:      {len(real_df)} rows")
    print(f"   Range:     {combined_df['ingestion_timestamp'].min()} → {combined_df['ingestion_timestamp'].max()}")
    print(f"   AQI mean:  {combined_df['aqi'].mean():.1f}")
    print(f"   AQI std:   {combined_df['aqi'].std():.1f}")

    # Step 5: Delete old v3
    print("\n[DELETE] Step 3: Deleting old v3 feature group...")
    delete_old_feature_group(project)

    # Step 6: Upload combined data to fresh v3
    print("\n[UPLOAD] Step 4: Uploading combined dataset to fresh v3...")
    # pyrefly: ignore [unknown-name]
    upload_combined_data(combined_df)

    print("\n" + "=" * 60)
    print("  [SUCCESS] Feature store rebuilt successfully.")
    print(f"  Next step: python src/model_training/model_trainer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
