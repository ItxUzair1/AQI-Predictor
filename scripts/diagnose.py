"""Diagnostic: check what model the app loads and what the data looks like."""
import os, sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
load_dotenv()

p = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
)

# ── 1. Check model registry ──
print("=" * 60)
print("MODEL REGISTRY")
print("=" * 60)
mr = p.get_model_registry()
models = mr.get_models("aqi_predictor_karachi")
print(f"Total models: {len(models)}")
for m in sorted(models, key=lambda x: x.version):
    desc = (m.description or "")[:80]
    mae = m.training_metrics.get("mae", "N/A") if m.training_metrics else "N/A"
    print(f"  v{m.version} | MAE={mae} | {desc}")

# Which one would the app pick?
multi = [m for m in models if m.description and "multi-output" in m.description.lower()]
if multi:
    best = min(multi, key=lambda m: m.training_metrics.get("mae", float("inf")))
    print(f"\nApp would select: v{best.version} (MAE={best.training_metrics.get('mae','N/A')})")

# ── 2. Check feature store data ──
print("\n" + "=" * 60)
print("FEATURE STORE DATA (v6)")
print("=" * 60)
fs = p.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=6)
df = fg.select_all().read()

print(f"Total rows: {len(df)}")
print(f"Unique cities: {df['city'].unique()}")

df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("ingestion_timestamp")

print(f"\nIngestion timestamp range: {df['ingestion_timestamp'].min()} → {df['ingestion_timestamp'].max()}")
print(f"API timestamp range:      {df['timestamp'].min()} → {df['timestamp'].max()}")

# Check for duplicate AQI values (stale AQICN data)
print(f"\nAQI stats: mean={df['aqi'].mean():.1f}, std={df['aqi'].std():.1f}, min={df['aqi'].min():.0f}, max={df['aqi'].max():.0f}")

# Show the last 10 rows to see if AQICN pipeline rows are stale
print("\nLast 10 rows (sorted by ingestion_timestamp):")
cols = ["ingestion_timestamp", "timestamp", "aqi", "pm25", "temperature", "humidity", "city"]
cols = [c for c in cols if c in df.columns]
print(df[cols].tail(10).to_string(index=False))

# Check how many unique AQI values in the last 48 rows
last_48 = df.tail(48)
print(f"\nLast 48 rows: {last_48['aqi'].nunique()} unique AQI values out of {len(last_48)}")
print(f"Last 48 AQI value counts:\n{last_48['aqi'].value_counts().head(5)}")
