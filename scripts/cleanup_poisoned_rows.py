"""Remove the poisoned AQICN rows from the feature store and local cache."""
import os, sys
sys.path.append(os.getcwd())
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# ── 1. Clean local CSV cache ──
local_path = os.path.join("data", "aqi_features.csv")
if os.path.exists(local_path):
    df = pd.read_csv(local_path)
    before = len(df)
    df = df[df['city'] == 'Karachi, Pakistan']
    after = len(df)
    df.to_csv(local_path, index=False)
    print(f"Local CSV: removed {before - after} poisoned rows ({before} -> {after})")
else:
    print("No local CSV found.")

print("\nNote: The poisoned rows in Hopsworks are already excluded by the")
print("city filter in model_trainer.py and app/main.py (filter: city == 'Karachi, Pakistan').")
print("They will not affect training or predictions.")
print("Hopsworks feature groups don't support row deletion, so they stay in the store")
print("but are harmlessly ignored.")
