import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style to dark
plt.style.use('dark_background')
BG_COLOR = '#0f1116'  # Dark slate background
TEXT_COLOR = '#f8fafc'    # Light text
BAR_COLOR = '#38bdf8'     # Sky blue

# Load the model
model_path = 'artifacts/model.joblib'
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
    exit(1)

model = joblib.load(model_path)

# Load feature names
feature_names_path = 'artifacts/feature_names.json'
if not os.path.exists(feature_names_path):
    print(f"Error: {feature_names_path} does not exist.")
    exit(1)

with open(feature_names_path, 'r') as f:
    feature_names = json.load(f)

# Rename features for better layout readability
rename_dict = {
    'pm25': 'PM2.5',
    'pm10': 'PM10',
    'o3': 'O3',
    'no2': 'NO2',
    'so2': 'SO2',
    'co': 'CO',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'pressure': 'Pressure',
    'wind_speed': 'Wind Speed',
    'hour': 'Hour of Day',
    'day': 'Day of Month',
    'month': 'Month',
    'day_of_week': 'Day of Week',
    'aqi_lag_24h': 'AQI Lag (24h)',
    'aqi_lag_48h': 'AQI Lag (48h)',
    'aqi_lag_72h': 'AQI Lag (72h)',
    'aqi_rolling_mean_24h': 'AQI Roll Mean (24h)',
    'aqi_rolling_std_24h': 'AQI Roll Std (24h)',
    'aqi_rolling_mean_72h': 'AQI Roll Mean (72h)',
    'aqi_diff_24h': 'AQI Change (24h)',
    'pm25_rolling_mean_24h': 'PM2.5 Roll Mean (24h)',
    'wind_speed_rolling_mean_24h': 'Wind Speed Roll Mean (24h)'
}
display_names = [rename_dict.get(name, name) for name in feature_names]

# Get feature importances from each estimator and average them
importances = []
for idx, estimator in enumerate(model.estimators_):
    if hasattr(estimator, 'feature_importances_'):
        importances.append(estimator.feature_importances_)
    else:
        print(f"Estimator {idx} does not have feature_importances_ attribute.")
        exit(1)

avg_importances = np.mean(importances, axis=0)

# Create a DataFrame for sorting and plotting
feat_imp_df = pd.DataFrame({
    'Feature': display_names,
    'Importance': avg_importances
}).sort_values(by='Importance', ascending=True) # Ascending for horizontal bar chart plot order

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Generate bars with edge coloring
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=BAR_COLOR, alpha=0.9, height=0.6, edgecolor='#0284c7', linewidth=1.2)

# Grid and Labels
ax.grid(axis='x', linestyle='--', alpha=0.3, color='#1e293b')
ax.tick_params(colors=TEXT_COLOR, labelsize=10)
ax.set_xlabel('Average Feature Importance (Gini Split Metric)', fontsize=11, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel('Model Features', fontsize=11, color=TEXT_COLOR, labelpad=10)
ax.set_title('XGBoost Multi-Output Feature Importance\n(Averaged across Day 1, 2, and 3 Forecasts)', fontsize=14, color=TEXT_COLOR, fontweight='bold', pad=20)

# Annotate bars with values
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2.0, f'{width:.3f}', 
            ha='left', va='center', color=TEXT_COLOR, fontsize=9, fontweight='semibold')

plt.tight_layout()

# Save image
os.makedirs('artifacts', exist_ok=True)
output_path = 'artifacts/feature_importance.png'
plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, edgecolor='none')
print(f"Feature importance chart successfully saved to {output_path}")
