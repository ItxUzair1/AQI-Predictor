import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style to match the dark slate theme
plt.style.use('dark_background')
BG_COLOR = '#0f1116'  # Dark slate background
TEXT_COLOR = '#f8fafc'    # Light text

# Load features from CSV
csv_path = 'data/aqi_features.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} does not exist.")
    exit(1)

df = pd.read_csv(csv_path)

# Filter data to match notebook
if 'city' in df.columns:
    df = df[df['city'] == 'Karachi, Pakistan']

# Filter out zero-filled weather rows
df = df[df['temperature'] > 0.0]

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number])
cols_to_drop = ['target_aqi', 'hour', 'day', 'month', 'day_of_week']
numerical_cols = numerical_cols.drop(columns=cols_to_drop, errors='ignore')

# Rename columns for cleaner display
rename_dict = {
    'pm25': 'PM2.5',
    'pm10': 'PM10',
    'o3': 'O3',
    'no2': 'NO2',
    'so2': 'SO2',
    'co': 'CO',
    'aqi': 'AQI (Base)',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'pressure': 'Pressure',
    'wind_speed': 'Wind Speed'
}
numerical_cols = numerical_cols.rename(columns=rename_dict)

# Compute correlation matrix
corr_matrix = numerical_cols.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Generate a custom diverging colormap (coolwarm style, matching slate aesthetics)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Plot heatmap with clean annotations and lines
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=1,
    linecolor=BG_COLOR,
    annot_kws={"size": 10, "weight": "bold"},
    cbar_kws={"shrink": 0.8}
)

# Customize title and labels
plt.title('Correlation Heatmap (Pollutants & Meteorological Features)', fontsize=15, color=TEXT_COLOR, fontweight='bold', pad=20)
plt.xticks(fontsize=11, color=TEXT_COLOR, rotation=45, ha='right')
plt.yticks(fontsize=11, color=TEXT_COLOR, rotation=0)

# Adjust layout
plt.tight_layout()

# Ensure target directories exist
os.makedirs('artifacts', exist_ok=True)
output_path = 'artifacts/correlation_heatmap.png'
plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, edgecolor='none')
print(f"Heatmap successfully saved to {output_path}")
