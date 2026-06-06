import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style to dark
plt.style.use('dark_background')

# Colors matching Streamlit glassmorphic dark theme (harmonious, modern colors)
# We use cool tech gradients or colors: a bright cyan/blue for metrics, maybe orange/accent
BG_COLOR = '#0f1116'  # Dark slate background
GRID_COLOR = '#1e293b' # Slate grid
BAR_COLOR_MAE = '#38bdf8' # Sky blue
BAR_COLOR_R2 = '#34d399'  # Emerald green
TEXT_COLOR = '#f8fafc'    # Light text

# Data
models = ['XGBoost\n(Best)', 'Gradient\nBoosting', 'Random\nForest', 'Linear\nRegression']
mae_values = [2.66, 3.88, 6.26, 10.35]
r2_values = [0.98, 0.96, 0.92, 0.37]

# Create figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(BG_COLOR)

# Subplot 1: MAE (Lower is better)
ax1.set_facecolor(BG_COLOR)
bars1 = ax1.bar(models, mae_values, color=BAR_COLOR_MAE, alpha=0.9, width=0.5, edgecolor='#0284c7', linewidth=1.5)
ax1.set_title('Mean Absolute Error (MAE)\n(Lower is Better)', fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=15)
ax1.set_ylabel('MAE Score', fontsize=11, color=TEXT_COLOR)
ax1.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_COLOR)
ax1.tick_params(colors=TEXT_COLOR, labelsize=10)

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.2f}', 
             ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=10)

# Subplot 2: R2 Score (Higher is better)
ax2.set_facecolor(BG_COLOR)
bars2 = ax2.bar(models, r2_values, color=BAR_COLOR_R2, alpha=0.9, width=0.5, edgecolor='#059669', linewidth=1.5)
ax2.set_title('R² Score (Model Fit)\n(Higher is Better)', fontsize=13, color=TEXT_COLOR, fontweight='bold', pad=15)
ax2.set_ylabel('R² Score', fontsize=11, color=TEXT_COLOR)
ax2.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_COLOR)
ax2.tick_params(colors=TEXT_COLOR, labelsize=10)
ax2.set_ylim(0, 1.15) # Leave room for labels

# Add values on top of bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', 
             ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=10)

# Main Title & Styling
plt.suptitle('Karachi AQI Prediction: Model Performance Comparison', fontsize=16, color=TEXT_COLOR, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Make sure output directory exists
os.makedirs('artifacts', exist_ok=True)
output_path = 'artifacts/model_metrics_comparison.png'
plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, edgecolor='none')
print(f"Chart successfully saved to {output_path}")
