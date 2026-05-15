import os
import sys
import json
from datetime import datetime, timedelta

# sys.path MUST be set before any src.* imports
sys.path.append(os.getcwd())

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor Pro",
    page_icon="🌬️",
    layout="wide"
)

# ──────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b27 100%); }

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

.prediction-card {
    background: rgba(255, 255, 255, 0.04);
    padding: 40px 30px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    text-align: center;
    margin-top: 20px;
    backdrop-filter: blur(10px);
}
.aqi-value {
    font-size: 96px;
    font-weight: 900;
    line-height: 1;
    margin: 16px 0 8px;
    letter-spacing: -4px;
}
.aqi-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.aqi-sub { color: #8b949e; font-size: 0.9rem; margin-top: 12px; }

.aqi-good            { color: #00e676; }
.aqi-moderate        { color: #ffee58; }
.aqi-sensitive       { color: #ffa726; }
.aqi-unhealthy       { color: #ef5350; }
.aqi-very-unhealthy  { color: #ab47bc; }
.aqi-hazardous       { color: #b71c1c; }

.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 10px;
}
.pill-good            { background: #00e67622; color: #00e676; }
.pill-moderate        { background: #ffee5822; color: #ffee58; }
.pill-sensitive       { background: #ffa72622; color: #ffa726; }
.pill-unhealthy       { background: #ef535022; color: #ef5350; }
.pill-very-unhealthy  { background: #ab47bc22; color: #ab47bc; }
.pill-hazardous       { background: #b71c1c22; color: #ff6060; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: AQI category
# ──────────────────────────────────────────────
def get_aqi_info(aqi: int):
    if aqi <= 50:
        return "Good", "aqi-good", "pill-good", "🟢", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "aqi-moderate", "pill-moderate", "🟡", "Acceptable for most people."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "aqi-sensitive", "pill-sensitive", "🟠", "Sensitive groups should limit outdoor exposure."
    elif aqi <= 200:
        return "Unhealthy", "aqi-unhealthy", "pill-unhealthy", "🔴", "Everyone may experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "aqi-very-unhealthy", "pill-very-unhealthy", "🟣", "Health alert – avoid prolonged outdoor activity."
    else:
        return "Hazardous", "aqi-hazardous", "pill-hazardous", "⚫", "Emergency conditions – stay indoors."


# ──────────────────────────────────────────────
# Hopsworks helpers
# ──────────────────────────────────────────────
def _hopsworks_login():
    """Connects to Hopsworks using env variables. Returns project handle."""
    api_key      = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_ML_System")
    host         = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not set in environment / .env file")

    # Windows tmp workaround
    if sys.platform == "win32":
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        os.environ["TMP"] = tmp_dir
        os.environ["TEMP"] = tmp_dir

    return hopsworks.login(host=host, project=project_name, api_key_value=api_key)


@st.cache_resource(show_spinner="Loading model from registry…")
def load_model(city_name: str):
    """
    Downloads the BEST (latest) registered model from Hopsworks Model Registry.
    Returns (model, feature_names_list).
    """
    project = _hopsworks_login()
    mr = project.get_model_registry()

    model_name = f"aqi_predictor_{city_name}"
    # get_best_model picks the version with lowest 'mae' metric
    try:
        model_meta = mr.get_best_model(model_name, metric="mae", direction="min")
    except Exception:
        # Fallback: get the latest version if no metric is set
        model_meta = mr.get_model(model_name)

    model_dir = model_meta.download()

    # Load the XGBoost model
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    # Load saved feature names (saved during training)
    feature_names_file = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(feature_names_file):
        with open(feature_names_file) as f:
            feature_names = json.load(f)
    else:
        feature_names = None  # will handle gracefully later

    return model, feature_names


@st.cache_data(show_spinner="Fetching latest data from feature store…", ttl=3600)
def load_feature_data(city_name: str):
    """
    Fetches the most recent rows from the Hopsworks feature store.
    Returns a DataFrame sorted newest-first.
    """
    project = _hopsworks_login()
    fs = project.get_feature_store()

    aqi_fg = fs.get_feature_group(name="aqi_features", version=2)
    df = aqi_fg.select_all().read()

    if 'city' in df.columns:
        df = df[df['city'] == city_name]

    df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
    df = df.sort_values(by="ingestion_timestamp", ascending=False).reset_index(drop=True)
    return df


def build_feature_row(latest_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """
    Drops non-feature columns from the latest row and aligns to the trained feature list.
    This prevents column-order mismatch bugs with XGBoost.
    """
    non_feature_cols = {'city', 'timestamp', 'target_day', 'target_aqi', 'ingestion_timestamp'}
    row_dict = {k: v for k, v in latest_row.items() if k not in non_feature_cols}

    if feature_names:
        # Align to training column order, fill any missing columns with 0
        aligned = {col: row_dict.get(col, 0) for col in feature_names}
        return pd.DataFrame([aligned], columns=feature_names)
    else:
        return pd.DataFrame([row_dict])


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
city = st.sidebar.selectbox(
    "Select City",
    ["karachi", "lahore", "islamabad"],
    index=0,
    format_func=str.capitalize
)

if st.sidebar.button("🔄 Refresh Data"):
    load_feature_data.clear()
    load_model.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline schedule**\n"
    "- 📥 Feature ingestion: **every hour**\n"
    "- 🧠 Model training: **every day at midnight**"
)

# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
st.title("🌬️ AQI Predictor Pro")
st.markdown(f"### Real-time Air Quality Intelligence · **{city.capitalize()}**")
st.markdown("---")

# Load data
col_status1, col_status2 = st.columns(2)

try:
    with st.spinner("Connecting to Hopsworks…"):
        model, feature_names = load_model(city)
        df = load_feature_data(city)

    if df.empty:
        st.warning("No data found for this city in the feature store. Run the ingestion pipeline first.")
        st.stop()

    latest = df.iloc[0]

    # ── Current Conditions ──────────────────────────────────────────────
    st.subheader("📍 Current Conditions")
    c1, c2, c3, c4, c5 = st.columns(5)

    def safe_metric(col, label, value, fmt="{:.0f}", suffix=""):
        try:
            col.metric(label, f"{fmt.format(float(value))}{suffix}")
        except Exception:
            col.metric(label, "N/A")

    safe_metric(c1, "🌡 AQI Now",       latest.get('aqi', 0),         "{:.0f}")
    safe_metric(c2, "💨 PM2.5",         latest.get('pm25', 0),        "{:.1f}")
    safe_metric(c3, "🌡️ Temperature",   latest.get('temperature', 0), "{:.1f}", "°C")
    safe_metric(c4, "💧 Humidity",      latest.get('humidity', 0),    "{:.0f}", "%")
    safe_metric(c5, "🌬 Wind Speed",    latest.get('wind_speed', 0),  "{:.1f}", " m/s")

    st.markdown("---")

    # ── 3-Day Prediction ────────────────────────────────────────────────
    st.subheader("🔮 3-Day AQI Forecast")

    feature_row = build_feature_row(latest, feature_names)
    raw_pred = model.predict(feature_row)[0]
    pred_aqi = int(max(0, round(raw_pred)))

    label, css_class, pill_class, icon, advice = get_aqi_info(pred_aqi)
    target_date = (datetime.now() + timedelta(days=3)).strftime("%A, %B %d %Y")

    st.markdown(f"""
    <div class="prediction-card">
        <p style="color:#8b949e; font-size:0.9rem; margin-bottom:4px;">FORECAST DATE</p>
        <p style="font-size:1.1rem; font-weight:600; margin:0">{target_date}</p>
        <div class="aqi-value {css_class}">{pred_aqi}</div>
        <div class="aqi-label {css_class}">{icon} {label}</div>
        <span class="pill {pill_class}">{label}</span>
        <p class="aqi-sub">{advice}</p>
        <p class="aqi-sub" style="margin-top:16px">
            Predicted by XGBoost trained on historical AQI patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Historical Trend ────────────────────────────────────────────────
    st.subheader("📈 Recent AQI Trend (Last 48 Hours)")

    trend_df = df.head(48).copy().sort_values('ingestion_timestamp')
    trend_df = trend_df.set_index('ingestion_timestamp')[['aqi', 'pm25']].dropna()

    st.line_chart(trend_df, use_container_width=True)

    # ── Pollutant Breakdown ─────────────────────────────────────────────
    st.subheader("🔬 Pollutant Snapshot")
    pollutants = {
        "PM2.5": latest.get('pm25', 0),
        "PM10":  latest.get('pm10', 0),
        "O₃":   latest.get('o3', 0),
        "NO₂":  latest.get('no2', 0),
        "SO₂":  latest.get('so2', 0),
        "CO":   latest.get('co', 0),
    }
    poll_df = pd.DataFrame.from_dict(
        pollutants, orient='index', columns=['Value']
    ).sort_values('Value', ascending=False)
    st.bar_chart(poll_df, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Could not load data: {e}")
    st.info("Make sure your `.env` file has valid `HOPSWORKS_API_KEY` and the pipeline has been run at least once.")

st.markdown("---")
st.caption("Powered by XGBoost · Hopsworks Feature Store & Model Registry · AQICN API")
