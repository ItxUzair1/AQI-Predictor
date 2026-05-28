import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_aqi(city="Karachi", lat=24.8607, lon=67.0011, days=30):
    print(f"Fetching {days} days of historical air quality data for {city} from Open-Meteo...")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Open-Meteo Air Quality API
    # We fetch European AQI, PM10, and PM2.5
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["european_aqi", "pm10", "pm2_5"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data: {response.text}")
        return None
        
    data = response.json()
    
    # Convert to Pandas DataFrame
    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "aqi": hourly["european_aqi"],
        "pm10": hourly["pm10"],
        "pm25": hourly["pm2_5"]
    })
    
    # Drop rows where AQI is null (can happen for future hours in the current day)
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    df = fetch_historical_aqi()
    
    if df is not None:
        print("\n=== Data Preview (First 5 Rows) ===")
        print(df.head().to_string(index=False))
        
        print("\n=== Data Preview (Last 5 Rows) ===")
        print(df.tail().to_string(index=False))
        
        print(f"\nTotal Real Rows Fetched: {len(df)}")
        print("\n=== AQI Statistics ===")
        print(df['aqi'].describe().to_string())
