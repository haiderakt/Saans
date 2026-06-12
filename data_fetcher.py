import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import requests

def fetch_data(latitude=31.5204, longitude=74.3587, start_date=None, end_date=None, output_path="data/raw_data.csv"):
    """
    Fetches historical weather (ERA5) and air quality data for the specified coordinates
    and time range from Open-Meteo APIs, merges them, and saves to a CSV.
    """
    # 1. Determine date range (default to last 3 years up to yesterday)
    today = datetime.now()
    if not end_date:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    if not start_date:
        start_date = (today - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        
    print(f"Fetching data for Lahore (Lat: {latitude}, Lon: {longitude})")
    print(f"Period: {start_date} to {end_date}")

    # 2. Fetch Historical Weather Data (ERA5)
    # Variables: temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,boundary_layer_height",
        "timezone": "Asia/Karachi"
    }
    
    print("Requesting weather data from Open-Meteo Archive API...")
    weather_response = requests.get(weather_url, params=weather_params)
    if weather_response.status_code != 200:
        raise Exception(f"Failed to fetch weather data: {weather_response.text}")
    
    weather_data = weather_response.json()
    hourly_weather = weather_data.get("hourly", {})
    df_weather = pd.DataFrame(hourly_weather)
    if df_weather.empty:
        raise Exception("No weather data returned from the API.")
    
    # Rename weather columns for clarity
    df_weather = df_weather.rename(columns={
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
        "boundary_layer_height": "boundary_layer_height"
    })
    print(f"Retrieved {len(df_weather)} weather records.")

    # 3. Fetch Historical Air Quality Data
    # Variables: pm2_5, pm10, nitrogen_dioxide, ozone, carbon_monoxide
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": "Asia/Karachi"
    }
    
    print("Requesting air quality data from Open-Meteo Air Quality API...")
    aq_response = requests.get(aq_url, params=aq_params)
    if aq_response.status_code != 200:
        raise Exception(f"Failed to fetch air quality data: {aq_response.text}")
        
    aq_data = aq_response.json()
    hourly_aq = aq_data.get("hourly", {})
    df_aq = pd.DataFrame(hourly_aq)
    if df_aq.empty:
        raise Exception("No air quality data returned from the API.")
        
    df_aq = df_aq.rename(columns={
        "time": "timestamp",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "carbon_monoxide": "co"
    })
    print(f"Retrieved {len(df_aq)} air quality records.")

    # 4. Merge datasets on timestamp
    df_merged = pd.merge(df_weather, df_aq, on="timestamp", how="outer")
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save raw data
    df_merged.to_csv(output_path, index=False)
    print(f"Successfully merged data and saved to {output_path}")
    print(f"Total merged records: {len(df_merged)}")
    print(f"Columns: {list(df_merged.columns)}")
    return df_merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical weather and AQI data from Open-Meteo.")
    parser.pragma = "Saans Data Fetcher"
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/raw_data.csv", help="Path to save raw CSV")
    
    args = parser.parse_args()
    fetch_data(start_date=args.start_date, end_date=args.end_date, output_path=args.output)
