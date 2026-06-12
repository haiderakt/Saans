import os
import json
import torch
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from preprocessing import DataPreprocessor, pm25_to_aqi, get_aqi_category
from model import TransformerTimeSeriesModel

def get_live_forecast_inputs(latitude=31.5204, longitude=74.3587):
    """
    Fetches the last 5 days of historical weather and air quality data from Open-Meteo
    to form the 72-hour lookback window needed for inference (accounting for lag/rolling features).
    """
    print(f"Fetching live inputs for Lahore (Lat: {latitude}, Lon: {longitude})...")
    
    # 1. Fetch weather forecast + past 5 days
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,boundary_layer_height",
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Karachi"
    }
    
    weather_response = requests.get(weather_url, params=weather_params)
    if weather_response.status_code != 200:
        raise Exception(f"Failed to fetch live weather forecast: {weather_response.text}")
        
    weather_data = weather_response.json().get("hourly", {})
    df_weather = pd.DataFrame(weather_data).rename(columns={
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
        "boundary_layer_height": "boundary_layer_height"
    })
    
    # 2. Fetch AQ forecast + past 5 days
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Karachi"
    }
    
    aq_response = requests.get(aq_url, params=aq_params)
    if aq_response.status_code != 200:
        raise Exception(f"Failed to fetch live air quality forecast: {aq_response.text}")
        
    aq_data = aq_response.json().get("hourly", {})
    df_aq = pd.DataFrame(aq_data).rename(columns={
        "time": "timestamp",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "carbon_monoxide": "co"
    })
    
    # 3. Merge weather and air quality
    df_merged = pd.merge(df_weather, df_aq, on="timestamp", how="outer")
    df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    
    return df_merged

def run_inference(config_path="models/model_config.json", weights_path="models/best_model.pth"):
    # 1. Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}. Train the model first.")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    lookback = config["lookback"]
    horizon = config["forecast_horizon"]
    num_layers = config["num_layers"]
    feature_cols = config["feature_cols"]
    target_col = config["target_col"]
    
    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTimeSeriesModel(
        input_dim=len(feature_cols),
        d_model=64,
        nhead=4,
        num_layers=num_layers,
        forecast_horizon=horizon,
        seq_len=lookback
    ).to(device)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Load preprocessor and scalers
    preprocessor = DataPreprocessor(feature_cols=feature_cols, target_col=target_col)
    preprocessor.load_scalers()
    
    # 4. Fetch live data
    df_live = get_live_forecast_inputs()
    
    # Add cyclical time features (engineers all 46 features)
    df_live = preprocessor.add_cyclical_time_features(df_live)
    
    # Handle missing values if any
    cols_to_fill = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure", "boundary_layer_height", "pm2_5", "pm10", "no2", "o3", "co"]
    for col in cols_to_fill:
        if col in df_live.columns:
            df_live[col] = df_live[col].interpolate(method="linear").bfill().ffill()
            
    # We need the last `lookback` hours as input.
    current_time = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
    
    # Find records that are in the past or current hour
    df_past = df_live[df_live["timestamp"] <= current_time].copy()
    
    # Take the last `lookback` hours of history
    if len(df_past) < lookback:
        df_input = df_live.head(lookback).copy()
    else:
        df_input = df_past.tail(lookback).copy()
        
    print(f"\nLookback window input starts: {df_input['timestamp'].iloc[0]}")
    print(f"Lookback window input ends:   {df_input['timestamp'].iloc[-1]}")
    print(f"Total input steps: {len(df_input)} hours")
    
    # Transform input dataframe
    X_scaled, _ = preprocessor.transform_df(df_input)
    
    # Shape for model: [1, lookback, num_features]
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Run prediction
    with torch.no_grad():
        output_scaled, weights = model(input_tensor)
        
    output_scaled = output_scaled.cpu().numpy()  # [1, horizon]
    weights = weights.squeeze(0).cpu().numpy()  # [lookback, 1]
    
    # Inverse transform to PM2.5 concentrations
    predicted_pm25 = preprocessor.inverse_transform_targets(output_scaled)[0]
    
    # Convert to AQI
    predicted_aqi = np.array([pm25_to_aqi(val) for val in predicted_pm25])
    
    # Print results
    print("\n==========================================================")
    print(f"         SAANS TRANSFORMER AQI FORECAST FOR LAHORE        ")
    print(f"          Forecast generated at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}   ")
    print("==========================================================")
    print(f"{'Time (Lahore)':<20} | {'PM2.5 (ug/m³)':<14} | {'Est. AQI':<8} | {'Category':<20}")
    print("-" * 72)
    
    # The forecast starts at the hour immediately following our input window end
    forecast_start_time = df_input['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
    
    for i in range(horizon):
        forecast_time = forecast_start_time + pd.Timedelta(hours=i)
        pm_val = predicted_pm25[i]
        aqi_val = predicted_aqi[i]
        category = get_aqi_category(aqi_val)
        
        # Determine warning level indicator
        color_indicator = ""
        if category in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
            color_indicator = " [WARN]"
            
        print(f"{forecast_time.strftime('%Y-%m-%d %H:%M'):<20} | {pm_val:<14.2f} | {round(aqi_val):<8} | {category}{color_indicator}")
        
    print("==========================================================")
    
    # Display attention highlight (top 3 hours that influenced the model most)
    top_indices = np.argsort(weights.flatten())[::-1][:3]
    print("\nModel Analysis - Top 3 historical times influencing this forecast:")
    for rank, idx in enumerate(top_indices, 1):
        hist_time = df_input['timestamp'].iloc[idx]
        hist_pm = df_input['pm2_5'].iloc[idx]
        weight_pct = weights.flatten()[idx] * 100
        print(f"  {rank}. {hist_time.strftime('%Y-%m-%d %H:%M')} | PM2.5: {hist_pm:.1f} ug/m3 (Self-Attention Weight: {weight_pct:.1f}%)")

if __name__ == "__main__":
    run_inference()
