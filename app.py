import os
import json
import torch
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from preprocessing import DataPreprocessor, pm25_to_aqi, get_aqi_category
from model import TransformerTimeSeriesModel

# Page configuration
st.set_page_config(
    page_title="Saans - Lahore AQI Forecaster",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark theme glassmorphism)
st.markdown("""
    <style>
    /* Main app styles */
    .stApp {
        background-color: #090a0f;
        background-image: radial-gradient(circle at 10% 20%, rgba(20, 24, 38, 0.3) 0%, rgba(9, 10, 15, 0.8) 90%);
        color: #e0e5f5;
    }
    
    /* Headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #64ffda 0%, #00b0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
        text-shadow: 0 4px 12px rgba(0, 176, 255, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #8f9bb3;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    /* Custom container cards (Glassmorphism) */
    .glass-card {
        background: rgba(20, 24, 35, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Metric Cards */
    .aqi-card {
        text-align: center;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .aqi-value {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 10px;
    }
    
    .aqi-cat {
        font-size: 1.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Category specific colors */
    .cat-good { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); }
    .cat-moderate { background: linear-gradient(135deg, #f57f17 0%, #f9a825 100%); color: #000000; }
    .cat-sensitive { background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%); }
    .cat-unhealthy { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); }
    .cat-veryunhealthy { background: linear-gradient(135deg, #4a148c 0%, #6a1b9a 100%); }
    .cat-hazardous { background: linear-gradient(135deg, #310d0d 0%, #5d0f0f 100%); }
    
    /* Recommendations styling */
    .rec-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00b0ff;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
    }
    
    .rec-title {
        font-weight: 700;
        color: #00b0ff;
        margin-bottom: 5px;
    }
    
    /* Tables */
    .dataframe {
        background-color: transparent !important;
        color: #e0e5f5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Title Banner
st.markdown('<div class="main-title">💨 SAANS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">State-of-the-Art Transformer AQI Forecasting Dashboard for Lahore, Pakistan</div>', unsafe_allow_html=True)

# Setup Sidebar
st.sidebar.markdown("### ⚙️ System Controls")
latitude = 31.5204
longitude = 74.3587
st.sidebar.info(f"📍 **Location**: Lahore\n🌐 **Coordinates**: {latitude}° N, {longitude}° E")

# Load configuration and initialize
@st.cache_resource
def load_forecast_system():
    config_path = "models/model_config.json"
    weights_path = "models/best_model.pth"
    scaler_path = "models/scalers.pkl"
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path) or not os.path.exists(scaler_path):
        return None, None, None
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTimeSeriesModel(
        input_dim=config["input_dim"],
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_layers=config["num_layers"],
        forecast_horizon=config["forecast_horizon"],
        seq_len=config["lookback"]
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    preprocessor = DataPreprocessor(feature_cols=config["feature_cols"], target_col=config["target_col"], scaler_path=scaler_path)
    preprocessor.load_scalers()
    
    return model, preprocessor, config

model, preprocessor, config = load_forecast_system()

if model is None:
    st.error("🚨 **Error**: Trained model files not found in the `models/` directory. Please run training on your system first to generate `best_model.pth`, `scalers.pkl`, and `model_config.json`.")
    st.stop()

# Live data fetch caching
@st.cache_data(ttl=3600)  # Cache forecast results for 1 hour to prevent API spamming
def fetch_and_predict():
    # Fetch live weather and AQ inputs (past 5 days + forecast)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,boundary_layer_height",
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Karachi"
    }
    weather_response = requests.get(weather_url, params=weather_params).json().get("hourly", {})
    df_weather = pd.DataFrame(weather_response).rename(columns={
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
        "boundary_layer_height": "boundary_layer_height"
    })
    
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Karachi"
    }
    aq_response = requests.get(aq_url, params=aq_params).json().get("hourly", {})
    df_aq = pd.DataFrame(aq_response).rename(columns={
        "time": "timestamp",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "carbon_monoxide": "co"
    })
    
    df_live = pd.merge(df_weather, df_aq, on="timestamp", how="outer")
    df_live["timestamp"] = pd.to_datetime(df_live["timestamp"])
    df_live = df_live.sort_values("timestamp").reset_index(drop=True)
    
    # Preprocess
    df_live = preprocessor.add_cyclical_time_features(df_live)
    cols_to_fill = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure", "boundary_layer_height", "pm2_5", "pm10", "no2", "o3", "co"]
    for col in cols_to_fill:
        if col in df_live.columns:
            df_live[col] = df_live[col].interpolate(method="linear").bfill().ffill()
            
    # Filter up to current hour to form history
    current_time = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
    df_past = df_live[df_live["timestamp"] <= current_time].copy()
    
    lookback = config["lookback"]
    if len(df_past) < lookback:
        df_input = df_live.head(lookback).copy()
    else:
        df_input = df_past.tail(lookback).copy()
        
    X_scaled, _ = preprocessor.transform_df(df_input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output_scaled, weights = model(input_tensor)
        
    output_scaled = output_scaled.cpu().numpy()
    weights = weights.squeeze(0).cpu().numpy()
    
    predicted_pm25 = preprocessor.inverse_transform_targets(output_scaled)[0]
    predicted_aqi = np.array([pm25_to_aqi(val) for val in predicted_pm25])
    
    return df_input, predicted_pm25, predicted_aqi, weights, current_time

# Run predictions
with st.spinner("Fetching real-time weather and pollution inputs..."):
    try:
        df_input, predicted_pm25, predicted_aqi, weights, current_time = fetch_and_predict()
    except Exception as e:
        st.error(f"🚨 **Network Error**: Failed to fetch live data from Open-Meteo API. Please reload the page. Details: {e}")
        st.stop()

# Layout splits
col1, col2 = st.columns([1, 2])

# Left column: Metrics card and Advisory
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📍 Current Forecast Summary")
    
    # Use the first hour of forecast as "current" predicted AQI
    next_hour_pm25 = predicted_pm25[0]
    next_hour_aqi = round(predicted_aqi[0])
    category = get_aqi_category(next_hour_aqi)
    
    # Match styles
    cat_class = "cat-good"
    rec_text = "Air quality is satisfactory. Outdoor activities are safe."
    if category == "Moderate":
        cat_class = "cat-moderate"
        rec_text = "Unusually sensitive people should consider reducing prolonged or heavy exertion outdoors."
    elif category == "Unhealthy for Sensitive Groups":
        cat_class = "cat-sensitive"
        rec_text = "Members of sensitive groups may experience health effects. Limit prolonged outdoor exposure."
    elif category == "Unhealthy":
        cat_class = "cat-unhealthy"
        rec_text = "Everyone may begin to experience health effects. Wear an **N95 mask** if going outdoors; keep windows closed."
    elif category == "Very Unhealthy":
        cat_class = "cat-veryunhealthy"
        rec_text = "Health alert: everyone may experience more serious health effects. **Avoid all outdoor physical activity**; run air purifiers."
    elif category == "Hazardous":
        cat_class = "cat-hazardous"
        rec_text = "Emergency conditions: entire population is likely to be affected. **Stay indoors, keep all doors/windows sealed**, and continuously run air purifiers."

    st.markdown(f"""
        <div class="aqi-card {cat_class}">
            <div class="aqi-cat">{category}</div>
            <div class="aqi-value">{next_hour_aqi}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Est. US EPA AQI index</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**PM2.5 Concentration**: `{next_hour_pm25:.2f} µg/m³`")
    
    st.markdown(f"""
        <div class="rec-box">
            <div class="rec-title">📢 Public Health Advisory</div>
            <div class="rec-text">{rec_text}</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Self-attention explainability card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Self-Attention Analysis")
    st.write("Below are the top 3 historical hours that the Transformer model attended to most heavily to generate this 24-hour forecast:")
    
    top_indices = np.argsort(weights.flatten())[::-1][:3]
    for rank, idx in enumerate(top_indices, 1):
        hist_time = df_input['timestamp'].iloc[idx]
        hist_pm = df_input['pm2_5'].iloc[idx]
        weight_pct = weights.flatten()[idx] * 100
        
        st.markdown(f"""
            **{rank}. {hist_time.strftime('%b %d, %H:%M')}** (Lahore Time)
            * Observed PM2.5: `{hist_pm:.1f} µg/m³`
            * Attention Weight: `{weight_pct:.1f}%`
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# Right column: Graphical chart & download breakdown
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Predicted 24-Hour Forecast Timeline")
    
    forecast_start_time = df_input['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
    forecast_times = [forecast_start_time + pd.Timedelta(hours=i) for i in range(len(predicted_pm25))]
    
    # Custom matplotlib design to match dark premium styling
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#090a0f')
    ax.set_facecolor('#0d0f17')
    
    ax.plot(forecast_times, predicted_pm25, color='#00b0ff', linewidth=3, label="Predicted PM2.5", marker='o', markersize=4)
    ax.fill_between(forecast_times, predicted_pm25, color='#00b0ff', alpha=0.1)
    
    ax.set_title("Forecasted PM2.5 Concentration for Next 24 Hours", color='#ffffff', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel("PM2.5 Concentration (µg/m³)", color='#8f9bb3', fontsize=10)
    ax.set_xlabel("Time (Lahore)", color='#8f9bb3', fontsize=10)
    
    ax.tick_params(colors='#8f9bb3', labelsize=8)
    ax.grid(True, color='#1e2230', linestyle=':', alpha=0.7)
    
    # Rotate x labels nicely
    plt.xticks(rotation=15)
    
    # Remove border lines (spines)
    for spine in ax.spines.values():
        spine.set_color('#1e2230')
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Table breakdown and CSV Export
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Hourly Table Forecast Breakdown")
    
    forecast_df = pd.DataFrame({
        "Time (Lahore)": [t.strftime('%Y-%m-%d %H:%M') for t in forecast_times],
        "PM2.5 (ug/m3)": [round(val, 2) for val in predicted_pm25],
        "Estimated AQI": [round(val) for val in predicted_aqi],
        "AQI Category": [get_aqi_category(val) for val in predicted_aqi]
    })
    
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    # Export csv button
    csv_data = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast Data as CSV",
        data=csv_data,
        file_name=f"lahore_aqi_forecast_{current_time.strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv',
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #5a647d; font-size: 0.8rem; margin-top: 50px; padding: 15px; border-top: 1px solid rgba(255, 255, 255, 0.05);">
        Saans forecasting pipeline is powered by a PyTorch Transformer Encoder utilizing 46 meteorological & historical variables. 
        Data provided dynamically by Copernicus Atmospheric Monitoring Service (CAMS) & Open-Meteo free feeds.
    </div>
""", unsafe_allow_html=True)
