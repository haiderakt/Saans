import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def pm25_to_aqi(pm25):
    """
    Converts PM2.5 concentration (ug/m3) to US EPA AQI sub-index.
    """
    if pd.isna(pm25) or pm25 < 0:
        return 0
    # US EPA PM2.5 Breakpoints
    if pm25 <= 12.0:
        return ((50.0 - 0.0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0.0
    elif pm25 <= 35.4:
        return ((100.0 - 51.0) / (35.4 - 12.1)) * (pm25 - 12.1) + 51.0
    elif pm25 <= 55.4:
        return ((150.0 - 101.0) / (55.4 - 35.5)) * (pm25 - 35.5) + 101.0
    elif pm25 <= 150.4:
        return ((200.0 - 151.0) / (150.4 - 55.5)) * (pm25 - 55.5) + 151.0
    elif pm25 <= 250.4:
        return ((300.0 - 201.0) / (250.4 - 150.5)) * (pm25 - 150.5) + 201.0
    elif pm25 <= 350.4:
        return ((400.0 - 301.0) / (350.4 - 250.5)) * (pm25 - 250.5) + 301.0
    elif pm25 <= 500.4:
        return ((500.0 - 401.0) / (500.4 - 350.5)) * (pm25 - 350.5) + 401.0
    else:
        # Extrapolate beyond 500.4 using the slope of the Hazardous category
        return ((500.0 - 401.0) / (500.4 - 350.5)) * (pm25 - 350.5) + 401.0

def get_aqi_category(aqi):
    """
    Returns US EPA AQI category label based on AQI value.
    """
    aqi_val = round(aqi)
    if aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Moderate"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_val <= 200:
        return "Unhealthy"
    elif aqi_val <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

class DataPreprocessor:
    def __init__(self, feature_cols=None, target_col="pm2_5", scaler_path="models/scalers.pkl"):
        if feature_cols is None:
            # Default features (including engineered stagnation & wind components)
            self.feature_cols = [
                "temperature", "humidity", "pressure", "boundary_layer_height",
                "pm2_5", "pm10", "no2", "o3", "co",
                "wind_u", "wind_v",
                "hour_sin", "hour_cos",
                "dayofweek_sin", "dayofweek_cos",
                "month_sin", "month_cos",
                "dayofyear_sin", "dayofyear_cos",
                "wind_dir_sin", "wind_dir_cos",
                "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_lag_6",
                "pm2_5_lag_12", "pm2_5_lag_24", "pm2_5_lag_48",
                "pm2_5_roll_mean_3", "pm2_5_roll_std_3", "pm2_5_roll_max_3", "pm2_5_roll_min_3",
                "pm2_5_roll_mean_6", "pm2_5_roll_std_6", "pm2_5_roll_max_6", "pm2_5_roll_min_6",
                "pm2_5_roll_mean_12", "pm2_5_roll_std_12", "pm2_5_roll_max_12", "pm2_5_roll_min_12",
                "pm2_5_roll_mean_24", "pm2_5_roll_std_24", "pm2_5_roll_max_24", "pm2_5_roll_min_24",
                "wind_speed_blh_interaction", "temp_humidity_interaction"
            ]
        else:
            self.feature_cols = feature_cols
            
        self.target_col = target_col
        self.scaler_path = scaler_path
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    def load_and_clean(self, file_path):
        """
        Loads the raw merged CSV, handles missing values, and returns clean df.
        """
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Handle missing values using linear interpolation, then backward/forward fill for boundary values
        cols_to_interpolate = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure", "boundary_layer_height", "pm2_5", "pm10", "no2", "o3", "co"]
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").bfill().ffill()
                
        return df

    def add_cyclical_time_features(self, df):
        """
        Calculates cyclical time features, cyclical wind direction features,
        lags, rolling averages, and interaction features.
        """
        # Time components
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        
        # 1. Cyclical time encodings (8 features)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        
        # 2. Wind Direction Cyclical Encoding (2 features)
        wind_dir_rad = np.radians(df["wind_direction"])
        df["wind_dir_sin"] = np.sin(wind_dir_rad)
        df["wind_dir_cos"] = np.cos(wind_dir_rad)
        
        # Wind U/V components (2 features)
        df["wind_u"] = df["wind_speed"] * np.cos(wind_dir_rad)
        df["wind_v"] = df["wind_speed"] * np.sin(wind_dir_rad)
        
        # 3. PM2.5 Lags (7 features)
        df["pm2_5_lag_1"] = df["pm2_5"].shift(1)
        df["pm2_5_lag_2"] = df["pm2_5"].shift(2)
        df["pm2_5_lag_3"] = df["pm2_5"].shift(3)
        df["pm2_5_lag_6"] = df["pm2_5"].shift(6)
        df["pm2_5_lag_12"] = df["pm2_5"].shift(12)
        df["pm2_5_lag_24"] = df["pm2_5"].shift(24)
        df["pm2_5_lag_48"] = df["pm2_5"].shift(48)
        
        # 4. PM2.5 Rolling Statistics (16 features)
        for w in [3, 6, 12, 24]:
            roll = df["pm2_5"].rolling(window=w)
            df[f"pm2_5_roll_mean_{w}"] = roll.mean()
            df[f"pm2_5_roll_std_{w}"] = roll.std()
            df[f"pm2_5_roll_max_{w}"] = roll.max()
            df[f"pm2_5_roll_min_{w}"] = roll.min()
            
        # 5. Interactions (2 features)
        df["wind_speed_blh_interaction"] = df["wind_speed"] * df["boundary_layer_height"]
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
        
        # Clean up NaNs via backfill and fillna to preserve rows
        cols_to_clean = [
            "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12", "pm2_5_lag_24", "pm2_5_lag_48",
            "pm2_5_roll_mean_3", "pm2_5_roll_std_3", "pm2_5_roll_max_3", "pm2_5_roll_min_3",
            "pm2_5_roll_mean_6", "pm2_5_roll_std_6", "pm2_5_roll_max_6", "pm2_5_roll_min_6",
            "pm2_5_roll_mean_12", "pm2_5_roll_std_12", "pm2_5_roll_max_12", "pm2_5_roll_min_12",
            "pm2_5_roll_mean_24", "pm2_5_roll_std_24", "pm2_5_roll_max_24", "pm2_5_roll_min_24"
        ]
        for col in cols_to_clean:
            df[col] = df[col].bfill().fillna(0.0)
            
        return df

    def split_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        Splits data chronologically into train, validation, and test sets.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        
        return train_df, val_df, test_df

    def fit_transform(self, train_df, val_df, test_df):
        """
        Fits scalers on train_df and transforms all data splits.
        Saves scalers to disk.
        """
        # Select features and targets
        X_train = train_df[self.feature_cols].values
        y_train = train_df[[self.target_col]].values
        
        # Fit
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)
        
        # Save scalers
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump({
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "feature_cols": self.feature_cols,
                "target_col": self.target_col
            }, f)
            
        print(f"Scalers saved to {self.scaler_path}")
        
        # Transform
        X_train_scaled = self.feature_scaler.transform(X_train)
        y_train_scaled = self.target_scaler.transform(y_train)
        
        X_val_scaled = self.feature_scaler.transform(val_df[self.feature_cols].values)
        y_val_scaled = self.target_scaler.transform(val_df[[self.target_col]].values)
        
        X_test_scaled = self.feature_scaler.transform(test_df[self.feature_cols].values)
        y_test_scaled = self.target_scaler.transform(test_df[[self.target_col]].values)
        
        return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled)

    def load_scalers(self):
        """
        Loads feature and target scalers from disk.
        """
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scalers file not found at {self.scaler_path}")
            
        with open(self.scaler_path, "rb") as f:
            scalers = pickle.load(f)
            self.feature_scaler = scalers["feature_scaler"]
            self.target_scaler = scalers["target_scaler"]
            self.feature_cols = scalers["feature_cols"]
            self.target_col = scalers["target_col"]
            
        print(f"Scalers loaded from {self.scaler_path}")

    def transform_df(self, df):
        """
        Transforms a dataframe using the loaded scalers (for inference).
        """
        X = df[self.feature_cols].values
        X_scaled = self.feature_scaler.transform(X)
        if self.target_col in df.columns:
            y = df[[self.target_col]].values
            y_scaled = self.target_scaler.transform(y)
            return X_scaled, y_scaled
        return X_scaled, None

    def create_sequences(self, X_scaled, y_scaled, lookback=72, horizon=24):
        """
        Creates 3D inputs and 2D targets for time series sequencing.
        X_scaled: [num_timesteps, num_features]
        y_scaled: [num_timesteps, 1]
        
        Returns:
            X_seq: [num_samples, lookback, num_features]
            y_seq: [num_samples, horizon]
        """
        X_seq, y_seq = [], []
        # We need enough steps for the lookback + horizon
        limit = len(X_scaled) - lookback - horizon + 1
        
        for i in range(limit):
            # Input window: lookback steps
            X_seq.append(X_scaled[i : i + lookback])
            # Target window: next horizon steps
            y_seq.append(y_scaled[i + lookback : i + lookback + horizon, 0])
            
        return np.array(X_seq), np.array(y_seq)
        
    def inverse_transform_targets(self, y_scaled):
        """
        Converts scaled target predictions back to raw concentrations.
        y_scaled: 2D array [samples, horizon] or 1D array
        """
        original_shape = y_scaled.shape
        # Target scaler was fitted on 2D array [samples, 1]
        # Reshape to 2D if needed, run inverse transform, and restore shape
        y_flat = y_scaled.reshape(-1, 1)
        y_inv = self.target_scaler.inverse_transform(y_flat)
        return y_inv.reshape(original_shape)

if __name__ == "__main__":
    # Test preprocessing pipeline on raw data
    preprocessor = DataPreprocessor()
    if os.path.exists("data/raw_data.csv"):
        df = preprocessor.load_and_clean("data/raw_data.csv")
        df = preprocessor.add_cyclical_time_features(df)
        print("Data loaded & features added. Columns:", list(df.columns))
        print("Total rows:", len(df))
        
        train_df, val_df, test_df = preprocessor.split_data(df)
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(train_df, val_df, test_df)
        print("Scale shapes - X_train:", X_train.shape, "y_train:", y_train.shape)
        
        # Test sequences
        X_seq, y_seq = preprocessor.create_sequences(X_train, y_train, lookback=72, horizon=24)
        print("Sequence shapes - X_seq:", X_seq.shape, "y_seq:", y_seq.shape)
        print("Preprocessing checks out successfully!")
    else:
        print("raw_data.csv not found, run data_fetcher.py first.")
