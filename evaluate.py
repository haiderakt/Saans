import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import DataPreprocessor, pm25_to_aqi, get_aqi_category
from model import TransformerTimeSeriesModel
from train import AQIDataset
from torch.utils.data import DataLoader

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-5):
    """
    Computes MAPE, protecting against division by zero.
    """
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100

def evaluate_model(data_path="data/raw_data.csv", config_path="models/model_config.json", weights_path="models/best_model.pth", plot_dir="plots"):
    # 1. Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration not found at {config_path}. Train the model first.")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    lookback = config["lookback"]
    horizon = config["forecast_horizon"]
    num_layers = config["num_layers"]
    feature_cols = config["feature_cols"]
    target_col = config["target_col"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Re-create and load model
    model = TransformerTimeSeriesModel(
        input_dim=len(feature_cols),
        d_model=64,
        nhead=4,
        num_layers=num_layers,
        forecast_horizon=horizon,
        seq_len=lookback
    ).to(device)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
    
    # 3. Load test data
    preprocessor = DataPreprocessor(feature_cols=feature_cols, target_col=target_col)
    preprocessor.load_scalers()
    
    print("Loading test dataset...")
    df = preprocessor.load_and_clean(data_path)
    df = preprocessor.add_cyclical_time_features(df)
    
    _, _, test_df = preprocessor.split_data(df)
    print(f"Test data size: {len(test_df)} rows")
    
    X_test_scaled, y_test_scaled = preprocessor.transform_df(test_df)
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled, y_test_scaled, lookback, horizon)
    
    test_dataset = AQIDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. Predict
    all_preds = []
    all_targets = []
    
    print("Running evaluation predictions...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds, _ = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    preds_scaled = np.vstack(all_preds)
    targets_scaled = np.vstack(all_targets)
    
    # 5. Inverse scaling to raw PM2.5 concentrations
    preds_raw = preprocessor.inverse_transform_targets(preds_scaled)
    targets_raw = preprocessor.inverse_transform_targets(targets_scaled)
    
    # 6. Calculate Metrics (RMSE, MAE, R2, MAPE, AQI Category Accuracy)
    overall_rmse = np.sqrt(mean_squared_error(targets_raw, preds_raw))
    overall_mae = mean_absolute_error(targets_raw, preds_raw)
    overall_r2 = r2_score(targets_raw, preds_raw)
    overall_mape = mean_absolute_percentage_error(targets_raw, preds_raw)
    
    # Calculate AQI Category Accuracy
    targets_aqi = np.vectorize(pm25_to_aqi)(targets_raw)
    preds_aqi = np.vectorize(pm25_to_aqi)(preds_raw)
    
    target_categories = np.vectorize(get_aqi_category)(targets_aqi)
    pred_categories = np.vectorize(get_aqi_category)(preds_aqi)
    
    category_matches = (target_categories == pred_categories).sum()
    total_elements = target_categories.size
    category_accuracy = (category_matches / total_elements) * 100
    
    # Spike detection threshold (top 10% actual PM2.5 values)
    p90_threshold = np.percentile(targets_raw, 90)
    spike_mask = targets_raw > p90_threshold
    spike_rmse = np.sqrt(np.mean((targets_raw[spike_mask] - preds_raw[spike_mask]) ** 2))
    spike_mae = np.mean(np.abs(targets_raw[spike_mask] - preds_raw[spike_mask]))
    
    print("\n================ Evaluation Results (PM2.5 concentration in ug/m3) ================")
    print(f"Overall Test RMSE: {overall_rmse:.4f}")
    print(f"Overall Test MAE : {overall_mae:.4f}")
    print(f"Overall Test R²  : {overall_r2:.4f}")
    print(f"Overall Test MAPE: {overall_mape:.2f}%")
    print(f"90th Percentile Spike RMSE (Actual > {p90_threshold:.2f} µg/m³): {spike_rmse:.4f}")
    print(f"90th Percentile Spike MAE  (Actual > {p90_threshold:.2f} µg/m³): {spike_mae:.4f}")
    
    # Specific Lead-time Metrics
    lead_times = [0, 5, 11, 23]  # indices for 1h, 6h, 12h, 24h ahead
    print("\nLead-Time Metrics:")
    for idx in lead_times:
        if idx < horizon:
            actual_step = targets_raw[:, idx]
            pred_step = preds_raw[:, idx]
            step_rmse = np.sqrt(mean_squared_error(actual_step, pred_step))
            step_mae = mean_absolute_error(actual_step, pred_step)
            step_r2 = r2_score(actual_step, pred_step)
            
            p90_step = np.percentile(actual_step, 90)
            step_spike_mask = actual_step > p90_step
            step_spike_rmse = np.sqrt(np.mean((actual_step[step_spike_mask] - pred_step[step_spike_mask]) ** 2))
            
            print(f"  t+{idx+1:02d} Hours Ahead | RMSE: {step_rmse:.4f} | MAE: {step_mae:.4f} | R²: {step_r2:.4f} | P90 Spike RMSE (>{p90_step:.1f}): {step_spike_rmse:.4f}")
            
    # AQI Metrics
    overall_aqi_rmse = np.sqrt(mean_squared_error(targets_aqi, preds_aqi))
    overall_aqi_mae = mean_absolute_error(targets_aqi, preds_aqi)
    
    print("\n================ Evaluation Results (US EPA AQI) ================")
    print(f"Overall AQI RMSE: {overall_aqi_rmse:.4f}")
    print(f"Overall AQI MAE : {overall_aqi_mae:.4f}")
    print(f"AQI Category Exact Match Accuracy: {category_accuracy:.2f}%")
    print("==================================================================")
    
    # 7. Plots Generation
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Actual vs. Predicted comparison line plot (for a slice, t+24h)
    plot_step = min(23, horizon - 1)
    plot_slice = -300
    timestamps = test_df["timestamp"].iloc[lookback + plot_step : lookback + plot_step + len(targets_raw)].values
    
    slice_time = timestamps[plot_slice:]
    slice_actual = targets_raw[plot_slice:, plot_step]
    slice_pred = preds_raw[plot_slice:, plot_step]
    
    plt.figure(figsize=(14, 6))
    plt.plot(slice_time, slice_actual, label="Actual PM2.5", color="#2c3e50", linewidth=2, alpha=0.8)
    plt.plot(slice_time, slice_pred, label=f"Predicted PM2.5 (t+{plot_step+1}h)", color="#e67e22", linewidth=2, linestyle="--")
    plt.title(f"Saans Transformer Model - Lahore PM2.5 Forecast ({plot_step+1} Hours Ahead)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "evaluation_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 2: Scatter Plot (Actual vs. Predicted)
    # Flatten across all horizons to show complete scatter distribution
    y_true_flat = targets_raw.flatten()
    y_pred_flat = preds_raw.flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_flat, y_pred_flat, color="#3498db", alpha=0.15, edgecolors='none', label="Predictions")
    # Draw reference line y = x
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Perfect Forecast")
    
    plt.title("Saans Transformer Model - Predicted vs. Actual PM2.5", fontsize=14, fontweight="bold")
    plt.xlabel("Actual PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.ylabel("Predicted PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "scatter_plot.png"), dpi=150)
    plt.close()
    
    # Plot 3: Residual Plot
    residuals = y_pred_flat - y_true_flat
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_flat, residuals, color="#9b59b6", alpha=0.15, edgecolors='none')
    plt.axhline(0, color="#2c3e50", linestyle="--", linewidth=2)
    
    plt.title("Saans Transformer Model - Residual Analysis", fontsize=14, fontweight="bold")
    plt.xlabel("Actual PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.ylabel("Residual (Predicted - Actual) (µg/m³)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "residual_plot.png"), dpi=150)
    plt.close()
    
    print(f"Evaluation plots saved to: {plot_dir}/")
    
    # Save metrics as a JSON for reference
    metrics = {
        "pm25": {
            "overall_rmse": float(overall_rmse),
            "overall_mae": float(overall_mae),
            "overall_r2": float(overall_r2),
            "overall_mape": float(overall_mape)
        },
        "aqi": {
            "overall_rmse": float(overall_aqi_rmse),
            "overall_mae": float(overall_aqi_mae),
            "category_accuracy": float(category_accuracy)
        }
    }
    with open(os.path.join(plot_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    evaluate_model()
