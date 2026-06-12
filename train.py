import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocessing import DataPreprocessor
from model import TransformerTimeSeriesModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AQIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AsymmetricWeightedHuberLoss(nn.Module):
    def __init__(self, target_min, target_scale, threshold=100.0, asymmetry_factor=5.0, delta=0.1):
        super(AsymmetricWeightedHuberLoss, self).__init__()
        self.target_min = target_min
        self.target_scale = target_scale
        self.threshold = threshold
        self.asymmetry_factor = asymmetry_factor
        self.delta = delta

    def forward(self, pred, target):
        # Reconstruct raw target concentrations to compute weights
        raw_target = target * self.target_scale + self.target_min
        error = pred - target
        
        # Huber Loss component
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        huber_loss = 0.5 * (quadratic ** 2) + self.delta * linear
        
        # Asymmetry mask: underpredicting spikes (actual > threshold and pred < actual)
        underprediction_mask = (error < 0) & (raw_target > self.threshold)
        
        # Apply asymmetry penalty (5x)
        weights = torch.ones_like(error)
        weights[underprediction_mask] = self.asymmetry_factor
        
        # Scale weight: penalize errors on higher values
        scale_weight = 1.0 + (raw_target / 150.0)
        
        loss = weights * scale_weight * huber_loss
        return loss.mean()

class WeightedMSELoss(nn.Module):
    def __init__(self, target_min, target_scale):
        super(WeightedMSELoss, self).__init__()
        self.target_min = target_min
        self.target_scale = target_scale

    def forward(self, pred, target):
        raw_target = target * self.target_scale + self.target_min
        weight = 1.0 + (raw_target / 100.0)
        loss = weight * (pred - target) ** 2
        return loss.mean()

def train_model(data_path="data/raw_data.csv", lookback=72, horizon=24, epochs=50, batch_size=64, lr=0.001, hidden_dim=128, num_layers=2, patience=7, loss_type="asymmetric"):
    # Set seed for strict reproducibility
    set_seed(42)
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Preprocess Data
    preprocessor = DataPreprocessor(scaler_path="models/scalers.pkl")
    print("Loading and cleaning raw data...")
    df = preprocessor.load_and_clean(data_path)
    df = preprocessor.add_cyclical_time_features(df)
    
    print("Splitting data chronologically...")
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    print("Scaling features and target...")
    (X_train_s, y_train_s), (X_val_s, y_val_s), (X_test_s, y_test_s) = preprocessor.fit_transform(train_df, val_df, test_df)
    
    # Create sequence windows
    print("Creating sequence windows...")
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_s, y_train_s, lookback, horizon)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_s, y_val_s, lookback, horizon)
    
    print(f"Train shapes: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Val shapes: X={X_val_seq.shape}, y={y_val_seq.shape}")
    
    # DataLoaders
    train_dataset = AQIDataset(X_train_seq, y_train_seq)
    val_dataset = AQIDataset(X_val_seq, y_val_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model Initialization (Transformer)
    input_dim = X_train_seq.shape[2]
    model = TransformerTimeSeriesModel(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=num_layers,
        forecast_horizon=horizon,
        seq_len=lookback,
        dropout=0.1
    ).to(device)
    
    print(model)
    
    # Save Model Parameters & Configurations for Evaluation/Inference
    model_config = {
        "input_dim": input_dim,
        "d_model": 64,
        "nhead": 4,
        "num_layers": num_layers,
        "forecast_horizon": horizon,
        "lookback": lookback,
        "feature_cols": preprocessor.feature_cols,
        "target_col": preprocessor.target_col
    }
    os.makedirs("models", exist_ok=True)
    with open("models/model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    print("Model configuration saved to models/model_config.json")
    
    # 4. Loss & Optimizer
    if loss_type == "asymmetric":
        target_min = float(preprocessor.target_scaler.data_min_[0])
        target_scale = float(preprocessor.target_scaler.data_range_[0])
        criterion = AsymmetricWeightedHuberLoss(target_min, target_scale)
        print("Using custom AsymmetricWeightedHuberLoss (Huber + 5x spike underprediction penalty)")
    elif loss_type == "weighted":
        target_min = float(preprocessor.target_scaler.data_min_[0])
        target_scale = float(preprocessor.target_scaler.data_range_[0])
        criterion = WeightedMSELoss(target_min, target_scale)
        print("Using custom WeightedMSELoss")
    else:
        criterion = nn.MSELoss()
        print("Using standard MSELoss")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (Reduce LR on Plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    # Mixed precision training scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # 5. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision autocast
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                
            # Backward pass & Optimize
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model weights
            torch.save(model.state_dict(), "models/best_model.pth")
            print("--> Validation loss improved. Saving checkpoint...")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best Val Loss: {best_val_loss:.6f}")
                break
                
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model for Lahore AQI.")
    parser.add_argument("--data", type=str, default="data/raw_data.csv", help="Path to raw CSV data")
    parser.add_argument("--lookback", type=int, default=72, help="Input lookback window in hours")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours (24 or 48)")
    parser.add_argument("--epochs", type=int, default=30, help="Max number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer Encoder layers")
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    parser.add_argument("--loss", type=str, default="asymmetric", choices=["asymmetric", "weighted", "mse"], help="Loss function")
    
    args = parser.parse_args()
    train_model(
        data_path=args.data,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_layers=args.num_layers,
        patience=args.patience,
        loss_type=args.loss
    )
