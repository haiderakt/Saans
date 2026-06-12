import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, seq_len, d_model]
        # Multi-head self-attention
        attn_out, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        
        # Feedforward network
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        
        return src, attn_weights

class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src):
        # Stacks encoder layers and passes weights upward
        all_weights = []
        for layer in self.layers:
            src, weights = layer(src)
            all_weights.append(weights)
        # Return final layer attention weights
        return src, all_weights[-1]

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, forecast_horizon=24, seq_len=72, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 1. Feature projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Custom Transformer Encoder
        self.transformer_encoder = CustomTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # 4. Decoder (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, forecast_horizon)
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns:
            out: predictions of shape [batch_size, forecast_horizon]
            attention_weights: mean self-attention weights of shape [batch_size, seq_len, 1]
        """
        # Project inputs: [batch_size, seq_len, d_model]
        x_proj = self.input_projection(x)
        
        # Add Positional Encoding
        x_pos = self.pos_encoder(x_proj)
        
        # Transformer encoding: [batch_size, seq_len, d_model]
        # weights: [batch_size, seq_len, seq_len]
        enc_out, weights = self.transformer_encoder(x_pos)
        
        # Flatten sequence length and feature space
        # shape: [batch_size, seq_len * d_model]
        flat_out = enc_out.reshape(enc_out.size(0), -1)
        
        # Run forecast decoder
        out = self.decoder(flat_out)
        
        # Compress weights to shape [batch_size, seq_len, 1] by averaging across query heads
        # weights: [batch_size, seq_len, seq_len] -> average over dim 1 -> shape [batch_size, seq_len]
        mean_weights = weights.mean(dim=1).unsqueeze(-1)
        
        return out, mean_weights

if __name__ == "__main__":
    # Test shape compatibility
    batch_size = 8
    seq_len = 72
    num_features = 46
    horizon = 24
    
    model = TransformerTimeSeriesModel(input_dim=num_features, d_model=64, forecast_horizon=horizon, seq_len=seq_len)
    print(model)
    
    dummy_input = torch.randn(batch_size, seq_len, num_features)
    output, weights = model(dummy_input)
    
    print("\nTransformer Output shape:", output.shape)  # [8, 24]
    print("Attention Weights shape:", weights.shape)  # [8, 72, 1]
    print("Transformer forward pass successful!")
