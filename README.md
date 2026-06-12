# Saans 💨

Saans is a state-of-the-art PM2.5 and AQI prediction system for Lahore, Pakistan. Built using a PyTorch Transformer Encoder and an asymmetric safety-first loss, it maps 46 weather and pollution variables to forecast air quality 24 hours ahead, deployed as a real-time, interactive Streamlit dashboard.

## Features

1. **Transformer Encoder Model** (`model.py`): Models global historical sequence dependencies without the single-vector information compression bottleneck of classic LSTMs.
2. **Asymmetric Risk-Averse Loss** (`train.py`): Employs a custom `AsymmetricWeightedHuberLoss` applying a 5x penalty for underpredicting severe spikes ($>100$ µg/m³), prioritizing public health safety.
3. **46-Feature Pipeline** (`preprocessing.py`): Engineers lags (1h to 48h), rolling stats (mean, std, max, min over multiple windows), wind U/V vector components, and atmospheric stagnation interaction metrics.
4. **Explainable AI (Self-Attention)**: Extracts the inner encoder attention weights to identify and visualize which past hours most heavily influenced the forecast.
5. **Interactive Web Dashboard** (`app.py`): A premium glassmorphic dark theme dashboard featuring real-time data fetching, forecast charts, and health precautions.

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Fetch Historical Data
Download 3.5+ years of historical meteorological and air quality CAMS data for Lahore:
```bash
python data_fetcher.py --start-date 2022-08-01
```

### 3. Train the Model
Train the Transformer model using the asymmetric loss function:
```bash
python train.py --epochs 30 --horizon 24 --loss asymmetric
```

### 4. Run Evaluation
Verify metrics ($R^2$, MAPE, category match accuracy) and generate scatter/residual diagnostic plots:
```bash
python evaluate.py
```

### 5. Launch the Dashboard Locally
```bash
streamlit run app.py
```

---

## Free Cloud Deployment (Streamlit Community Cloud)

You can host this dashboard live on the internet at no cost:
1. Push this folder to a public GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository and select `app.py` as the entrypoint.
4. Deploy! Your app will be live at `https://<your-app-name>.streamlit.app`.
