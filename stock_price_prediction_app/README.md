## Stock Price Prediction using LSTM and Machine Learning

An end-to-end Streamlit app to predict stock closing prices using two models:
- LSTM neural network (deep learning)
- RandomForestRegressor (baseline ML)

Data is pulled from Yahoo Finance via the `yfinance` API. The app provides interactive charts, model training, historical backtesting, and 7-day forecasts with CSV download.

### Project Structure
```
stock_price_prediction_app/
  ├─ app.py                # Streamlit app
  ├─ requirements.txt      # Python dependencies
  ├─ README.md             # This file
  ├─ setup.sh              # Optional setup for some hosts
  ├─ data/                 # Saved CSVs (runtime)
  │   └─ stock_data.csv    # Last downloaded default CSV
  └─ models/               # Trained models and scalers
```

### Installation
1. Python 3.10+
2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run app.py
```
Open the provided local URL in your browser.

### Usage
- Use the sidebar to set:
  - Ticker (e.g., AAPL, TSLA, MSFT, RELIANCE.NS)
  - Date range
  - Model (LSTM or RandomForest)
- Click "Train Model" to train and save the selected model for the given ticker.
- Click "Predict Future" to generate a 7-day forecast and download the forecast CSV.

### Features
- Yahoo Finance data download
- Technical indicators: MA20, MA50, RSI(14)
- Two models:
  - RandomForestRegressor: sequence of 60 closes → next-day close
  - LSTM: sequence of 60 closes → next-day close
- Visualizations:
  - Candlestick with MA overlays
  - RSI
  - Actual vs Predicted (historical)
  - 7-day forecast
- Download forecast CSV

### Model Comparison (quick reference)
| Model | Input Window | Target | Notes |
|------|--------------|--------|-------|
| RandomForest | 60-day close (scaled), flattened | Next-day close | Fast baseline |
| LSTM | 60-day close (scaled), 3D [samples, 60, 1] | Next-day close | Captures temporal patterns |

### Deployment
#### Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to Streamlit Community Cloud → "New app" → select the repo.
3. Set the app file path to `stock_price_prediction_app/app.py`.
4. Deploy. Ensure `requirements.txt` is present at repo root or same folder.

#### Alternatives
- Render or Railway: Create a web service, run `streamlit run stock_price_prediction_app/app.py --server.port $PORT --server.address 0.0.0.0` and provide a start command and `requirements.txt`.

### Notes
- Models and scalers are saved per ticker in `models/` (e.g., `lstm_model_AAPL.h5`, `rf_model_AAPL.joblib`).
- Default CSV `data/stock_data.csv` is overwritten with the latest downloaded dataset (for convenience). For multiple tickers, data is fetched live each session.


