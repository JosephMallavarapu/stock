import os
import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

# TensorFlow can be heavy; import lazily in train_lstm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_TICKER = "AAPL"
SEQUENCE_LENGTH = 60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    df.rename(columns={"Date": "Date"}, inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # RSI (14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    data["RSI14"] = 100 - (100 / (1 + rs))
    data["RSI14"] = data["RSI14"].fillna(50)
    return data


def save_default_csv(df: pd.DataFrame):
    csv_path = os.path.join(DATA_DIR, "stock_data.csv")
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        pass


def create_sequences(values: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def train_random_forest(close_prices: np.ndarray) -> tuple[RandomForestRegressor, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - SEQUENCE_LENGTH:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, scaler


def train_lstm(close_prices: np.ndarray) -> tuple[Sequential, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))
    X, y = create_sequences(scaled.flatten(), SEQUENCE_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    callbacks = [EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)]
    model.fit(X, y, epochs=30, batch_size=32, verbose=0, callbacks=callbacks)
    return model, scaler


def save_models(model_type: str, model, scaler: MinMaxScaler, ticker: str):
    if model_type == "RandomForest":
        joblib.dump(model, os.path.join(MODELS_DIR, f"rf_model_{ticker}.joblib"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"rf_scaler_{ticker}.joblib"))
    else:
        model.save(os.path.join(MODELS_DIR, f"lstm_model_{ticker}.h5"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"lstm_scaler_{ticker}.joblib"))


def load_models(model_type: str, ticker: str):
    if model_type == "RandomForest":
        model_path = os.path.join(MODELS_DIR, f"rf_model_{ticker}.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"rf_scaler_{ticker}.joblib")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            return joblib.load(model_path), joblib.load(scaler_path)
    else:
        model_path = os.path.join(MODELS_DIR, f"lstm_model_{ticker}.h5")
        scaler_path = os.path.join(MODELS_DIR, f"lstm_scaler_{ticker}.joblib")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            return tf.keras.models.load_model(model_path), joblib.load(scaler_path)
    return None, None


def predict_series(model_type: str, model, scaler: MinMaxScaler, close_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = scaler.transform(close_prices.reshape(-1, 1)).flatten()
    if model_type == "RandomForest":
        X, y_true = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled)):
            X.append(scaled[i - SEQUENCE_LENGTH:i])
            y_true.append(scaled[i])
        X = np.array(X)
        preds_scaled = model.predict(X)
    else:
        X, y_true = create_sequences(scaled, SEQUENCE_LENGTH)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        preds_scaled = model.predict(X, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
    return y_true, preds


def forecast_future(model_type: str, model, scaler: MinMaxScaler, close_prices: np.ndarray, days: int = 7) -> np.ndarray:
    scaled = scaler.transform(close_prices.reshape(-1, 1)).flatten()
    window = list(scaled[-SEQUENCE_LENGTH:])
    future_scaled = []
    for _ in range(days):
        if model_type == "RandomForest":
            x = np.array(window[-SEQUENCE_LENGTH:]).reshape(1, -1)
            next_scaled = float(model.predict(x)[0])
        else:
            x = np.array(window[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
            next_scaled = float(model.predict(x, verbose=0).flatten()[0])
        future_scaled.append(next_scaled)
        window.append(next_scaled)
    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    return future


def plot_candlestick(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles")])
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50", line=dict(color="orange")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI14", line=dict(color="purple")))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI", template="plotly_white")
    return fig


def plot_actual_vs_predicted(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted", line=dict(color="teal")))
    fig.update_layout(title="Actual vs Predicted Close", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    return fig


def main():
    st.set_page_config(page_title="Stock Price Prediction", layout="wide")
    st.title("Stock Price Prediction using LSTM and RandomForest")

    # Sidebar inputs
    st.sidebar.header("Controls")
    ticker = st.sidebar.text_input("Stock Ticker", value=DEFAULT_TICKER)
    default_start = datetime(2015, 1, 1)
    default_end = datetime.today()
    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=datetime(1990, 1, 1), max_value=default_end)
    end_date = st.sidebar.date_input("End Date", value=default_end, min_value=start_date, max_value=default_end)
    model_choice = st.sidebar.selectbox("Model", ["LSTM", "RandomForest"], index=0)

    col_train, col_predict = st.sidebar.columns(2)
    train_clicked = col_train.button("Train Model")
    predict_clicked = col_predict.button("Predict Future")

    # Data section
    with st.spinner("Downloading data..."):
        df = download_data(ticker, start_date.strftime("%Y-%m-%d"), (end_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    if df.empty:
        st.error("No data found for the given ticker/date range.")
        return
    df = compute_indicators(df)
    save_default_csv(df)

    st.subheader(f"{ticker} Price and Indicators")
    st.plotly_chart(plot_candlestick(df, f"{ticker} Candlestick with MAs"), use_container_width=True)
    st.plotly_chart(plot_rsi(df), use_container_width=True)

    # Model paths and data prep
    close_prices = df["Close"].values.astype(float)
    available_for_sequences = len(close_prices) - SEQUENCE_LENGTH
    if available_for_sequences < 30:
        st.warning("Not enough data to train/predict (need at least 90 days).")
        return

    # Training
    model_key = "RandomForest" if model_choice == "RandomForest" else "LSTM"
    if train_clicked:
        with st.spinner(f"Training {model_key} model..."):
            if model_key == "RandomForest":
                model, scaler = train_random_forest(close_prices)
            else:
                model, scaler = train_lstm(close_prices)
            save_models(model_key, model, scaler, ticker)
        st.success(f"{model_key} model trained and saved.")

    # Load model (post-train or for prediction)
    model, scaler = load_models(model_key, ticker)
    if model is None or scaler is None:
        st.info("No saved model found for this ticker and model type. Click 'Train Model' to train.")
        return

    # Backtest-style prediction over historical period
    y_true, y_pred = predict_series(model_key, model, scaler, close_prices)
    pred_dates = df["Date"].iloc[SEQUENCE_LENGTH:]
    st.subheader("Model Performance (Historical)")
    st.plotly_chart(plot_actual_vs_predicted(pred_dates, y_true, y_pred), use_container_width=True)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    st.caption(f"RMSE: {rmse:,.4f}")

    # Future forecast
    if predict_clicked:
        with st.spinner("Generating 7-day forecast..."):
            future = forecast_future(model_key, model, scaler, close_prices, days=7)
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(7)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future})

        st.subheader("7-Day Forecast")
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="History", line=dict(color="gray")))
        fig_future.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], name="Forecast", line=dict(color="crimson")))
        fig_future.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_future, use_container_width=True)

        st.dataframe(forecast_df, use_container_width=True)
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv_bytes, file_name=f"{ticker}_forecast.csv", mime="text/csv")


if __name__ == "__main__":
    main()


