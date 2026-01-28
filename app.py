import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Stock Market Tracker & Predictor (LSTM)")

# User input: multiple tickers
tickers = st.text_input("Enter Stock Tickers (comma separated, e.g., AAPL, TSLA, MSFT)", "AAPL, TSLA").split(",")
start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.now())

# Function to prepare data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

# Function to build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape,1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Loop through tickers
for ticker in tickers:
    ticker = ticker.strip()
    st.subheader(f"📊 {ticker} Stock Data")

    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write(data.tail())

    # Plot historical prices
    st.line_chart(data['Close'])

    # Prepare data
    X, y, scaler = prepare_data(data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train LSTM
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predictions
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    # Accuracy metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds))
    mae = mean_absolute_error(y_test_rescaled, preds)
    r2 = r2_score(y_test_rescaled, preds)

    st.write(f"✅ Accuracy Metrics for {ticker}:")
    st.write(f"- RMSE: {rmse:.2f}")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    # Future prediction (next 30 days)
    last_60 = data['Close'].values[-60:]
    scaled_last = scaler.transform(last_60.reshape(-1,1))
    X_future = [scaled_last[:,0]]
    X_future = np.array(X_future).reshape(1,60,1)

    future_preds = []
    for _ in range(30):
        pred = model.predict(X_future)
        future_preds.append(pred[0][0])
        X_future = np.append(X_future[:,1:,:], pred.reshape(1,1,1), axis=1)
        
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=30)

    # Plot predictions
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label="Historical")
    ax.plot(future_dates, future_preds, label="Predicted", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.write("📌 Prediction for next 30 days:")
    st.write(pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds.flatten()}))
