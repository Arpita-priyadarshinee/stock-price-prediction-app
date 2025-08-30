import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor

from utils.data import fetch_data, build_features, train_test_split_time, compute_metrics, recursive_forecast

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction – End-to-End App")

with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Example: AAPL, MSFT, TSLA, INFY.NS, TCS.NS, RELIANCE.NS")
    period = st.selectbox("History Period", options=["2y", "5y", "8y", "max"], index=2)
    interval = st.selectbox("Interval", options=["1d", "1h", "1wk"], index=0)
    lookback_lags = st.slider("Lag Features (days)", min_value=3, max_value=15, value=5, step=1)
    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    n_estimators = st.slider("RF Trees", min_value=100, max_value=800, value=400, step=50)
    steps_ahead = st.slider("Forecast Days Ahead", min_value=1, max_value=10, value=5)
    use_lstm = st.checkbox("Enable LSTM (optional, requires TensorFlow)")

    st.caption("💡 Tip: If deployment struggles, keep LSTM off and use RandomForest.")

@st.cache_data(show_spinner=False)
def cached_fetch(ticker, period, interval):
    return fetch_data(ticker, period=period, interval=interval)

def load_saved_model(ticker: str):
    model_path = os.path.join("models", f"{ticker}_rf.joblib")
    meta_path = os.path.join("models", f"{ticker}_meta.json")
    if os.path.exists(model_path):
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        lookback_lags_saved = bundle.get("lookback_lags", 5)
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return model, feature_names, lookback_lags_saved, meta
    return None, None, None, None

def save_model(ticker: str, model, feature_names, lookback_lags, metrics, period, interval):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{ticker}_rf.joblib")
    meta_path = os.path.join("models", f"{ticker}_meta.json")
    joblib.dump({"model": model, "feature_names": feature_names, "lookback_lags": lookback_lags}, model_path)
    with open(meta_path, "w") as f:
        json.dump({"ticker": ticker, "period": period, "interval": interval, "metrics": metrics}, f, indent=2)

tab1, tab2, tab3 = st.tabs(["📊 Data & EDA", "🧠 Train & Evaluate", "🔮 Forecast & Visualize"])

with tab1:
    st.subheader("Download & Preview")
    try:
        df = cached_fetch(ticker, period, interval)
        st.write(f"Data shape: {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.tail(10))
        st.line_chart(df['Close'])
        st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

with tab2:
    st.subheader("Feature Engineering")
    with st.spinner("Building features..."):
        X, y, prev_close, feature_names = build_features(df, lookback_lags=lookback_lags)
    st.write(f"Features shape: {X.shape}")
    feature_names = [str(f) for f in feature_names]

    st.code(
       "Features used ({}):\n{}".format(len(feature_names), ", ".join(feature_names)),
       language="markdown"
    )
    X_train, X_test, y_train, y_test, prev_train, prev_test = train_test_split_time(X, y, prev_close, test_size=test_size)

    model = None
    metrics = None

    colA, colB = st.columns(2)
    with colA:
        if st.button("Train RandomForest", use_container_width=True):
            with st.spinner("Training RandomForest..."):
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred, prev_test)
                st.success(f"Trained! RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE_%']:.2f}%, DirAcc={metrics['Directional_Accuracy_%']:.1f}%")

                save_model(ticker, model, feature_names, lookback_lags, metrics, period, interval)

    with colB:
        if st.button("Load Saved Model", use_container_width=True):
            model, feature_names_saved, lookback_lags_saved, meta = load_saved_model(ticker)
            if model is None:
                st.warning("No saved model found. Train first.")
            else:
                st.success(f"Loaded saved model for {ticker}.")
                st.json(meta)

    if use_lstm:
        st.info("LSTM option is available via a separate script to keep the app responsive. See README for notes.")
        st.code("# Example skeleton (run locally if TensorFlow installed):\n"
                "# from tensorflow.keras.models import Sequential\n"
                "# from tensorflow.keras.layers import LSTM, Dense, Dropout\n"
                "# ... (prepare 3D sequences) ...\n"
                "# model = Sequential([...])\n"
                "# model.fit(...)\n"
                "# model.save('models/<ticker>_lstm.h5')", language="python")

with tab3:
    st.subheader("Predict & Visualize")
    model_loaded, feature_names_loaded, lookback_lags_saved, meta = load_saved_model(ticker)
    if model_loaded is None:
        st.warning("Train or load a model first in the previous tab.")
        st.stop()

    # Rebuild features to match saved lookback
    X_all, y_all, prev_all, feats_all = build_features(df, lookback_lags=lookback_lags_saved)

    # 🔧 Ensure column names are flat strings
    X_all.columns = [str(c) for c in X_all.columns]

    # Also flatten saved feature names (in case they were tuples)
    feature_names_loaded = [str(f) for f in feature_names_loaded]

    test_fraction = test_size
    n = len(X_all)
    n_test = int(n * test_fraction)
    X_test_all = X_all.iloc[-n_test:][feature_names_loaded]

    y_test_all = y_all.iloc[-n_test:]
    prev_test_all = prev_all.iloc[-n_test:]

    y_pred_all = model_loaded.predict(X_test_all)
    met = compute_metrics(y_test_all, y_pred_all, prev_test_all)
    st.write("**Test Metrics (Saved Model):**")
    st.json(met)

    # Plot actual vs predicted for test segment
    fig1, ax1 = plt.subplots()
    ax1.plot(y_test_all.index, y_test_all.values, label="Actual Close")
    ax1.plot(y_test_all.index, y_pred_all, label="Predicted Close")
    ax1.set_title("Test Set: Actual vs Predicted Close")
    ax1.legend()
    st.pyplot(fig1)

    # Multi-step forecast
    with st.spinner(f"Forecasting next {steps_ahead} business days..."):
        fc_series = recursive_forecast(df, model_loaded, feature_names_loaded, steps=steps_ahead, lookback_lags=lookback_lags_saved)

    st.write("**Next Days Forecast:**")
    st.dataframe(fc_series.to_frame())

    # Plot historical + forecast
    combined = pd.concat([df['Close'].iloc[-200:], fc_series])
    fig2, ax2 = plt.subplots()
    ax2.plot(combined.index, combined.values, label="Close / Forecast")
    ax2.axvline(df.index[-1], linestyle="--", label="Forecast Start")
    ax2.set_title("Historical Close with Forecast Overlay")
    ax2.legend()
    st.pyplot(fig2)

st.caption("⚠️ Educational project. Not financial advice.")