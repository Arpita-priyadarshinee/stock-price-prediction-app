from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Dict
from .indicators import add_indicators

def fetch_data(ticker: str, period: str = "8y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}. Check symbol/period/interval.")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def build_features(df: pd.DataFrame, lookback_lags: int = 5) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    '''
    Creates supervised learning features with technical indicators and lagged closes.
    Target is next day's Close.
    Returns:
      X, y, prev_close_for_direction, feature_names
    '''
    df_feat = add_indicators(df)

    # Lag features of Close (t-1..t-n)
    for lag in range(1, lookback_lags + 1):
        df_feat[f"Close_lag_{lag}"] = df_feat['Close'].shift(lag)

    # Drop rows with NaNs introduced by indicators/lags
    df_feat = df_feat.dropna().copy()

    # Target: next day's close
    df_feat['y_next_close'] = df_feat['Close'].shift(-1)

    # prev close for direction accuracy computation
    df_feat['prev_close'] = df_feat['Close']

    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c not in ['y_next_close', 'prev_close']]
    X = df_feat[feature_cols]
    y = df_feat['y_next_close']
    prev_close = df_feat['prev_close']

    return X, y, prev_close, feature_cols

def train_test_split_time(X: pd.DataFrame, y: pd.Series, prev_close: pd.Series, test_size: float = 0.2):
    n = len(X)
    n_test = int(n * test_size)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    prev_train, prev_test = prev_close.iloc[:-n_test], prev_close.iloc[-n_test:]
    return X_train, X_test, y_train, y_test, prev_train, prev_test

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, prev_close: pd.Series) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (np.abs((y_true - y_pred) / y_true).replace([np.inf, -np.inf], np.nan)).dropna().mean() * 100

    # Directional accuracy: did we get the sign of change right?
    true_dir = np.sign(y_true.values - prev_close.values)
    pred_dir = np.sign(y_pred - prev_close.values)
    dir_acc = (true_dir == pred_dir).mean() * 100.0
    return {"RMSE": float(rmse), "MAPE_%": float(mape), "Directional_Accuracy_%": float(dir_acc)}

def recursive_forecast(df_original: pd.DataFrame, model, feature_names, steps: int = 5, lookback_lags: int = 5):
    '''
    Forecast next N days by appending predictions and recomputing features.
    '''
    df_work = df_original.copy()
    preds = []
    dates = []
    last_date = df_work.index[-1]

    for i in range(steps):
        # Recompute features on the fly
        X_all, y_all, prev_all, feats = build_features(df_work, lookback_lags=lookback_lags)
        # Take the last available row as the input for the next prediction
        X_last = X_all.iloc[[-1]][feature_names]
        next_pred = float(model.predict(X_last)[0])
        preds.append(next_pred)
        # Append a new row with predicted close (approx approach: carry ffill other cols)
        new_date = last_date + pd.tseries.offsets.BDay(1)
        last_row = df_work.iloc[-1].copy()
        last_row['Close'] = next_pred
        df_work.loc[new_date] = last_row
        dates.append(new_date)
        last_date = new_date

    return pd.Series(preds, index=pd.DatetimeIndex(dates), name="Predicted_Close")