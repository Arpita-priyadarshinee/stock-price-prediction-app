import numpy as np
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period: int = 14):
    # Ensure close is 1D
    if isinstance(close, (pd.DataFrame, np.ndarray)):
        close = close.squeeze()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def rolling_volatility(close: pd.Series, window: int = 20):
    return close.pct_change().rolling(window=window, min_periods=window).std() * np.sqrt(252)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out['Close']

    out['SMA_10'] = sma(close, 10)
    out['SMA_20'] = sma(close, 20)
    out['EMA_10'] = ema(close, 10)
    out['EMA_20'] = ema(close, 20)

    out['RSI_14'] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    out['MACD'] = macd_line
    out['MACD_SIGNAL'] = signal_line
    out['MACD_HIST'] = hist

    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
    out['BB_UPPER'] = bb_u
    out['BB_MIDDLE'] = bb_m
    out['BB_LOWER'] = bb_l

    out['RET_1'] = close.pct_change(1)
    out['RET_5'] = close.pct_change(5)
    out['RET_10'] = close.pct_change(10)

    out['ROLL_VOL_20'] = rolling_volatility(close, 20)

    return out