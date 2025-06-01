import pandas as pd
import numpy as np
import talib

# ========================
# 1. Individual Signal Functions
# ========================
# RSI动量
def rsi_momentum(close, window=14, overbought=70, oversold=30):
    rsi = talib.RSI(close, window)
    return np.where(rsi > overbought, -1, np.where(rsi < oversold, 1, 0))

# MACD动量
def macd_momentum(close, fast=12, slow=26, signal=9):
    macd, signal, _ = talib.MACD(close, fast, slow, signal)
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))

# CCI动量
def cci_momentum(high, low, close, window=20, threshold=100):
    cci = talib.CCI(high, low, close, window)
    return np.where(cci > threshold, -1, np.where(cci < -threshold, 1, 0))

# 随机振荡器(Stochastic Oscillator)
def stochastic_momentum(high, low, close, k_window=14, d_window=3):
    slowk, slowd = talib.STOCH(high, low, close, k_window, d_window)
    return np.where(slowk > slowd, 1, -1)

# 价格动量
def price_momentum(close, lookback=20):
    ret = close.pct_change(lookback)
    return np.sign(ret)

# 量价突破
def volume_breakout(close, volume, window=20, multiplier=2):
    vol_ma = volume.rolling(window).mean()
    close_ma = close.rolling(window).mean()
    signal = (close > close_ma) & (volume > multiplier * vol_ma)
    return signal.astype(int)

# OBV能量潮
def obv_strategy(close, volume):
    obv = talib.OBV(close, volume)
    return np.where(obv > obv.rolling(20).mean(), 1, -1)

# 成交量加权MACD
def volume_weighted_macd(close, volume, fast=12, slow=26):
    vwma_fast = (close * volume).rolling(fast).sum() / volume.rolling(fast).sum()
    vwma_slow = (close * volume).rolling(slow).sum() / volume.rolling(slow).sum()
    return np.where(vwma_fast > vwma_slow, 1, -1)

# ATR通道突破
def atr_breakout(high, low, close, window=14, multiplier=2):
    atr = talib.ATR(high, low, close, window)
    upper = close.rolling(window).mean() + multiplier * atr
    lower = close.rolling(window).mean() - multiplier * atr
    return np.where(close > upper, 1, np.where(close < lower, -1, 0))

# 布林带收缩
def bollinger_squeeze(close, window=20, std_dev=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    bandwidth = (ma + std_dev*std) - (ma - std_dev*std)
    return (bandwidth / ma).rank(pct=True) < 0.2  # 识别波动率极低时期

# 波动率聚类策略
def volatility_regime(close, short_window=10, long_window=50):
    short_vol = close.pct_change().rolling(short_window).std()
    long_vol = close.pct_change().rolling(long_window).std()
    return (short_vol > long_vol).astype(int)  # 高波动 regime=1

# 三重EMA
def triple_ema(close, short=5, medium=20, long=50):
    ema1 = talib.EMA(close, short)
    ema2 = talib.EMA(close, medium)
    ema3 = talib.EMA(close, long)
    return ((ema1 > ema2) & (ema2 > ema3)).astype(int)

# ADX趋势强度
def adx_trend(high, low, close, window=14, threshold=25):
    adx = talib.ADX(high, low, close, window)
    return (adx > threshold).astype(int)

# 抛物线SAR
def sar_strategy(high, low, acceleration=0.02, maximum=0.2):
    sar = talib.SAR(high, low, acceleration, maximum)
    return (close > sar).astype(int)

def multi_pattern(open_p, high_p, low_p, close_p):
    patterns = {
        'hammer': talib.CDLHAMMER(open_p, high_p, low_p, close_p),
        'engulfing': talib.CDLENGULFING(open_p, high_p, low_p, close_p),
        'doji': talib.CDLDOJI(open_p, high_p, low_p, close_p)
    }
    df = pd.DataFrame(patterns)
    
    # Take the pattern with the **largest absolute value** each row
    dominant_signal = df.apply(lambda row: row[row.abs().idxmax()], axis=1)
    
    # Normalize to {+1, 0, -1}
    return np.sign(dominant_signal)

# ========================
# 2. Signal Aggregator
# ========================
def generate_all_signals(ohlcv_data):
    signals = pd.DataFrame(index=ohlcv_data.index)
    
    # 动量类
    signals['rsi'] = rsi_momentum(ohlcv_data['close'])
    signals['macd'] = macd_momentum(ohlcv_data['close'])
    signals['cci'] = cci_momentum(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
    
    # 成交量类
    signals['vol_break'] = volume_breakout(ohlcv_data['close'], ohlcv_data['volume'])
    signals['obv'] = obv_strategy(ohlcv_data['close'], ohlcv_data['volume'])
    
    # 波动率类
    signals['atr_break'] = atr_breakout(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
    
    # 趋势类
    signals['triple_ema'] = triple_ema(ohlcv_data['close'])
    
    # K线形态
    signals['candle_pattern'] = multi_pattern(
        ohlcv_data['open'], ohlcv_data['high'],
        ohlcv_data['low'], ohlcv_data['close'])
    
    return signals

# ========================
# 3. Feature Generator
# ========================
def create_ml_features(signals, price_data):
    features = signals.copy()
    
    # 添加交互特征
    features['momentum_ensemble'] = signals[['rsi','macd','cci']].mean(axis=1)
    features['vol_trend_interaction'] = signals['vol_break'] * signals['triple_ema']
    
    # 添加统计特征
    for col in signals.columns:
        features[f'{col}_zscore'] = (signals[col] - signals[col].rolling(30).mean()) / signals[col].rolling(30).std()
    
    # 添加滞后特征
    for lag in [1, 3, 5]:
        for col in ['momentum_ensemble', 'atr_break']:
            features[f'{col}_lag{lag}'] = features[col].shift(lag)
    
    return features.dropna()