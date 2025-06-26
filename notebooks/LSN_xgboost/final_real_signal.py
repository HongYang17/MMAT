import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import os
import numpy as np
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from binance.client import Client
from config.load_env import load_keys

from binance.client import Client
#sys.path.append(os.path.abspath(".."))  # root /PycharmProjects/MMAT
from config.load_env import load_keys

try:
    vol_model = joblib.load("volatility_model.pkl")
    vol_features = joblib.load("volatility_features.pkl")
    print(f"Loaded volatility model with {len(vol_features)} features")
except Exception as e:
    print(f"Error loading volatility model: {e}")
    vol_model = None
    vol_features = []


MODEL_PATH = 'final_signal_model.pkl'
FEATURE_NAMES_PATH = 'selected_features.pkl'


try:
    xgb_model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURE_NAMES_PATH)
    print(f"Loaded XGBoost model with {len(selected_features)} features")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None
    selected_features = []

keys = load_keys()
client = Client(keys['api_key'], keys['secret_key'])

from binance.client import Client
from dotenv import load_dotenv
try:
    from config.load_env import load_keys
except ImportError:
    # Fallback if config.load_env is unavailable
    def load_keys():
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        if api_key and api_secret:
            return {'api_key': api_key, 'secret_key': api_secret}
        raise ValueError("No API keys found. Set BINANCE_API_KEY and BINANCE_SECRET_KEY in environment or .env file.")

def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        print(f"Total K-lines loaded: {len(df)}")
        print("First 5 rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"CSV file '{csv_path}' not found.")
        return None

def fetch_binance_data(api_key, api_secret, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000):

    try:
        client = Client(api_key, api_secret)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Singapore')  # Set to Singapore Time
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        print(f"Fetched {len(df)} K-lines from Binance API:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return None

def calculate_basic_indicators(df):

    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], 12, 26, 9)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)

    # Trend-Following
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA200'] = talib.EMA(df['close'], timeperiod=200)

    # Volume-Based
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['Volume_MA'] = talib.SMA(df['volume'], timeperiod=20)

    # Volatility
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
    df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=20)

    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['Volume_MA20'] = talib.SMA(df['volume'], timeperiod=20)
    df['mean_ATR'] = df['ATR'].rolling(20).mean()

    df.dropna(inplace=True)
    return df


def calculate_patterns(df):
    """
    Detect candlestick patterns and assign TA-Lib raw outputs for ±100 signals.
    Also generates Signal_ columns that map strong bullish (1), strong bearish (-1), and neutral (0).
    """

    import talib
    import numpy as np

    patterns = {
        'Hammer': talib.CDLHAMMER,
        'InvertedHammer': talib.CDLINVERTEDHAMMER,
        'BullishEngulfing': talib.CDLENGULFING,
        'PiercingLine': talib.CDLPIERCING,
        'MorningStar': talib.CDLMORNINGSTAR,
        'DragonflyDoji': talib.CDLDRAGONFLYDOJI,
        'LongLine': talib.CDLLONGLINE,
        'ThreeLineStrike': talib.CDL3LINESTRIKE,

        'HangingMan': talib.CDLHANGINGMAN,
        'ShootingStar': talib.CDLSHOOTINGSTAR,
        'BearishEngulfing': talib.CDLENGULFING,
        'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
        'EveningDojiStar': talib.CDLEVENINGDOJISTAR,
        'EveningStar': talib.CDLEVENINGSTAR,
        'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
    }

    for name, func in patterns.items():
        df[name] = func(df['open'], df['high'], df['low'], df['close'])

    bullish_patterns_strong = ['Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
                               'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike']
    bearish_patterns_strong = ['HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
                               'EveningDojiStar', 'EveningStar', 'GravestoneDoji']

    for name in patterns.keys():
        if name == 'GravestoneDoji':
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == 100 else 0)
        elif name in bullish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x == 100 else 0)
        elif name in bearish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == -100 else 0)
        else:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    signal_cols = [f'Signal_{name}' for name in patterns.keys()]
    df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == 1), axis=1)
    df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == -1), axis=1)
    df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

    return df

def calculate_additional_features(df):

    features = {}

    features['close_to_high'] = (df['high'] - df['close']) / df['high']
    features['close_to_low'] = (df['close'] - df['low']) / df['close']
    features['price_range'] = (df['high'] - df['low']) / df['close']
    features['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)
    features['price_change'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()
    features['volume_ratio'] = df['volume'] / df['Volume_MA']
    features['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
    features['macd_hist'] = df['MACD'] - df['MACD_signal']
    features['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close']
    features['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close']
    features['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])

    for col in ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']:
        for lag in [1, 2, 3, 5, 10]:
            features[f'{col}_lag{lag}'] = df[col].shift(lag)

    for col in ['RSI', 'MACD', 'ATR', 'volume', 'close']:
        features[f'{col}_pct_change'] = df[col].pct_change()

    features['macd_histogram'] = df['MACD'] - df['MACD_signal']
    features['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)

    df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    for col in ['net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']:
        if col not in df.columns:
            df[col] = 0

    df.dropna(inplace=True)
    return df

def should_predict(df):

    if vol_model is None:
        print("Volatility model not loaded, default allows prediction")
        return True  # fallback: always predict

    try:
        df = calculate_basic_indicators(df)
        df = calculate_patterns(df)
        df = calculate_additional_features(df)

        i = len(df) - 2  # 倒数第二根K线（最后一根是未闭合的）
        if i < 0:
            return False

        vol_input = df.loc[[df.index[i]], vol_features].copy()
        if vol_input.isnull().any().any():
            print("The volatility model feature contains NaN, so skip this round of prediction")
            return False

        pred = vol_model.predict(vol_input)[0]
        proba = vol_model.predict_proba(vol_input)[0][1]
        print(f"Volatility model prediction: {pred} (p={proba:.2%})")

        return pred == 1  # 只有预测为“有波动”才允许后续预测
    except Exception as e:
        print(f"Incorrect volatility assessment: {e}")
        return False




def generate_xgboost_signals(df, signal_logger=None, confidence_threshold=0.85):

    if xgb_model is None:
        print("XGBoost model not loaded. Skipping signal generation.")
        return df

    # 初始化信号列
    df['xgboost_signal'] = 0
    df['xgboost_direction'] = 'NONE'
    df['xgboost_confidence'] = 0.0

    # 特征计算
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)
    df = calculate_additional_features(df)

    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features for XGBoost model: {missing_features}")
        return df

    i = len(df) - 2
    if i < 0 or i >= len(df):
        print("Not enough valid rows for prediction after feature engineering.")
        return df

    try:
        features_df = df.loc[[df.index[i]], selected_features].copy()

        if features_df.isnull().any().any():
            nan_cols = features_df.columns[features_df.isnull().any()].tolist()
            print(f"Skipping prediction: NaN found in features at index {df.index[i]} in columns: {nan_cols}")
            return df

        prediction_raw = xgb_model.predict(features_df)[0]
        proba = xgb_model.predict_proba(features_df)[0][1]

        if proba >= confidence_threshold:
            prediction = 1  # 看涨
        elif proba <= (1 - confidence_threshold):
            prediction = 0  # 看跌
        else:
            prediction = -1  # 不确定

        if prediction == 1:
            df.loc[df.index[i], 'xgboost_signal'] = 1
            df.loc[df.index[i], 'xgboost_direction'] = 'UP'
            df.loc[df.index[i], 'xgboost_confidence'] = proba
            print(f"XGBoost prediction at {df.index[i]}: UP (Confidence: {proba:.2%})")
            if signal_logger:
                signal_logger.add_signal('xgboost_bullish', df.index[i], df['close'].iloc[i],
                                         trigger=f"Confidence: {proba:.2%}")

        elif prediction == 0:
            df.loc[df.index[i], 'xgboost_signal'] = -1
            df.loc[df.index[i], 'xgboost_direction'] = 'DOWN'
            df.loc[df.index[i], 'xgboost_confidence'] = 1 - proba
            print(f"XGBoost prediction at {df.index[i]}: DOWN (Confidence: {(1 - proba):.2%})")
            if signal_logger:
                signal_logger.add_signal('xgboost_bearish', df.index[i], df['close'].iloc[i],
                                         trigger=f"Confidence: {(1 - proba):.2%}")

        else:
            print(f"Skipped low-confidence prediction at {df.index[i]} (Confidence: {proba:.2%})")

    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        import traceback
        traceback.print_exc()

    return df


def evaluate_patterns(df, patterns_dict, window=1, threshold=0.001):
    """
    Evaluate the accuracy of each candlestick pattern signal.

    Measures forward return after each signal and compares it against a defined threshold.

    Parameters:
    - df : DataFrame with candlestick signals
    - patterns_dict : dict of candlestick patterns used (as defined in calculate_patterns)
    - window : int, number of bars to look ahead
    - threshold : float, min return for a signal to be considered correct
    """
    results = {}

    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    for name in patterns_dict.keys():
        signal_col = f'Signal_{name}'
        if signal_col not in df.columns:
            continue

        signals = df[df[signal_col] != 0]
        total_signals = len(signals)

        if total_signals == 0:
            results[name] = {'accuracy': 0, 'total_signals': 0, 'correct_signals': 0}
            continue

        correct_signals = len(signals[
            ((signals[signal_col] == 1) & (signals['return'] >= threshold)) |
            ((signals[signal_col] == -1) & (signals['return'] <= -threshold))
        ])
        accuracy = correct_signals / total_signals * 100

        results[name] = {
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': correct_signals
        }

    return results

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signal_logger import SignalHistoryLogger


def bootstrap_accuracy_pvalue(correct_seq, baseline=0.55, B=1000, block_len=20, seed=42):
    """
    用 Block Bootstrap 检验预测准确率是否显著优于 baseline（默认是 50%）。

    Parameters:
        correct_seq: np.array 或 list，二元准确性序列（1=预测正确，0=错误）
        baseline: 随机基线准确率（如 0.5）
        B: 重采样次数
        block_len: 区块长度（控制时间相关性）
        seed: 随机种子

    Returns:
        p-value, 实际准确率, bootstrap 平均准确率
    """
    import numpy as np
    np.random.seed(seed)
    correct_seq = np.array(correct_seq)
    n = len(correct_seq)
    indices = np.arange(n)
    boot_means = []

    for _ in range(B):
        sample_idx = []
        nb = int(np.ceil(n / block_len))
        for _ in range(nb):
            start = np.random.randint(0, n - block_len + 1)
            sample_idx.extend(indices[start: start + block_len])
        sample_idx = sample_idx[:n]
        boot_means.append(correct_seq[sample_idx].mean())

    acc = correct_seq.mean()
    p_value = np.mean([acc <= m for m in boot_means])
    return p_value, acc, np.mean(boot_means)


def plot_realtime_signals(df, symbol='BTCUSDT', data_range=50, output_dir=r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\plots', signal_logger=None):
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if signal_logger is None:
        raise ValueError("signal_logger must be provided")

    df_plot = df.iloc[-data_range:].copy()

    # Latest forming candle hover text
    df_plot['hover_text'] = np.where(
        df_plot.index == df.index[-1],
        ' Latest forming candle (not evaluated)',
        ''
    )

    print(f"Plotting real-time chart for last {data_range} candles")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Candlestick + MA', 'RSI', 'ATR', 'Volume'],
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # === Candlestick + MAs ===
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA5'], mode='lines', name='5 MA', line=dict(color='blue')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA20'], mode='lines', name='20 MA', line=dict(color='purple')
    ), row=1, col=1)

    # === Forming candle marker ===
    forming_candle = df_plot[df_plot.index == df.index[-1]]
    if not forming_candle.empty:
        fig.add_trace(go.Scatter(
            x=forming_candle.index,
            y=forming_candle['close'] * 1.002,
            mode='markers',
            name='Forming Candle',
            marker=dict(symbol='circle', color='gray', size=8, opacity=0.3),
            text=forming_candle['hover_text'],
            hoverinfo='text+x+y',
            hoverlabel=dict(bgcolor='lightgray'),
            showlegend=False
        ), row=1, col=1)

    # === Historical signals from logger ===
    signal_df = signal_logger.get_history()
    print(f"Signal history loaded: {len(signal_df)} records")

    signal_map = {
        'xgboost_bullish': ('triangle-up', 'green', 1.005),
        'xgboost_bearish': ('triangle-down', 'red', 0.995),
    }

    if not signal_df.empty:
        try:
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], utc=True)
            signal_df['timestamp'] = signal_df['timestamp'].dt.tz_convert(df_plot.index.tz)
        except Exception as e:
            print(f"Error converting signal timestamps: {e}")
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])

        signal_df = signal_df[signal_df['timestamp'] >= df_plot.index[0]]
        signal_df = signal_df.drop_duplicates(subset=['timestamp', 'type'])

        for _, row in signal_df.iterrows():
            sig_type = row['type']
            if sig_type in signal_map and row['timestamp'] in df_plot.index:
                symbol_shape, color, y_factor = signal_map[sig_type]
                trigger_text = (
                    f"{sig_type.replace('_', ' ').capitalize()}: {row['trigger']}"
                    if 'trigger' in row and pd.notna(row['trigger']) else sig_type.replace('_', ' ').capitalize()
                )
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['price'] * y_factor],
                    mode='markers',
                    marker=dict(symbol=symbol_shape, color=color, size=12),
                    name='',
                    text=[trigger_text],
                    hoverinfo='text+x+y',
                    showlegend=False
                ), row=1, col=1)

    # === RSI ===
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='blue')
    ), row=2, col=1)
    fig.add_hline(y=50, line_dash='dash', line_color='black', row=2, col=1)

    # === ATR ===
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['ATR'],
        mode='lines',
        name='ATR',
        line=dict(color='orange')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['mean_ATR'] * 1.2,
        mode='lines',
        name='1.2 * mean_ATR',
        line=dict(color='red', dash='dash')
    ), row=3, col=1)

    # === Volume ===
    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['volume'],
        name='Volume',
        marker_color='blue'
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['Volume_MA20'] * 1.3,
        mode='lines',
        name='1.3 * Volume_MA20',
        line=dict(color='red', dash='dash')
    ), row=4, col=1)

    fig.update_layout(
        title=f'[Testing] Real-Time 1 Min Signals for {symbol}',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=800,
        template='plotly_white'
    )

    # Save to HTML with auto-refresh
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f'realtime_signals_{symbol}.html')
    try:
        html_content = fig.to_html(include_plotlyjs='cdn')
        html_content = html_content.replace(
            '<head>',
            '<head><meta http-equiv="refresh" content="60">'
        )
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Updated real-time plot saved to: {html_path}")
    except Exception as e:
        print(f"Error saving HTML: {e}")

import atexit
from prediction_logger import PredictionLogger
from signal_logger import SignalHistoryLogger
import time
import talib
from binance.client import Client


def update_signal(signal_logger, signal_type, timestamp, price, confidence_str):
    """更新信号记录：删除反向信号，再添加新信号"""
    signal_logger.remove_opposite_signal(timestamp, signal_type)
    if not signal_logger.has_signal(timestamp, signal_type):
        signal_logger.add_signal(signal_type, timestamp, price, trigger=confidence_str)

# Initialize loggers
logger = PredictionLogger()
signal_logger = SignalHistoryLogger(filename=r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\signal_history.csv')

# Register atexit handlers to save logs on program exit
atexit.register(lambda: logger.save_to_csv("TestLive_prediction_log.csv"))
atexit.register(lambda: signal_logger.save_to_csv("signal_history.csv"))


def run_realtime_signals(api_key, api_secret, symbol='BTCUSDT',
                         interval=Client.KLINE_INTERVAL_1MINUTE,
                         limit=1000, sleep_seconds=60,
                         signal_logger=None, prediction_logger=None,
                         vol_model=None, vol_features=None,
                         xgb_model=None, selected_features=None,
                         confidence_threshold=0.85,
                         debug=False):
    """
    实时运行：先判断是否波动期 -> 再调用方向模型
    """

    if signal_logger is None or prediction_logger is None:
        raise ValueError("signal_logger and prediction_logger must be provided")
    if vol_model is None or vol_features is None:
        raise ValueError("volatility model and feature list must be loaded")
    if xgb_model is None or selected_features is None:
        raise ValueError("XGBoost model and feature list must be loaded")

    print("Starting real-time signal loop...")

    while True:
        try:
            df = fetch_binance_data(api_key, api_secret, symbol, interval, limit)
            if df is None or df.empty:
                print("No data fetched, retrying...")
                time.sleep(sleep_seconds)
                continue

            df = calculate_basic_indicators(df)
            df = calculate_patterns(df)
            df = calculate_additional_features(df)

            latest = df.iloc[[-2]].copy()
            if any(f not in latest.columns for f in vol_features):
                print("Missing features for volatility model, skipping...")
                continue

            if latest[vol_features].isnull().any().any():
                print("NaN in volatility features, skipping...")
                continue

            vol_pred = vol_model.predict(latest[vol_features])[0]
            is_active = vol_pred == 1

            prediction = "NEUTRAL"
            if is_active:
                print("fluctuating state: ON → Try generating directional predictions")
                features_df = latest[selected_features]
                if features_df.isnull().any().any():
                    print("NaN in main features, skipping...")
                    continue

                proba = xgb_model.predict_proba(features_df)[0][1]
                if proba >= confidence_threshold:
                    prediction = "UP"
                    update_signal(signal_logger, 'xgboost_bullish', latest.index[0], latest['close'].iloc[0],
                                  f"Confidence: {proba:.2%}")
                    print(f"XGBoost: ↑ (confidence level: {proba:.2%})")
                elif proba <= 1 - confidence_threshold:
                    prediction = "DOWN"
                    update_signal(signal_logger, 'xgboost_bearish', latest.index[0], latest['close'].iloc[0],
                                  f"Confidence: {(1 - proba):.2%}")
                    print(f"XGBoost: ↓ (confidence level: {(1 - proba):.2%})")
                else:
                    print(f"XGBoost insufficient confidence: {proba:.2%}")
            else:
                print("Currently in a low volatility state, skip trading decisions.")

            # 命中记录
            ts = df.index[-1]
            close_now = df['close'].iloc[-2]
            close_future = df['close'].iloc[-1]
            prediction_logger.record_prediction(ts, prediction, close_future, close_now)
            print(f"Current hit rate: {prediction_logger.get_hit_rate():.2%}")


            df_pred = prediction_logger.to_dataframe()
            if len(df_pred) >= 30:
                pval, acc, boot_mean = bootstrap_accuracy_pvalue(df_pred['hit'].values, baseline=0.55)
                print(f"Real-time significance detection: acc={acc:.2%}, p={pval:.4f} | baseline={boot_mean:.2%}")
                if pval < 0.05:
                    print("Significantly better than random predictions (p < 0.05)")


            plot_realtime_signals(df, symbol=symbol, data_range=50, signal_logger=signal_logger)

            print(f"Sleeping for {sleep_seconds} seconds...\n")
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("Stop real-time signal generation")
            break
        except Exception as e:
            print(f"Error in live loop: {e}")
            time.sleep(sleep_seconds)


import atexit
from prediction_logger import PredictionLogger
from signal_logger import SignalHistoryLogger
from binance.client import Client
import pandas as pd
import talib
import os

def main(realtime=True, debug=False):
    import atexit
    from signal_logger import SignalHistoryLogger
    from prediction_logger import PredictionLogger
    import joblib

    symbol = 'BTCUSDT'
    csv_path = r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\btc_1min.csv'
    signal_log_path = r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\signal_history.csv'
    pred_log_path = r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\TestLive_prediction_log.csv'

    signal_logger = SignalHistoryLogger(filename=signal_log_path)
    prediction_logger = PredictionLogger()

    atexit.register(lambda: signal_logger.save_to_csv(signal_log_path))
    atexit.register(lambda: prediction_logger.save_to_csv(pred_log_path))

    # ========== 加载模型 ==========
    try:
        xgb_model = joblib.load("final_signal_model.pkl")
        selected_features = joblib.load("selected_features.pkl")
        print(f"Loaded XGBoost model with {len(selected_features)} features")
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")
        xgb_model = None
        selected_features = []

    try:
        vol_model = joblib.load("volatility_model.pkl")
        vol_features = joblib.load("volatility_features.pkl")
        print(f"Loaded volatility model with {len(vol_features)} features")
    except Exception as e:
        print(f"Failed to load volatility model: {e}")
        vol_model = None
        vol_features = None


    try:
        keys = load_keys()
        api_key = keys['api_key']
        api_secret = keys['secret_key']
    except Exception as e:
        print(f"Failed to load API keys: {e}")
        return

    if realtime:
        # === 运行实盘 ===
        run_realtime_signals(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            signal_logger=signal_logger,
            prediction_logger=prediction_logger,
            vol_model=vol_model,
            vol_features=vol_features,
            xgb_model=xgb_model,
            selected_features=selected_features,
            confidence_threshold=0.85,
            debug=debug
        )
        return

    try:
        df = load_data(csv_path)
        if df is None or df.empty:
            print("Failed to load CSV data, exiting.")
            return
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return

    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)
    df = calculate_additional_features(df)
    df = generate_xgboost_signals(df, signal_logger=signal_logger)

    accuracy_results = evaluate_patterns(df)
    print("\n--- Pattern Signal Accuracy ---")
    for name, metrics in sorted(accuracy_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name} - Accuracy: {metrics['accuracy']:.2%}, Total: {metrics['total_signals']}, "
              f"Correct: {metrics['correct_signals']}")

    plot_realtime_signals(df, symbol, data_range=50, signal_logger=signal_logger)


if __name__ == "__main__":
    main(realtime=True, debug=False)
