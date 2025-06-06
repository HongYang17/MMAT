import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import os
import numpy as np
import joblib  # 添加joblib用于加载模型
from binance.client import Client
from config.load_env import load_keys

from binance.client import Client
# sys.path.append(os.path.abspath(".."))  # root /PycharmProjects/MMAT
from config.load_env import load_keys

# 加载XGBoost模型和特征列表
MODEL_PATH = 'improved_signal_model.pkl'  # 替换为你的模型路径
FEATURE_NAMES_PATH = 'selected_features.pkl'  # 替换为你的特征列表路径

# 加载模型和特征
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
    """计算所有基础技术指标"""
    # Momentum & Oscillator
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

    # 清除NA值
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
        # Bullish
        'Hammer': talib.CDLHAMMER,
        'InvertedHammer': talib.CDLINVERTEDHAMMER,
        'BullishEngulfing': talib.CDLENGULFING,
        'PiercingLine': talib.CDLPIERCING,
        'MorningStar': talib.CDLMORNINGSTAR,
        'DragonflyDoji': talib.CDLDRAGONFLYDOJI,
        'LongLine': talib.CDLLONGLINE,
        'ThreeLineStrike': talib.CDL3LINESTRIKE,

        # Bearish
        'HangingMan': talib.CDLHANGINGMAN,
        'ShootingStar': talib.CDLSHOOTINGSTAR,
        'BearishEngulfing': talib.CDLENGULFING,
        'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
        'EveningDojiStar': talib.CDLEVENINGDOJISTAR,
        'EveningStar': talib.CDLEVENINGSTAR,
        'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
    }

    # 输出原始形态信号值（±100/0）
    for name, func in patterns.items():
        df[name] = func(df['open'], df['high'], df['low'], df['close'])

    # 定义形态分类
    bullish_patterns_strong = ['Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
                               'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike']
    bearish_patterns_strong = ['HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
                               'EveningDojiStar', 'EveningStar', 'GravestoneDoji']

    # Signal_列：标准化为 -1 / 0 / 1
    for name in patterns.keys():
        if name == 'GravestoneDoji':
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == 100 else 0)
        elif name in bullish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x == 100 else 0)
        elif name in bearish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == -100 else 0)
        else:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # 聚合信号：行内总 bullish、bearish 形态个数
    signal_cols = [f'Signal_{name}' for name in patterns.keys()]
    df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == 1), axis=1)
    df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == -1), axis=1)
    df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

    return df

def calculate_additional_features(df):
    """整合所有所需衍生特征，避免碎片化写入"""

    features = {}

    # === 基础衍生特征 ===
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

    # === 滞后特征 ===
    for col in ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']:
        for lag in [1, 2, 3, 5, 10]:
            features[f'{col}_lag{lag}'] = df[col].shift(lag)

    # === 变化率特征 ===
    for col in ['RSI', 'MACD', 'ATR', 'volume', 'close']:
        features[f'{col}_pct_change'] = df[col].pct_change()

    # === 交叉特征 ===
    features['macd_histogram'] = df['MACD'] - df['MACD_signal']
    features['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)

    # 一次性 concat 进 DataFrame，避免碎片化
    df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    # 确保关键聚合信号列存在
    for col in ['net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']:
        if col not in df.columns:
            df[col] = 0

    # 删除缺失值
    df.dropna(inplace=True)
    return df


def generate_xgboost_signals(df, signal_logger=None):
    """
    使用XGBoost模型生成交易信号（增强版，包含诊断日志）
    """
    # 确保模型已加载
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

    # 检查缺失特征
    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        print(f"⚠️ Warning: Missing features for XGBoost model: {missing_features}")
        return df  # 跳过预测

    # 定位最后一根已完成K线
    i = len(df) - 2
    if i < 0 or i >= len(df):
        print("⚠️ Not enough valid rows for prediction after feature engineering.")
        return df

    try:
        # 取出最新一根的所有模型特征 - 保持为DataFrame格式
        features_df = df.loc[[df.index[i]], selected_features].copy()

        # 检查是否有 NaN
        if features_df.isnull().any().any():
            nan_cols = features_df.columns[features_df.isnull().any()].tolist()
            print(f"⚠️ Skipping prediction: NaN found in features at index {df.index[i]} in columns: {nan_cols}")
            print(features_df[nan_cols])
            return df

        # 执行模型预测 - 直接传递DataFrame
        prediction = xgb_model.predict(features_df)[0]
        proba = xgb_model.predict_proba(features_df)[0][1]  # 预测为上涨的概率

        print(f"✅ XGBoost prediction at {df.index[i]}: "
              f"{'UP' if prediction == 1 else 'DOWN'} "
              f"(Confidence: {proba:.2%})")

        # 记录到DataFrame
        if prediction == 1:
            df.loc[df.index[i], 'xgboost_signal'] = 1
            df.loc[df.index[i], 'xgboost_direction'] = 'UP'
            df.loc[df.index[i], 'xgboost_confidence'] = proba

            if signal_logger:
                trigger_text = f"XGBoost UP (Confidence: {proba:.2%})"
                signal_logger.add_signal('xgboost_bullish', df.index[i], df['close'].iloc[i], trigger=trigger_text)

        elif prediction == 0:
            df.loc[df.index[i], 'xgboost_signal'] = -1
            df.loc[df.index[i], 'xgboost_direction'] = 'DOWN'
            df.loc[df.index[i], 'xgboost_confidence'] = 1 - proba

            if signal_logger:
                trigger_text = f"XGBoost DOWN (Confidence: {1 - proba:.2%})"
                signal_logger.add_signal('xgboost_bearish', df.index[i], df['close'].iloc[i], trigger=trigger_text)

        else:
            print("⚠️ XGBoost output neither 0 nor 1 — interpreted as NEUTRAL")

    except Exception as e:
        print(f"❌ Error in XGBoost prediction: {e}")
        import traceback
        traceback.print_exc()

    return df


def evaluate_patterns(df, patterns_dict, window=5, threshold=0.001):
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

    # Compute forward return
    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    # Evaluate each pattern signal
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
            print(f"⚠️ Error converting signal timestamps: {e}")
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
        title=f'[Testing] Real-Time 5 Min Signals for {symbol}',
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
            '<head><meta http-equiv="refresh" content="300">'
        )
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"✅ Updated real-time plot saved to: {html_path}")
    except Exception as e:
        print(f"❌ Error saving HTML: {e}")

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
                         limit=1000, sleep_seconds=300,
                         signal_logger=None, prediction_logger=None, debug=False):

    if signal_logger is None or prediction_logger is None:
        raise ValueError("signal_logger and prediction_logger must be provided")

    print("Starting real-time signal generation. Press Ctrl+C to stop.")

    while True:
        try:
            # === 获取实时数据 ===
            df = fetch_binance_data(api_key, api_secret, symbol, interval, limit)
            if df is None or df.empty:
                print("Failed to fetch data. Retrying in 60 seconds.")
                time.sleep(60)
                continue

            # === 生成XGBoost信号 ===
            df = generate_xgboost_signals(df, signal_logger=signal_logger)

            # === 获取当前信号数据点（倒数第二根K线）===
            latest_ts = df.index[-2]
            close_now = df['close'].iloc[-2]
            xgboost_signal = df['xgboost_signal'].iloc[-2]
            xgboost_confidence = df['xgboost_confidence'].iloc[-2]

            # === 判断并记录信号 ===
            prediction = "NEUTRAL"
            if xgboost_signal == 1:
                prediction = "UP"
                print(f"\nXGBoost Signal: BULLISH (Confidence: {xgboost_confidence:.2%})")
                update_signal(signal_logger, 'xgboost_bullish', latest_ts, close_now,
                              confidence_str=f"Confidence: {xgboost_confidence:.2%}")

            elif xgboost_signal == -1:
                prediction = "DOWN"
                print(f"\nXGBoost Signal: BEARISH (Confidence: {(1 - xgboost_confidence):.2%})")
                update_signal(signal_logger, 'xgboost_bearish', latest_ts, close_now,
                              confidence_str=f"Confidence: {(1 - xgboost_confidence):.2%}")
            else:
                print("\nXGBoost Signal: NEUTRAL")

            print(f"Prediction for next 5min: {prediction}")

            # === 记录预测命中情况 ===
            if len(df) >= 2:
                ts = df.index[-1]  # 最新的未来点
                close_future = df['close'].iloc[-1]
                prediction_logger.record_prediction(ts, prediction, close_future, close_now)
                print(f"Current Hit Rate: {prediction_logger.get_hit_rate():.2%}")

            # === 可视化 ===
            plot_realtime_signals(df, symbol, data_range=50, signal_logger=signal_logger)

            print(f"Waiting {sleep_seconds} seconds for next update...\n")
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("\nStopped real-time signal generation. Exiting...")
            break
        except Exception as e:
            print(f"Error in real-time loop: {e}. Retrying in 60 seconds.")
            time.sleep(60)


import atexit
from prediction_logger import PredictionLogger
from signal_logger import SignalHistoryLogger
from binance.client import Client
import pandas as pd
import talib
import os

def main(realtime=True, debug=False):
    """
    Main function to run candlestick pattern analysis and signal generation.
    Supports both real-time data from Binance API and historical data from CSV.
    Generates signals, evaluates accuracy, and plots results.

    Args:
        realtime (bool): If True, use real-time Binance API data; else, use CSV data (default: True).
        debug (bool): If True, print detailed debugging information (default: False).
    """
    # Configuration
    use_api = realtime
    symbol = 'BTCUSDT'
    csv_path = r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\btc_1min.csv'
    signal_log_path = r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\signal_history.csv'


    # Initialize loggers
    signal_logger = SignalHistoryLogger(filename=signal_log_path)
    prediction_logger = PredictionLogger()

    # Register atexit handlers to save logs on program exit
    atexit.register(lambda: prediction_logger.save_to_csv(r'C:\Users\86159\Desktop\MQF\mqf635\final_groupwork\TestLive_prediction_log.csv'))
    atexit.register(lambda: signal_logger.save_to_csv(signal_log_path))

    if use_api:
        try:
            # Load Binance API keys
            keys = load_keys()
            api_key = keys['api_key']
            api_secret = keys['secret_key']
        except Exception as e:
            print(f"Failed to load API keys: {e}")
            return

        if realtime:
            # Run real-time signal generation
            run_realtime_signals(api_key, api_secret, symbol=symbol, signal_logger=signal_logger,
                                 prediction_logger=prediction_logger, debug=debug)
            return

        # Fetch historical data from Binance API
        df = fetch_binance_data(api_key, api_secret, symbol=symbol)
        if df is None or df.empty:
            print("Failed to fetch Binance data, exiting.")
            return
    else:
        # Load historical data from CSV
        try:
            df = load_data(csv_path)
            if df is None or df.empty:
                print("Failed to load CSV data, exiting.")
                return
            # Resample to 15-minute intervals (uncomment if needed)
            # df = resample_to_15min(df)
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return

            # 计算技术指标和XGBoost信号
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)  # patterns dict 可用于评估
    df = calculate_additional_features(df)
    df = generate_xgboost_signals(df, signal_logger=signal_logger)

    # Evaluate pattern accuracy
    accuracy_results = evaluate_patterns(df)
    print("\n--- Pattern Signal Accuracy ---")
    for name, metrics in sorted(accuracy_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name} - Accuracy: {metrics['accuracy']:.2f}%, Total: {metrics['total_signals']}, "
              f"Correct: {metrics['correct_signals']}")

    # Plot results
    plot_realtime_signals(df, symbol, data_range=50, signal_logger=signal_logger)

if __name__ == "__main__":
    main(realtime=True, debug=False)
