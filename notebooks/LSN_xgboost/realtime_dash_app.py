import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import os
import numpy as np
import sys
import joblib
from binance.client import Client
import time
import datetime
import threading
import logging
import atexit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.load_env import load_keys



def load_models():
    models = {
        'vol_model': None,
        'vol_features': [],
        'xgb_model': None,
        'selected_features': []
    }

    try:
        models['vol_model'] = joblib.load("volatility_model.pkl")
        models['vol_features'] = joblib.load("volatility_features.pkl")
        logger.info(f"Loaded volatility model with {len(models['vol_features'])} features")
    except Exception as e:
        logger.error(f"Error loading volatility model: {e}")

    try:
        models['xgb_model'] = joblib.load('final_signal_model.pkl')
        models['selected_features'] = joblib.load('selected_features.pkl')
        logger.info(f"Loaded XGBoost model with {len(models['selected_features'])} features")
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")

    return models


models = load_models()


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
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        logger.info(f"Fetched {len(df)} K-lines from Binance API")
        return df
    except Exception as e:
        logger.error(f"Error fetching Binance data: {e}")
        return None


def calculate_basic_indicators(df):
    # 动量指标
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], 12, 26, 9)

    # 趋势指标
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)

    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['Volume_MA20'] = talib.SMA(df['volume'], timeperiod=20)
    df['mean_ATR'] = df['ATR'].rolling(20).mean()


    df.dropna(inplace=True)
    return df


def generate_signals(df, confidence_threshold=0.85):
    if models['xgb_model'] is None:
        logger.warning("XGBoost model not loaded. Skipping signal generation.")
        return df

    df['xgboost_signal'] = 0
    df['xgboost_direction'] = 'NONE'
    df['xgboost_confidence'] = 0.0

    i = len(df) - 2
    if i < 0 or i >= len(df):
        logger.warning("Not enough valid rows for prediction")
        return df

    try:
        features_df = df.loc[[df.index[i]], models['selected_features']].copy()

        if features_df.isnull().any().any():
            nan_cols = features_df.columns[features_df.isnull().any()].tolist()
            logger.warning(f"Skipping prediction: NaN found in features in columns: {nan_cols}")
            return df

        prediction_raw = models['xgb_model'].predict(features_df)[0]
        proba = models['xgb_model'].predict_proba(features_df)[0][1]

        if proba >= confidence_threshold:
            df.loc[df.index[i], 'xgboost_signal'] = 1
            df.loc[df.index[i], 'xgboost_direction'] = 'UP'
            df.loc[df.index[i], 'xgboost_confidence'] = proba
            logger.info(f"XGBoost prediction: UP (Confidence: {proba:.2%})")
        elif proba <= (1 - confidence_threshold):
            df.loc[df.index[i], 'xgboost_signal'] = -1
            df.loc[df.index[i], 'xgboost_direction'] = 'DOWN'
            df.loc[df.index[i], 'xgboost_confidence'] = 1 - proba
            logger.info(f"XGBoost prediction: DOWN (Confidence: {(1 - proba):.2%})")
        else:
            logger.info(f"Skipped low-confidence prediction (Confidence: {proba:.2%})")

    except Exception as e:
        logger.error(f"Error in XGBoost prediction: {e}")

    return df


class DataUpdater:
    def __init__(self, api_key, api_secret, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE,
                 update_interval=60):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.update_interval = update_interval
        self.df = pd.DataFrame()
        self.latest_signal = None
        self.running = False
        self.thread = None
        self.last_update = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _update_loop(self):
        while self.running:
            try:
                new_df = fetch_binance_data(self.api_key, self.api_secret, self.symbol, self.interval)
                if new_df is not None and not new_df.empty:

                    new_df = calculate_basic_indicators(new_df)


                    new_df = generate_signals(new_df)

                    self.df = new_df
                    self.last_update = datetime.datetime.now()

                    if not self.df.empty and 'xgboost_signal' in self.df.columns:
                        last_row = self.df.iloc[-2]  # 倒数第二根K线
                        if last_row['xgboost_signal'] != 0:
                            self.latest_signal = {
                                'timestamp': last_row.name,
                                'direction': last_row['xgboost_direction'],
                                'confidence': last_row['xgboost_confidence'],
                                'price': last_row['close']
                            }

                logger.info(f"Data updated at {self.last_update}")
            except Exception as e:
                logger.error(f"Error in update loop: {e}")

            time.sleep(self.update_interval)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "加密货币交易信号监控"

try:
    keys = load_keys()
    api_key = keys['api_key']
    api_secret = keys['secret_key']
except Exception as e:
    logger.error(f"Failed to load API keys: {e}")
    api_key = None
    api_secret = None

data_updater = DataUpdater(api_key, api_secret) if api_key and api_secret else None

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("加密货币实时交易信号", className="text-center my-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("系统状态", className="card-title"),
                    html.Div(id='system-status', className="mb-2"),
                    html.Div(id='last-update', className="mb-2"),
                    html.Div(id='latest-signal', className="mb-2"),
                    dbc.Button("启动监控", id='start-button', color="success", className="me-2"),
                    dbc.Button("停止监控", id='stop-button', color="danger"),
                ])
            ], className="mt-4")
        ], md=3),

        dbc.Col([
            dcc.Graph(id='live-graph', style={'height': '80vh'}),
            dcc.Interval(id='graph-update', interval=60 * 1000, n_intervals=0)  # 每分钟更新
        ], md=9),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("最近信号", className="card-title"),
                    html.Div(id='signal-history')
                ])
            ], className="mt-4")
        ], width=12)
    ]),

    dcc.Store(id='signal-store', data=[]),

], fluid=True)


@app.callback(
    Output('system-status', 'children'),
    Output('last-update', 'children'),
    Output('latest-signal', 'children'),
    Output('signal-store', 'data'),
    Input('graph-update', 'n_intervals'),
    State('signal-store', 'data')
)
def update_status(n, signal_data):
    if not data_updater or not data_updater.running:
        return "系统未启动", "无数据", "无信号", signal_data

    status = "运行中" if data_updater.running else "已停止"

    last_update = data_updater.last_update.strftime("%Y-%m-%d %H:%M:%S") if data_updater.last_update else "无数据"

    latest_signal = "无最新信号"
    if data_updater.latest_signal:
        signal = data_updater.latest_signal
        direction = signal['direction']
        color = "success" if direction == 'UP' else "danger" if direction == 'DOWN' else "warning"
        confidence = signal['confidence']
        price = signal['price']
        timestamp = signal['timestamp'].strftime("%H:%M:%S")

        latest_signal = [
            html.Span(f"{timestamp} - {direction}信号", className=f"text-{color} fw-bold"),
            html.Br(),
            html.Span(f"置信度: {confidence:.2%}, 价格: ${price:.2f}")
        ]

        new_signal = {
            'timestamp': timestamp,
            'direction': direction,
            'confidence': confidence,
            'price': price
        }
        signal_data = signal_data[-9:] + [new_signal]

    return status, f"最后更新: {last_update}", latest_signal, signal_data


@app.callback(
    Output('signal-history', 'children'),
    Input('signal-store', 'data')
)
def update_signal_history(signal_data):
    if not signal_data:
        return "暂无信号历史"

    rows = []
    for signal in reversed(signal_data):
        direction = signal['direction']
        color = "success" if direction == 'UP' else "danger" if direction == 'DOWN' else "warning"
        icon = "↑" if direction == 'UP' else "↓" if direction == 'DOWN' else "-"

        row = dbc.ListGroupItem([
            dbc.Row([
                dbc.Col(html.Span(icon, className=f"fs-4 text-{color}"), width=1),
                dbc.Col(signal['timestamp']),
                dbc.Col(direction),
                dbc.Col(f"{signal['confidence']:.2%}"),
                dbc.Col(f"${signal['price']:.2f}")
            ])
        ])
        rows.append(row)

    return dbc.ListGroup(rows)


@app.callback(
    Output('live-graph', 'figure'),
    Input('graph-update', 'n_intervals')
)
def update_graph(n):
    if not data_updater or not data_updater.df.empty:
        return no_update

    df = data_updater.df
    if df.empty:
        return go.Figure().update_layout(title="等待数据...")

    df_plot = df.iloc[-100:].copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='价格',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA5'],
        mode='lines', name='5 MA', line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA20'],
        mode='lines', name='20 MA', line=dict(color='purple')
    ), row=1, col=1)

    if 'xgboost_signal' in df_plot.columns:
        bullish_signals = df_plot[df_plot['xgboost_signal'] == 1]
        bearish_signals = df_plot[df_plot['xgboost_signal'] == -1]

        if not bullish_signals.empty:
            fig.add_trace(go.Scatter(
                x=bullish_signals.index,
                y=bullish_signals['close'] * 0.998,
                mode='markers',
                name='买入信号',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                hoverinfo='text',
                hovertext=[f"置信度: {c:.2%}" for c in bullish_signals['xgboost_confidence']]
            ), row=1, col=1)

        if not bearish_signals.empty:
            fig.add_trace(go.Scatter(
                x=bearish_signals.index,
                y=bearish_signals['close'] * 1.002,
                mode='markers',
                name='卖出信号',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                hoverinfo='text',
                hovertext=[f"置信度: {c:.2%}" for c in bearish_signals['xgboost_confidence']]
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['RSI'],
        mode='lines', name='RSI', line=dict(color='orange')
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot['volume'],
        name='成交量', marker_color='blue'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['Volume_MA20'],
        mode='lines', name='20 MA Volume', line=dict(color='red')
    ), row=3, col=1)

    fig.update_layout(
        title=f'{data_updater.symbol} 实时交易信号',
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        height=800,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=3, col=1)

    return fig


@app.callback(
    Output('start-button', 'disabled'),
    Output('stop-button', 'disabled'),
    Input('start-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
    prevent_initial_call=True
)
def control_monitoring(start_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button' and data_updater:
        data_updater.start()
        return True, False
    elif button_id == 'stop-button' and data_updater:
        data_updater.stop()
        return False, True

    return no_update


def cleanup():
    if data_updater and data_updater.running:
        data_updater.stop()
        logger.info("数据监控已停止")


atexit.register(cleanup)

if __name__ == "__main__":
    app.run(debug=True, port=8050)