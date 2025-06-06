# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 00:16:33 2025

@author: XIE
"""
import pandas as pd
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
from signal_logger import SignalHistoryLogger

# ============ Binance & Logger Setup ============
client = Client(api_key='your_api_key', api_secret='your_api_secret')
signal_logger = SignalHistoryLogger(filename="D:/My File/SMU/QF 635/MMAT/signal_history_xie.csv")

# ============ Fetch Data ============
def fetch_ohlcv_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_5MINUTE, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Singapore")  
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# ============ Strategy 2 + 4 ============
def generate_strategy24_signals(df, signal_logger=None):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['mean_ATR'] = df['ATR'].rolling(20).mean()
    df['Volume_MA20'] = df['volume'].rolling(20).mean()
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    macd, macdsignal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal

    df['bearish_combined'] = 0
    df['bearish_trigger'] = ''
    i = -2
    ts = df.index[i]

    strat2 = df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['CDLSHOOTINGSTAR'].iloc[i] < 0
    strat4 = df['ATR'].iloc[i] > df['mean_ATR'].iloc[i] * 1.1 and df['volume'].iloc[i] > df['Volume_MA20'].iloc[i] * 1.2 and df['RSI'].iloc[i] < 35

    if strat2 or strat4:
        df.at[ts, 'bearish_combined'] = -1
        reason = []
        if strat2: reason.append("Strategy 2")
        if strat4: reason.append("Strategy 4")
        df.at[ts, 'bearish_trigger'] = ' + '.join(reason)
        if signal_logger is not None:
            signal_logger.add_signal('bearish', ts, df['close'].iloc[i], df.at[ts, 'bearish_trigger'])

    return df

# ============ Dash Layout ============
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H4("[Strategy 2+4] BTCUSDT 5-Minute Live Chart (Singapore Time)"),
    dcc.Interval(id='interval-component', interval=300 * 1000, n_intervals=0),
    dcc.Graph(id='live-chart')
])

@app.callback(
    Output('live-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_chart(n):
    df = fetch_ohlcv_data()
    df = generate_strategy24_signals(df, signal_logger=signal_logger)
    df_plot = df.iloc[-50:].copy()
    signals = df_plot[df_plot['bearish_combined'] == -1]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=['Candlestick', 'RSI', 'ATR', 'Volume'],
                        row_heights=[0.4, 0.2, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'],
                                 low=df_plot['low'], close=df_plot['close'], name='Candlestick'), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals.index, y=signals['close'] * 0.995,
                             mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Bearish Signal', text=signals['bearish_trigger']), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dash', line_color='gray', row=2, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ATR'], mode='lines', name='ATR'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['mean_ATR'], mode='lines', name='mean_ATR', line=dict(dash='dash')), row=3, col=1)

    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Volume_MA20'], mode='lines', name='Volume_MA20', line=dict(dash='dash')), row=4, col=1)

    fig.update_layout(height=800, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

# ============ Run App ============
if __name__ == '__main__':
    app.run(debug=True, port=8051)

# http://127.0.0.1:8051

