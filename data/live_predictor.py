import requests
import time
import joblib
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from model_utils import generate_all_signals, create_ml_features

# Load trained model
model = joblib.load("../notebooks/xgb_model.pkl")

# Binance API endpoint
URL = 'https://fapi.binance.com'
METHOD = '/fapi/v1/klines'

console = Console()

# Global tracking for delayed verification
prev_prediction = None
prev_price = None
prev_timestamp = None

# Collect latest 500 1min candles
def get_latest_ohlcv():
    params = {'symbol': 'BTCUSDT', 'interval': '1m', 'limit': 500}
    response = requests.get(URL + METHOD, params=params)
    response.raise_for_status()
    ohlcv = response.json()

    df = pd.DataFrame(ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def predict_live():
    global prev_prediction, prev_price, prev_timestamp

    while True:
        try:
            df = get_latest_ohlcv()
            signals = generate_all_signals(df)
            features = create_ml_features(signals, df['close'])

            if features.empty:
                console.print("Not enough data to generate features.", style="yellow")
                time.sleep(10)
                continue

            latest = features.iloc[-1:]
            prediction = model.predict(latest)[0]  # 0=down, 1=up
            proba = model.predict_proba(latest)[0]
            confidence = max(proba)

            action = "UP" if prediction == 1 else "DOWN"
            if confidence < 0.6:
                action = "UNCERTAIN"

            current_price = df['close'].iloc[-1]
            timestamp = df.index[-1]
            rsi = signals['rsi'].iloc[-1]
            macd = signals['macd'].iloc[-1]
            atr = signals['atr_break'].iloc[-1]

            # 延迟验证上一次预测是否准确
            if prev_prediction is not None:
                price_change = current_price - prev_price
                if prev_prediction == 1 and price_change > 0:
                    direction_result = "correct"
                elif prev_prediction == 0 and price_change < 0:
                    direction_result = "correct"
                else:
                    direction_result = "wrong"

            # 更新历史状态
            prev_prediction = prediction
            prev_price = current_price
            prev_timestamp = timestamp

            # --- Display table ---
            table = Table(title="Live BTC/USDT Direction Prediction")
            table.add_column("timestamp", justify="left")
            table.add_column("price", justify="right")
            table.add_column("RSI", justify="right")
            table.add_column("MACD", justify="right")
            table.add_column("ATR", justify="right")
            table.add_column("prediction", justify="center")
            table.add_column("result", justify="center")

            table.add_row(
                str(timestamp),
                f"{current_price:.2f}",
                str(rsi),
                str(macd),
                str(atr),
                action,
                direction_result
            )

            console.clear()
            console.print(table)

        except Exception as e:
            console.print(f"Error: {e}", style="red")

        time.sleep(60)

if __name__ == "__main__":
    try:
        predict_live()
    except KeyboardInterrupt:
        console.print("Live prediction manually stopped.", style="bold cyan")
