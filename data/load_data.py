import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

def get_binance_1m_data(symbol="BTCUSDT", start_str="2024-05-01", end_str="2025-05-01"):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000

    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    current_ts = start_ts

    print("正在从 Binance 获取数据，请稍候...")
    pbar = tqdm(total=(end_ts - start_ts) // (60 * 1000 * limit) + 1)

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "limit": limit
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        current_ts = data[-1][0] + 60 * 1000  # +1分钟
        time.sleep(0.1)  # 防止被限流
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df

# 示例：拉取 BTC/USDT 的 2024.05.01 到 2025.05.01 的 1分钟数据
df = get_binance_1m_data("BTCUSDT", "2024-05-01", "2025-05-01")
print(df.head())
print(f"共获取 {len(df)} 条 1 分钟K线")

# 保存为 CSV
df.to_csv("BTCUSDT_1min_2024-05-01_to_2025-05-01.csv")