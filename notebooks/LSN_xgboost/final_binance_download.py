import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm


def get_binance_1m_data(symbol="BTCUSDT", start_time=None, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000

    all_data = []
    current_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    print("Retrieving data from Binance, please wait...")
    pbar = tqdm(total=(end_ts - current_ts) // (60 * 1000 * limit) + 1)

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
        current_ts = data[-1][0] + 60 * 1000
        time.sleep(0.1)
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


def update_binance_csv(symbol="BTCUSDT", file_path="BTCUSDT_1min_data.csv"):
    now = datetime.utcnow()

    if os.path.exists(file_path):
        # 读取已有数据，找到最后时间戳
        existing_df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        last_timestamp = existing_df.index[-1]
        start_time = last_timestamp + timedelta(minutes=1)
        print(f"Local data detected, last time was: {last_timestamp}，From now on, time will continue to be updated")
    else:
        # 文件不存在，从默认时间开始
        existing_df = pd.DataFrame()
        start_time = datetime(2024, 5, 1)
        print("No local data detected, will download from scratch")

    # 下载更新数据
    new_df = get_binance_1m_data(symbol, start_time=start_time, end_time=now)

    if not new_df.empty:
        # 合并并去重（避免重复）
        updated_df = pd.concat([existing_df, new_df])
        updated_df = updated_df[~updated_df.index.duplicated(keep='first')]
        updated_df.sort_index(inplace=True)

        updated_df.to_csv(file_path)
        print(f"Updated and saved to：{file_path}，with {len(updated_df)} records")
    else:
        print("No new data to update")


# 运行更新函数
update_binance_csv("BTCUSDT", "BTCUSDT_1min_2024-05-01_to_now.csv")
