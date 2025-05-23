import os
import sys
import time
import csv
from datetime import datetime, timedelta
import pytz
from binance.client import Client

# Load API keys
sys.path.append(os.path.abspath(".."))
from config.load_env import load_keys

keys = load_keys()
client = Client(api_key=keys['api_key'], api_secret=keys['secret_key'])

def align_to_3min(dt):
    dt = dt.astimezone(pytz.utc) if dt.tzinfo else dt.replace(tzinfo=pytz.utc)
    aligned_minute = (dt.minute // 3) * 3
    return dt.replace(minute=aligned_minute, second=0, microsecond=0)

def fetch_batch_range(client, symbol, start_time, end_time, limit=1500):
    interval = Client.KLINE_INTERVAL_3MINUTE
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        endTime=end_ts,
        limit=limit
    )
    return klines

def fetch_7day_klines_in_3_batches(client, symbol):
    total_hours = 7 * 24  # 168 hours
    chunk_hours = 56  # 168 / 3 = 56 hours per batch

    def align_to_3min(dt):
        aligned_minute = (dt.minute // 3) * 3
        return dt.replace(minute=aligned_minute, second=0, microsecond=0)

    end_time = align_to_3min(datetime.now(pytz.utc))
    start_time = end_time - timedelta(hours=total_hours)

    batch_ranges = []
    current_start = start_time
    for i in range(3):
        if i < 2:
            current_end = current_start + timedelta(hours=chunk_hours)
        else:
            current_end = end_time

        if current_end > end_time:
            current_end = end_time
        if current_start >= current_end:
            current_start = current_end - timedelta(minutes=3)

        batch_ranges.append((current_start, current_end))
        current_start = current_end

    print("Batch Time Ranges (UTC):")
    for i, (s, e) in enumerate(batch_ranges):
        print(f"  Batch {i+1}: {s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M')} "
              f"(Duration: {(e-s).total_seconds()/3600:.2f} hours)")

    all_klines = []
    for i, (start, end) in enumerate(batch_ranges):
        print(f"Fetching batch {i+1}: {start.strftime('%m-%d %H:%M')} to {end.strftime('%m-%d %H:%M')}")
        try:
            batch = fetch_batch_range(client, symbol, start, end)
            print(f"  Fetched {len(batch)} candles (Expected: {(end - start).total_seconds() / 180:.0f})")
            all_klines.extend(batch)
        except Exception as e:
            print(f"  Request failed: {str(e)}")
        time.sleep(0.3)

    if not all_klines:
        return []

    all_klines.sort(key=lambda x: x[0])
    start_timestamp = int(start_time.timestamp() * 1000)
    final_data = [k for k in all_klines if k[0] >= start_timestamp]

    seen = set()
    unique_data = []
    for k in final_data:
        if k[0] not in seen:
            seen.add(k[0])
            unique_data.append(k)

    final_data = unique_data[-3360:]

    if final_data:
        first = datetime.utcfromtimestamp(final_data[0][0] / 1000)
        last = datetime.utcfromtimestamp(final_data[-1][0] / 1000)
        print(f"Final Data Range: {first.strftime('%Y-%m-%d %H:%M')} to {last.strftime('%Y-%m-%d %H:%M')}")

    return final_data

def save_merged_to_csv(symbol, merged_klines, days):
    filename = f"{symbol}_3min_{days}days.csv"
    headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for k in merged_klines:
            timestamp = datetime.utcfromtimestamp(k[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, k[1], k[2], k[3], k[4], k[5]])

    print(f"Saved {len(merged_klines)} rows to {filename}")

# Run for each symbol
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
days = 7

for symbol in symbols:
    print(f"\nFetching 7 days of data for {symbol}...")
    klines = fetch_7day_klines_in_3_batches(client, symbol)
    print(f"{symbol}: Total merged candles = {len(klines)}")
    save_merged_to_csv(symbol, klines, days)
