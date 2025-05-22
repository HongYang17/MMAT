# import os
# import sys
# from datetime import datetime, timedelta
# from binance.client import Client
# import csv


# # Go up one folder from /notebooks to project root
# sys.path.append(os.path.abspath(".."))

# from config.load_env import load_keys

# keys = load_keys()
# client = Client(api_key=keys['api_key'], api_secret=keys['secret_key'])

# function to get yesterday's 3-minute candles
# def fetch_yesterday_3min_klines(client, symbol):
#     end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
#     start_time = end_time - timedelta(days=1)

#     klines = client.futures_klines(
#         symbol=symbol,
#         interval=Client.KLINE_INTERVAL_3MINUTE,
#         start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
#         end_str=end_time.strftime('%Y-%m-%d %H:%M:%S')
#     )
#     return klines

# # Step 4: Write klines to CSV
# def save_klines_to_csv(symbol, klines):
#     filename = f"{symbol}_3min_20_may.csv"
#     headers = ['timestamp', 'close', 'volume']

#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(headers)

#         for k in klines:
#             timestamp = datetime.utcfromtimestamp(k[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
#             close = k[4]
#             volume = k[5]
#             writer.writerow([timestamp, close, volume])

#     print(f"Saved {len(klines)} rows to {filename}")

# # Step 5: Run for all symbols
# symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']

# for symbol in symbols:
#     klines = fetch_yesterday_3min_klines(client, symbol)
#     print(f"\n{symbol} - Total candles: {len(klines)}")
#     for k in klines:
#         ts = datetime.utcfromtimestamp(k[0] / 1000.0)
#         print(f"{ts} | Close: {k[4]} Volume: {k[5]}")
#     save_klines_to_csv(symbol, klines)

#-----

import os
import sys
import time
from datetime import datetime, timedelta
from binance.client import Client
import csv

sys.path.append(os.path.abspath(".."))
from config.load_env import load_keys

keys = load_keys()
client = Client(api_key=keys['api_key'], api_secret=keys['secret_key'])

def fetch_3min_klines_paged(client, symbol, days=7):
    interval = Client.KLINE_INTERVAL_3MINUTE
    limit = 1500
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    all_klines = []
    current_start = start_time

    while current_start < end_time:
        # Convert to string format for API
        start_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            limit=limit
        )

        if not klines:
            break

        all_klines.extend(klines)

        # Advance to the next start time based on last candle timestamp
        last_kline_ts = klines[-1][0]  # milliseconds
        current_start = datetime.utcfromtimestamp(last_kline_ts / 1000.0) + timedelta(minutes=3)

        time.sleep(0.2)  # respect API rate limits

    return all_klines

def save_klines_to_csv(symbol, klines, days):
    filename = f"{symbol}_3min_{days}days.csv"
    headers = ['timestamp', 'close', 'volume']

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for k in klines:
            timestamp = datetime.utcfromtimestamp(k[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
            close = k[4]
            volume = k[5]
            writer.writerow([timestamp, close, volume])

    print(f" Saved {len(klines)} rows to {filename}")

# Run for all symbols
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
days = 7

for symbol in symbols:
    klines = fetch_3min_klines_paged(client, symbol, days)
    print(f"\n{symbol} - Total candles: {len(klines)}")
    for k in klines[:5]:  # preview first 5 rows
        ts = datetime.utcfromtimestamp(k[0] / 1000.0)
        print(f"{ts} | Close: {k[4]} Volume: {k[5]}")
    save_klines_to_csv(symbol, klines, days)

#---

# import os
# import sys
# import time
# import argparse
# from datetime import datetime, timedelta
# from binance.client import Client
# import csv

# # Go up one folder from /notebooks to project root
# sys.path.append(os.path.abspath(".."))
# from config.load_env import load_keys
# keys = load_keys()
# client = Client(api_key=keys['api_key'], api_secret=keys['secret_key'])

# def fetch_3min_klines_paged(client, symbol, days=7):
#     interval = Client.KLINE_INTERVAL_3MINUTE
#     limit = 1500
#     end_time = datetime.utcnow()
#     start_time = end_time - timedelta(days=days)

#     all_klines = []
#     current_start = start_time

#     while current_start < end_time:
#         start_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
#         klines = client.futures_klines(
#             symbol=symbol,
#             interval=interval,
#             start_str=start_str,
#             limit=limit
#         )

#         if not klines:
#             break

#         all_klines.extend(klines)
#         last_kline_ts = klines[-1][0]
#         current_start = datetime.utcfromtimestamp(last_kline_ts / 1000.0) + timedelta(minutes=3)

#         time.sleep(0.2)  # Avoid hitting rate limit

#     return all_klines

# def save_klines_to_csv(symbol, klines, days):
#     filename = f"{symbol}_3min_{days}days.csv"
#     headers = ['timestamp', 'close', 'volume']

#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(headers)

#         for k in klines:
#             timestamp = datetime.utcfromtimestamp(k[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
#             close = k[4]
#             volume = k[5]
#             writer.writerow([timestamp, close, volume])

#     print(f"\n Saved {len(klines)} rows to {filename}")

# def main():
#     parser = argparse.ArgumentParser(description="Fetch historical 3-minute Kline data for a given symbol")
#     parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol, e.g. BTCUSDT')
#     parser.add_argument('--days', type=int, default=7, help='Number of past days to fetch (default: 7)')

#     args = parser.parse_args()
#     symbol = args.symbol.upper()
#     days = args.days

#     keys = load_keys()
#     client = Client(api_key=keys['api_key'], api_secret=keys['secret_key'])

#     print(f"\n Fetching {days} days of 3m data for {symbol}...")
#     klines = fetch_3min_klines_paged(client, symbol, days)

#     print(f" {symbol} - Total candles fetched: {len(klines)}")
#     for k in klines[:5]:  # Preview first 5 rows
#         ts = datetime.utcfromtimestamp(k[0] / 1000.0)
#         print(f"{ts} | Close: {k[4]} Volume: {k[5]}")

#     save_klines_to_csv(symbol, klines, days)

# if __name__ == "__main__":
#     main()
