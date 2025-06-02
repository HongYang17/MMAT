### Originally, the signal history was only stored in memory, so once we stopped or restarted the program, all detected signals disappeared.
### With signal_logger.py, we save each signal’s timestamp, type, and price into a CSV.
### When we plot the chart, we read from that file to redraw all historical signals. That’s why they persist over time.


import pandas as pd
import os

class SignalHistoryLogger:
    def __init__(self, filename='signal_history.csv'):
        self.filename = filename
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        else:
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price'])

    def add_signal(self, signal_type, timestamp, price):
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'type': signal_type,
            'price': price
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.filename, index=False)

    def get_history(self):
        if os.path.exists(self.filename):
            return pd.read_csv(self.filename, parse_dates=['timestamp'])
        else:
            return pd.DataFrame(columns=['timestamp', 'type', 'price'])

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)
