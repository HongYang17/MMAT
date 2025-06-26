import pandas as pd
import os

class SignalHistoryLogger:
    def __init__(self, filename='signal_history.csv'):
        self.filename = filename
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        else:
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])

    def add_signal(self, signal_type, timestamp, price, trigger=None):
        # 如果已经有该时间戳的该类型信号，则先删除（防止重复）
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

        # 添加新信号
        new_row = {
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'trigger': trigger
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def remove_by_timestamp(self, timestamp):
        """移除所有该时间戳的记录（无论方向）"""
        self.df = self.df[self.df['timestamp'] != timestamp]

    def has_signal(self, timestamp, signal_type):
        """检查是否已有该时间戳的该类型信号"""
        if self.df.empty:
            return False
        return not self.df[
            (self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type)
        ].empty

    def get_history(self):
        return self.df.copy()

    def save_to_csv(self, filename=None):
        path = filename if filename else self.filename
        self.df.to_csv(path, index=False)

    def remove_by_type_and_timestamp(self, signal_type, timestamp):
        """移除指定类型和时间戳的信号"""
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

    def remove_opposite_signal(self, timestamp, signal_type):
        """根据信号类型自动删除相反方向信号"""
        opposite = {
            'xgboost_bullish': 'xgboost_bearish',
            'xgboost_bearish': 'xgboost_bullish'
        }
        if signal_type in opposite:
            self.remove_by_type_and_timestamp(opposite[signal_type], timestamp)


