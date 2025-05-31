# prediction_logger.py

import pandas as pd
from datetime import datetime

class PredictionLogger:
    def __init__(self):
        self.log = []

    def record_prediction(self, timestamp, prediction, close_now, close_prev):
        """
        記錄一筆預測資料：方向、報酬率、是否命中。
        """
        if prediction.startswith("UP") or prediction.startswith("DOWN"):
            try:
                ret = (close_now - close_prev) / close_prev
                hit = (
                    (prediction == "UP" and ret > 0) or
                    (prediction == "DOWN" and ret < 0)
                )
                self.log.append({
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'return': ret,
                    'hit': int(hit)
                })
            except Exception as e:
                print(f"[Logger] Error recording prediction: {e}")

    def get_hit_rate(self):
        """計算目前的命中率。"""
        if not self.log:
            return 0.0
        return sum(entry['hit'] for entry in self.log) / len(self.log)

    def to_dataframe(self):
        return pd.DataFrame(self.log)

    def save_to_csv(self, path='prediction_log.csv'):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[Logger] Saved {len(df)} predictions to {path}")
