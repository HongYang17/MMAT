import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import os
import numpy as np

import pandas as pd
import numpy as np
import talib
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_patterns(df):
    """
    Detect candlestick patterns and assign TA-Lib raw outputs for ±100 signals.
    Also generates Signal_ columns that map strong bullish (1), strong bearish (-1), and neutral (0).
    """

    import talib
    import numpy as np

    patterns = {
        # Bullish
        'Hammer': talib.CDLHAMMER,
        'InvertedHammer': talib.CDLINVERTEDHAMMER,
        'BullishEngulfing': talib.CDLENGULFING,
        'PiercingLine': talib.CDLPIERCING,
        'MorningStar': talib.CDLMORNINGSTAR,
        'DragonflyDoji': talib.CDLDRAGONFLYDOJI,
        'LongLine': talib.CDLLONGLINE,
        'ThreeLineStrike': talib.CDL3LINESTRIKE,

        # Bearish
        'HangingMan': talib.CDLHANGINGMAN,
        'ShootingStar': talib.CDLSHOOTINGSTAR,
        'BearishEngulfing': talib.CDLENGULFING,
        'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
        'EveningDojiStar': talib.CDLEVENINGDOJISTAR,
        'EveningStar': talib.CDLEVENINGSTAR,
        'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
    }

    for name, func in patterns.items():
        df[name] = func(df['open'], df['high'], df['low'], df['close'])

    # 定义形态分类
    bullish_patterns_strong = ['Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
                               'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike']
    bearish_patterns_strong = ['HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
                               'EveningDojiStar', 'EveningStar', 'GravestoneDoji']

    for name in patterns.keys():
        if name == 'GravestoneDoji':
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == 100 else 0)
        elif name in bullish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x == 100 else 0)
        elif name in bearish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == -100 else 0)
        else:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    signal_cols = [f'Signal_{name}' for name in patterns.keys()]
    df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == 1), axis=1)
    df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == -1), axis=1)
    df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

    return df


def prepare_model_data(df, window=5, threshold=0.001):


    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    volatility = df['ATR'] / df['close']
    adjusted_threshold = threshold + 0.5 * volatility
    df['target'] = np.select(
        [df['return'] >= adjusted_threshold, df['return'] <= -adjusted_threshold],
        [1, 0], default=-1
    )
    df = df[df['target'] != -1].copy()

    df = calculate_additional_features(df)

    base_features = ['open', 'high', 'low', 'close', 'volume']
    momentum_features = ['RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D', 'CCI', 'MOM']
    trend_features = ['ADX', 'PLUS_DI', 'MINUS_DI', 'EMA20', 'SMA20', 'EMA200']
    volatility_features = ['ATR', 'NATR', 'SAR', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'STDDEV']
    volume_features = ['OBV', 'AD', 'ADOSC', 'MFI', 'Volume_MA']
    derived_features = ['close_to_high', 'close_to_low', 'price_range',
                        'volatility_ratio', 'price_change', 'volume_change',
                        'rsi_divergence', 'macd_hist', 'distance_to_upper_bb',
                        'distance_to_lower_bb', 'trend_power']

    lag_features = []
    for col in ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']:
        for lag in [1, 2, 3, 5, 10]:
            new_col = f'{col}_lag{lag}'
            df[new_col] = df[col].shift(lag)
            lag_features.append(new_col)

    change_features = []
    for col in ['RSI', 'MACD', 'ATR', 'volume', 'close']:
        new_col = f'{col}_pct_change'
        df[new_col] = df[col].pct_change()
        change_features.append(new_col)

    df['macd_histogram'] = df['MACD'] - df['MACD_signal']
    df['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)
    cross_features = ['macd_histogram', 'di_crossover']

    all_features = (base_features + momentum_features + trend_features +
                    volatility_features + volume_features + derived_features +
                    lag_features + change_features + cross_features)

    pattern_signals = [col for col in df.columns if col.startswith('Signal_') or col in [
        'net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']]
    all_features += pattern_signals

    duplicates = set([x for x in all_features if all_features.count(x) > 1])
    if duplicates:
        raise ValueError(f"Duplicate feature names found: {duplicates}")

    available_features = [col for col in all_features if col in df.columns]
    missing = set(all_features) - set(available_features)
    if missing:
        print(f"Warning: Missing feature: {missing}")

    X = df[available_features]
    y = df['target']
    valid_idx = y.notna() & X.notna().all(axis=1)

    print(f"final dataset: {sum(valid_idx)}samples, {len(available_features)}features")
    print(f"target: up {sum(y[valid_idx] == 1)} | down {sum(y[valid_idx] == 0)}")
    return X[valid_idx], y[valid_idx], available_features


def calculate_additional_features(df):

    df['close_to_high'] = (df['high'] - df['close']) / df['high']
    df['close_to_low'] = (df['close'] - df['low']) / df['close']
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)
    df['price_change'] = df['close'].pct_change()

    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['Volume_MA']

    df['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
    df['macd_hist'] = df['MACD'] - df['MACD_signal']
    df['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])
    df['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close']
    df['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close']

    return df

def calculate_basic_indicators(df):

    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], 12, 26, 9)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)

    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA200'] = talib.EMA(df['close'], timeperiod=200)

    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['Volume_MA'] = talib.SMA(df['volume'], timeperiod=20)

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
    df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=20)

    df.dropna(inplace=True)
    return df

def add_volatility_label(df, future_window=15, pct_threshold=0.01):

    df['future_high'] = df['high'].rolling(window=future_window, min_periods=1).max().shift(-future_window)
    df['future_low'] = df['low'].rolling(window=future_window, min_periods=1).min().shift(-future_window)

    df['future_range'] = (df['future_high'] - df['future_low']) / df['close']

    df['vol_target'] = (df['future_range'].abs() > pct_threshold).astype(int)

    return df



def train_improved_model(X, y, importance_threshold=0.01):

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = 0
    best_pipeline = None
    selected_features = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, X.columns)
        ])

        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"Fold {fold + 1} AUC: {auc:.4f}")

        if auc > best_score:
            best_score = auc
            best_pipeline = pipeline
            importances = model.feature_importances_
            selected_idx = np.where(importances >= importance_threshold)[0]
            selected_features = X.columns[selected_idx].tolist()

    print(f"\nNumber of selected features: {len(selected_features)}: {selected_features}")
    return best_pipeline, selected_features


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

def retrain_final_model(best_pipeline, selected_features, X_recent, y_recent):

    valid_features = [f for f in selected_features if f in X_recent.columns]
    if not valid_features:
        raise ValueError("None of the selected features can be found in X_recent. Please check whether the feature generation process is consistent.")

    print(f"\nRetrain the final model using the latest data, number of features: {len(valid_features)}")

    classifier = clone(best_pipeline.named_steps['classifier'])

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, valid_features)
    ])

    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    final_model.fit(X_recent[valid_features], y_recent)

    print("The final model training is complete!")
    return final_model


def evaluate_model(pipeline, X, y):
    if pipeline is None:
        print("No models available for evaluation")
        return

    print("\nOverall model assessment:")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    print(f"accuracy_score: {accuracy_score(y, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y, y_proba):.4f}")
    print("\nclassification_report:")
    print(classification_report(y, y_pred))

    plot_roc_curve(y, y_proba)


def plot_roc_curve(y_true, y_proba):

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def main(df):

    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)

    X, y, feature_names = prepare_model_data(df, window=5, threshold=0.001)

    pipeline = train_improved_model(X, y)

    if pipeline is not None:
        evaluate_model(pipeline, X, y)

        joblib.dump(pipeline, 'trading_model_pipeline.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        print("\nModel training is complete and has been saved!")

    return pipeline




def plot_feature_importance(model, feature_names, top_n=20):

    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    n_features = len(feature_names)
    top_n = min(top_n, n_features)  # <-- 加这个限制
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} features importance')
    plt.barh(range(top_n), importance[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score
    )

    data_path = 'BTCUSDT_1min_2024-05-01_to_now.csv'
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    print("\nData preprocessing...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    print("\nCalculate technical indicators and candlestick patterns...")
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)  # 插入K线形态特征

    print("\nPrepare model data...")
    X, y, feature_names = prepare_model_data(df, window=5, threshold=0.001)

    cutoff_date = X.index.max() - pd.Timedelta(days=30)
    X_old = X[X.index < cutoff_date]
    y_old = y[X.index < cutoff_date]
    X_recent = X[X.index >= cutoff_date]
    y_recent = y[X.index >= cutoff_date]

    print(f"\nData split：old data {X_old.shape}, recent data {X_recent.shape}")

    print("\n Stage 1: Cross-validation on old data, selecting model structure and features")
    cv_model, selected_features = train_improved_model(X_old, y_old, importance_threshold=0.01)

    print("\n Stage 2: Train the final model using the most recent data")
    final_model = retrain_final_model(cv_model, selected_features, X_recent, y_recent)

    print("\n Model evaluation (based on recent data)")
    X_eval = X_recent[selected_features]
    y_eval = y_recent
    y_pred = final_model.predict(X_eval)
    y_proba = final_model.predict_proba(X_eval)[:, 1]

    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred))

    print("\nDrawing a confusion matrix...")
    cm = confusion_matrix(y_eval, y_pred)
    labels = sorted(list(set(y_eval)))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Recent Data)')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    print("\nFeature importance visualization.")
    plot_feature_importance(final_model.named_steps['classifier'], selected_features)

    print("\nSave model and feature list...")
    joblib.dump(final_model, 'final_signal_model.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')

    print("\nAll processes completed!")
