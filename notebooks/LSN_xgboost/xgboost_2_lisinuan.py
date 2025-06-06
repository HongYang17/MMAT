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

    # 输出原始形态信号值（±100/0）
    for name, func in patterns.items():
        df[name] = func(df['open'], df['high'], df['low'], df['close'])

    # 定义形态分类
    bullish_patterns_strong = ['Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
                               'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike']
    bearish_patterns_strong = ['HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
                               'EveningDojiStar', 'EveningStar', 'GravestoneDoji']

    # Signal_列：标准化为 -1 / 0 / 1
    for name in patterns.keys():
        if name == 'GravestoneDoji':
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == 100 else 0)
        elif name in bullish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x == 100 else 0)
        elif name in bearish_patterns_strong:
            df[f'Signal_{name}'] = df[name].apply(lambda x: -1 if x == -100 else 0)
        else:
            df[f'Signal_{name}'] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # 聚合信号：行内总 bullish、bearish 形态个数
    signal_cols = [f'Signal_{name}' for name in patterns.keys()]
    df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == 1), axis=1)
    df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(1 for x in row if x == -1), axis=1)
    df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

    return df


def prepare_model_data(df, window=5, threshold=0.001):
    """
    重构数据准备函数 - 确保所有特征列名唯一
    """
    # 计算未来收益率
    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    # 创建目标变量
    volatility = df['ATR'] / df['close']
    adjusted_threshold = threshold + 0.5 * volatility
    df['target'] = np.select(
        [df['return'] >= adjusted_threshold, df['return'] <= -adjusted_threshold],
        [1, 0], default=-1
    )
    df = df[df['target'] != -1].copy()

    # 计算更多技术指标
    df = calculate_additional_features(df)

    # 核心特征 - 确保列名唯一
    base_features = ['open', 'high', 'low', 'close', 'volume']
    momentum_features = ['RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D', 'CCI', 'MOM']
    trend_features = ['ADX', 'PLUS_DI', 'MINUS_DI', 'EMA20', 'SMA20', 'EMA200']
    volatility_features = ['ATR', 'NATR', 'SAR', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'STDDEV']
    volume_features = ['OBV', 'AD', 'ADOSC', 'MFI', 'Volume_MA']
    derived_features = ['close_to_high', 'close_to_low', 'price_range',
                        'volatility_ratio', 'price_change', 'volume_change',
                        'rsi_divergence', 'macd_hist', 'distance_to_upper_bb',
                        'distance_to_lower_bb', 'trend_power']

    # 添加滞后特征 (确保唯一命名)
    lag_features = []
    for col in ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']:
        for lag in [1, 2, 3, 5, 10]:
            new_col = f'{col}_lag{lag}'
            df[new_col] = df[col].shift(lag)
            lag_features.append(new_col)

    # 添加变化率特征 (确保唯一命名)
    change_features = []
    for col in ['RSI', 'MACD', 'ATR', 'volume', 'close']:
        new_col = f'{col}_pct_change'
        df[new_col] = df[col].pct_change()
        change_features.append(new_col)

    # 添加交叉特征
    df['macd_histogram'] = df['MACD'] - df['MACD_signal']
    df['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)
    cross_features = ['macd_histogram', 'di_crossover']


    # 合并所有特征
    all_features = (base_features + momentum_features + trend_features +
                    volatility_features + volume_features + derived_features +
                    lag_features + change_features + cross_features)

    # 添加K线形态信号列
    pattern_signals = [col for col in df.columns if col.startswith('Signal_') or col in [
        'net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']]
    all_features += pattern_signals

    # 检查重复列名
    duplicates = set([x for x in all_features if all_features.count(x) > 1])
    if duplicates:
        raise ValueError(f"发现重复特征名: {duplicates}")

    # 确保所有特征都存在
    available_features = [col for col in all_features if col in df.columns]
    missing = set(all_features) - set(available_features)
    if missing:
        print(f"警告: 缺失特征: {missing}")

    X = df[available_features]
    y = df['target']
    valid_idx = y.notna() & X.notna().all(axis=1)

    print(f"final dataset: {sum(valid_idx)}samples, {len(available_features)}features")
    print(f"target: up {sum(y[valid_idx] == 1)} | down {sum(y[valid_idx] == 0)}")
    return X[valid_idx], y[valid_idx], available_features


def calculate_additional_features(df):
    """计算衍生特征，确保列名唯一"""
    # 价格衍生特征
    df['close_to_high'] = (df['high'] - df['close']) / df['high']
    df['close_to_low'] = (df['close'] - df['low']) / df['close']
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)
    df['price_change'] = df['close'].pct_change()

    # 成交量衍生特征
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['Volume_MA']

    # 技术指标衍生特征
    df['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
    df['macd_hist'] = df['MACD'] - df['MACD_signal']
    df['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])
    df['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close']
    df['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close']

    return df


def train_improved_model(X, y, test_size=0.2, importance_threshold=0.01):
    """
    改进的模型训练方法，增加特征筛选功能
    """
    # 时间序列分割
    tscv = TimeSeriesSplit(n_splits=3)

    best_score = 0
    best_pipeline = None
    selected_features = None  # 存储最终选中的特征

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 预处理管道
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, X.columns)],
            remainder='drop'
        )

        # 处理类别不平衡
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        # 创建模型
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric=['auc', 'logloss']
        )

        # 训练管道
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 训练
        pipeline.fit(X_train, y_train)

        # 评估
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        print(f"\nFold {fold + 1} 评估结果:")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        if auc > best_score:
            best_score = auc
            best_pipeline = pipeline
            # 获取特征重要性
            model = best_pipeline.named_steps['classifier']
            importances = model.feature_importances_

            # 筛选特征：重要性 >= 阈值
            selected_idx = np.where(importances >= importance_threshold)[0]
            selected_features = X.columns[selected_idx].tolist()

            print(f"\n发现新最佳模型 (AUC={auc:.4f}), 特征筛选结果:")
            print(f"原始特征数: {len(X.columns)}, 筛选后特征数: {len(selected_features)}")
            print("筛选后的特征:", selected_features)

    # 使用筛选后的特征重新训练最终模型
    if selected_features and len(selected_features) > 0:
        print("\n使用筛选后的特征重新训练最终模型...")
        X_selected = X[selected_features]

        # 重新训练最终模型（使用全部数据）
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, selected_features)],
            remainder='drop'
        )

        # 处理类别不平衡
        scale_pos_weight = sum(y == 0) / sum(y == 1)

        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', final_model)
        ])

        final_pipeline.fit(X_selected, y)
        print("筛选特征后的模型训练完成!")
        return final_pipeline, selected_features
    else:
        print("\n警告：未筛选出重要特征，使用原始模型")
        return best_pipeline, X.columns.tolist()


def evaluate_model(pipeline, X, y):
    """评估模型性能"""
    if pipeline is None:
        print("没有可用的模型进行评估")
        return

    print("\n模型整体评估:")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    print(f"accuracy_score: {accuracy_score(y, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y, y_proba):.4f}")
    print("\nclassification_report:")
    print(classification_report(y, y_pred))

    # 绘制ROC曲线
    plot_roc_curve(y, y_proba)


def plot_roc_curve(y_true, y_proba):
    """绘制ROC曲线"""
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
    # 计算基础指标
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)

    # 准备机器学习数据
    X, y, feature_names = prepare_model_data(df, window=5, threshold=0.001)

    # 训练模型
    pipeline = train_improved_model(X, y)

    # 评估模型
    if pipeline is not None:
        evaluate_model(pipeline, X, y)

        # 保存整个Pipeline
        joblib.dump(pipeline, 'trading_model_pipeline.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        print("\n模型训练完成并已保存!")

    return pipeline


def calculate_basic_indicators(df):
    """计算所有基础技术指标"""
    # Momentum & Oscillator
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], 12, 26, 9)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)

    # Trend-Following
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA200'] = talib.EMA(df['close'], timeperiod=200)

    # Volume-Based
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['Volume_MA'] = talib.SMA(df['volume'], timeperiod=20)

    # Volatility
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
    df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=20)

    # 清除NA值
    df.dropna(inplace=True)
    return df


def plot_feature_importance(model, feature_names, top_n=20):
    """可视化特征重要性"""
    import matplotlib.pyplot as plt

    importance = model.feature_importances_
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
    # 1. 加载数据
    data_path = 'BTCUSDT_1min_2024-05-01_to_2025-06-01.csv'
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path)

    # 2. 数据预处理
    print("\n数据预处理中...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 3. 计算基础技术指标
    print("\n计算基础技术指标...")
    df = calculate_basic_indicators(df)

    # 2. 计算K线形态等模式识别特征
    print("\n计算K线形态特征...")
    df = calculate_patterns(df)  # ← 插入你定义的模式识别函数

    # 4. 准备模型数据
    print("\n准备模型数据...")
    X, y, feature_names = prepare_model_data(df, window=5, threshold=0.001)

    # 6. 模型评估
    print("\n模型评估结果:")

    # 修改这里：接收两个返回值（模型和特征列表）
    pipeline, selected_features = train_improved_model(X, y, importance_threshold=0.01)

    # 使用筛选后的特征进行评估
    if selected_features:
        X_selected = X[selected_features]
    else:
        X_selected = X

    # 使用 pipeline 直接预测
    y_pred = pipeline.predict(X_selected)
    y_proba = pipeline.predict_proba(X_selected)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    evaluate_model(pipeline, X, y)

    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nclassification_report:")
    print(classification_report(y, y_pred))

    # === 8. 混淆矩阵可视化 ===
    print("\n绘制混淆矩阵...")
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_pred)
    labels = sorted(list(set(y)))  # 应该是 [0, 1]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('predicted value')
    plt.ylabel('actual value')
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    # 7. 特征重要性可视化 - 使用筛选后的特征
    print("\n特征重要性可视化...")
    plot_feature_importance(pipeline.named_steps['classifier'], selected_features)

    # 8. 保存模型和选中的特征
    print("\n保存模型和特征名...")
    joblib.dump(pipeline, 'improved_signal_model.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')  # 保存筛选后的特征列表

    print("\n所有流程完成!")
