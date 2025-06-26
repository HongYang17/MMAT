if __name__ == "__main__":
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score
    )

    from final_signal_model import (
        calculate_basic_indicators,
        calculate_patterns,
        calculate_additional_features,
        add_volatility_label,
        train_improved_model,
        retrain_final_model,
        plot_feature_importance
    )

    data_path = 'BTCUSDT_1min_2024-05-01_to_now.csv'
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    print("\nData preprocessing...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    print("\nCalculate technical indicators and candlestick patterns...")
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)
    df = calculate_additional_features(df)

    print("\nCreate future volatility labels...")
    df = add_volatility_label(df, future_window=45, pct_threshold=0.003)
    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ['target', 'vol_target', 'next_close', 'return', 'future_range', 'future_high', 'future_low']]
    X = df[feature_cols].copy()
    y = df['vol_target'].copy()

    print(f"\n total sample size: {len(y)}, Volatility label 1 ratio: {y.mean():.2%}")

    cutoff_date = X.index.max() - pd.Timedelta(days=30)
    X_old = X[X.index < cutoff_date]
    y_old = y[X.index < cutoff_date]
    X_recent = X[X.index >= cutoff_date]
    y_recent = y[X.index >= cutoff_date]

    print(f"\nData set partitioning: Old data {X_old.shape}, Recent data {X_recent.shape}")

    print("\nStage 1: Cross-validation on old data, selecting model structure and features")
    cv_model, selected_features = train_improved_model(X_old, y_old, importance_threshold=0.01)

    print("\nStage 2: Train the final model using the most recent data")
    final_model = retrain_final_model(cv_model, selected_features, X_recent, y_recent)

    print("\nModel evaluation (based on recent data)")
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
    plt.title('Confusion Matrix (Volatility Model)')
    plt.tight_layout()
    plt.savefig("volatility_confusion_matrix.png", dpi=300)
    plt.show()

    print("\nFeature importance visualization...")
    plot_feature_importance(final_model.named_steps['classifier'], selected_features)

    print("\nSaving volatility models and feature lists...")
    joblib.dump(final_model, 'volatility_model.pkl')
    joblib.dump(selected_features, 'volatility_features.pkl')

    print("\nVolatility model process completed!")
