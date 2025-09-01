import pandas as pd
import numpy as np
import pytest
from src.strategy import (
    create_labels,
    williams_r,
    feature_engineering,
    compute_rsi,
    train_model,
    predict_signals,
    backtest,
    strategy_analysis,
)
from src.labels import create_forward_return_labels
from src.combined_analysis import merge_sentiment_with_features

def test_create_labels():
    df = pd.DataFrame({'Close': [100, 102, 104, 106, 108]})
    result = create_labels(df, horizon=2, threshold=0.01)
    assert 'label' in result.columns
    assert result['label'].dtype == int

def test_williams_r():
    df = pd.DataFrame({
        'High': [10, 12, 13, 14, 15],
        'Low': [5, 6, 7, 8, 9],
        'Close': [7, 8, 9, 10, 11]
    })
    wr = williams_r(df, period=3)
    assert isinstance(wr, pd.Series)
    assert wr.isnull().sum() > 0  # Rolling window should produce NaNs

def test_feature_engineering():
    df = pd.DataFrame({
        'Open': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'High': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        'Low': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        'Close': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'Volume': [100]*15
    })
    result = feature_engineering(df)
    assert 'sma_5' in result.columns
    assert 'Williams_%R' in result.columns
    assert 'Doji' in result.columns

def test_compute_rsi():
    series = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    rsi = compute_rsi(series, window=5)
    assert isinstance(rsi, pd.Series)
    assert rsi.isnull().sum() > 0

def test_create_forward_return_labels():
    df = pd.DataFrame({'Close': [100, 102, 104, 106, 108]})
    result = create_forward_return_labels(df, horizon=2, threshold=0.01)
    assert 'forward_return' in result.columns
    assert 'direction' in result.columns

def test_merge_sentiment_with_features():
    df = pd.DataFrame({'Date': pd.date_range('2022-01-01', periods=5), 'Close': [1,2,3,4,5]})
    sentiment_dict = {
        '2022-01-01': 0.1,
        '2022-01-02': -0.2,
        '2022-01-03': 0.0,
        '2022-01-04': 0.3,
        '2022-01-05': -0.1
    }
    merged = merge_sentiment_with_features(df, sentiment_dict)
    assert 'sentiment_score' in merged.columns

def test_train_predict_backtest():
    df = pd.DataFrame({
        'sma_5': np.random.rand(20),
        'sma_20': np.random.rand(20),
        'rsi_14': np.random.rand(20),
        'macd': np.random.rand(20),
        'macd_signal': np.random.rand(20),
        'stoch_k': np.random.rand(20),
        'stoch_d': np.random.rand(20),
        'VWAP': np.random.rand(20),
        'rolling_vol20': np.random.rand(20),
        'volume_z': np.random.rand(20),
        'roc5': np.random.rand(20),
        'Williams_%R': np.random.rand(20),
        'label': np.random.randint(0, 2, 20),
        'Close': np.random.rand(20)
    })
    features = [
        'sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal', 'stoch_k',
        'stoch_d', 'VWAP', 'rolling_vol20', 'volume_z', 'roc5', 'Williams_%R'
    ]
    X = df[features]
    y = df['label']
    model = train_model(X, y)
    signals, prob = predict_signals(model, X)
    results = backtest(df, signals)
    assert isinstance(results, list)

def test_strategy_analysis_and_visualization():
    results = [0.01, -0.02, 0.03, 0.05, -0.01]
    strategy_analysis(results)
    # visualize_returns(results)  # Uncomment to visually check plots

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))