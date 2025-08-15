import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

# For now this scaffold covers:

# Label creation
# Feature engineering
# Model training and prediction
# Backtesting logic

def create_labels(df, horizon=5, threshold=0.015):
    # Creating binary labels, returns 1 if teh forward return over the next [horizon] days is >= threshold, else 0
    df = df.copy()
    df['forward_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['label'] = (df['forward_return'] >= threshold).astype(int)
    return df

def feature_engineering(df):
    # Adding technical indictaors as teh features for the ML model
    df = df.copy()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi_14'] = compute_rsi(df['close'], window=14)
    return df


def compute_rsi(series, window=14):
    # Calculating RSI, teh realtiev STrength Index (this is a momentum indicator), 
    # using price changes to compute the average gains/losses over a window tome frame
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(X, y):
    # Training a classifier using the scikit-learn pipeline; teh pipeline inclydes includes imputation (handling missing data), 
    # scaling (normalizing features), and a classifier (Logistic Regression).
    pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    # tscv = variable name for an instance of TimeSeriesSplit from the scikit-learn library
    # TimeSeriesSplit is a cross-validation splitter designed for time series data, here I use 5 tiem splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Optionally add: cross-validation, hyperparameter tuning, etc.
    pipeline.fit(X, y)
    return pipeline


def predict_signals(model, X, threshold=0.6):
    # Using the trained model to predict the probabilities for stocks each day;
    # If a probability exceeds teh threshold (specified here as 0.6), then: generate "buy" signal
    # Baiscally jsut converting the model's output into actionable trading signals to output on the app
    prob = model.predict_proba(X)[:,1]
    signals = (prob > threshold).astype(int)
    return signals, prob

def backtests(df, signals, horizon=5):
    pass


# The plan for how to use these:
# df = create_labels(df)
# df = feature_engineering(df)
# X = df[['sma_5', 'sma_20', 'rsi_14']].dropna()
# y = df['label'].dropna()
# model = train_model(X, y)
# signals, prob = predict_signals(model, X)
# results = backtest(df, signals)