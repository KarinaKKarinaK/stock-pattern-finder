import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import matplotlib.pyplot as plt
# import ta
# df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
# For now this scaffold covers:

# Label creation
# Feature engineering
# Model training and prediction
# Backtesting logic

def create_labels(df, horizon=5, threshold=0.015):
    # Creating binary labels, returns 1 if teh forward return over the next [horizon] days is >= threshold, else 0
    df = df.copy()
    df['forward_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['label'] = (df['forward_return'] >= threshold).astype(int)
    return df

def feature_engineering(df):
    # Adding technical indictaors as teh features for the ML model
    df = df.copy()

    # SImploe Moving Average (SMA)
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    df['rsi_14'] = compute_rsi(df['Close'], window=14)

    # Exponential Moving Average (EMA) = "a type of moving average that gives more weight to recent prices,
    #  making it more responsive to new information than the SMA"
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
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

def backtest(df, signals, horizon=5):
    # Here I just simulate trading based on signals in order to evaluate the strategy's performance
    #How: for each signal, enter a trade and exit after the horizon, then calculate the return
    
    trades = []
    
    for i in range(len(signals)):
        if signals[i] == 1:
            entry_price = df['Close'].iloc[i]
            if i + horizon < len(df):
                exit_price = df['Close'].iloc[i + horizon]
                ret = (exit_price / entry_price) - 1
                trades.append(ret)

    return trades


# The plan for how to use these:
df = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
df = create_labels(df)
df = feature_engineering(df)
df.columns = [col[0] for col in df.columns]  # Flattens to 'Close', 'High', etc.
print(df.columns)

# Add or subtract the indicators included in teh feature_engineering function as you see fit for highest returns

# Add a feature to UI that allows the usre to modify which indicators to turn on and which to turn off
features = ['sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal']
#dropna() removes rows/columns that contain Not a Number (NaN) values
df = df.dropna(subset=features + ['label'])  # Remove rows with missing feature or label values

X = df[features]
y = df['label']

model = train_model(X, y)
signals, prob = predict_signals(model, X)
results = backtest(df, signals)

print(results)

for result in results:
    # result = (result / 100)
    print(f"Trade return: {result:.2%}")

def strategy_analysis(results):
    # Analyze returns (mean, median, win rate, etc.).
    mean = 0
    for result in results:
        # result = (result / 100)
        mean += result
    mean /= len(results)
    print(f"Mean trade return: {mean:.2%}")

def visualize_returns(results):
    # Visualize them (histogram, cumulative return plot).
    plt.figure(figsize=(12, 6))
    plt.hist(results, bins=30, alpha=0.7, color='blue')
    plt.title('Trade Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

strategy_analysis(results)
visualize_returns(results)