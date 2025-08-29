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
import pandas_ta as ta
from src.labels import create_forward_return_labels

# import ta
# df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)


def create_labels(df, horizon=5, threshold=0.005):
    # Creating binary labels, returns 1 if teh forward return over the next [horizon] days is >= threshold, else 0
    df = df.copy()
    df['forward_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['label'] = (df['forward_return'] >= threshold).astype(int)
    return df

# Williams %R - Similar to Stochastic, shows overbought/oversold.
# How to calculate: (Highest High - Close) / (Highest High - Lowest Low) * -100
def williams_r(df, period=14):
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    wr = (high - df['Close']) / (high - low) * -100
    return wr

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

    # ATR (Average True Range, window=14)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = df['TR'].rolling(14).mean().shift(1)

    # Stochastic Oscillator
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # RRolling Volatility
    df['rolling_vol20'] = df['Close'].pct_change().rolling(20).std().shift(1)

    # Volume Z-Score:
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['volume_z'] = (df['Volume'] - vol_mean) / vol_std

    # Rate of Change (Momentum):
    df['roc5'] = df['Close'].pct_change(5).shift(1)

    # WIlliams R%
    df['Williams_%R'] = williams_r(df)

    # Candle Patterns
    # Doji, Hammer, Engulfing patterns as binary features
    df['Doji'] = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])
    df['Hammer'] = ta.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
    df['Engulfing'] = ta.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])

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

    # Check if y contains at least two classes
    if len(np.unique(y)) < 2:
        print("Error: Not enough classes in labels to train a classifier. Found only:", np.unique(y))
        # Optionally, handle this case (skip training, adjust data, etc.)
    else:
        # Optionally add: cross-validation, hyperparameter tuning, etc
        pipeline.fit(X, y)

    print("Label distribution:", np.unique(y, return_counts=True))
    

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
df = create_forward_return_labels(df, horizon=5, threshold=0.01)
df = create_labels(df)
df = feature_engineering(df)
df.columns = [col[0] for col in df.columns]  # Flattens to 'Close', 'High', etc.
print(df.columns)

# Add or subtract the indicators included in teh feature_engineering function as you see fit for highest returns

# Add a feature to UI that allows the usre to modify which indicators to turn on and which to turn off
features = ['sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'VWAP', 'rolling_vol20', 'volume_z', 'roc5', 'Williams_%R']
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