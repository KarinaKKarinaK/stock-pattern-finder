# What this Streamlit app does
This app is a Stock Pattern Finder that helps users analyze historical stock price data and discover trading patterns using machine learning and technical analysis. It fetches stock data (e.g., from Yahoo Finance), computes a wide range of technical indicators, creates trading signals, and simulates trades to evaluate strategy performance.


# Features
- Stock price pattern detection using ML (Logistic Regression, Random Forest, etc.)
- Technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands
- Backtesting with realistic trading rules (transaction costs, stop-loss, take-profit)
- Sentiment analysis from news headlines (NewsAPI + TextBlob)
- Streamlit web interface for interactive analysis

# How this app works
1. Data Fetching:
Downloads historical price data for a selected stock ticker (e.g., NVDA, MSFT, or AAPL).

2. Feature Engineering:
Calculates technical indicators such as:
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands, ATR
- Stochastic Oscillator, VWAP, Volatility, Volume Z-score, Rate of Change
- Williams %R (overbought/oversold)
- Candlestick patterns (Doji, Hammer, Engulfing)

4. Label Creation:
Assigns binary labels for supervised learning (e.g., "buy" if the forward return exceeds a threshold).

5. Model Training:
Trains a machine learning classifier (Logistic Regression) to predict trading signals based on the engineered features.

6. Signal Prediction:
Uses the trained model to generate buy/sell signals.

7. Backtesting:
Simulates trades using the predicted signals and calculates returns to evaluate strategy performance.

8. Visualization:
Displays results, trade returns, and indicator values for user analysis.

# Pattern Recognition Strategy (ML)
## Supervised “is this a good entry now?”

**Goal of this approach:** turn patterns into a **label** and train a classifier/regressor.

- **Label ideas:**
    - **Classification:** `1` if **forward return** over next *H* days ≥ threshold (e.g., +1.5% in 5 days), else `0`.
    - **Regression:** predict **forward return** over next *H* days, later threshold it for signals.
- **Models:** Start simple → **LogisticRegression**, **RandomForest**, **GradientBoosting** (or XGBoost if allowed).
- **Pipeline:** `ColumnSelector → Imputer → Scaler → Model` (scikit-learn Pipeline).
- **CV:** `TimeSeriesSplit(n_splits=5)` with **purged** windows (no overlapping leakage when labels use horizons).
- **Model selection:** compare simple baselines to avoid overfitting; tune only a few hyperparams.
- **Interpretation:** permutation importance; partial dependence profiles for top features.

**Outputs I'm using:** probability/score per day → **convert to trades** with simple rules (e.g., score > 0.6 enter long; stop/TP rules; max concurrent positions = 1 for MVP).

# Feature Engineering Rules/ Guiding Principles:

The features must be:
- Predictive (contain information about future returns)
- Available in real-time (no lookahead bias — always shift by +1 bar)
- Stable/out-of-sample robust (not just fit noise in history)

-> Rule: Only including indicators that can be computed from past and present data, never future.

# Thsi project fetaures teh following ML techniques:

### Supervised Learning:
- Creating labels (classification or regression) based on future returns and train models to predict them.

### Classification Models:

- Logistic Regression, Random Forest, Gradient Boosting (and optionally XGBoost) to predict “good entry” points.

### Feature Engineering:

- Generating technical indicators (moving averages, RSI, etc.) as input features for the models.

## Down-the-line Additions:
### Pipeline Processing:

- Using scikit-learn Pipelines for preprocessing (imputation, scaling) and modeling.

### Time Series Cross-Validation:

- Using TimeSeriesSplit to evaluate model performance without data leakage.

### Model Interpretation:

- Techniques like permutation importance and partial dependence profiles to understand feature impact.


## The Feature Engineering Function

The `feature_engineering` function in `src/strategy.py` automatically generates a rich set of technical indicators from your stock price data. These features are designed to help machine learning models identify patterns and make predictions about future price movements.

**What it does:**
- Computes moving averages (SMA, EMA) for trend detection.
- Calculates Relative Strength Index (RSI) for momentum analysis.
- Derives MACD and its signal line for trend and momentum.
- Adds Bollinger Bands for volatility and price extremes.
- Computes Average True Range (ATR) for volatility measurement.
- Generates Stochastic Oscillator values for overbought/oversold signals.
- Calculates VWAP for average price weighted by volume.
- Adds rolling volatility and volume z-score for market activity.
- Computes Rate of Change (ROC) for momentum.

**How it works:**
- All indicators are calculated using only past and present data (no lookahead bias).
- Each feature is added as a new column to your DataFrame.
- The function is used as a preprocessing step before model training, ensuring that all relevant technical signals are available for the ML pipeline.

## Sentiment Analysis
- Using newsapi to fetch news headlines in a given timeframe
- Using textblob for sentiment analysis of the news

# GEneral Use Info
## Project Overview

Stock Pattern Finder is an interactive web app for analyzing historical stock price data, discovering trading patterns, and generating actionable signals using machine learning and technical analysis. The app is designed for both beginners and experienced traders who want to explore quantitative strategies and evaluate their performance.

## How to Use the App

1. **Enter a Stock Ticker:**  
   Use the sidebar to input a stock symbol (e.g., AAPL, MSFT, NVDA).

2. **Select Date Range:**  
   Choose the time period for analysis. The app will fetch historical price data for the selected range.

3. **Fetch Data & Visualize:**  
   Click "Fetch Data" to load and plot the stock’s price history and technical indicators.

4. **Explore Features:**  
   The app automatically computes a variety of technical indicators (moving averages, RSI, MACD, Williams %R, candlestick patterns, etc.) and displays them for analysis.

5. **Run Pattern Recognition:**  
   The app trains a machine learning model to detect buy/sell signals based on historical patterns and displays predicted signals.

6. **Backtest Strategies:**  
   Simulate trades using the predicted signals and review performance metrics such as returns and drawdowns.

7. **Sentiment Analysis (Optional):**  
   Fetch recent news headlines and analyze sentiment to see how news may impact trading signals.

## Customization

- Users can select which technical indicators to include in the analysis.
- Thresholds for signal generation and model parameters can be adjusted for personalized strategies.
- Future updates will allow users to choose investment horizons and simulate portfolio growth.

## Getting Started

1. Clone the repository and install required packages from `requirements.txt`.
2. Run `streamlit run app.py` in your terminal.
3. Open the app in your browser and start exploring stock patterns!

## Limitations

- NewsAPI usage is limited by free tier restrictions.
- Only daily price bars are supported for now.
- Model predictions are for educational purposes and not financial advice.

---

For more details and future plans, see [`brainstormin_ideas.md`](brainstormin_ideas.md).