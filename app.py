import streamlit as st
import pandas as pd
import yfinance as yf
from src.sentiment_analysis import build_news_dict, aggregate_daily_sentiment, sentiment_to_label
from src.config import NEWSAPI_KEY
from datetime import date
import matplotlib.pyplot as plt
from src.strategy import create_labels, feature_engineering, train_model, predict_signals, backtest
import datetime

st.title("Stock Pattern Finder")

st.header("Welcome :) This app helps you find stock patterns in historical data.")

st.subheader("Instructions")

st.markdown("""
## Stock Pattern Finder

Welcome! This app helps you analyze historical stock price data and discover patterns using machine learning.

**How to use:**
1. Enter a stock ticker symbol (e.g., `AAPL`).
2. Select a date range for analysis.
3. Click "Fetch Data" to load and visualize the stock’s price history.
4. Explore detected patterns and try the moving average crossover strategy with backtesting.

Use the sidebar to adjust your inputs. Results and charts will appear here!
""")


st.sidebar.header("Sentiment Analysis")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=today, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=today, value=today)
run_sentiment = st.sidebar.button("Analyze Sentiment")

def visualize_returns(results):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(results, bins=30, alpha=0.7, color='blue')
    ax.set_title('Trade Returns Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid()
    return fig

today = datetime.date.today()
min_date = datetime.date(2025, 7, 23)  # Update this if NewsAPI error message changes

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=today, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=today, value=today)

start_date_str = max(start_date, min_date).strftime("%Y-%m-%d")
end_date_str = min(end_date, today).strftime("%Y-%m-%d")

if run_sentiment:
    try:
        st.info("Fetching news and analyzing sentiment...")
        news_dict = build_news_dict(ticker, start_date_str, end_date_str, NEWSAPI_KEY)
        sentiment = aggregate_daily_sentiment(news_dict)
        sentiment_labels = {d: sentiment_to_label(s) for d, s in sentiment.items()}
        sentiment_series = pd.Series(sentiment)
        sentiment_series.index = pd.to_datetime(sentiment_series.index)
        st.line_chart(sentiment_series)
        st.write("Buy/Hold/Sell signals by date:")
        st.table(sentiment_labels)
    except Exception as e:
        st.warning(f"NewsAPI error: {e}. Sentiment analysis is unavailable for the selected dates.")

st.sidebar.header("ML Strategy")
ml_ticker = st.sidebar.text_input("ML Ticker Symbol", "AAPL", key="ml_ticker")
ml_start_date = st.sidebar.date_input("ML Start Date", date(2025, 7, 21), key="ml_start")
ml_end_date = st.sidebar.date_input("ML End Date", date(2025, 8, 19), key="ml_end")
run_strategy = st.sidebar.button("Run ML Strategy")

if run_strategy:
    st.info(f"Running ML strategy for {ml_ticker} from {ml_start_date} to {ml_end_date}...")
    df = yf.download(ml_ticker, start=ml_start_date, end=ml_end_date)
    if df.empty:
        st.error("No data found. Please check the ticker symbol and date range.")
    else:
        df = create_labels(df)
        df = feature_engineering(df)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        print(df.columns)
        features = ['sma_5', 'sma_20', 'rsi_14', 'macd', 'macd_signal']
        df = df.dropna(subset=features + ['label'])  # Remove rows with missing feature or label values

        X = df[features]
        y = df['label']

        model = train_model(X, y)
        signals, prob = predict_signals(model, X)
        results = backtest(df, signals)
        st.write("Trade returns:", results)
        st.write(f"Mean trade return: {pd.Series(results).mean():.2%}")
        # Visualize returns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(results, bins=30, alpha=0.7, color='blue')
        ax.set_title('Trade Returns Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.grid()
        st.pyplot(fig)


        # df = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
