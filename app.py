import streamlit as st
import pandas as pd
import yfinance as yf
from src.sentiment_analysis import build_news_dict, aggregate_daily_sentiment, sentiment_to_label
from src.config import NEWSAPI_KEY
from datetime import date
import matplotlib.pyplot as plt

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
start_date = st.sidebar.date_input("Start Date", date(2025, 7, 20))
end_date = st.sidebar.date_input("End Date", date(2025, 8, 19))
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

if run_sentiment:
    st.info("Fetching news and analyzing sentiment...")
    news_dict = build_news_dict(ticker, start_date, end_date, NEWSAPI_KEY)
    sentiment = aggregate_daily_sentiment(news_dict)
    sentiment_labels = {d: sentiment_to_label(s) for d, s in sentiment.items()}
    st.write("Daily Sentiment Scores:", sentiment)
    st.write("Daily Sentiment Labels:", sentiment_labels)
    st.line_chart(list(sentiment.values()))
    st.write("Buy/Hold/Sell signals by date:")
    st.table(sentiment_labels)

    sentiment_series = pd.Series(sentiment)
    sentiment_series.index = pd.to_datetime(sentiment_series.index)
    st.line_chart(sentiment_series)

    fig = visualize_returns(results)
    st.pyplot(fig)
