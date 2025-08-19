import streamlit as st
import pandas as pd
import yfinance as yf
from src.sentiment_analysis import build_news_dict, aggregate_daily_sentiment, sentiment_to_label
from config import NEWSAPI_KEY
from datetime import date

st.title("Stock Pattern Finder")

st.header("Welcome! This app helps you find stock patterns in historical data.")

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

# st.sidebar.header("User Input")
# ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL")
# if ticker_symbol not in yf.Ticker(ticker_symbol).info:
#     st.error("Invalid ticker symbol.")
#     st.stop()

# date_range = st.sidebar.date_input("Date Range", [pd.to_datetime("2020-01-01"), pd.to_datetime("2023-01-01")])
# fetch_data_btn = st.sidebar.button("Fetch Data")


# if fetch_data_btn:
#     start_date = pd.to_datetime(date_range[0])
#     end_date = pd.to_datetime(date_range[1])
#     df = yf.download(ticker_symbol, start=start_date, end=end_date)
#     df = yf.download(ticker_symbol, start=start_date, end=end_date)
#     if df.empty:
#         st.error("No data found. Please check the ticker symbol and date range.")
#     else:
#         st.write(f"Fetching data for {ticker_symbol} from {date_range[0]} to {date_range[1]}")
#         st.dataframe(df)
#         st.line_chart(df['Close'])

st.pyplot()