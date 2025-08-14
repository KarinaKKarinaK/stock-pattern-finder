import streamlit as st
import pandas as pd
import yfinance as yf

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

st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL")
if ticker_symbol not in yf.Ticker(ticker_symbol).info:
    st.error("Invalid ticker symbol.")
    st.stop()

date_range = st.sidebar.date_input("Date Range", [pd.to_datetime("2020-01-01"), pd.to_datetime("2023-01-01")])
fetch_data_btn = st.sidebar.button("Fetch Data")

if fetch_data_btn:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found. Please check the ticker symbol and date range.")
    else:
        st.write(f"Fetching data for {ticker_symbol} from {date_range[0]} to {date_range[1]}")
        st.dataframe(df)
        st.line_chart(df['Close'])

st.pyplot()