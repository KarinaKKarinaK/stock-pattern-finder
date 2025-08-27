from textblob import TextBlob
import requests
import numpy as np
from src.config import NEWSAPI_KEY
from newsapi import NewsApiClient
from datetime import timedelta, date, datetime


newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# for this use a dictionary that looks like this: news_dict = {'2022-01-01': ['Apple launches new product', ...], ...}

# Fetching news data =========================================
def fetch_news_headlines(ticker, from_date, to_date, api_key):
    newsapi = NewsApiClient(api_key=api_key)
    response = newsapi.get_everything(q=ticker, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=100)
    headlines = [article['title'] for article in response.get('articles', [])]
    return headlines

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def build_news_dict(ticker, start_date, end_date, api_key):
    # Only convert if input is a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    # Clamp start_date to min_date
    min_date = date(2025, 7, 23)  # Update as needed
    news_dict = {}
    for single_date in daterange(start_date, end_date):
        if single_date < min_date:
            continue  # Skip dates before allowed minimum
        date_str = single_date.strftime("%Y-%m-%d")
        headlines = fetch_news_headlines(ticker, date_str, date_str, api_key)
        news_dict[date_str] = headlines
    return news_dict

# Sentiment analysis =========================================
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity  # Range: [-1, 1]

def aggregate_daily_sentiment(news_dict):
    sentiment_by_date = {}
    for date, headlines in news_dict.items():
        scores = [get_sentiment_score(h) for h in headlines]
        sentiment_by_date[date] = np.mean(scores) if scores else 0
    return sentiment_by_date

def sentiment_to_label(sentiment_score, buy_threshold=0.1, sell_threshold=-0.1):
    if sentiment_score > buy_threshold:
        return "buy"
    elif sentiment_score < sell_threshold:
        return "sell"
    else:
        return "hold"

ticker = "MSFT"
today = datetime.today()
start_date = today - timedelta(days=28)
start_date_str = start_date.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD' for NewsAPI
end_date_str = today.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD' for NewsAPI
news_dict = build_news_dict(ticker, start_date_str, end_date_str, NEWSAPI_KEY)
sentiment = aggregate_daily_sentiment(news_dict)

labeled_sentiment = {date: sentiment_to_label(score) for date, score in sentiment.items()}
print(labeled_sentiment)