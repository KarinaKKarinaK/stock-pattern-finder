from textblob import TextBlob
import requests
from config import NEWSAPI_KEY

# for this use a dictionary that looks like this: news_dict = {'2022-01-01': ['Apple launches new product', ...], ...}


def fetch_news_headlines(ticker, from_date, to_date, api_key):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
        "pageSize": 100
    }
    response = requests.get(url, params=params)
    data = response.json()
    headlines = []
    if "articles" in data:
        for article in data["articles"]:
            headlines.append(article["title"])
    return headlines

from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def build_news_dict(ticker, start_date, end_date, api_key):
    news_dict = {}
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        headlines = fetch_news_headlines(ticker, date_str, date_str, api_key)
        news_dict[date_str] = headlines
    return news_dict

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity  # Range: [-1, 1]

def aggregate_daily_sentiment(news_dict):
    sentiment_by_date = {}
    for date, headlines in news_dict.items():
        scores = [get_sentiment_score(h) for h in headlines]
        sentiment_by_date[date] = np.mean(scores) if scores else 0
    return sentiment_by_date

api_key = NEWSAPI_KEY
ticker = "AAPL"
start_date = date(2022, 1, 1)
end_date = date(2022, 1, 10)
news_dict = build_news_dict(ticker, start_date, end_date, api_key)