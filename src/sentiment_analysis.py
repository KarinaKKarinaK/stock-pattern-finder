from textblob import TextBlob
# for this use a dictionary that looks like this: news_dict = {'2022-01-01': ['Apple launches new product', ...], ...}

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity  # Range: [-1, 1]

def aggregate_daily_sentiment(news_dict):
    sentiment_by_date = {}
    for date, headlines in news_dict.items():
        scores = [get_sentiment_score(h) for h in headlines]
        sentiment_by_date[date] = np.mean(scores) if scores else 0
    return sentiment_by_date