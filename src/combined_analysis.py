import pandas as pd

def merge_sentiment_with_features(df, sentiment_dict):
    """
    Merge daily sentiment scores into the main feature DataFrame.
    Assumes df has a DatetimeIndex or a 'Date' column.
    sentiment_dict: {date_str: sentiment_score}
    """
    sentiment_df = pd.DataFrame(list(sentiment_dict.items()), columns=['Date', 'sentiment_score'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        merged = pd.merge(df, sentiment_df, on='Date', how='left')
    else:
        merged = df.copy()
        merged['sentiment_score'] = merged.index.map(sentiment_dict)
    return merged