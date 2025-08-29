import pandas as pd
import ta

def create_forward_return_labels(df, horizon=5, threshold=0.01):
    """
    Adds columns for forward return and direction label.
    - forward_return: (future_close / current_close) - 1
    - direction: 1 if forward_return >= threshold, -1 if <= -threshold, else 0
    """
    df['forward_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['direction'] = df['forward_return'].apply(
        lambda x: 1 if x >= threshold else (-1 if x <= -threshold else 0)
    )
    return df