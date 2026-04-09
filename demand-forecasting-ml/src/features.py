import pandas as pd

def create_features(df):
    df = df.copy()

    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)

    df['rolling_mean_7'] = df['value'].rolling(7).mean()

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    df['lag_14'] = df['value'].shift(14)
    df['rolling_mean_14'] = df['value'].rolling(14).mean()

    df = df.dropna()
    return df