import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_hdf(path, key='/building1/elec/meter1')
    df = df.reset_index()
    df.columns = ['timestamp', 'power']
    df = df.sort_values('timestamp').dropna()

    df = df.iloc[:100000]

    return df

def preprocess(df):
    scaler = MinMaxScaler()
    df['power_scaled'] = scaler.fit_transform(df[['power']]).astype(float)

    df['power_scaled'] += np.random.normal(0, 0.02, len(df))
    df['power_scaled'] = np.clip(df['power_scaled'], 0, 1)

    df['label'] = 0

    for _ in range(100):
        s = np.random.randint(0, len(df)-50)
        seg = df.loc[s:s+50, 'power_scaled'].values
        drift = np.linspace(0.1, 0.25, len(seg))
        noise = np.random.normal(0, 0.03, len(seg))

        df.loc[s:s+50, 'power_scaled'] = np.clip(seg + drift + noise, 0, 1)
        df.loc[s:s+50, 'label'] = 1

    return df

def create_sequences(df, seq_len=20):
    values = df['power_scaled'].values
    labels = df['label'].values

    X, y = [], []

    for i in range(len(df)-seq_len):
        X.append(values[i:i+seq_len])
        y.append(1 if np.any(labels[i:i+seq_len]==1) else 0)

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    return X, y