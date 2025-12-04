import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
import os

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def preprocess(csv_path, window=60):
    df = pd.read_csv(csv_path, index_col=0)

    # remove linhas totalmente vazias
    df = df.dropna(how="all")

    # força o tipo numérico e remove valores inválidos
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # remove linhas onde Close virou NaN (ex: 'PETR4.SA')
    df = df.dropna(subset=["Close"])

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])

    joblib.dump(scaler, "models/scaler.pkl")

    X, y = create_sequences(scaled, window)

    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

    print("Pré-processamento concluído!")
    print("Gerado: data/X.npy, data/y.npy, models/scaler.pkl")

if __name__ == "__main__":
    preprocess(sys.argv[1])
