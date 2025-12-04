import yfinance as yf
import pandas as pd
import sys
import os

def fetch_stock(symbol, start_date, end_date):
    # baixa os dados
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

    # reseta index para trazer a coluna Date
    df.reset_index(inplace=True)

    # se vier MultiIndex, achatamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # garante as colunas válidas
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[expected_cols]

    return df


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python fetch_yfinance.py SYMBOL START_DATE END_DATE")
        sys.exit(1)

    symbol = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]

    df = fetch_stock(symbol, start, end)

    os.makedirs("data", exist_ok=True)
    output_path = f"data/current.csv"
    df.to_csv(output_path, index=False)

    print(f"Arquivo gerado: {output_path}")
