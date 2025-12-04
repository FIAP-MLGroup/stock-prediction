import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

# ===========================
# CONFIGURAÇÕES FIXAS
# ===========================

DATA_PATH = "data/current.csv"
MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"
WINDOW_SIZE = 60

# ===========================
# CARGA DOS ARTEFATOS
# ===========================

print("Carregando modelo...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Carregando scaler...")
scaler = joblib.load(SCALER_PATH)

print("Carregando métricas...")
with open(METRICS_PATH) as f:
    training_metrics = json.load(f)

# ===========================
# FASTAPI
# ===========================

app = FastAPI(title="Stock Prediction API")

# ===========================
# HEALTHCHECK
# ===========================

@app.get("/health")
def health():
    return {"status": "ok"}

# ===========================
# PREDICT (ÚNICO ATIVO)
# ===========================

@app.get("/predict")
def predict():
    start = time.time()

    try:
        df = pd.read_csv(DATA_PATH)

        if "Close" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV sem coluna 'Close'")

        closes = df["Close"].astype(float).values

        if len(closes) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"CSV precisa ter no mínimo {WINDOW_SIZE} valores"
            )

        last_window = closes[-WINDOW_SIZE:].reshape(-1, 1)
        last_window_scaled = scaler.transform(last_window)

        X = np.array([last_window_scaled])
        X = X.reshape((1, WINDOW_SIZE, 1))

        prediction_scaled = model.predict(X)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        latency = time.time() - start

        return {
            "prediction": float(prediction),
            "latency_seconds": latency
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# MÉTRICAS PROMETHEUS
# ===========================

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    prometheus_metrics = f"""
# HELP model_mae Mean Absolute Error
# TYPE model_mae gauge
model_mae {training_metrics['mae']}

# HELP model_rmse Root Mean Squared Error
# TYPE model_rmse gauge
model_rmse {training_metrics['rmse']}

# HELP model_mape Mean Absolute Percentage Error
# TYPE model_mape gauge
model_mape {training_metrics['mape']}
"""
    return prometheus_metrics.strip()
