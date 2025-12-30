import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# -----------------------------------------------------------------------------
# Configurações
# -----------------------------------------------------------------------------

DATA_PATH = "data/current.csv"
MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"

app = FastAPI(title="LSTM Stock Prediction API")

# -----------------------------------------------------------------------------
# Carregamento de modelo e scaler
# -----------------------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

WINDOW_SIZE = model.input_shape[1]

# -----------------------------------------------------------------------------
# Métricas Prometheus
# -----------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latência das requisições HTTP",
    ["endpoint"]
)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict():
    start = time.time()

    try:
        df = pd.read_csv(DATA_PATH)

        if "Close" not in df.columns:
            REQUEST_COUNT.labels("/predict", "GET", "400").inc()
            raise HTTPException(
                status_code=400,
                detail="Arquivo CSV não contém a coluna 'Close'"
            )

        closes = df["Close"].astype(float).values

        if len(closes) < WINDOW_SIZE:
            REQUEST_COUNT.labels("/predict", "GET", "400").inc()
            raise HTTPException(
                status_code=400,
                detail=f"CSV precisa conter ao menos {WINDOW_SIZE} valores"
            )

        last_window = closes[-WINDOW_SIZE:].reshape(-1, 1)
        last_window_scaled = scaler.transform(last_window)

        X = np.array([last_window_scaled])
        X = X.reshape((1, WINDOW_SIZE, 1))

        prediction_scaled = model.predict(X)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        latency = time.time() - start

        REQUEST_LATENCY.labels("/predict").observe(latency)
        REQUEST_COUNT.labels("/predict", "GET", "200").inc()

        return {
            "prediction": float(prediction)
        }

    except HTTPException:
        raise

    except Exception as e:
        REQUEST_COUNT.labels("/predict", "GET", "500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
