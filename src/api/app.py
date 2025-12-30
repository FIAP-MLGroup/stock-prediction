import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# ===========================
# CONFIGURAÇÕES
# ===========================

DATA_PATH = "data/current.csv"
MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"
WINDOW_SIZE = 60

# ===========================
# PROMETHEUS METRICS
# ===========================

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições no endpoint /predict",
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latência das requisições de inferência",
    registry=registry
)

# ===========================
# CARGA DOS ARTEFATOS
# ===========================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

with open(METRICS_PATH) as f:
    training_metrics = json.load(f)

# ===========================
# FASTAPI
# ===========================

app = FastAPI(title="Stock Prediction API")

# ===========================
# HEALTH
# ===========================

@app.get("/health")
def health():
    return {"status": "ok"}

# ===========================
# PREDICT
# ===========================

@app.get("/predict")
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        df = pd.read_csv(DATA_PATH)

        if "Close" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV não contém a coluna 'Close'"
            )

        closes = df["Close"].astype(float).values

        if len(closes) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"O CSV precisa ter pelo menos {WINDOW_SIZE} valores"
            )

        last_window = closes[-WINDOW_SIZE:].reshape(-1, 1)
        last_window_scaled = scaler.transform(last_window)

        X = np.array([last_window_scaled])
        X = X.reshape((1, WINDOW_SIZE, 1))

        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        return {
            "prediction": float(prediction)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)

# ===========================
# METRICS
# ===========================

@app.get("/metrics")
def metrics():
    """
    Exposição Prometheus:
    - Métricas do modelo (offline)
    - Métricas de runtime (latência e request rate)
    """

    metrics_payload = f"""
# HELP model_mae Mean Absolute Error
# TYPE model_mae gauge
model_mae {training_metrics["mae"]}

# HELP model_rmse Root Mean Squared Error
# TYPE model_rmse gauge
model_rmse {training_metrics["rmse"]}

# HELP model_mape Mean Absolute Percentage Error
# TYPE model_mape gauge
model_mape {training_metrics["mape"]}
"""

    prometheus_metrics = generate_latest(registry).decode("utf-8")

    return PlainTextResponse(
        metrics_payload + "\n" + prometheus_metrics,
        media_type=CONTENT_TYPE_LATEST
    )
