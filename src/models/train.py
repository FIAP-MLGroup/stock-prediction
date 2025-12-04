import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ===========================
# CONFIGURAÇÃO FIXA DO ATIVO
# ===========================

DATA_PATH = "data/current.csv"   # Sempre um único ativo por imagem
MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"

WINDOW_SIZE = 60
TEST_SPLIT = 0.2
EPOCHS = 25
BATCH_SIZE = 32

# ===========================
# FUNÇÕES
# ===========================

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ===========================
# PIPELINE DE TREINAMENTO
# ===========================

print("Carregando dados...")

df = pd.read_csv(DATA_PATH)

if "Close" not in df.columns:
    raise ValueError("O CSV precisa conter a coluna 'Close'.")

prices = df["Close"].astype(float).values.reshape(-1, 1)

print(f"Total de amostras: {len(prices)}")

# ===========================
# NORMALIZAÇÃO
# ===========================

scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# ===========================
# JANELAMENTO
# ===========================

X, y = create_sequences(prices_scaled, WINDOW_SIZE)

X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")

# ===========================
# SPLIT TREINO / VALIDAÇÃO
# ===========================

split_index = int(len(X) * (1 - TEST_SPLIT))

X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"Treino: {len(X_train)} amostras")
print(f"Validação: {len(X_val)} amostras")

# ===========================
# MODELO LSTM
# ===========================

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(WINDOW_SIZE, 1)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# ===========================
# TREINAMENTO
# ===========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("✅ Iniciando treinamento...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ===========================
# AVALIAÇÃO
# ===========================

print("Avaliando modelo...")

y_val_pred = model.predict(X_val)

# Desnormaliza
y_val_real = scaler.inverse_transform(y_val)
y_val_pred_real = scaler.inverse_transform(y_val_pred)

mae = mean_absolute_error(y_val_real, y_val_pred_real)
rmse = np.sqrt(mean_squared_error(y_val_real, y_val_pred_real))
mape = np.mean(np.abs((y_val_real - y_val_pred_real) / y_val_real)) * 100

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape)
}

print("Métricas de Validação:")
print(json.dumps(metrics, indent=2))

# ===========================
# SALVAMENTO
# ===========================

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Modelo salvo em:", MODEL_PATH)
print("Scaler salvo em:", SCALER_PATH)
print("Métricas salvas em:", METRICS_PATH)
print("Treinamento finalizado com sucesso!")
