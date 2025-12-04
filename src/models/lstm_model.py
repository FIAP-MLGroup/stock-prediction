from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(window_size, n_features=1):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
