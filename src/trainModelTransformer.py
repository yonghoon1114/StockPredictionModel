import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from config import companyCode, sequenceLength, data_columns, dataNumber, Date

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

def create_sequences(data, sequenceLength):
    X, y = [], []
    for i in range(len(data) - sequenceLength):
        X.append(data[i:i + sequenceLength])
        y.append(data[i + sequenceLength, 0])
    return np.array(X), np.array(y)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def train_transformer_model(df: pd.DataFrame):
    df = df[data_columns].dropna()

    scalers = {col: MinMaxScaler() for col in data_columns}
    scaled_data = [scalers[col].fit_transform(df[[col]]) for col in data_columns]
    scaled_features = np.hstack(scaled_data)

    X, y = create_sequences(scaled_features, sequenceLength)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    inputs = Input(shape=(sequenceLength, dataNumber))
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    os.makedirs("models/scalers", exist_ok=True)
    model.save(os.path.join("models", f"transformer_model_for_{companyCode}_elaborated.h5"))
    for col in data_columns:
        joblib.dump(scalers[col], f"models/scalers/elaborated_scaler_transformer_{col}_{companyCode}.joblib")

    return model

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_elaborated_merged.csv")
    df = load_data(data_path)
    trained_model = train_transformer_model(df)
