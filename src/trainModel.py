import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from config import companyCode

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # 예측할 대상은 stock_close (0번째 열)
    return np.array(X), np.array(y)

def train_lstm_model(df: pd.DataFrame, sequence_length=200):
    df = df[["stock_close", "rate_close", "nasdaq_close", "target", "Revenue", "NetIncome", "TotalAssets"]].dropna()

    # 정규화
    scaler_stock = MinMaxScaler()
    scaler_rate = MinMaxScaler()
    scaler_nasdaq = MinMaxScaler()
    scaler_Revenue = MinMaxScaler()
    scaler_NetIncome = MinMaxScaler()
    scaler_TotalAssets = MinMaxScaler()

    scaled_stock = scaler_stock.fit_transform(df[["stock_close"]])
    scaled_rate = scaler_rate.fit_transform(df[["rate_close"]])
    scaled_nasdaq = scaler_nasdaq.fit_transform(df[["nasdaq_close"]])
    scaled_Revenue = scaler_Revenue.fit_transform(df[["Revenue"]])
    scaled_NetIncome = scaler_NetIncome.fit_transform(df[["NetIncome"]])
    scaled_totalAssets = scaler_TotalAssets.fit_transform(df[["TotalAssets"]])

    # 시퀀스 생성
    scaled_features = np.hstack((scaled_stock, scaled_rate, scaled_nasdaq,scaled_Revenue, scaled_NetIncome,scaled_totalAssets))
    X, y = create_sequences(scaled_features, sequence_length)

    # 학습/테스트 분할
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM 입력 형태: (samples, time steps, features)
    # X_train = X_train[:, :, :3]
    # X_test = X_test[:, :, :3]

    # 모델 정의
    # 모델 정의 시 input_shape 수정
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 6)),
        Dense(1)
    ])  
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    # 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    # 모델 및 스케일러 저장
    os.makedirs("models/scalers", exist_ok=True)
    model.save(os.path.join("models", f"lstm_model_for_{companyCode}.h5"))
    joblib.dump(scaler_stock, f"models/scalers/scaler_stock_{companyCode}.joblib")
    joblib.dump(scaler_rate, "models/scalers/scaler_rate.joblib")
    joblib.dump(scaler_nasdaq, "models/scalers/scaler_nasdaq.joblib")
    joblib.dump(scaler_TotalAssets,f"models/scalers/scaler_TotalAssets_{companyCode}.joblib")
    joblib.dump(scaler_Revenue,f"models/scalers/scaler_Revenue_{companyCode}.joblib")
    joblib.dump(scaler_NetIncome,f"models/scalers/scaler_NetIncome_{companyCode}.joblib")
    return model

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", f"{companyCode}_merged.csv")
    df = load_data(data_path)
    trained_model = train_lstm_model(df)
