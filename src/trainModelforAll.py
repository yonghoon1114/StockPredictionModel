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
from config import sequenceLength, data_columns, dataNumber, Date, sp500_top100

def load_data(path: str) -> pd.DataFrame:
    """CSV 파일을 로드하고 Date를 인덱스로 설정"""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

def create_sequences(data, sequenceLength):
    """시퀀스 데이터를 생성하는 함수"""
    X, y = [], []
    for i in range(len(data) - sequenceLength):
        X.append(data[i:i + sequenceLength])  # X는 sequenceLength 길이의 데이터
        y.append(data[i + sequenceLength, 0])  # y는 X 이후의 값 (첫 번째 컬럼을 예측)
    return np.array(X), np.array(y)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer Encoder 블록 정의"""
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def train_transformer_model(df: pd.DataFrame, companyCode: str):
    """Transformer 모델을 학습시키고 예측하는 함수"""
    df = df[data_columns].dropna()  # 필요한 컬럼만 선택하고 결측값 제거

    # MinMaxScaler로 데이터 스케일링
    scalers = {col: MinMaxScaler() for col in data_columns}
    scaled_data = [scalers[col].fit_transform(df[[col]]) for col in data_columns]
    scaled_features = np.hstack(scaled_data)  # 스케일링된 데이터 병합

    X, y = create_sequences(scaled_features, sequenceLength)  # 시퀀스 데이터 생성

    # 학습 데이터와 테스트 데이터 분리
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Transformer 모델 정의
    inputs = Input(shape=(sequenceLength, dataNumber))
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    # 모델 학습
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {companyCode}: {mse:.4f}")

    # 모델 및 스케일러 저장
    os.makedirs("models/scalers", exist_ok=True)
    model.save(os.path.join("models", f"transformer_model_for_{companyCode}.h5"))
    for col in data_columns:
        joblib.dump(scalers[col], f"models/scalers/scaler_transformer_{col}_{companyCode}.joblib")

    return model

if __name__ == "__main__":
    # 각 회사 코드에 대해 모델을 학습
    for companyCode in sp500_top100:
        data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")
        if os.path.exists(data_path):
            df = load_data(data_path)
            trained_model = train_transformer_model(df, companyCode)
        else:
            print(f"데이터 파일 {data_path}가 존재하지 않습니다.")
