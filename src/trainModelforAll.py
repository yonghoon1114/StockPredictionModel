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

# 시퀀스 길이: 과거 40일 데이터를 보고 예측
sequenceLength = 40

# 사용할 특성(피처) 목록: 종가, 금리, 나스닥, 재무제표, RSI, 금 가격, 선거 여부, 상대 수익률 등
data_columns = ["stock_close", "rate_close", "nasdaq_close", "Revenue", "NetIncome", "TotalAssets", 
                "RSI", "gold_close", "election_marker", "relative", "sector_close"]
dataNumber = len(data_columns)

# CSV 파일 불러오기 및 날짜를 인덱스로 지정
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

# 시계열 데이터를 Transformer에 맞게 슬라이딩 윈도우 방식으로 나눔
def create_sequences(data, sequenceLength):
    X, y = [], []
    for i in range(len(data) - sequenceLength):
        X.append(data[i:i + sequenceLength])  # X: sequenceLength 길이의 연속된 구간
        y.append(data[i + sequenceLength, 0])  # y: 그 다음 날의 주가 (첫 번째 열: stock_close)
    return np.array(X), np.array(y)

# Transformer Encoder 구성 (Self-Attention + Feed Forward 네트워크)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Multi-Head Self Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])  # Residual Connection (정보 유실 방지)
    x = LayerNormalization(epsilon=1e-6)(x)  # 정규화 (학습 안정화)

    # 2. Position-wise Feed Forward Network
    x_ff = Dense(ff_dim, activation="relu")(x)  # 차원 확장 (복잡한 표현 학습)
    x_ff = Dense(inputs.shape[-1])(x_ff)        # 원래 차원으로 복원
    x = Add()([x, x_ff])  # Residual Connection
    x = LayerNormalization(epsilon=1e-6)(x)  # 정규화
    return x

# 전체 학습 및 평가 함수
def train_transformer_model(df: pd.DataFrame, companyCode: str):
    df = df[data_columns].dropna()  # 필요한 컬럼만 남기고 결측값 제거

    # 각 특성에 대해 MinMax 스케일링 (0~1 범위로 정규화)
    scalers = {col: MinMaxScaler() for col in data_columns}
    scaled_data = [scalers[col].fit_transform(df[[col]]) for col in data_columns]
    scaled_features = np.hstack(scaled_data)  # (행, 열) 기준으로 붙임

    # 시퀀스 데이터 생성 (예: 40일간 데이터를 보고 다음 날 예측)
    X, y = create_sequences(scaled_features, sequenceLength)

    # 학습 데이터와 테스트 데이터 분리 (80%/20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Transformer 기반 모델 정의
    inputs = Input(shape=(sequenceLength, dataNumber))  # 입력: (40일, 11개 특성)
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)  # 시계열 평균 (차원 축소)
    x = Dropout(0.1)(x)  # 과적합 방지용 드롭아웃
    x = Dense(64, activation="relu")(x)  # 비선형 변환 (차원 확장)
    outputs = Dense(1)(x)  # 출력: 하나의 예측값 (예: 주가)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")  # 손실함수: 평균 제곱 오차

    # 모델 학습 시작
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # 테스트 데이터 예측 및 성능 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {companyCode}: {mse:.4f}")

    # 모델 및 스케일러 저장
    os.makedirs("models/scalers", exist_ok=True)
    model.save(os.path.join("models", f"transformer_model_for_{companyCode}.h5"))
    for col in data_columns:
        joblib.dump(scalers[col], f"models/scalers/scaler_transformer_{col}_{companyCode}.joblib")

    return model

# 여러 회사에 대해 학습을 반복 실행하는 함수
def trainModel(Companies_for_prediction: list, Date): 
    for companyCode in Companies_for_prediction:
        data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")
        if os.path.exists(data_path):
            df = load_data(data_path)
            train_transformer_model(df, companyCode)
        else:
            print(f"데이터 파일 {data_path}가 존재하지 않습니다.")
