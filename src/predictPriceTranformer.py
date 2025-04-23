import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from config import companyCode, sequenceLength, data_columns, dataNumber, Date

def load_transformer_model(model_path: str):
    return load_model(model_path, compile=False)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

def load_data_for_prediction(
    path: str,
    scalers: dict
) -> np.ndarray:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    # 가장 최근 시퀀스 추출
    recent_seq = df[data_columns].iloc[-sequenceLength:]

    # 각 feature별로 scaler를 이용해 변환
    scaled_features = np.hstack([scalers[col].transform(recent_seq[[col]]) for col in data_columns])

    # LSTM 입력 형태: (samples, time steps, features)
    return scaled_features.reshape(1, sequenceLength, dataNumber)

def predict_future_days(
    model, df, scalers, days=50
):
    df_sorted = df.sort_values("Date").copy()
    recent_seq = df_sorted[data_columns].iloc[-sequenceLength:]

    predictions = []

    for _ in range(days):
        # 각 feature별로 scaler를 이용해 변환
        scaled_features = np.hstack([scalers[col].transform(recent_seq[[col]]) for col in data_columns])

        # 모델 입력 형태에 맞게 reshape
        X_input = scaled_features.reshape(1, sequenceLength, dataNumber)

        # 예측 수행
        pred_scaled = model.predict(X_input)[0][0]
        pred_price = scalers['stock_close'].inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)

        # 예측된 주가로 다음 날짜 예측 준비
        new_row = recent_seq.iloc[-1].copy()
        new_row['stock_close'] = pred_price

        # pandas concat()을 사용하여 새로운 행을 추가
        recent_seq = pd.concat([recent_seq.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

    return predictions

# 예측 결과 시각화
def plot_predictions(predictions, last_actual_price):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(predictions)+1), predictions, label="Predicted Price", marker='o')
    plt.axhline(y=last_actual_price, color='r', linestyle='--', label=f"Last Actual Price: {last_actual_price:.2f}")
    plt.title("Stock Price Prediction for Next 30 Days")
    plt.xlabel("Days Ahead")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    model_path = os.path.join("models", f"transformer_model_for_{companyCode}.h5")
    scaler_dir = os.path.join("models", "scalers")
    data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")

    # 모델 로드
    model = load_transformer_model(model_path)

    # 스케일러 로드
    scalers = {col: load_scaler(os.path.join(scaler_dir, f"scaler_transformer_{col}_{companyCode}.joblib")) for col in data_columns}

    # 예측용 데이터 준비
    X_pred = load_data_for_prediction(data_path, scalers)

    # 예측 (예: 30일 예측)
    predictions = predict_future_days(model, pd.read_csv(data_path, parse_dates=["Date"]), scalers, days=30)

    for i, p in enumerate(predictions, 1):
        print(f"Day {i} predicted price: {p:.2f}")

    # 마지막 실제 주가와 비교하여 수익률 계산
    last_price = pd.read_csv(data_path)["stock_close"].iloc[-1]
    profit = (predictions[-1] - last_price) / last_price * 100
    print(f"Current price: {last_price:.2f}")
    print(f"Profit on predicted price after 50 days: {profit:.2f}%")
    
    # 마지막 부분에 추가
    plot_predictions(predictions, last_price)   
