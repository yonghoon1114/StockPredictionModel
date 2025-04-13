import pandas as pd
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from config import companyCode, sequenceLength, data_columns, dataNumber


def load_lstm_model(model_path: str):
    return load_model(model_path, compile=False)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

def load_data_for_prediction(
    path: str,
    scaler_stock: MinMaxScaler,
    scaler_rate: MinMaxScaler,
    scaler_nasdaq: MinMaxScaler,
    scaler_Revenue: MinMaxScaler,
    scaler_NetIncome: MinMaxScaler,
    scaler_TotalAssets: MinMaxScaler,
    scaler_RSI: MinMaxScaler

) -> np.ndarray:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    # 가장 최근 시퀀스 추출
    recent_seq = df[data_columns].iloc[-sequenceLength:]

    # 각 feature별로 scaler를 이용해 변환
    scaled_stock = scaler_stock.transform(recent_seq[["stock_close"]])
    scaled_rate = scaler_rate.transform(recent_seq[["rate_close"]])
    scaled_nasdaq = scaler_nasdaq.transform(recent_seq[["nasdaq_close"]])
    scaled_Revenue = scaler_Revenue.transform(recent_seq[["Revenue"]])
    scaled_NetIncome = scaler_NetIncome.transform(recent_seq[["NetIncome"]])
    scaled_TotalAssets = scaler_TotalAssets.transform(recent_seq[["TotalAssets"]])
    scaled_RSI = scaler_RSI.transform(recent_seq[["RSI"]])

    # scaled_features를 합침
    scaled_features = np.hstack((
        scaled_stock,
        scaled_rate,
        scaled_nasdaq,
        scaled_Revenue,
        scaled_NetIncome,
        scaled_TotalAssets,
        scaled_RSI
    ))

    # LSTM 입력 형태: (samples, time steps, features)
    return scaled_features.reshape(1, sequenceLength, dataNumber)

def predict_future_days(
    model, df, scalers, days=30
):
    scaler_stock, scaler_rate, scaler_nasdaq, scaler_Revenue, scaler_NetIncome, scaler_TotalAssets,scaler_RSI = scalers

    df_sorted = df.sort_values("Date").copy()

    recent_seq = df_sorted[data_columns].iloc[-sequenceLength:]

    predictions = []

    for _ in range(days):
        # 스케일링
        scaled_stock = scaler_stock.transform(recent_seq[["stock_close"]])
        scaled_rate = scaler_rate.transform(recent_seq[["rate_close"]])
        scaled_nasdaq = scaler_nasdaq.transform(recent_seq[["nasdaq_close"]])
        scaled_Revenue = scaler_Revenue.transform(recent_seq[["Revenue"]])
        scaled_NetIncome = scaler_NetIncome.transform(recent_seq[["NetIncome"]])
        scaled_TotalAssets = scaler_TotalAssets.transform(recent_seq[["TotalAssets"]])
        scaled_RSI = scaler_RSI.transform(recent_seq[["RSI"]])

        X = np.hstack((scaled_stock, scaled_rate, scaled_nasdaq, scaled_Revenue, scaled_NetIncome, scaled_TotalAssets, scaled_RSI))
        X_input = X.reshape(1, sequenceLength, dataNumber)

        pred_scaled = model.predict(X_input)[0][0]
        pred_price = scaler_stock.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)

        # 새로운 예측값을 가진 행 추가 (나머지 변수는 그대로 사용)
        last_row = recent_seq.iloc[-1].copy()
        new_row = last_row.copy()
        new_row["stock_close"] = pred_price
        recent_seq = pd.concat([recent_seq.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

    return predictions

if __name__ == "__main__":
    model_path = os.path.join("models", f"lstm_model_for_{companyCode}.h5")
    scaler_dir = os.path.join("models", "scalers")
    data_path = os.path.join("data", "processed", f"{companyCode}_merged.csv")

    # 스케일러 로드
    scaler_stock = load_scaler(os.path.join(scaler_dir, f"scaler_stock_{companyCode}.joblib"))
    scaler_rate = load_scaler(os.path.join(scaler_dir, "scaler_rate.joblib"))
    scaler_nasdaq = load_scaler(os.path.join(scaler_dir, "scaler_nasdaq.joblib"))
    scaler_Revenue = load_scaler(os.path.join(scaler_dir, f"scaler_Revenue_{companyCode}.joblib"))
    scaler_NetIncome = load_scaler(os.path.join(scaler_dir, f"scaler_NetIncome_{companyCode}.joblib"))
    scaler_TotalAssets = load_scaler(os.path.join(scaler_dir, f"scaler_TotalAssets_{companyCode}.joblib"))
    scaler_RSI = load_scaler(os.path.join(scaler_dir, f"scaler_RSI_{companyCode}.joblib"))

    # 모델 로드
    model = load_lstm_model(model_path)

    # 예측용 데이터 준비
    X_pred = load_data_for_prediction(
        data_path,
        scaler_stock,
        scaler_rate,
        scaler_nasdaq,
        scaler_Revenue,
        scaler_NetIncome,
        scaler_TotalAssets,
        scaler_RSI
    )

    # 예측
    predicted_scaled = model.predict(X_pred)[0][0]

    # 결과를 원래 가격대로 되돌림
    predicted_price = scaler_stock.inverse_transform([[predicted_scaled]])[0][0]

    # print(f"Predicted stock price: {predicted_price:.2f}")
    
    df = pd.read_csv(data_path, parse_dates=["Date"])

    predictions = predict_future_days(
        model, df,
        scalers=[scaler_stock, scaler_rate, scaler_nasdaq, scaler_Revenue, scaler_NetIncome, scaler_TotalAssets, scaler_RSI],
        days=50
    )

    # for i, p in enumerate(predictions, 1):
    #     print(f"Day {i} predicted price: {p:.2f}")
    
    print(f"Estimated value of stock of {companyCode} : {predictions[-1]:.2f}")