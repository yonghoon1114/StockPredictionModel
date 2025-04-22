import pandas as pd
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from config import companyCode, sequenceLength, data_columns, dataNumber, Date
import matplotlib.pyplot as plt

SCALER_KEYS = [
    "stock", "rate", "nasdaq", "Revenue",
    "NetIncome", "TotalAssets", "RSI", "Gold"
]

def load_model_for_prediction(model_path: str):
    return load_model(model_path, compile=False)

def load_scalers(scaler_dir: str, company_code: str):
    scalers = {}
    for key in SCALER_KEYS:
        fname = f"scaler_{key}_{company_code}.joblib" if key in ["stock", "Revenue", "NetIncome", "TotalAssets", "RSI"] else f"scaler_{key}.joblib"
        path = os.path.join(scaler_dir, fname)
        scalers[key] = joblib.load(path)
    return scalers

def preprocess_sequence(df, scalers):
    recent_seq = df[data_columns].iloc[-sequenceLength:]
    scaled_features = np.hstack([
        scalers["stock"].transform(recent_seq[["stock_close"]]),
        scalers["rate"].transform(recent_seq[["rate_close"]]),
        scalers["nasdaq"].transform(recent_seq[["nasdaq_close"]]),
        scalers["Revenue"].transform(recent_seq[["Revenue"]]),
        scalers["NetIncome"].transform(recent_seq[["NetIncome"]]),
        scalers["TotalAssets"].transform(recent_seq[["TotalAssets"]]),
        scalers["RSI"].transform(recent_seq[["RSI"]]),
        scalers["Gold"].transform(recent_seq[["gold_close"]])
    ])
    return scaled_features.reshape(1, sequenceLength, dataNumber)

def predict_future_prices(model, df, scalers, days=30):
    df = df.sort_values("Date").copy()
    recent_seq = df[data_columns].iloc[-sequenceLength:].copy()
    predictions = []

    for _ in range(days):
        temp_df = pd.concat([df.iloc[-(sequenceLength + 1):-1], recent_seq.tail(1)], ignore_index=True)
        X_input = preprocess_sequence(temp_df, scalers)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        pred_price = scalers["stock"].inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)

        new_row = recent_seq.iloc[-1].copy()
        new_row["stock_close"] = pred_price
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
    base_dir = "models"
    scaler_dir = os.path.join(base_dir, "scalers")
    model_path = os.path.join(base_dir, f"lstm_model_for_{companyCode}.h5")
    data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")

    model = load_model_for_prediction(model_path)
    scalers = load_scalers(scaler_dir, companyCode)
    df = pd.read_csv(data_path, parse_dates=["Date"])

    # 예측 수행
    predictions = predict_future_prices(model, df, scalers, days=30)

    # 수익률 계산
    last_price = round(float(df["stock_close"].iloc[-1]), 2)
    final_predicted_price = predictions[-1]
    profit = (final_predicted_price - last_price) / last_price * 100

    print(f"Estimated value of stock of {companyCode} after 30 days: {final_predicted_price:.2f}")
    print(f"Current price : {last_price}")
    print(f"Profit: {profit:.2f}%")

    # 그래프 출력
    plot_predictions(predictions, last_price)
