import pandas as pd
from datetime import datetime
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from trainModelforAll import sequenceLength, data_columns, dataNumber

def load_transformer_model(model_path: str):
    return load_model(model_path, compile=False)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

def load_data_for_prediction(path: str, scalers: dict) -> np.ndarray:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    recent_seq = df[data_columns].iloc[-sequenceLength:]
    scaled_features = np.hstack([scalers[col].transform(recent_seq[[col]]) for col in data_columns])
    return scaled_features.reshape(1, sequenceLength, dataNumber)

def predict_future_days(model, df, scalers, days=50):
    df_sorted = df.sort_values("Date").copy()
    recent_seq = df_sorted[data_columns].iloc[-sequenceLength:]
    predictions = []
    for _ in range(days):
        scaled_features = np.hstack([scalers[col].transform(recent_seq[[col]]) for col in data_columns])
        X_input = scaled_features.reshape(1, sequenceLength, dataNumber)
        pred_scaled = model.predict(X_input)[0][0]
        pred_price = scalers['stock_close'].inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)
        new_row = recent_seq.iloc[-1].copy()
        new_row['stock_close'] = pred_price
        recent_seq = pd.concat([recent_seq.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    return predictions

def runPrediction(Companies_for_prediction: list, Date: str):
    all_results = []
    today_str = datetime.today().strftime("%Y-%m-%d")

    for companyCode in Companies_for_prediction:
        model_path = os.path.join("models", f"transformer_model_for_{companyCode}.h5")
        scaler_dir = os.path.join("models", "scalers")
        data_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")

        if os.path.exists(model_path) and os.path.exists(data_path):
            model = load_transformer_model(model_path)
            scalers = {col: load_scaler(os.path.join(scaler_dir, f"scaler_transformer_{col}_{companyCode}.joblib")) for col in data_columns}
            load_data_for_prediction(data_path, scalers)
            df = pd.read_csv(data_path, parse_dates=["Date"])
            predictions = predict_future_days(model, df, scalers, days=30)
            last_price = df["stock_close"].iloc[-1]
            profit = (predictions[-1] - last_price) / last_price * 100
            print(f"Company: {companyCode}")
            for i, p in enumerate(predictions, 1):
                print(f"Day {i} predicted price: {p:.2f}")
            print(f"Current price: {last_price:.2f}")
            print(f"Profit on predicted price after 30 days: {profit:.2f}%\n")
            all_results.append({
                "Company": companyCode,
                "Last Price": last_price,
                "Predicted Price": predictions[-1],
                "Profit Percent": profit
            })
        else:
            print(f"Model or data for {companyCode} is missing.")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="Profit Percent", ascending=False)

    # 텍스트 형식으로 누적 저장
    output_txt = "predicted_profits_sorted.txt"
    with open(output_txt, "a") as f:
        f.write(f"{today_str}\n\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Company']},{row['Last Price']:.10f},{row['Predicted Price']:.10f},{row['Profit Percent']:.10f}\n")
        f.write("\n")

    print(f"Predicted profits saved to '{output_txt}'.")
