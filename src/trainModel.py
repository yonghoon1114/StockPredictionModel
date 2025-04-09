import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

def train_model(data: pd.DataFrame):
    # RSI 제거하고 feature 설정
    X = data[["stock_close", "rate_close", "nasdaq_close"]]
    y = data["target"]

    # 시간 순서 유지한 채로 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    # 모델 저장
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "linear_model.joblib"))

    return model

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "merged.csv")
    df = load_data(data_path)
    trained_model = train_model(df)
