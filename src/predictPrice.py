import pandas as pd
import os
import joblib  # 학습한 모델을 저장하고 불러오는 데 사용되는 라이브러리

# 저장된 모델을 불러오는 함수
def load_model(model_path: str):
    return joblib.load(model_path)

# 예측에 사용할 데이터를 불러오고 전처리하는 함수
def load_data_for_prediction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])  # Date 컬럼을 날짜형으로 파싱
    df = df.sort_values("Date")  # 날짜 순 정렬

    last_row = df.iloc[-1]  # 가장 마지막 행 선택

    # 예측에 필요한 feature들을 포함한 입력 데이터 구성 (rsi 추가)
    X_pred = pd.DataFrame([{
        "stock_close": last_row["stock_close"],
        "rate_close": last_row["rate_close"],
        "nasdaq_close": last_row["nasdaq_close"],
        "rsi": last_row["rsi"]  # rsi도 포함시킴
    }])
    return X_pred

if __name__ == "__main__":
    # 모델 파일 경로
    model_path = os.path.join("models", "linear_model.joblib")
    # 예측용 데이터 파일 경로
    data_path = os.path.join("data", "processed", "merged.csv")

    model = load_model(model_path)
    X_pred = load_data_for_prediction(data_path)

    predicted_price = model.predict(X_pred)[0]
    print(f"Predicted stock price for 2025-04-07: {predicted_price:.2f}")
