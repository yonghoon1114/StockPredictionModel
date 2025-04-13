import pandas as pd
import os
import matplotlib.pyplot as plt
from config import companyCode

# 데이터 로딩 함수
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df

# 금리와 나스닥 지수 시각화 함수
def plot_nasdaq_and_interest_rate(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # 금리 (Interest Rate) 그래프
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Interest Rate (%)', color='tab:red')
    ax1.plot(df['Date'], df['rate_close'], color='tab:red', label='Interest Rate')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 나스닥 (NASDAQ Index) 그래프
    ax2 = ax1.twinx()
    ax2.set_ylabel('NASDAQ Index', color='tab:blue')
    ax2.plot(df['Date'], df['nasdaq_close'], color='tab:blue', label='NASDAQ Index')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 그래프 제목
    plt.title(f"Interest Rate and NASDAQ Index over Time")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# 메인 실행 부분
if __name__ == "__main__":
    data_path = os.path.join("data", "processed", f"{companyCode}_merged.csv")

    # 데이터 로드
    df = load_data(data_path)

    # 금리와 나스닥 그래프 출력
    plot_nasdaq_and_interest_rate(df)
