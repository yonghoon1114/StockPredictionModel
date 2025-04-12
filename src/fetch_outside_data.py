import yfinance as yf
import os

def fetch_interest_rate_data(ticker="^IRX", start="1910-01-01", end="2026-04-11", save_path="data/raw/INTEREST"):
    # yfinance로 금리 데이터 다운로드
    data = yf.download(ticker, start=start, end=end)

    # 저장 경로 없으면 생성
    os.makedirs(save_path, exist_ok=True)

    # 저장 파일명
    file_path = os.path.join(save_path, f"{ticker}_{start}_{end}.csv".replace("^", ""))

    # CSV로 저장
    data.to_csv(file_path, index_label="Date")
    print(f"Saved: {file_path}")
    return data

def fetch_nasdaq_index_data(ticker="^IXIC", start="1910-01-01", end="2026-04-11", save_path="data/raw/NASDAQ"):
    # yfinance로 NASDAQ Composite Index 데이터 다운로드
    data = yf.download(ticker, start=start, end=end)

    # 저장 경로 없으면 생성
    os.makedirs(save_path, exist_ok=True)

    # 저장 파일명
    file_path = os.path.join(save_path, f"{ticker}_{start}_{end}.csv".replace("^", ""))

    # CSV로 저장
    data.to_csv(file_path, index_label="Date")
    print(f"Saved: {file_path}")
    return data

if __name__ == "__main__":
    fetch_interest_rate_data()
    fetch_nasdaq_index_data()

