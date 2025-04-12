import yfinance as yf
import pandas as pd
import os
from config import companyCode

os.makedirs(f"data/raw/Companies/{companyCode}", exist_ok=True)
def fetch_stock_data(ticker: str, start: str, end: str, save_path: str = f"data/raw/Companies/{companyCode}"):
    # yfinance에서 데이터 다운로드
    data = yf.download(ticker, start=start, end=end)

    # 저장 경로 없으면 생성
    os.makedirs(save_path, exist_ok=True)

    # 파일 경로
    file_path = os.path.join(save_path, f"{ticker}_{start}_{end}.csv")

    # CSV로 저장 (Date를 컬럼으로 저장)
    data.to_csv(file_path, index_label="Date")
    print(f"Saved: {file_path}")

if __name__ == "__main__":
    fetch_stock_data(f"{companyCode}", "1910-01-01", "2026-04-11")
