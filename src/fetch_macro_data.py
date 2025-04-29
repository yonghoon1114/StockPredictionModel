import yfinance as yf
import os
from config import Date

def fetch_data(ticker, start="1910-01-01", end=Date, save_path="data/raw", prefix="DATA"):
    # yfinance로 데이터 다운로드
    data = yf.download(ticker, start=start, end=end)

    # 저장 경로 없으면 생성
    full_path = os.path.join(save_path, prefix)
    os.makedirs(full_path, exist_ok=True)

    # 저장 파일명
    safe_ticker = ticker.replace("^", "").replace("=", "")
    file_path = os.path.join(full_path, f"{safe_ticker}_{start}_{end}.csv")

    # CSV로 저장
    data.to_csv(file_path, index_label="Date")
    print(f"Saved: {file_path}")
    return data

def fetch_interest_rate_data():
    return fetch_data(ticker="^IRX", save_path="data/raw", prefix="INTEREST")

def fetch_nasdaq_index_data():
    return fetch_data(ticker="^IXIC", save_path="data/raw", prefix="NASDAQ")

def fetch_gold_price_data():
    return fetch_data(ticker="GC=F", start="2000-01-01", end=Date, save_path="data/raw", prefix="GOLD")

def fetch_semiCondocter_data():
    return fetch_data(ticker="^SOX", start ="1910-01-01", end=Date, prefix="SEMICOND")

def fetch_entertainment():
    return fetch_data(ticker= "XLC", start="2000-01-01", end=Date, prefix="ENTERTAINMENT")

if __name__ == "__main__":
    fetch_interest_rate_data()
    fetch_nasdaq_index_data()
    fetch_gold_price_data()
    fetch_semiCondocter_data()
    fetch_entertainment()