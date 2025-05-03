import yfinance as yf
import os

def fetch_macro_data(ticker, start="1910-01-01", end = "2026-01-01" , save_path="data/raw", prefix="DATA"):
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

