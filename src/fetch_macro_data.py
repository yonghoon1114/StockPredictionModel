import yfinance as yf
import os
import pandas as pd
from curl_cffi import requests
from datetime import date, timedelta

session = requests.Session(impersonate="edge")  # 크롬처럼 위장

def fetch_macro_data(ticker, start="1910-01-01", end=None, save_path="data/raw", prefix="DATA"):
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    safe_ticker = ticker.replace("^", "").replace("=", "")
    full_path = os.path.join(save_path, prefix)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, f"{safe_ticker}_macro.csv")

    # 기존 데이터가 있는 경우
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, parse_dates=["Date"])
        last_date = existing["Date"].max().date()
        new_start = last_date + timedelta(days=1)

        if pd.Timestamp(new_start) >= pd.Timestamp(end):
            print(f"{ticker}: up to date ({last_date})")
            return existing

        print(f"{ticker}: updating from {new_start} to {end}")
        new_data = yf.download(
            ticker,
            start=new_start.strftime("%Y-%m-%d"),
            end=end,
            session=session,
            auto_adjust=True
        )

        if not new_data.empty:
            new_data.reset_index(inplace=True)
            
            # MultiIndex 컬럼 처리
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [col[0] if col[0] != '' else col[1] for col in new_data.columns]
            
            # 컬럼명 강제 설정
            new_data = new_data[["Date", "Close", "High", "Low", "Open", "Volume"]]

            merged = pd.concat([existing, new_data], ignore_index=True)
            merged.drop_duplicates(subset="Date", keep="last", inplace=True)
            merged.sort_values("Date", inplace=True)
            merged.to_csv(file_path, index=False)
            print(f"Updated: {file_path}")
            return merged
        else:
            print(f"{ticker}: no new data found")
            return existing

    else:
        # 새 파일일 경우 전체 다운로드
        print(f"{ticker}: downloading full history...")
        data = yf.download(ticker, start=start, end=end, session=session, auto_adjust=True)
        data.reset_index(inplace=True)

        # MultiIndex 컬럼 처리
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]

        # 컬럼명 강제 설정
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]

        data.to_csv(file_path, index=False)
        print(f"Saved new file: {file_path}")
        return data
