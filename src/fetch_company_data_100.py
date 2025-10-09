import yfinance as yf
import pandas as pd
from curl_cffi import requests
import os
from datetime import date

# Edge 브라우저 세션 위장 (yfinance 접속 안정성 향상)
session = requests.Session(impersonate="edge")

def fetch_data(company_code: str, end_date: str = None):
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    save_dir = f"data/raw/Companies/{company_code}"
    os.makedirs(save_dir, exist_ok=True)

    # 분기별 기간 계산 함수
    def get_quarter_range(date_obj: pd.Timestamp):
        quarter = (date_obj.month - 1) // 3 + 1
        year = date_obj.year
        if quarter == 1:
            return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-03-31")
        elif quarter == 2:
            return pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-06-30")
        elif quarter == 3:
            return pd.Timestamp(f"{year}-07-01"), pd.Timestamp(f"{year}-09-30")
        else:
            return pd.Timestamp(f"{year}-10-01"), pd.Timestamp(f"{year}-12-31")

    # 재무제표 수집
    def fetch_quarterly_financials_merged(ticker: str):
        stock = yf.Ticker(ticker, session=session)
        income_stmt = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow_stmt = stock.quarterly_cashflow.T

        expanded_records = []

        for report_date in income_stmt.index:
            start_date, end_date = get_quarter_range(report_date)
            record = {
                "Revenue": income_stmt.loc[report_date].get("Total Revenue", None),
                "NetIncome": income_stmt.loc[report_date].get("Net Income", None),
                "TotalAssets": balance_sheet.loc[report_date].get("Total Assets", None),
                "TotalLiabilities": balance_sheet.loc[report_date].get("Total Liab", None),
                "OperatingCashFlow": cashflow_stmt.loc[report_date].get("Total Cash From Operating Activities", None),
                "CapitalExpenditures": cashflow_stmt.loc[report_date].get("Capital Expenditures", None),
            }

            for d in pd.date_range(start=start_date, end=end_date):
                expanded_record = {"Date": d.strftime("%Y-%m-%d")}
                expanded_record.update(record)
                expanded_records.append(expanded_record)

        df = pd.DataFrame(expanded_records)
        df.sort_values("Date", inplace=True)

        filepath = os.path.join(save_dir, f"{ticker}_quarterly_financials_expanded.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Saved expanded financials: {filepath}")

    # 주가 데이터 수집
    def fetch_stock_data(ticker: str, start: str = "1910-01-01", end: str = end_date):
        file_path = os.path.join(save_dir, f"{ticker}_stock.csv")

        if os.path.exists(file_path):
            existing = pd.read_csv(file_path, parse_dates=["Date"])
            last_date = existing["Date"].max().strftime("%Y-%m-%d")
            new_start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
            new_start_str = new_start.strftime("%Y-%m-%d")

            if new_start_str > end:
                print(f"⚙️ Up to date: {ticker}")
                return

            print(f"⬆️ Updating {ticker}: {new_start_str} → {end}")
            new_data = yf.download(ticker, start=new_start_str, end=end, session=session, auto_adjust=True)

            if not new_data.empty:
                new_data.reset_index(inplace=True)
                merged = pd.concat([existing, new_data], ignore_index=True)
                merged.drop_duplicates(subset="Date", keep="last", inplace=True)
                merged.sort_values("Date", inplace=True)
                merged.to_csv(file_path, index=False)
                print(f"✅ Updated: {file_path}")
            else:
                print(f"⚠️ No new data found for {ticker}")
        else:
            print(f"⬇️ Downloading full history for {ticker}")
            data = yf.download(ticker, start=start, end=end, session=session, auto_adjust=True)
            data.reset_index(inplace=True)
            data.to_csv(file_path, index=False)
            print(f"✅ Saved new stock data: {file_path}")

    # 실행
    fetch_quarterly_financials_merged(company_code)
    fetch_stock_data(company_code)

