import yfinance as yf
import pandas as pd
from curl_cffi import requests
import os
from datetime import date, datetime

# Edge 브라우저 세션 위장 (yfinance 접속 안정성 향상)
session = requests.Session(impersonate="edge")

def fetch_data(company_code: str, end_date: str = None):
    if end_date is None:
        end_date = date.today()

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
        stock = yf.Ticker(ticker)
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

        new_df = pd.DataFrame(expanded_records)
        new_df.sort_values("Date", inplace=True)

        filepath = os.path.join(save_dir, f"{ticker}_quarterly_financials_expanded.csv")

        # ✅ 기존 파일 유지 + 새 데이터 병합
        if os.path.exists(filepath):
            old_df = pd.read_csv(filepath)
            old_df["Date"] = pd.to_datetime(old_df["Date"], errors="coerce")
            new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")

            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset="Date", keep="last", inplace=True)
            combined_df.sort_values("Date", inplace=True)
            combined_df.to_csv(filepath, index=False)
            print(f"🔄 Updated financials: {filepath}")
        else:
            new_df.to_csv(filepath, index=False)
            print(f"✅ Saved new financials: {filepath}")
            
    # 주가 데이터 수집
    def fetch_stock_data(ticker: str, start: str = "1910-01-01", end: str = end_date):
        file_path = os.path.join(save_dir, f"{ticker}_stock.csv")
        if isinstance(end, str):
            end = datetime.strptime(end,"%Y-%m-%d").date()
            
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path, parse_dates=["Date"])
            last_date = existing["Date"].max().strftime("%Y-%m-%d")
            new_start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

            if pd.Timestamp(new_start) > pd.Timestamp(end):
                print(f"⚙️ Up to date: {ticker}")
                return

            print(f"⬆️ Updating {ticker}: {new_start} → {end}")
            new_data = yf.download(ticker, start=new_start, end=end, session=session, auto_adjust=True)

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
                print(f"✅ Updated: {file_path}")
            else:
                print(f"⚠️ No new data found for {ticker}")
        else:
            print(f"⬇️ Downloading full history for {ticker}")
            data = yf.download(ticker, start=start, end=end, session=session, auto_adjust=True)
            data.reset_index(inplace=True)

            # MultiIndex 컬럼 처리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]
            
            # 컬럼명 강제 설정
            data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]

            data.to_csv(file_path, index=False)
            print(f"✅ Saved new stock data: {file_path}")


    # 실행
    fetch_quarterly_financials_merged(company_code)
    fetch_stock_data(company_code)

