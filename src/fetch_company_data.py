import yfinance as yf
import pandas as pd
import os
from config import companyCode

os.makedirs(f"data/raw/Companies/{companyCode}", exist_ok=True)
def get_quarter_range(date: pd.Timestamp):

    # 분기 시작/종료일 구하기
    quarter = (date.month - 1) // 3 + 1
    year = date.year
    if quarter == 1:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-03-31")
    elif quarter == 2:
        start = pd.Timestamp(f"{year}-04-01")
        end = pd.Timestamp(f"{year}-06-30")
    elif quarter == 3:
        start = pd.Timestamp(f"{year}-07-01")
        end = pd.Timestamp(f"{year}-09-30")
    else:
        start = pd.Timestamp(f"{year}-10-01")
        end = pd.Timestamp(f"{year}-12-31")
    return start, end

def fetch_quarterly_financials_merged(ticker: str, save_dir: str = f"data/raw/Companies/{companyCode}"):
    os.makedirs(save_dir, exist_ok=True)
    
    stock = yf.Ticker(ticker)

    income_stmt = stock.quarterly_financials.T
    balance_sheet = stock.quarterly_balance_sheet.T
    cashflow_stmt = stock.quarterly_cashflow.T

    expanded_records = []

    for report_date in income_stmt.index:
        start_date, end_date = get_quarter_range(report_date)

        record = {
            "Revenue": income_stmt.loc[report_date].get("Total Revenue") if report_date in income_stmt.index else None,
            "NetIncome": income_stmt.loc[report_date].get("Net Income") if report_date in income_stmt.index else None,
            "TotalAssets": balance_sheet.loc[report_date].get("Total Assets") if report_date in balance_sheet.index else None,
            "TotalLiabilities": balance_sheet.loc[report_date].get("Total Liab") if report_date in balance_sheet.index else None,
            "OperatingCashFlow": cashflow_stmt.loc[report_date].get("Total Cash From Operating Activities") if report_date in cashflow_stmt.index else None,
            "CapitalExpenditures": cashflow_stmt.loc[report_date].get("Capital Expenditures") if report_date in cashflow_stmt.index else None
        }

        # 해당 분기의 각 날짜에 같은 데이터를 복제
        date_range = pd.date_range(start=start_date, end=end_date)
        for d in date_range:
            expanded_record = {"Date": d.strftime("%Y-%m-%d")}
            expanded_record.update(record)
            expanded_records.append(expanded_record)

    df = pd.DataFrame(expanded_records)
    df = df.sort_values("Date")

    filepath = os.path.join(save_dir, f"{ticker}_quarterly_financials_expanded.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved expanded file: {filepath}")

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
    fetch_quarterly_financials_merged(f"{companyCode}")
    fetch_stock_data(f"{companyCode}", "1910-01-01", "2026-04-11")
