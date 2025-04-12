import yfinance as yf
import pandas as pd
import os

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

def fetch_quarterly_financials_merged(ticker: str, save_dir: str = "data/raw/AAPL"):
    os.makedirs(save_dir, exist_ok=True)
    
    stock = yf.Ticker(ticker)

    income_stmt = stock.quarterly_financials.T
    balance_sheet = stock.quarterly_balance_sheet.T
    cashflow_stmt = stock.quarterly_cashflow.T

    expanded_records = []

    for report_date in income_stmt.index:
        start_date, end_date = get_quarter_range(report_date)

        record = {
            "Revenue": income_stmt.loc[report_date].get("Total Revenue"),
            "NetIncome": income_stmt.loc[report_date].get("Net Income"),
            "TotalAssets": balance_sheet.loc[report_date].get("Total Assets"),
            "TotalLiabilities": balance_sheet.loc[report_date].get("Total Liab"),
            "OperatingCashFlow": cashflow_stmt.loc[report_date].get("Total Cash From Operating Activities"),
            "CapitalExpenditures": cashflow_stmt.loc[report_date].get("Capital Expenditures")
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

if __name__ == "__main__":
    fetch_quarterly_financials_merged("AAPL")
