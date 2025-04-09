import yfinance as yf
import pandas as pd
import os

def fetch_quarterly_financials_merged(ticker: str, save_dir: str = "data/raw/AAPL"):
    os.makedirs(save_dir, exist_ok=True)
    
    stock = yf.Ticker(ticker)

    # 분기별 재무제표 (Transposed for easier row-based access)
    income_stmt = stock.quarterly_financials.T
    balance_sheet = stock.quarterly_balance_sheet.T
    cashflow_stmt = stock.quarterly_cashflow.T

    # 모든 분기 데이터를 담을 리스트
    records = []

    # 분기별로 반복
    for date in income_stmt.index:
        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Revenue": income_stmt.loc[date].get("Total Revenue"),
            "NetIncome": income_stmt.loc[date].get("Net Income"),
            "TotalAssets": balance_sheet.loc[date].get("Total Assets"),
            "TotalLiabilities": balance_sheet.loc[date].get("Total Liab"),
            "OperatingCashFlow": cashflow_stmt.loc[date].get("Total Cash From Operating Activities"),
            "CapitalExpenditures": cashflow_stmt.loc[date].get("Capital Expenditures")
        })

    # 병합 후 저장
    df = pd.DataFrame(records)
    df = df.sort_values("Date")  # 날짜순 정렬

    filepath = os.path.join(save_dir, f"{ticker}_quarterly_financials_merged.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved merged file: {filepath}")

if __name__ == "__main__":
    fetch_quarterly_financials_merged("AAPL")
