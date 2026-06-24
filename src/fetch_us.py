import yfinance as yf
import pandas as pd
import os
from curl_cffi import requests
from datetime import date, timedelta

session = requests.Session(impersonate="edge")


def fetch_stock_price(ticker: str, save_dir: str) -> pd.DataFrame:
    """주가 데이터 수집 (신규 or 업데이트)"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{ticker}_stock.csv")
    end = date.today().strftime("%Y-%m-%d")

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
        existing = existing.dropna(subset=["Date"])

        # 빈 파일이면 전체 다운로드
        if existing.empty:
            print(f"[{ticker}] CSV 비어있음, 전체 다운로드")
            os.remove(file_path)
            return fetch_stock_price(ticker, save_dir)

        last_date = existing["Date"].max().date()
        new_start = last_date + timedelta(days=1)

        if pd.Timestamp(new_start) >= pd.Timestamp(end):
            print(f"[{ticker}] 주가 데이터 최신 상태")
            return existing

        print(f"[{ticker}] 주가 업데이트: {new_start} → {end}")
        new_data = yf.download(ticker, start=str(new_start), end=end, session=session, auto_adjust=True)

        if new_data.empty:
            print(f"[{ticker}] 새 주가 데이터 없음")
            return existing

        new_data.reset_index(inplace=True)
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] for col in new_data.columns]
        new_data = new_data[["Date", "Close", "High", "Low", "Open", "Volume"]]

        merged = pd.concat([existing, new_data], ignore_index=True)
        merged.drop_duplicates(subset="Date", keep="last", inplace=True)
        merged.sort_values("Date", inplace=True)
        merged.to_csv(file_path, index=False)
        return merged

    else:
        print(f"[{ticker}] 전체 주가 이력 다운로드")
        data = yf.download(ticker, start="2000-01-01", end=end, session=session, auto_adjust=True)
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
        data.to_csv(file_path, index=False)
        print(f"[{ticker}] 주가 저장 완료: {file_path}")
        return data


def fetch_financials(ticker: str, save_dir: str) -> dict:
    """
    재무제표 수집
    - 손익계산서, 대차대조표, 현금흐름표 (연간 + 분기)
    - info에서 PER, PBR, ROE 등 이미 계산된 지표도 함께 수집
    """
    os.makedirs(save_dir, exist_ok=True)
    stock = yf.Ticker(ticker)

    result = {}

    # 1. 손익계산서 (연간)
    try:
        income_annual = stock.financials.T
        income_annual.index = income_annual.index.strftime("%Y-%m-%d")
        path = os.path.join(save_dir, f"{ticker}_income_annual.csv")
        income_annual.to_csv(path)
        result["income_annual"] = income_annual
        print(f"[{ticker}] 연간 손익계산서 저장")
    except Exception as e:
        print(f"[{ticker}] 연간 손익계산서 실패: {e}")

    # 2. 손익계산서 (분기)
    try:
        income_quarterly = stock.quarterly_financials.T
        income_quarterly.index = income_quarterly.index.strftime("%Y-%m-%d")
        path = os.path.join(save_dir, f"{ticker}_income_quarterly.csv")
        income_quarterly.to_csv(path)
        result["income_quarterly"] = income_quarterly
        print(f"[{ticker}] 분기 손익계산서 저장")
    except Exception as e:
        print(f"[{ticker}] 분기 손익계산서 실패: {e}")

    # 3. 대차대조표 (연간)
    try:
        balance_annual = stock.balance_sheet.T
        balance_annual.index = balance_annual.index.strftime("%Y-%m-%d")
        path = os.path.join(save_dir, f"{ticker}_balance_annual.csv")
        balance_annual.to_csv(path)
        result["balance_annual"] = balance_annual
        print(f"[{ticker}] 연간 대차대조표 저장")
    except Exception as e:
        print(f"[{ticker}] 연간 대차대조표 실패: {e}")

    # 4. 대차대조표 (분기)
    try:
        balance_quarterly = stock.quarterly_balance_sheet.T
        balance_quarterly.index = balance_quarterly.index.strftime("%Y-%m-%d")
        path = os.path.join(save_dir, f"{ticker}_balance_quarterly.csv")
        balance_quarterly.to_csv(path)
        result["balance_quarterly"] = balance_quarterly
        print(f"[{ticker}] 분기 대차대조표 저장")
    except Exception as e:
        print(f"[{ticker}] 분기 대차대조표 실패: {e}")

    # 5. 현금흐름표 (연간)
    try:
        cashflow_annual = stock.cashflow.T
        cashflow_annual.index = cashflow_annual.index.strftime("%Y-%m-%d")
        path = os.path.join(save_dir, f"{ticker}_cashflow_annual.csv")
        cashflow_annual.to_csv(path)
        result["cashflow_annual"] = cashflow_annual
        print(f"[{ticker}] 연간 현금흐름표 저장")
    except Exception as e:
        print(f"[{ticker}] 연간 현금흐름표 실패: {e}")

    # 6. info (PER, PBR, ROE 등 이미 계산된 지표)
    try:
        info = stock.info
        info_keys = [
            "currentPrice", "marketCap", "trailingPE", "forwardPE",
            "priceToBook", "returnOnEquity", "returnOnAssets",
            "debtToEquity", "currentRatio", "quickRatio",
            "revenueGrowth", "earningsGrowth", "operatingMargins",
            "profitMargins", "grossMargins", "revenuePerShare",
            "trailingEps", "forwardEps", "bookValue",
            "totalRevenue", "netIncomeToCommon", "totalDebt",
            "totalCash", "freeCashflow", "operatingCashflow",
            "sector", "industry", "longName", "currency"
        ]
        info_filtered = {k: info.get(k) for k in info_keys}
        info_df = pd.DataFrame([info_filtered])
        path = os.path.join(save_dir, f"{ticker}_info.csv")
        info_df.to_csv(path, index=False)
        result["info"] = info_filtered
        print(f"[{ticker}] 기업 정보 저장")
    except Exception as e:
        print(f"[{ticker}] 기업 정보 실패: {e}")

    return result


def fetch_us(ticker: str, base_dir: str = "data/raw/us") -> dict:
    """미국 종목 전체 데이터 수집 진입점"""
    save_dir = os.path.join(base_dir, ticker)

    stock_df = fetch_stock_price(ticker, save_dir)
    financials = fetch_financials(ticker, save_dir)

    return {
        "ticker": ticker,
        "stock": stock_df,
        **financials
    }


if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "MSFT"]
    for t in tickers:
        fetch_us(t)