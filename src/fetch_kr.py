import os
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from datetime import date, timedelta


# -----------------------------------------------------------------------
# 주가 데이터
# -----------------------------------------------------------------------

def fetch_stock_price_kr(stock_code: str, save_dir: str) -> pd.DataFrame:
    """FinanceDataReader로 한국 주가 수집 (백업용)"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{stock_code}_stock.csv")
    end = date.today().strftime("%Y-%m-%d")

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, parse_dates=["Date"])
        last_date = existing["Date"].max().date()
        new_start = last_date + timedelta(days=1)

        if pd.Timestamp(new_start) >= pd.Timestamp(end):
            print(f"[{stock_code}] 주가 최신 상태")
            return existing

        print(f"[{stock_code}] 주가 업데이트: {new_start} → {end}")
        new_data = fdr.DataReader(stock_code, start=str(new_start), end=end)

        if new_data.empty:
            print(f"[{stock_code}] 새 주가 데이터 없음")
            return existing

        new_data = new_data.reset_index().rename(columns={"index": "Date"})
        new_data = new_data[["Date", "Close", "High", "Low", "Open", "Volume"]]

        merged = pd.concat([existing, new_data], ignore_index=True)
        merged.drop_duplicates(subset="Date", keep="last", inplace=True)
        merged.sort_values("Date", inplace=True)
        merged.to_csv(file_path, index=False)
        return merged

    else:
        print(f"[{stock_code}] 전체 주가 이력 다운로드")
        data = fdr.DataReader(stock_code, start="2000-01-01", end=end)
        data = data.reset_index().rename(columns={"index": "Date"})
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
        data.to_csv(file_path, index=False)
        print(f"[{stock_code}] 주가 저장 완료: {file_path}")
        return data


# -----------------------------------------------------------------------
# yfinance 재무데이터
# -----------------------------------------------------------------------

def fetch_yf_kr(stock_code: str, save_dir: str) -> dict:
    """
    yfinance로 한국 주식 재무데이터 수집
    .KS (코스피) → .KQ (코스닥) 순서로 시도
    """
    os.makedirs(save_dir, exist_ok=True)

    for suffix in [".KS", ".KQ"]:
        try:
            ticker = f"{stock_code}{suffix}"
            info = yf.Ticker(ticker).info

            if not info.get("currentPrice"):
                continue

            result = {
                "ticker_yf": ticker,
                "name": info.get("longName"),
                "current_price": info.get("currentPrice"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "net_income": info.get("netIncomeToCommon"),
                "market_cap": None,  # 아래서 직접 계산
                "PER": None,         # 아래서 직접 계산
                "PBR": None,         # bookValue 없어서 생략
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "total_revenue": info.get("totalRevenue"),
                "free_cashflow": info.get("freeCashflow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }

            # 직접 계산
            price = result["current_price"]
            shares = result["shares_outstanding"]
            net_income = result["net_income"]

            if price and shares:
                result["market_cap"] = price * shares
            if price and shares and net_income:
                eps = net_income / shares
                result["PER"] = round(price / eps, 2) if eps > 0 else None

            # 저장
            path = os.path.join(save_dir, f"{stock_code}_info.csv")
            pd.DataFrame([result]).to_csv(path, index=False)
            print(f"[{stock_code}] yfinance 데이터 저장 완료 ({ticker})")
            return result

        except Exception as e:
            print(f"[{stock_code}] yfinance {suffix} 실패: {e}")

    print(f"[{stock_code}] yfinance 데이터 수집 실패")
    return {}


# -----------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------

def fetch_kr(stock_code: str, base_dir: str = "data/raw/kr") -> dict:
    """
    한국 종목 전체 데이터 수집 진입점
    stock_code: 6자리 종목코드 (예: "005930" 삼성전자)
    """
    save_dir = os.path.join(base_dir, stock_code)

    stock_df = fetch_stock_price_kr(stock_code, save_dir)
    yf_data = fetch_yf_kr(stock_code, save_dir)

    return {
        "stock_code": stock_code,
        "stock": stock_df,
        "yf_data": yf_data,
    }


if __name__ == "__main__":
    stocks = ["005930", "000660"]
    for code in stocks:
        fetch_kr(code)