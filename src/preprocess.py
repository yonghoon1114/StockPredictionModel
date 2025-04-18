import pandas as pd
import os
from config import companyCode, Date

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_and_merge_data(stock_path: str, rate_path: str, nasdaq_path: str, financial_path: str, gold_path: str) -> pd.DataFrame:
    def read_file(path: str, col_prefix: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            header=2,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )[[ "Close" ]].rename(columns={"Close": f"{col_prefix}_close"})

    stock = read_file(stock_path, "stock")
    rate = read_file(rate_path, "rate")
    nasdaq = read_file(nasdaq_path, "nasdaq")
    gold = read_file(gold_path, "gold")

    # RSI 계산 후 열 추가
    stock["RSI"] = calculate_rsi(stock["stock_close"])

    # 재무 데이터 로드
    financials = pd.read_csv(financial_path, parse_dates=["Date"])
    financials = financials.set_index("Date")


    # 세 개 데이터프레임 병합
    df = stock.join(rate, how="inner").join(nasdaq, how="inner").join(gold, how="left")

    # 재무 데이터 병합 (가장 가까운 날짜의 재무제표로 보간)
    df = df.join(financials, how="left").ffill()

    
    # 타겟값 생성 (다음 날 주가)
    df["target"] = df["stock_close"].shift(-1)

    # NaN 제거 (RSI, target 없는 행들 제거)
    df = df.dropna(subset=["RSI", "target"])

    return df

if __name__ == "__main__":
    stock_file = os.path.join("data", "raw","Companies", f"{companyCode}", f"{companyCode}_1910-01-01_{Date}.csv")
    rate_file = os.path.join("data", "raw", "INTEREST", f"IRX_1910-01-01_{Date}.csv")
    gold_file = os.path.join("data", "raw", "GOLD", f"GCF_2000-01-01_{Date}.csv")
    nasdaq_file = os.path.join("data", "raw", "NASDAQ", f"IXIC_1910-01-01_{Date}.csv")
    financial_file = os.path.join("data", "raw","Companies", f"{companyCode}", f"{companyCode}_quarterly_financials_expanded.csv")
    
    merged_df = load_and_merge_data(stock_file, rate_file, nasdaq_file, financial_file, gold_file)
    save_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path)
    print(f"Merged data saved to {save_path}")
    print(merged_df.head())
