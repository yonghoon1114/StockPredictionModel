import pandas as pd
import os
from config import companyCode, Date
from itertools import chain


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_relative(stock_close: pd.Series, semi_close: pd.Series) -> pd.Series:
    return stock_close / semi_close

def read_file(path: str, col_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=2,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        index_col="Date"
    )
    return df[["Close"]].rename(columns={"Close": f"{col_prefix}_close"})

def preprocess(stock_df: pd.DataFrame) -> pd.DataFrame:
    stock_df["RSI"] = calculate_rsi(stock_df["stock_close"])
    stock_df["relative"] = calculate_relative(stock_df["stock_close"], stock_df["semiCond_close"])
    stock_df["target"] = stock_df["stock_close"].shift(-1)

    # 이벤트 마커: 대선일 ±30일
    election_dates = ["2020-11-03", "2024-11-05"]
    election_ranges = [
        pd.date_range(start=pd.to_datetime(day) - pd.DateOffset(days=30), end=pd.to_datetime(day) + pd.DateOffset(days=30))
        for day in election_dates
    ]
    # 날짜 리스트 펼쳐서 중복 제거 후 인덱스로 만들기
    extended_election_days = pd.DatetimeIndex(sorted(set(chain.from_iterable(election_ranges))))
    stock_df["election_marker"] = stock_df.index.isin(extended_election_days)

    return stock_df.dropna(subset=["RSI", "target"])


def load_and_merge_data(paths: dict, financial_path: str) -> pd.DataFrame:
    dfs = {prefix: read_file(path, prefix) for prefix, path in paths.items()}

    # stock과 semicond는 먼저 결합한 후 전처리
    stock = dfs["stock"].join(dfs["semiCond"], how="left")
    stock = preprocess(stock)

    # 기타 macro data 병합
    macro_keys = ["rate", "nasdaq", "gold"]
    for key in macro_keys:
        stock = stock.join(dfs[key], how="inner" if key != "gold" else "left")

    # 재무 데이터
    financials = pd.read_csv(financial_path, parse_dates=["Date"]).set_index("Date")
    full_df = stock.join(financials, how="left").ffill()

    return full_df

if __name__ == "__main__":
    data_root = "data/raw"
    company_path = os.path.join(data_root, "Companies", companyCode)
    paths = {
        "stock": os.path.join(company_path, f"{companyCode}_1910-01-01_{Date}.csv"),
        "rate": os.path.join(data_root, "INTEREST", f"IRX_1910-01-01_{Date}.csv"),
        "nasdaq": os.path.join(data_root, "NASDAQ", f"IXIC_1910-01-01_{Date}.csv"),
        "gold": os.path.join(data_root, "GOLD", f"GCF_2000-01-01_{Date}.csv"),
        "semiCond": os.path.join(data_root, "SEMICOND", f"SOX_1910-01-01_{Date}.csv"),
    }
    financial_file = os.path.join(company_path, f"{companyCode}_quarterly_financials_expanded.csv")

    merged_df = load_and_merge_data(paths, financial_file)
    save_path = os.path.join("data", "processed", f"{companyCode}_{Date}_merged.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path)
    print(f"Merged data saved to {save_path}")
    print(merged_df.head())
