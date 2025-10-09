import pandas as pd
import os
from itertools import chain

# --- 유틸리티 함수들 (그대로 유지) ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_relative(stock_close: pd.Series, index: pd.Series) -> pd.Series:
    return stock_close / index

def read_file(path: str, col_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=2,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        index_col="Date"
    )
    return df[["Close"]].rename(columns={"Close": f"{col_prefix}_close"})

def preprocess(stock_df: pd.DataFrame, sector:str) -> pd.DataFrame:
    stock_df["RSI"] = calculate_rsi(stock_df["stock_close"])
    stock_df["relative"] = calculate_relative(stock_df["stock_close"], stock_df["sector_close"])
    stock_df["target"] = stock_df["stock_close"].shift(-1)
    election_dates = ["2017-01-20", "2025-01-20"]
    election_ranges = [
        pd.date_range(start=pd.to_datetime(day), 
                    end=pd.to_datetime(day) + pd.Timedelta(days=1461))
        for day in election_dates
    ]
    extended_election_days = pd.DatetimeIndex(sorted(set(chain.from_iterable(election_ranges))))
    stock_df["election_marker"] = stock_df.index.isin(extended_election_days).astype(int)

    return stock_df.dropna(subset=["RSI", "target"])


def load_and_merge_data(paths: dict, financial_path: str, sector:str) -> pd.DataFrame:
    dfs = {prefix: read_file(path, prefix) for prefix, path in paths.items()}

    stock = dfs["stock"].join(dfs["nasdaq"], how="left").join(dfs["sector"], how="left")
    stock = preprocess(stock, sector)

    for key in ["rate", "gold"]:
        stock = stock.join(dfs[key], how="left")

    financials = pd.read_csv(financial_path, parse_dates=["Date"]).set_index("Date")

    return stock.join(financials, how="left").ffill()


# --- 최적화된 실행 로직 ---
def process_company(company_code: str, date: str, sector:str) -> None:
    data_root = "data/raw"
    company_path = os.path.join(data_root, "Companies", company_code)

    paths = {
        "stock": os.path.join(company_path, f"{company_code}_1910-01-01_{date}.csv"),
        "rate": os.path.join(data_root, "MACRO", f"IRX_1910-01-01_{date}.csv"),
        "nasdaq": os.path.join(data_root, "MACRO", f"IXIC_1910-01-01_{date}.csv"),
        "gold": os.path.join(data_root, "MACRO", f"GCF_1910-01-01_{date}.csv"),
        "sector": os.path.join(data_root, "MACRO", f"{sector}_1910-01-01_{date}.csv"),
    }

    financial_file = os.path.join(company_path, f"{company_code}_quarterly_financials_expanded.csv")
    merged_df = load_and_merge_data(paths, financial_file, sector)

    save_path = os.path.join("data", "processed", f"{company_code}_{date}_merged.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path)
    print(f"Merged data saved to {save_path}")
    print(merged_df.head())
