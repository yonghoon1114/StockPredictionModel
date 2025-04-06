import pandas as pd
import os

def load_and_merge_data(stock_path: str, rate_path: str, nasdaq_path: str) -> pd.DataFrame:
    def read_file(path: str, col_prefix: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            skiprows=3,
            header=None,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )[[ "Close" ]].rename(columns={"Close": f"{col_prefix}_close"})

    stock = read_file(stock_path, "stock")
    rate = read_file(rate_path, "rate")
    nasdaq = read_file(nasdaq_path, "nasdaq")

    # 세 개 데이터프레임 병합
    df = stock.join(rate, how="inner").join(nasdaq, how="inner")

    # 타겟값 생성 (다음 날 주가)
    df["target"] = df["stock_close"].shift(-1)

    # NaN 제거
    df = df.dropna()

    return df

if __name__ == "__main__":
    stock_file = os.path.join("data", "raw", "AAPL_2020-01-01_2025-04-05.csv")
    rate_file = os.path.join("data", "raw", "IRX_2020-01-01_2025-04-01.csv")
    nasdaq_file = os.path.join("data", "raw", "IXIC_2020-01-01_2025-04-01.csv")

    merged_df = load_and_merge_data(stock_file, rate_file, nasdaq_file)
    print(merged_df.head())
