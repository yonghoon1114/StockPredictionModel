import yfinance as yf
from collections import defaultdict
from fetch_company_data_100 import fetchdata
from fetch_macro_data import fetch_macro_data
from preprocessForAll import process_company
import os
from trainModelforAll import load_data, train_transformer_model, trainModel
from predictPriceForAll import runPrediction

companyCode = "TSLA"  
Date = "2026-01-01"

Companies_for_prediction = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'ADBE', 'INTC', 'AMD'
]

# 섹터별 매크로 지수 맵핑
sector_index_map = {
    "Technology": "^NDXT",
    "Communication Services": "^XLC",
    "Consumer Cyclical": "^XLY",
    "Financial Services": "^XLF",
    "Health Care": "^XLV",
    "Energy": "^XLE",
    "Consumer Defensive": "^XLP",
    "Basic Materials": "^XLB",
    "Utilities": "^XLU",
    "Industrials": "^XLI",
    "Real Estate": "^XLRE",
    "Semiconductors": "^SOX"
}

# ✅ 섹터 정보 저장용 딕셔너리
ticker_to_sector_ticker = {}

# ✅ 섹터 정보 수집
for ticker in Companies_for_prediction:
    try:
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get("sector", None)
        if sector and sector in sector_index_map:
            ticker_to_sector_ticker[ticker] = sector_index_map[sector]
        else:
            ticker_to_sector_ticker[ticker] = None
    except Exception as e:
        print(f"{ticker} sector fetch failed: {e}")
        ticker_to_sector_ticker[ticker] = None

# 해당 회사들의 섹터 정보를 수집하고 매크로 지수 준비
def get_macro_sources(companies, date):
    tickers_set = set()
    for ticker in companies:
        ticker_to_sector_ticker.get(ticker)


    macro_sources = [
        {"name": "interest_rate", "ticker": "^IRX", "save_path": "data/raw", "prefix": "MACRO"},
        {"name": "nasdaq_index", "ticker": "^IXIC", "save_path": "data/raw", "prefix": "MACRO"},
        {"name": "gold_price", "ticker": "GC=F", "start": "2000-01-01", "end": date, "save_path": "data/raw", "prefix": "MACRO"}
    ]

    for ticker in tickers_set:
        macro_sources.append({
            "name": ticker, "ticker": ticker,
            "start": "2000-01-01", "end": date,
            "save_path": "data/raw", "prefix": "MACRO"
        })

    return macro_sources

def fetch_all_macro_data(macro_sources: list):
    for source in macro_sources:
        kwargs = {k: v for k, v in source.items() if k != "name"}
        fetch_macro_data(**kwargs)

if __name__ == "__main__":

    # 1. 매크로 지표 수집
    # macro_sources = get_macro_sources(Companies_for_prediction, Date)
    # fetch_all_macro_data(macro_sources)

    # # 2. 각 기업 개별 데이터 수집
    # for companyCode in Companies_for_prediction:
    #     fetchdata(companyCode, Date)

    # 3. 데이터 전처리 (sector ticker 정보 함께 전달)
    # for code in Companies_for_prediction:
    #     sector_ticker = ticker_to_sector_ticker.get(code, "Unknown")
    #     sector_ticker = sector_ticker.replace("^","")
    #     process_company(code, Date, sector_ticker)

    # 4. 각 기업별 Transformer 모델 훈련
    trainModel(Companies_for_prediction, Date)

    # 5. 예측 수행
    runPrediction(Companies_for_prediction, Date)

# Technology 20
# consumer cyclical 7
# communication services 7
# financial services 17
# health care 22
# Energy 2
# Consumer Defensive 8
# Basic Materials 2
# Utilities 1
# Industrials 11
# Real Estate 2
