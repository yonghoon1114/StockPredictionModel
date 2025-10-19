from fetch_company_data_100 import fetch_data
from fetch_macro_data import fetch_macro_data
from preprocessForAll import process_company
from trainModelforAll import trainModel
from predictPriceForAll import runPrediction
from curl_cffi import requests
from datetime import date

Date = date.today()

Companies_for_prediction = [
   "AMD"
]

session = requests.Session(impersonate="edge")

#======================================================================================
# 더 자동화 된 코드 api호출 문제 때문에 주석 해놓음
# ✅ 섹터 정보 수집

# for ticker in Companies_for_prediction:

#     try:
#         stock_info = yf.Ticker(ticker).info
#         if isinstance(stock_info, dict):
#             sector = stock_info.get("sector", None)
#             if sector and sector in sector_index_map:
#                 ticker_to_sector_ticker[ticker] = sector_index_map[sector]
#             else:
#                 ticker_to_sector_ticker[ticker] = None
#         else:
#             print(f"[경고] {ticker}의 info 타입이 이상함: {type(stock_info)}")
#             ticker_to_sector_ticker[ticker] = None
#     except Exception as e:
#         print(f"{ticker} sector fetch failed: {e}")
#         ticker_to_sector_ticker[ticker] = None
#=====================================================================================

# ✅ 각 티커의 섹터 수동 지정
ticker_to_sector_name = {
    "TSM": "Semiconductors",
    "AAPL": "Technology",
    "NVDA": "Semiconductors",
    "TSLA": "Consumer Cyclical",
    "PLTR": "Technology",
    "AMD": "Semiconductors",
    "AVGO": "Semiconductors",
    "META": "Communication Services",
    "MSFT": "Technology",
    "GOOG": "Communication Services"
}

# ✅ 섹터 이름 → 섹터별 인덱스 티커 매핑
sector_index_map = {
    "Technology": "^NDXT",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Defensive": "XLP",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Semiconductors": "^SOX"
}

# ✅ 티커 → 섹터별 인덱스 티커 매핑 생성
ticker_to_sector_ticker = {
    ticker: sector_index_map.get(sector_name)
    for ticker, sector_name in ticker_to_sector_name.items()
}


# ✅ 섹터별 매크로 소스 리스트 생성
def get_macro_sources(companies, date):
    sector_tickers = {
        ticker_to_sector_ticker.get(ticker)
        for ticker in companies
        if ticker_to_sector_ticker.get(ticker) is not None
    }

    # 기본 매크로 소스 3개
    macro_sources = [
        {"ticker": "^IRX", "start": "1910-01-01", "save_path": "data/raw", "prefix": "MACRO"},
        {"ticker": "^IXIC", "start": "1910-01-01", "save_path": "data/raw", "prefix": "MACRO"},
        {"ticker": "GC=F",  "start": "1910-01-01", "save_path": "data/raw", "prefix": "MACRO"},
    ]

    # 섹터별 매크로 인덱스 추가
    for ticker in sector_tickers:
        macro_sources.append({
            "ticker": ticker,
            "start": "1910-01-01",
            "end": date,
            "save_path": "data/raw",
            "prefix": "MACRO"
        })

    return macro_sources


# ✅ 매크로 데이터 수집
def fetch_all_macro_data(macro_sources: list):
    for source in macro_sources:
        kwargs = {k: v for k, v in source.items() if k != "name"}
        fetch_macro_data(**kwargs)


# ✅ 메인 실행 흐름
if __name__ == "__main__":

    # # 1. 지표 수집
    # macro_sources = get_macro_sources(Companies_for_prediction, Date)
    # fetch_all_macro_data(macro_sources)

    # # 2. 각 기업의 개별 데이터 수집
    # for companyCode in Companies_for_prediction:
    #     fetch_data(companyCode)

    # # 3. 전처리
    # for code in Companies_for_prediction:
    #     sector_ticker = ticker_to_sector_ticker.get(code)
    #     if sector_ticker:
    #         sector_ticker = sector_ticker.replace("^", "")
    #         process_company(code, Date, sector_ticker)
    #     else:
    #         print(f"[경고] {code}의 섹터 정보를 찾을 수 없어 전처리를 건너뜁니다.")
    
    for i in range(3):
        # 4. 모델 훈련 (주석 해제하면 실행됨!)
        trainModel(Companies_for_prediction,Date)

        # 5. 예측 수행 (옵션)
        runPrediction(Companies_for_prediction,Date)
