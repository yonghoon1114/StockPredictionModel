
import yfinance as yf
from collections import defaultdict

companyCode = "NVDA"  
sequenceLength = 40 # 테스트 용 데이터 개수
Date = "2026-01-01"
data_columns = ["stock_close", "rate_close", "nasdaq_close", "Revenue", "NetIncome", "TotalAssets", "RSI", "gold_close", "semiCond_close","election_marker","relative"] #데이터 종류
# data_columns = ["stock_close","nasdaq_close","RSI","relative","target","election_marker","rate_close","gold_close","Revenue","NetIncome","TotalAssets"]

# data_columns = ["stock_close","nasdaq_close","RSI","relative","target",
#                "election_marker","rate_close","gold_close","fin_Revenue","fin_NetIncome","fin_TotalAssets","fin_TotalLiabilities","fin_OperatingCashFlow",
#                "fin_CapitalExpenditures","company"]
# data_columns = ["stock_close","nasdaq_close","RSI","relative","target","election_marker","rate_close","gold_close","fin_Revenue",
#                 "fin_NetIncome","fin_TotalAssets","fin_TotalLiabilities","fin_OperatingCashFlow","fin_CapitalExpenditures"]

dataNumber = len(data_columns)

# # 시가총액이 높은 상위 100개 기업 티커 목록 (예시로, S&P 500 티커들 사용)
# tickers = [
#     'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'V', 'JNJ', 
#     'WMT', 'PYPL', 'DIS', 'BA', 'INTC', 'NFLX', 'CSCO', 'HD', 'UNH', 'MS', 
#     # 이 리스트는 예시로 20개만 넣었으며, 실제로는 더 많은 티커들이 필요
# ]
sp500_top100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "GOOG", "LLY", "UNH",
    "JPM", "V", "XOM", "JNJ", "WMT", "MA", "AVGO", "PG", "CVX", "MRK",
    "COST", "HD", "ABBV", "ADBE", "CRM", "PEP", "NFLX", "TMO", "BAC", "AMD",
    "KO", "CSCO", "LIN", "ORCL", "ACN", "ABT", "MCD", "DHR", "INTC", "WFC",
    "VZ", "BMY", "TXN", "NEE", "AMGN", "PFE", "DIS", "PM", "UNP", "MS",
    "RTX", "LOW", "INTU", "HON", "CAT", "SPGI", "QCOM", "AMT", "T", "NOW",
    "SCHW", "IBM", "GE", "ISRG", "ELV", "MDT", "CVS", "PLD", "ADP", "GILD",
    "DE", "SYK", "LRCX", "BKNG", "ZTS", "BLK", "AXP", "TJX", "CI", "C",
    "PGR", "BDX", "MU", "MO", "APD", "ADI", "NSC", "MMC", "CME", "USB",
    "REGN", "ETN", "VRTX", "GM", "FDX", "ITW", "NOC", "TGT", "PNC",'TSLA'
]

# 통합해서 한 번에 가져오기
company_info = []

for ticker in sp500_top100:
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info  # 빠른 데이터 접근 (기본적인 것만 있을 경우)

        stock_info = stock.info  # 자세한 데이터 접근

        company_info.append({
            'Ticker': ticker,
            'Market Cap': stock_info.get('marketCap', 'N/A'),
            'Sector': stock_info.get('sector', 'N/A'),
            'Name': stock_info.get('longName', 'N/A'),
        })
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

# 섹터별로 그룹화
sector_group = defaultdict(list)

for company in company_info:
    sector = company['Sector']
    sector_group[sector].append(company)

sector_arrays = {}

# 결과 출력
for sector, companies in sector_group.items():
    # 공백이나 특수문자가 들어간 Sector 이름은 변수 이름으로 부적합할 수 있어서 수정
    sector_name = sector.replace(' ', '_').replace('&', 'and').replace('-', '_')
    sector_arrays[sector_name] = companies

# 예시: Technology 섹터만 출력해보기
for company in sector_arrays.get('Technology', []):
    print(f"{company['Ticker']} - {company['Name']} (Market Cap: {company['Market Cap']})")

# print(sp500_top100)

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