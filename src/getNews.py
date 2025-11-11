# import feedparser
# import pandas as pd
# from datetime import datetime, timedelta

# # === 설정 ===
# TICKER = "AAPL"  # 검색할 종목
# START_DATE = "2025-10-01"
# END_DATE = "2025-10-19"
# OUTPUT_FILE = "rss_news_aapl.csv"

# # Yahoo Finance RSS URL
# RSS_URL = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={TICKER}&region=US&lang=en-US"

# # 날짜 범위 생성
# start = datetime.strptime(START_DATE, "%Y-%m-%d")
# end = datetime.strptime(END_DATE, "%Y-%m-%d")
# delta = timedelta(days=1)

# all_articles = []

# # RSS 피드 파싱
# feed = feedparser.parse(RSS_URL)

# while start <= end:
#     date_str = start.strftime("%Y-%m-%d")
#     for entry in feed.entries:
#         # 기사 게시 날짜
#         published = entry.get("published", entry.get("updated"))
#         if not published:
#             continue
#         pub_date = datetime(*entry.published_parsed[:6])
#         pub_date_str = pub_date.strftime("%Y-%m-%d")

#         # 날짜가 맞는 기사만 저장
#         if pub_date_str == date_str:
#             all_articles.append({
#                 "date": date_str,
#                 "ticker": TICKER,
#                 "title": entry.get("title"),
#                 "content": entry.get("summary"),
#                 "source": entry.get("source", {}).get("title", "Yahoo Finance"),
#                 "url": entry.get("link")
#             })
#     start += delta

# # CSV 저장
# df = pd.DataFrame(all_articles)
# df.to_csv(OUTPUT_FILE, index=False)
# print(f"Saved {len(df)} articles to {OUTPUT_FILE}")
import yfinance as yf
import pandas as pd

# Apple 티커
ticker = "AAPL"
stock = yf.Ticker(ticker)

# 1. 연간 재무제표
annual_financials = stock.financials  # 연간 재무제표
print("=== 연간 재무제표 ===")
print(annual_financials)

# 2. 분기별 재무제표
quarterly_financials = stock.quarterly_financials  # 분기별 재무제표
print("\n=== 분기별 재무제표 ===")
print(quarterly_financials)

# 3. 최근 분기별 실적 (매출, EPS)
quarterly_earnings = stock.quarterly_earnings
print("\n=== 최근 분기별 실적 (Revenue, EPS) ===")
print(quarterly_earnings)

# 4. 연간 실적 (Revenue, EPS)
annual_earnings = stock.earnings
print("\n=== 연간 실적 (Revenue, EPS) ===")
print(annual_earnings)

# 5. 기업 정보 요약
info = stock.info
print("\n=== 기업 정보 요약 ===")
print(f"회사명: {info.get('longName', 'N/A')}")
print(f"산업: {info.get('industry', 'N/A')}")
print(f"섹터: {info.get('sector', 'N/A')}")
print(f"시장: {info.get('exchange', 'N/A')}")

# 6. 최근 실적 발표일
earnings_dates = stock.calendar
print("\n=== 최근 실적 발표일 ===")
print(earnings_dates)

