import feedparser
import pandas as pd
from datetime import datetime, timedelta

# === 설정 ===
TICKER = "AAPL"  # 검색할 종목
START_DATE = "2025-10-01"
END_DATE = "2025-10-19"
OUTPUT_FILE = "rss_news_aapl.csv"

# Yahoo Finance RSS URL
RSS_URL = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={TICKER}&region=US&lang=en-US"

# 날짜 범위 생성
start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")
delta = timedelta(days=1)

all_articles = []

# RSS 피드 파싱
feed = feedparser.parse(RSS_URL)

while start <= end:
    date_str = start.strftime("%Y-%m-%d")
    for entry in feed.entries:
        # 기사 게시 날짜
        published = entry.get("published", entry.get("updated"))
        if not published:
            continue
        pub_date = datetime(*entry.published_parsed[:6])
        pub_date_str = pub_date.strftime("%Y-%m-%d")

        # 날짜가 맞는 기사만 저장
        if pub_date_str == date_str:
            all_articles.append({
                "date": date_str,
                "ticker": TICKER,
                "title": entry.get("title"),
                "content": entry.get("summary"),
                "source": entry.get("source", {}).get("title", "Yahoo Finance"),
                "url": entry.get("link")
            })
    start += delta

# CSV 저장
df = pd.DataFrame(all_articles)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} articles to {OUTPUT_FILE}")
