from fetch_us import fetch_us
from fetch_kr import fetch_kr
from preprocess import preprocess
from analyze import analyze


# -----------------------------------------------------------------------
# 종목 설정
# -----------------------------------------------------------------------

US_STOCKS = ["WDC","GOOGL","MRVL"]
KR_STOCKS = [""]  # 삼성전자, SK하이닉스


# -----------------------------------------------------------------------
# 실행 흐름
# -----------------------------------------------------------------------

def run(us_stocks: list = US_STOCKS, kr_stocks: list = KR_STOCKS):
    results = []

    # 미국 종목
    for ticker in us_stocks:
        print(f"\n{'='*50}")
        print(f"[미국] {ticker} 처리 중...")
        print(f"{'='*50}")

        try:
            raw = fetch_us(ticker)
            clean = preprocess(raw, market="US")
            report = analyze(clean)
            results.append({"ticker": ticker, "market": "US", "report": report})
            print(f"\n📊 {ticker} 분석 결과\n")
            print(report)
        except Exception as e:
            print(f"[{ticker}] 오류: {e}")

    # 한국 종목
    for stock_code in kr_stocks:
        print(f"\n{'='*50}")
        print(f"[한국] {stock_code} 처리 중...")
        print(f"{'='*50}")

        try:
            raw = fetch_kr(stock_code)
            clean = preprocess(raw, market="KR")
            report = analyze(clean)
            results.append({"ticker": stock_code, "market": "KR", "report": report})
            print(f"\n📊 {stock_code} 분석 결과\n")
            print(report)
        except Exception as e:
            print(f"[{stock_code}] 오류: {e}")

    return results


if __name__ == "__main__":
    run()