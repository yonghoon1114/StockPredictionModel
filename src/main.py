from fetch_us import fetch_us
from fetch_kr import fetch_kr
from preprocess import preprocess
from analyze import analyze, save_report

US_STOCKS = [
    # "NVDA", "MSFT", "WDC", "GOOGL", "ADBE", "TSLA", "META"
    # "MU"
    ]
KR_STOCKS = ["005930", "000660"]  # 삼성전자, SK하이닉스


def run(us_stocks: list = US_STOCKS, kr_stocks: list = KR_STOCKS):

    # 미국 종목
    for ticker in us_stocks:
        print(f"\n{'='*50}")
        print(f"[미국] {ticker} 처리 중...")
        print(f"{'='*50}")
        try:
            raw = fetch_us(ticker)
            clean = preprocess(raw, market="US")
            result = analyze(clean)
            save_report(ticker, clean, result)
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
            result = analyze(clean)
            save_report(stock_code, clean, result)
        except Exception as e:
            print(f"[{stock_code}] 오류: {e}")


if __name__ == "__main__":
    run()