from fetch_us import fetch_us
from fetch_kr import fetch_kr
from preprocess import preprocess
from analyze import analyze_stock, analyze_portfolio, save_report
from portfolio import get, get_all, summary, get_profile


# -----------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------

MODE = "portfolio"  # "analyze" or "portfolio"

US_STOCKS = ["PLTR"]
KR_STOCKS = ["005930"]


# -----------------------------------------------------------------------
# 종목별 재무 분석
# -----------------------------------------------------------------------

def run_analyze(us_stocks: list = US_STOCKS, kr_stocks: list = KR_STOCKS):

    if not us_stocks and not kr_stocks:
        print("분석할 종목이 없습니다.")
        return

    if not us_stocks:
        print("미국 종목 없음, 건너뜀")
    for ticker in us_stocks:
        print(f"\n{'='*50}")
        print(f"[미국] {ticker} 분석 중...")
        print(f"{'='*50}")
        try:
            raw = fetch_us(ticker)
            clean = preprocess(raw, market="US")
            portfolio = get(ticker)
            result = analyze_stock(clean, portfolio)
            save_report(ticker, clean, result)
        except Exception as e:
            print(f"[{ticker}] 오류: {e}")

    if not kr_stocks:
        print("한국 종목 없음, 건너뜀")
    for stock_code in kr_stocks:
        print(f"\n{'='*50}")
        print(f"[한국] {stock_code} 분석 중...")
        print(f"{'='*50}")
        try:
            raw = fetch_kr(stock_code)
            clean = preprocess(raw, market="KR")
            portfolio = get(stock_code)
            result = analyze_stock(clean, portfolio)
            save_report(stock_code, clean, result)
        except Exception as e:
            print(f"[{stock_code}] 오류: {e}")


# -----------------------------------------------------------------------
# 전체 포트폴리오 진단
# -----------------------------------------------------------------------

def run_portfolio():
    portfolio = get_all()

    if not portfolio:
        print("포트폴리오가 비어있습니다.")
        return

    print(f"\n{'='*50}")
    print(f"포트폴리오 전체 진단 중... ({len(portfolio)}개 종목)")
    print(f"{'='*50}")

    current_prices = {}
    financial_data = {}  # 재무데이터도 같이 수집

    for ticker, data in portfolio.items():
        try:
            market = data.get("market") or ("KR" if ticker.isdigit() and len(ticker) == 6 else "US")
            if market == "US":
                raw = fetch_us(ticker)
                clean = preprocess(raw, market="US")
            else:
                raw = fetch_kr(ticker)
                clean = preprocess(raw, market="KR")

            current_prices[ticker] = clean.get("current_price")
            financial_data[ticker] = clean  # 재무데이터 저장
            print(f"[{ticker}] 데이터 수집 완료")
        except Exception as e:
            print(f"[{ticker}] 데이터 수집 실패: {e}")

    # 손익 계산
    portfolio_summary = summary(current_prices)

    # 투자자 프로필 읽기
    profile = get_profile()

    # AI 진단 (손익 + 재무데이터 + 프로필 전달)
    result = analyze_portfolio(portfolio_summary, financial_data, profile)
    save_report("portfolio", {}, result)

    print("\n📊 포트폴리오 진단 완료")
    print(f"결과: data/reports/portfolio.json")


# -----------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if MODE == "analyze":
        run_analyze()
    elif MODE == "portfolio":
        run_portfolio()
    else:
        print(f"알 수 없는 모드: {MODE}")