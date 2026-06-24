import json
import os

PORTFOLIO_PATH = "data/portfolio.json"
PROFILE_PATH = "data/profile.json"


# -----------------------------------------------------------------------
# 기본 유틸
# -----------------------------------------------------------------------

def _load() -> dict:
    if not os.path.exists(PORTFOLIO_PATH):
        return {}
    with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile() -> dict:
    """투자자 프로필 조회"""
    if not os.path.exists(PROFILE_PATH):
        print("profile.json 없음, 프로필 없이 진행")
        return {}
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(portfolio: dict):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------
# CRUD
# -----------------------------------------------------------------------

def add(ticker: str, quantity: float, avg_price: float, market: str):
    """
    종목 추가 또는 평단가 업데이트
    market: "US" or "KR"
    """
    portfolio = _load()

    if ticker in portfolio:
        # 기존 보유분과 합산해서 평단가 재계산
        existing = portfolio[ticker]
        total_quantity = existing["quantity"] + quantity
        total_cost = (existing["quantity"] * existing["avg_price"]) + (quantity * avg_price)
        avg = total_cost / total_quantity

        portfolio[ticker] = {
            "quantity": total_quantity,
            "avg_price": round(avg, 4),
            "market": market,
        }
        print(f"[{ticker}] 추가 매수 반영 → 수량: {total_quantity}, 평단가: {round(avg, 4)}")
    else:
        portfolio[ticker] = {
            "quantity": quantity,
            "avg_price": avg_price,
            "market": market,
        }
        print(f"[{ticker}] 포트폴리오 추가 완료")

    _save(portfolio)


def remove(ticker: str):
    """종목 제거"""
    portfolio = _load()

    if ticker not in portfolio:
        print(f"[{ticker}] 포트폴리오에 없는 종목")
        return

    del portfolio[ticker]
    _save(portfolio)
    print(f"[{ticker}] 포트폴리오에서 제거 완료")


def update(ticker: str, quantity: float = None, avg_price: float = None):
    """수량 또는 평단가 직접 수정"""
    portfolio = _load()

    if ticker not in portfolio:
        print(f"[{ticker}] 포트폴리오에 없는 종목")
        return

    if quantity is not None:
        portfolio[ticker]["quantity"] = quantity
    if avg_price is not None:
        portfolio[ticker]["avg_price"] = avg_price

    _save(portfolio)
    print(f"[{ticker}] 수정 완료")


def get_all() -> dict:
    """전체 포트폴리오 조회"""
    return _load()


def get(ticker: str) -> dict | None:
    """특정 종목 조회"""
    return _load().get(ticker)


def _detect_market(ticker: str) -> str:
    """종목코드로 시장 자동 판단 (6자리 숫자면 한국, 아니면 미국)"""
    return "KR" if ticker.isdigit() and len(ticker) == 6 else "US"


def summary(current_prices: dict) -> list:
    portfolio = _load()
    result = []

    for ticker, data in portfolio.items():
        current_price = current_prices.get(ticker)
        quantity = data["quantity"]
        avg_price = data["avg_price"]
        market = data.get("market") or _detect_market(ticker)

        if current_price:
            profit = (current_price - avg_price) * quantity
            profit_pct = (current_price - avg_price) / avg_price * 100
        else:
            profit = None
            profit_pct = None

        result.append({
            "ticker": ticker,
            "market": market,
            "quantity": quantity,
            "avg_price": avg_price,
            "current_price": current_price,
            "profit": round(profit, 2) if profit is not None else None,
            "profit_pct": round(profit_pct, 2) if profit_pct is not None else None,
        })

    return result


if __name__ == "__main__":
    # 테스트
    add("AAPL", 10, 180.0, "US")
    add("AAPL", 5, 200.0, "US")       # 추가 매수 → 평단가 재계산
    add("005930", 50, 62000, "KR")

    print("\n전체 포트폴리오:")
    print(json.dumps(get_all(), ensure_ascii=False, indent=2))

    print("\n손익 요약:")
    prices = {"AAPL": 210.0, "005930": 66000}
    for row in summary(prices):
        print(row)