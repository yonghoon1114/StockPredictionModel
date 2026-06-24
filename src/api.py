from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os

from fetch_us import fetch_us
from fetch_kr import fetch_kr
from preprocess import preprocess
from analyze import analyze_stock, analyze_portfolio
from portfolio import get_all, summary, get_profile, add, remove, update

app = FastAPI(title="Stock Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 중엔 전체 허용, 배포 시 도메인 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# 요청 모델
# -----------------------------------------------------------------------

class StockRequest(BaseModel):
    ticker: str
    market: str  # "US" or "KR"


class PortfolioItem(BaseModel):
    ticker: str
    quantity: float
    avg_price: float
    market: str


# -----------------------------------------------------------------------
# 포트폴리오 조회/수정
# -----------------------------------------------------------------------

@app.get("/portfolio")
def get_portfolio():
    """포트폴리오 + 현재가 + 손익 반환"""
    portfolio = get_all()
    if not portfolio:
        return {"holdings": [], "profile": get_profile()}

    current_prices = {}
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
        except Exception as e:
            print(f"[{ticker}] 현재가 수집 실패: {e}")

    holdings = summary(current_prices)
    return {"holdings": holdings, "profile": get_profile()}


@app.post("/portfolio")
def add_to_portfolio(item: PortfolioItem):
    """종목 추가"""
    add(item.ticker, item.quantity, item.avg_price, item.market)
    return {"status": "ok"}


@app.delete("/portfolio/{ticker}")
def delete_from_portfolio(ticker: str):
    """종목 삭제"""
    remove(ticker)
    return {"status": "ok"}


@app.patch("/portfolio/{ticker}")
def update_portfolio_item(ticker: str, quantity: Optional[float] = None, avg_price: Optional[float] = None):
    """수량/평단가 수정"""
    update(ticker, quantity, avg_price)
    return {"status": "ok"}


# -----------------------------------------------------------------------
# 종목 분석
# -----------------------------------------------------------------------

@app.post("/analyze/stock")
def analyze_single_stock(req: StockRequest):
    """단일 종목 분석"""
    try:
        if req.market == "US":
            raw = fetch_us(req.ticker)
            clean = preprocess(raw, market="US")
        elif req.market == "KR":
            raw = fetch_kr(req.ticker)
            clean = preprocess(raw, market="KR")
        else:
            raise HTTPException(status_code=400, detail="market은 US 또는 KR이어야 합니다")

        portfolio = get_all().get(req.ticker)
        result = analyze_stock(clean, portfolio)

        return {"ticker": req.ticker, "data": clean, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------
# 포트폴리오 전체 진단
# -----------------------------------------------------------------------

@app.post("/analyze/portfolio")
def analyze_full_portfolio():
    """전체 포트폴리오 진단 (재무데이터 + 프로필 포함)"""
    portfolio = get_all()
    if not portfolio:
        raise HTTPException(status_code=400, detail="포트폴리오가 비어있습니다")

    current_prices = {}
    financial_data = {}

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
            financial_data[ticker] = clean
        except Exception as e:
            print(f"[{ticker}] 데이터 수집 실패: {e}")

    portfolio_summary = summary(current_prices)
    profile = get_profile()
    result = analyze_portfolio(portfolio_summary, financial_data, profile)

    # 저장
    os.makedirs("data/reports", exist_ok=True)
    with open("data/reports/portfolio.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {"holdings": portfolio_summary, "analysis": result}


# -----------------------------------------------------------------------
# 기존 리포트 조회
# -----------------------------------------------------------------------

@app.get("/reports/{ticker}")
def get_report(ticker: str):
    """저장된 분석 리포트 조회 (가장 최근 것)"""
    path = f"data/reports/{ticker}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="리포트가 없습니다")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[-1] if data else {}
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)