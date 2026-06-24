import pandas as pd


def _pct(val):
    """yfinance 소수 비율 → 퍼센트 변환 (0.18855 → 18.86%)"""
    if val is None:
        return None
    return round(val * 100, 2)


# -----------------------------------------------------------------------
# 미국 (yfinance)
# -----------------------------------------------------------------------

def preprocess_us(raw: dict) -> dict:
    info = raw.get("info", {})
    stock_df = raw.get("stock", pd.DataFrame())

    current_price = stock_df["Close"].iloc[-1] if not stock_df.empty else info.get("currentPrice")

    return {
        "ticker": raw.get("ticker"),
        "name": info.get("longName"),
        "market": "US",
        "currency": "USD",
        "current_price": current_price,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "valuation": {
            "PER": info.get("trailingPE"),
            "forward_PER": info.get("forwardPE"),
            "PBR": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
        },
        "profitability": {
            "ROE": _pct(info.get("returnOnEquity")),
            "ROA": _pct(info.get("returnOnAssets")),
            "operating_margin": _pct(info.get("operatingMargins")),
            "net_margin": _pct(info.get("profitMargins")),
            "gross_margin": _pct(info.get("grossMargins")),
        },
        "growth": {
            "revenue_yoy": _pct(info.get("revenueGrowth")),
            "earnings_yoy": _pct(info.get("earningsGrowth")),
        },
        "stability": {
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
        },
        "size": {
            "total_revenue": info.get("totalRevenue"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
        },
    }


# -----------------------------------------------------------------------
# 한국 (DART + yfinance 보완)
# -----------------------------------------------------------------------

def preprocess_kr(raw: dict) -> dict:
    yf = raw.get("yf_data", {})
    stock_df = raw.get("stock", pd.DataFrame())

    current_price = yf.get("current_price") or (
        stock_df["Close"].iloc[-1] if not stock_df.empty else None
    )

    return {
        "ticker": raw.get("stock_code"),
        "name": yf.get("name"),
        "market": "KR",
        "currency": "KRW",
        "current_price": current_price,
        "sector": yf.get("sector"),
        "industry": yf.get("industry"),
        "valuation": {
            "PER": yf.get("PER"),
            "PBR": yf.get("PBR"),
            "market_cap": yf.get("market_cap"),
        },
        "profitability": {
            "ROE": _pct(yf.get("ROE")),
            "ROA": _pct(yf.get("ROA")),
            "operating_margin": _pct(yf.get("operating_margin")),
            "net_margin": _pct(yf.get("net_margin")),
        },
        "growth": {
            "revenue_yoy": _pct(yf.get("revenue_growth")),
            "earnings_yoy": _pct(yf.get("earnings_growth")),
        },
        "stability": {
            "debt_to_equity": yf.get("debt_to_equity"),
            "current_ratio": yf.get("current_ratio"),
        },
        "size": {
            "total_revenue": yf.get("total_revenue"),
            "free_cashflow": yf.get("free_cashflow"),
        },
    }


# -----------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------

def preprocess(raw: dict, market: str) -> dict:
    if market == "US":
        return preprocess_us(raw)
    elif market == "KR":
        return preprocess_kr(raw)
    else:
        raise ValueError(f"지원하지 않는 시장: {market}")