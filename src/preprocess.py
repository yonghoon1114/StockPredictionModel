import pandas as pd


# -----------------------------------------------------------------------
# 미국 (yfinance)
# -----------------------------------------------------------------------

def preprocess_us(raw: dict) -> dict:
    """
    fetch_us() 결과를 AI 입력용 공통 포맷으로 변환
    yfinance info에서 이미 계산된 지표를 그대로 활용
    """
    info = raw.get("info", {})
    stock_df = raw.get("stock", pd.DataFrame())

    current_price = info.get("currentPrice") or (
        stock_df["Close"].iloc[-1] if not stock_df.empty else None
    )

    return {
        "ticker": raw.get("ticker"),
        "name": info.get("longName"),
        "market": "US",
        "currency": info.get("currency", "USD"),
        "current_price": current_price,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "valuation": {
            "PER": info.get("trailingPE"),
            "forward_PER": info.get("forwardPE"),
            "PBR": info.get("priceToBook"),
            "PSR": _calc_psr(info),
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
            "market_cap": _billion(info.get("marketCap")),
            "total_revenue": _billion(info.get("totalRevenue")),
            "total_debt": _billion(info.get("totalDebt")),
            "free_cashflow": _billion(info.get("freeCashflow")),
        },
    }


def _calc_psr(info: dict):
    market_cap = info.get("marketCap")
    total_revenue = info.get("totalRevenue")
    if market_cap and total_revenue and total_revenue != 0:
        return round(market_cap / total_revenue, 2)
    return None


# -----------------------------------------------------------------------
# 한국 (DART)
# -----------------------------------------------------------------------

# AI에 넘길 주요 계정과목만 추출
KR_ACCOUNT_MAP = {
    "매출액": ("size", "total_revenue"),
    "영업이익": ("profitability", "operating_income"),
    "당기순이익": ("profitability", "net_income"),
    "자산총계": ("stability", "total_assets"),
    "부채총계": ("stability", "total_liabilities"),
    "자본총계": ("stability", "total_equity"),
    "유동자산": ("stability", "current_assets"),
    "유동부채": ("stability", "current_liabilities"),
    "영업활동현금흐름": ("size", "operating_cashflow"),
}


def preprocess_kr(raw: dict) -> dict:
    """
    fetch_kr() 결과를 AI 입력용 공통 포맷으로 변환
    DART raw 데이터에서 주요 계정과목만 추출 후 지표 계산
    """
    info = raw.get("info", {})
    stock_df = raw.get("stock", pd.DataFrame())
    financials_df = raw.get("financials", pd.DataFrame())

    current_price = stock_df["Close"].iloc[-1] if not stock_df.empty else None

    # 가장 최근 연도 사업보고서 데이터만 추출
    accounts = _extract_accounts(financials_df)

    # 지표 계산
    revenue = accounts.get("total_revenue")
    net_income = accounts.get("net_income")
    operating_income = accounts.get("operating_income")
    total_assets = accounts.get("total_assets")
    total_equity = accounts.get("total_equity")
    total_liabilities = accounts.get("total_liabilities")
    current_assets = accounts.get("current_assets")
    current_liabilities = accounts.get("current_liabilities")

    return {
        "ticker": raw.get("stock_code"),
        "name": info.get("name"),
        "market": "KR",
        "currency": "KRW",
        "current_price": current_price,
        "sector": info.get("industry"),
        "industry": info.get("industry"),
        "valuation": {
            "PER": None,   # 시가총액 데이터 없으면 계산 불가 → AI가 맥락으로 판단
            "PBR": None,
            "PSR": None,
        },
        "profitability": {
            "ROE": _ratio(net_income, total_equity),
            "ROA": _ratio(net_income, total_assets),
            "operating_margin": _ratio(operating_income, revenue),
            "net_margin": _ratio(net_income, revenue),
            "gross_margin": None,
        },
        "growth": {
            "revenue_yoy": None,    # 전년도 데이터 있으면 추후 계산 가능
            "earnings_yoy": None,
        },
        "stability": {
            "debt_to_equity": _ratio(total_liabilities, total_equity),
            "current_ratio": _ratio(current_assets, current_liabilities),
            "quick_ratio": None,
            **accounts,             # 원본 수치도 함께 전달 (억원 단위)
        },
        "size": {
            "total_revenue": _billion_krw(revenue),
            "operating_cashflow": _billion_krw(accounts.get("operating_cashflow")),
        },
    }


def _extract_accounts(df: pd.DataFrame) -> dict:
    """DART 재무제표에서 최근 연도 주요 계정과목 추출"""
    if df.empty:
        return {}

    # 가장 최근 연도 사업보고서만
    latest_year = df["year"].max()
    df_latest = df[(df["year"] == latest_year) & (df["report"] == "사업보고서")]

    accounts = {}
    for _, row in df_latest.iterrows():
        account_name = str(row.get("account", "")).strip()
        if account_name in KR_ACCOUNT_MAP:
            _, key = KR_ACCOUNT_MAP[account_name]
            accounts[key] = row.get("current")

    return accounts


# -----------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------

def _pct(val):
    """소수 → 퍼센트 변환 (0.87 → 87.0)"""
    if val is None:
        return None
    return round(val * 100, 2)


def _ratio(numerator, denominator):
    """비율 계산 (분모 0 방지)"""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _billion(val):
    """달러 → 십억 달러 단위로 변환"""
    if val is None:
        return None
    return round(val / 1e9, 2)


def _billion_krw(val):
    """원 → 억원 단위로 변환"""
    if val is None:
        return None
    return round(val / 1e8, 0)


# -----------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------

def preprocess(raw: dict, market: str) -> dict:
    """
    market: "US" or "KR"
    """
    if market == "US":
        return preprocess_us(raw)
    elif market == "KR":
        return preprocess_kr(raw)
    else:
        raise ValueError(f"지원하지 않는 시장: {market}")