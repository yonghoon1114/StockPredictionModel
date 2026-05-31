import pandas as pd


# -----------------------------------------------------------------------
# 밸류에이션 직접 계산 (현재가 기준, 모든 계산은 원/달러 단위 그대로)
# -----------------------------------------------------------------------

def _calc_valuation(current_price, shares, net_income, total_equity, total_revenue):
    """
    현재가 기준으로 PER, PBR, PSR 직접 계산
    단위는 미국/한국 각자 통화 그대로 사용 (비율이라 단위 무관)
    """
    if not current_price or not shares or shares == 0:
        return {"PER": None, "PBR": None, "PSR": None, "market_cap": None}

    market_cap = current_price * shares
    eps = net_income / shares if net_income else None
    bps = total_equity / shares if total_equity else None

    per = round(current_price / eps, 2) if eps and eps > 0 else None
    pbr = round(current_price / bps, 2) if bps and bps > 0 else None
    psr = round(market_cap / total_revenue, 2) if total_revenue and total_revenue > 0 else None

    return {
        "PER": per,
        "PBR": pbr,
        "PSR": psr,
        "market_cap": market_cap,
    }


# -----------------------------------------------------------------------
# 미국 (yfinance)
# -----------------------------------------------------------------------

def preprocess_us(raw: dict) -> dict:
    info = raw.get("info", {})
    stock_df = raw.get("stock", pd.DataFrame())

    current_price = stock_df["Close"].iloc[-1] if not stock_df.empty else info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    net_income = info.get("netIncomeToCommon")
    bps = info.get("bookValue")                          # yfinance bookValue = 주당 자기자본(BPS), 달러
    total_equity = bps * shares if bps and shares else None
    total_revenue = info.get("totalRevenue")             # 달러

    valuation = _calc_valuation(current_price, shares, net_income, total_equity, total_revenue)

    return {
        "ticker": raw.get("ticker"),
        "name": info.get("longName"),
        "market": "US",
        "currency": "USD",
        "current_price": current_price,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "valuation": {
            "PER": valuation["PER"],
            "forward_PER": info.get("forwardPE"),
            "PBR": valuation["PBR"],
            "PSR": valuation["PSR"],
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
            # 표시용으로만 단위 변환 (계산 끝난 후)
            "market_cap_bil_usd": _fmt_bil(valuation["market_cap"]),
            "total_revenue_bil_usd": _fmt_bil(total_revenue),
            "total_debt_bil_usd": _fmt_bil(info.get("totalDebt")),
            "free_cashflow_bil_usd": _fmt_bil(info.get("freeCashflow")),
        },
    }


# -----------------------------------------------------------------------
# 한국 (DART + yfinance 보완)
# -----------------------------------------------------------------------

KR_ACCOUNT_MAP = {
    "매출액": "total_revenue",
    "영업이익": "operating_income",
    "당기순이익": "net_income",
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "유동자산": "current_assets",
    "유동부채": "current_liabilities",
    "영업활동현금흐름": "operating_cashflow",
}


def preprocess_kr(raw: dict) -> dict:
    info = raw.get("info", {})
    stock_df = raw.get("stock", pd.DataFrame())
    financials_df = raw.get("financials", pd.DataFrame())
    yf = raw.get("yf_supplement", {})

    # 현재가: yfinance 우선, 없으면 FinanceDataReader 종가
    current_price = yf.get("current_price") or (
        stock_df["Close"].iloc[-1] if not stock_df.empty else None
    )

    # DART에서 원 단위 그대로 추출
    accounts = _extract_accounts(financials_df)

    revenue = accounts.get("total_revenue")          # 원
    net_income = accounts.get("net_income")          # 원
    operating_income = accounts.get("operating_income")  # 원
    total_assets = accounts.get("total_assets")      # 원
    total_equity = accounts.get("total_equity")      # 원
    total_liabilities = accounts.get("total_liabilities")  # 원
    current_assets = accounts.get("current_assets")  # 원
    current_liabilities = accounts.get("current_liabilities")  # 원

    # 발행주식수: yfinance에서 가져옴
    shares = yf.get("shares_outstanding")

    # 모두 원 단위로 통일된 상태에서 계산
    valuation = _calc_valuation(current_price, shares, net_income, total_equity, revenue)

    return {
        "ticker": raw.get("stock_code"),
        "name": yf.get("long_name") or info.get("name"),
        "market": "KR",
        "currency": "KRW",
        "current_price": current_price,
        "sector": info.get("industry"),
        "industry": info.get("industry"),
        "valuation": {
            "PER": valuation["PER"],
            "forward_PER": yf.get("forward_PER"),
            "PBR": valuation["PBR"],
            "PSR": valuation["PSR"],
        },
        "profitability": {
            # 비율 계산은 원 단위끼리라 단위 무관
            "ROE": _ratio(net_income, total_equity),
            "ROA": _ratio(net_income, total_assets),
            "operating_margin": _ratio(operating_income, revenue),
            "net_margin": _ratio(net_income, revenue),
            "gross_margin": None,
        },
        "growth": {
            "revenue_yoy": None,
            "earnings_yoy": None,
        },
        "stability": {
            "debt_to_equity": _ratio(total_liabilities, total_equity),
            "current_ratio": _ratio(current_assets, current_liabilities),
            "quick_ratio": None,
        },
        "size": {
            # 표시용으로만 단위 변환 (계산 끝난 후)
            "market_cap_bil_krw": _fmt_bil_krw(valuation["market_cap"]),
            "total_revenue_bil_krw": _fmt_bil_krw(revenue),
            "operating_cashflow_bil_krw": _fmt_bil_krw(accounts.get("operating_cashflow")),
        },
    }


def _extract_accounts(df: pd.DataFrame) -> dict:
    """DART 재무제표에서 최근 연도 주요 계정과목 추출 (원 단위 그대로)"""
    if df.empty:
        return {}

    latest_year = df["year"].max()
    df_latest = df[(df["year"] == latest_year) & (df["report"] == "사업보고서")]

    accounts = {}
    for _, row in df_latest.iterrows():
        account_name = str(row.get("account", "")).strip()
        if account_name in KR_ACCOUNT_MAP:
            key = KR_ACCOUNT_MAP[account_name]
            accounts[key] = row.get("current")

    return accounts


# -----------------------------------------------------------------------
# 유틸 (표시용 단위 변환 - 계산 끝난 후에만 사용)
# -----------------------------------------------------------------------

def _pct(val):
    """소수 → 퍼센트 (0.87 → 87.0%)"""
    if val is None:
        return None
    return round(val * 100, 2)


def _ratio(numerator, denominator):
    """비율 계산"""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _fmt_bil(val):
    """달러 → 십억달러 (표시용)"""
    if val is None:
        return None
    return round(val / 1e9, 2)


def _fmt_bil_krw(val):
    """원 → 조원 (표시용)"""
    if val is None:
        return None
    return round(val / 1e12, 2)


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