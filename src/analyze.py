import os
import json
from datetime import datetime
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")


# -----------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------

def _fmt(val, suffix=""):
    return f"{val}{suffix}" if val is not None else "데이터 없음"


def _build_portfolio_context(portfolio: dict, current_price: float, currency: str) -> str:
    if not portfolio:
        return ""

    avg_price = portfolio.get("avg_price")
    quantity = portfolio.get("quantity")

    if avg_price and current_price:
        profit_pct = (current_price - avg_price) / avg_price * 100
        profit_amt = (current_price - avg_price) * quantity
        profit_str = f"{round(profit_pct, 2)}% ({'+' if profit_amt >= 0 else ''}{round(profit_amt, 2)} {currency})"
    else:
        profit_str = "데이터 없음"

    return f"""
[내 보유 현황]
- 보유 수량: {_fmt(quantity)}주
- 평균 매입가: {_fmt(avg_price)} {currency}
- 현재가: {_fmt(current_price)} {currency}
- 수익률: {profit_str}
"""


# -----------------------------------------------------------------------
# 종목 분석 프롬프트
# -----------------------------------------------------------------------

def _build_stock_prompt(data: dict, portfolio: dict = None) -> str:
    v = data.get("valuation", {})
    p = data.get("profitability", {})
    g = data.get("growth", {})
    s = data.get("stability", {})
    sz = data.get("size", {})
    currency = data.get("currency", "")

    portfolio_context = _build_portfolio_context(portfolio, data.get("current_price"), currency)
    has_portfolio = bool(portfolio)

    prompt = f"""
너는 피터 린치, 필립 피셔 (Philip Fisher) 투자 철학을 결합한 기관투자자 수준의 애널리스트다.

목표:
현재 주가가 저평가, 적정가, 고평가 상태인지 판단하라.

중요:
분석을 시작하기 전에 반드시 이 기업의 유형을 먼저 분류하라.

가능한 분류:
- 초고성장주
- 성장주
- 대형 우량주
- 가치주
- 배당주
- 경기순환주
- 턴어라운드주
- 자산주

복수 선택 가능하다.

분류 후에는 해당 기업 유형에 적합한 기준으로 평가하라.

예시:
- 성장주는 성장률과 시장 확대 가능성을 중요하게 평가
- 가치주는 자산가치와 현금흐름을 중요하게 평가
- 배당주는 배당 지속 가능성을 중요하게 평가
- 경기순환주는 현재 실적보다 산업 사이클을 중요하게 평가
- 우량주는 안정성과 경쟁우위를 중요하게 평가

모든 기업을 동일한 기준으로 평가하지 말 것.

================================
종목: {data.get('ticker')} ({data.get('name', '이름 없음')})
시장: {data.get('market')} | 통화: {currency}
섹터: {data.get('sector', '정보 없음')}
현재가: {_fmt(data.get('current_price'))} {currency}

[밸류에이션]
PER: {_fmt(v.get('PER'))} | Forward PER: {_fmt(v.get('forward_PER'))} | PBR: {_fmt(v.get('PBR'))} | 시가총액: {_fmt(v.get('market_cap'))} {currency}

[수익성] (%)
ROE: {_fmt(p.get('ROE'))} | ROA: {_fmt(p.get('ROA'))} | 영업이익률: {_fmt(p.get('operating_margin'))} | 순이익률: {_fmt(p.get('net_margin'))} | 매출총이익률: {_fmt(p.get('gross_margin'))}

[성장성] (%)
매출YoY: {_fmt(g.get('revenue_yoy'))} | 순이익YoY: {_fmt(g.get('earnings_yoy'))}

[안정성]
부채비율: {_fmt(s.get('debt_to_equity'))} | 유동비율: {_fmt(s.get('current_ratio'))} | 당좌비율: {_fmt(s.get('quick_ratio'))}

[규모]
총매출: {_fmt(sz.get('total_revenue'))} {currency} | 총부채: {_fmt(sz.get('total_debt'))} {currency} | FCF: {_fmt(sz.get('free_cashflow'))} {currency}
{portfolio_context}
================================

[1. 기업 유형 분류]

다음 중 어떤 유형에 해당하는지 판단하라.

- 초고성장주
- 성장주
- 대형 우량주
- 가치주
- 배당주
- 경기순환주
- 턴어라운드주
- 자산주

복수 선택 가능.

분류 근거를 설명하라.

================================

[2. 사업 분석]

- 이 사업이 10년 뒤에도 존재할 가능성
- 산업 성장성을 높게 평가
- 시장 규모
- 사업 모델 이해
- 고객 의존도
- 특정 제품 의존도

평가:
강함 / 보통 / 약함

================================

[3. 경쟁우위 분석]

다음을 평가하라.

- 브랜드 파워
- 기술력
- 특허
- 규모의 경제
- 네트워크 효과
- 전환비용(Switching Cost)
- 시장점유율
- 진입장벽

각 항목을
강함 / 보통 / 약함
으로 평가하라.

경쟁사가 쉽게 따라올 수 있는지도 평가하라.

================================

[4. 성장성 분석]

- 매출 성장률
- EPS 성장률
- 영업이익 성장률
- 성장 지속 가능성
- 시장 확대 가능성

현재 성장이

- 구조적 성장
- 일시적 성장
- 경기순환 효과

중 어디에 가까운지 판단하라.

================================

[5. 수익성 분석]

- 영업이익률
- 순이익률
- ROE
- ROIC

수익성이 업계 평균 대비 높은지 평가하라.

================================

[6. 재무 건전성 분석]

- 부채비율
- 유동비율
- 이자보상배율
- 현금 보유 수준

재무 위험도를 평가하라.

================================

[7. 현금흐름 분석]

- 영업현금흐름
- 잉여현금흐름(FCF)
- 순이익 대비 현금흐름

순이익이 실제 현금으로 이어지고 있는지 평가하라.

================================

[8. 산업 사이클 분석]

다음을 평가하라.

- 현재 산업 위치 (호황 / 중립 / 불황)
- 공급 증가 속도
- 경쟁사 투자(CAPEX) 동향
- 가격 경쟁 상황
- 수요 성장률

현재 실적이

1) 구조적 성장
2) 일시적 호황
3) 경기순환 효과

중 어디에 가까운지 판단하라.

================================

[9. 밸류에이션 분석]

- PER
- Forward PER
- PBR
- PSR
- EV/EBITDA

과거 평균 및 동종업계 대비 평가하라.

현재 시장이 지나치게 낙관적인지 또는 비관적인지 판단하라.

================================

[10. 리스크 분석]

다음 위험요인을 평가하라.

- 산업 위험
- 기술 변화 위험
- 경쟁 심화 위험
- 규제 위험
- 경영진 위험
- 고객 집중 위험

가장 큰 위험요인 3개를 선정하라.

================================

[11. 투자 대가 관점]

피터 린치 관점:
- 투자 여부
- 이유

Philip Fisher 관점:
- 투자 여부
- 이유


================================

[12. 성장주 투자자 관점]

다음을 평가하라.

- TAM(Total Addressable Market)
- 시장 확대 가능성
- 매출 성장 지속 가능성
- EPS 성장 지속 가능성
- 혁신성
- 기술 우위
- 산업 지배력 확대 가능성
- 네트워크 효과
- 플랫폼 효과
- 향후 5~10년 성장 여력

판단하라.

- 현재 고성장 기업인가
- 향후 초고성장 기업이 될 가능성이 있는가
- 현재 주가가 미래 성장을 얼마나 반영하고 있는가
- 현재 밸류에이션이 성장성을 고려할 때 정당화되는가

성장주 투자자 매력도:

- 매우 높음
- 높음
- 보통
- 낮음
- 매우 낮음

중 하나를 선택하라.

================================

[13. 최종 결론]

1. 기업 유형 분류
2. 산업 사이클 위치
3. 적정주가 범위
4. 현재 주가 대비 할인율
5. 투자 매력도 (0~100)
6. 성장주 투자 매력도 (0~100)
7. 가치투자 매력도 (0~100)
8. 가장 큰 강점 3개
9. 가장 큰 위험요소 3개
10. 투자하지 말아야 할 이유
11. 향후 5년 전망
12. 현재 주가 평가

- 매우 저평가
- 저평가
- 적정가
- 고평가
- 매우 고평가

13. 어떤 투자자에게 적합한가

- 성장투자자
- 가치투자자
- 배당투자자
- 경기순환 투자자
- 장기 보유 투자자

14. 최종 투자 의견

- 강력 매수
- 매수
- 보유
- 관망
- 매도

반드시 수치와 근거를 사용하여 설명하라.
감정적 표현 대신 데이터 기반으로 판단하라.
현재 실적이 일시적인지 지속 가능한지 반드시 구분하라.
성장주와 가치주의 평가 기준을 혼동하지 말 것.
기업 유형에 맞는 기준으로 평가할 것.
이상치가 발견된다면 이유를 찾아볼 것. 

{f'보유 중인 종목이므로 위 보유 현황을 고려하여 추가매수 / 보유 / 매도 중 하나의 action도 JSON에 포함하라.' if has_portfolio else ''}

아래 JSON 형식으로만 응답해줘. 다른 텍스트 절대 포함하지 마.

{{
  "company_type": ["기업 유형1", "기업 유형2"],
  "company_type_reason": "분류 근거",
  "business_analysis": "사업 분석 요약",
  "competitive_advantage": "경쟁우위 분석 요약",
  "growth_analysis": "성장성 분석 요약",
  "profitability_analysis": "수익성 분석 요약",
  "financial_health": "재무 건전성 분석 요약",
  "cashflow_analysis": "현금흐름 분석 요약",
  "industry_cycle": "산업 사이클 분석 요약",
  "valuation_analysis": "밸류에이션 분석 요약",
  "risks": ["리스크1", "리스크2", "리스크3"],
  "guru_views": {{
    "buffett": {{"invest": "예/아니오", "reason": "이유"}},
    "lynch": {{"invest": "예/아니오", "reason": "이유"}},
    "graham": {{"invest": "예/아니오", "reason": "이유"}}
  }},
  "growth_investor_score": "매우 높음 / 높음 / 보통 / 낮음 / 매우 낮음",
  "conclusion": {{
    "company_type": ["유형1"],
    "industry_cycle_position": "호황 / 중립 / 불황",
    "fair_value_range": "적정주가 범위",
    "discount_rate": "현재 주가 대비 할인율",
    "investment_score": 0,
    "growth_score": 0,
    "value_score": 0,
    "strengths": ["강점1", "강점2", "강점3"],
    "key_risks": ["위험1", "위험2", "위험3"],
    "reason_not_to_invest": "투자하지 말아야 할 이유",
    "five_year_outlook": "향후 5년 전망",
    "price_evaluation": "매우 저평가 / 저평가 / 적정가 / 고평가 / 매우 고평가",
    "suitable_investor": ["성장투자자", "가치투자자"],
    "final_opinion": "강력 매수 / 매수 / 보유 / 관망 / 매도"
  }},
  {'"action": "추가매수 / 보유 / 매도 중 하나",' if has_portfolio else ''}
  {'"action_reason": "보유 현황 고려한 판단 근거"' if has_portfolio else '"caution": "이 분석의 한계 및 추가 확인 사항"'}
}}

한국어로 작성해줘.
"""
    return prompt.strip()


# -----------------------------------------------------------------------
# 포트폴리오 전체 진단 프롬프트
# -----------------------------------------------------------------------

def _build_portfolio_prompt(portfolio_summary: list, financial_data: dict, profile: dict = {}) -> str:
    items = ""
    total_value = 0
    sector_map = {}

    for row in portfolio_summary:
        ticker = row["ticker"]
        fin = financial_data.get(ticker, {})
        v = fin.get("valuation", {})
        p = fin.get("profitability", {})
        g = fin.get("growth", {})
        s = fin.get("stability", {})
        currency = fin.get("currency", "")
        sector = fin.get("sector") or "기타"

        current_price = row.get("current_price") or 0
        quantity = row.get("quantity") or 0
        avg_price = row.get("avg_price") or 0
        current_value = current_price * quantity
        cost_value = avg_price * quantity
        total_value += current_value

        sector_map[sector] = sector_map.get(sector, 0) + current_value

        profit_pct = row.get("profit_pct")
        profit = row.get("profit")
        profit_str = f"{'+' if profit_pct and profit_pct >= 0 else ''}{round(profit_pct, 2)}% ({'+' if profit and profit >= 0 else ''}{round(profit, 2)} {currency})" if profit_pct is not None else "데이터 없음"

        items += f"""
▶ {ticker} ({fin.get('name', '이름 없음')}) | {row['market']} | {currency}
  섹터: {sector}
  보유: {quantity}주 / 평단가: {avg_price} {currency} / 현재가: {current_price} {currency}
  평가금액: {round(current_value, 2)} {currency} / 매입금액: {round(cost_value, 2)} {currency}
  수익률: {profit_str}
  [밸류에이션] PER: {_fmt(v.get('PER'))} | PBR: {_fmt(v.get('PBR'))} | 시가총액: {_fmt(v.get('market_cap'))} {currency}
  [수익성] ROE: {_fmt(p.get('ROE'))}% | 영업이익률: {_fmt(p.get('operating_margin'))}% | 순이익률: {_fmt(p.get('net_margin'))}%
  [성장성] 매출YoY: {_fmt(g.get('revenue_yoy'))}% | 순이익YoY: {_fmt(g.get('earnings_yoy'))}%
  [안정성] 부채비율: {_fmt(s.get('debt_to_equity'))} | 유동비율: {_fmt(s.get('current_ratio'))}
"""

    sector_breakdown = ""
    for sector, value in sorted(sector_map.items(), key=lambda x: -x[1]):
        pct = round(value / total_value * 100, 1) if total_value else 0
        sector_breakdown += f"  - {sector}: {pct}%\n"

    # 투자자 프로필 섹션
    profile_section = ""
    if profile:
        cash = profile.get('cash')
        savings = profile.get('savings')
        total_assets = total_value + (cash or 0) + (savings or 0)
        stock_ratio = round(total_value / total_assets * 100, 1) if total_assets else None

        profile_section = f"""
[투자자 프로필]
- 나이: {profile.get('age', '미입력')}세
- 소득 상황: {profile.get('income', '미입력')}
- 월 투자 가능액: {profile.get('monthly_investment', '미입력')}
- 리스크 허용도: {profile.get('risk_tolerance', '미입력')}
- 투자 기간: {profile.get('investment_horizon', '미입력')}
- 현금: {f"{cash:,}원" if cash else '미입력'}
- 적금/예금: {f"{savings:,}원" if savings else '미입력'}
- 총 자산 대비 주식 비중: {f"{stock_ratio}%" if stock_ratio else '계산 불가'}
- 기타: {profile.get('notes', '없음')}
"""

    prompt = f"""
너는 20년 경력의 전문 포트폴리오 매니저야.
아래 투자자 프로필과 포트폴리오를 재무 데이터, 섹터 비중, 시장 상황을 종합적으로 고려해서 깊이 있게 진단해줘.
{profile_section}
==============================================
[섹터별 비중]
{sector_breakdown}
[보유 종목 상세 현황]
{items}
==============================================

분석 시 아래 사항을 반드시 고려해줘.

1. 현 포트폴리오 분석 및 프로필을 통해서 투자자 성향을 분석한 후에 나이, 소득, 리스크 허용도에 맞는 포트폴리오인지
2. 자산 배분
3. 섹터 편중: 특정 섹터 집중 여부와 분산 필요성
4. 포트폴리오의 성장 가능성
5. 종목 간 상관관계: 비슷한 방향으로 움직이는 종목이 묶여있는지
6. 리밸런싱: 구체적으로 몇 주를 팔고 어떤 방향으로 조정할지

아래 JSON 형식으로만 응답해줘. 다른 텍스트 절대 포함하지 마.

{{
  "summary": "포트폴리오 전체 종합 평가 4~5줄 (강점과 약점 균형있게)",
  "risk_level": "낮음 / 보통 / 높음 중 하나",
  "profile_fit": "투자자 나이/성향/소득 대비 현재 포트폴리오가 적절한지 평가",
  "asset_allocation": "현금/적금 포함 전체 자산 배분 적절성 평가 및 제안",
  "sector_analysis": {{
    "concentration": "섹터 편중 여부 및 문제점",
    "market_cycle_fit": "현재 경기/금리 환경에서 이 섹터 비중이 적절한지",
    "suggestion": "섹터 비중 조정 방향"
  }},
  "correlation_risk": "종목 간 높은 상관관계로 인한 동반 하락 리스크 분석",
  "strengths": ["포트폴리오 강점1", "강점2", "강점3"],
  "risks": ["포트폴리오 리스크1", "리스크2", "리스크3"],
  "best_pick": {{
    "ticker": "가장 매력적인 종목 티커",
    "reason": "재무 수치 기반 구체적 이유"
  }},
  "worst_pick": {{
    "ticker": "가장 우려되는 종목 티커",
    "reason": "재무 수치 기반 구체적 이유"
  }},
  "rebalancing": [
    {{
      "action": "매도 / 매수 / 비중축소 / 비중확대 중 하나",
      "ticker": "종목 티커",
      "detail": "구체적으로 몇 주를 어떻게 조정할지, 그 이유"
    }}
  ],
  "recommendations": [
    {{
      "ticker": "추천 종목 티커",
      "market": "US 또는 KR",
      "reason": "추천 이유 (포트폴리오 보완 관점에서)"
    }}
  ],
  "caution": "이 분석의 한계, 실제 투자 전 추가 확인 사항"
}}

한국어로 작성해줘.
"""
    return prompt.strip()


# -----------------------------------------------------------------------
# Gemini 호출
# -----------------------------------------------------------------------

def _call_gemini(prompt: str) -> dict:
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON 파싱 실패", "raw": response.text}
    except Exception as e:
        return {"error": str(e)}


def analyze_stock(data: dict, portfolio: dict = None) -> dict:
    """종목 재무 분석 (포트폴리오 있으면 보유 맥락 추가)"""
    prompt = _build_stock_prompt(data, portfolio)
    return _call_gemini(prompt)


def analyze_portfolio(portfolio_summary: list, financial_data: dict, profile: dict = {}) -> dict:
    """전체 포트폴리오 종합 진단 (재무데이터 + 투자자 프로필 포함)"""
    prompt = _build_portfolio_prompt(portfolio_summary, financial_data, profile)
    return _call_gemini(prompt)


# -----------------------------------------------------------------------
# 저장
# -----------------------------------------------------------------------

def save_report(ticker: str, data: dict, result: dict, base_dir: str = "data/reports"):
    """분석 결과를 최신 1개만 저장 (덮어쓰기)"""
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{ticker}.json")

    record = {
        "date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "market": data.get("market"),
        "current_price": data.get("current_price"),
        "analysis": result
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[{ticker}] 분석 결과 저장: {file_path}")