import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash")


# -----------------------------------------------------------------------
# 프롬프트 구성
# -----------------------------------------------------------------------

def _build_prompt(data: dict) -> str:
    """preprocess 결과물을 Gemini 프롬프트로 변환"""

    def _fmt(val, suffix=""):
        return f"{val}{suffix}" if val is not None else "데이터 없음"

    v = data.get("valuation", {})
    p = data.get("profitability", {})
    g = data.get("growth", {})
    s = data.get("stability", {})
    sz = data.get("size", {})

    prompt = f"""
너는 전문 주식 애널리스트야. 아래 재무 데이터를 바탕으로 분석 리포트를 작성해줘. 부채와 재무안정성, 현금흐름을 중심으로 해주고 

==============================================
종목: {data.get('ticker')} ({data.get('name', '이름 없음')})
시장: {data.get('market')} | 통화: {data.get('currency')}
섹터: {data.get('sector', '정보 없음')}
현재가: {_fmt(data.get('current_price'))}
==============================================

[밸류에이션]
- PER (주가수익비율): {_fmt(v.get('PER'))}
- Forward PER: {_fmt(v.get('forward_PER'))}
- PBR (주가순자산비율): {_fmt(v.get('PBR'))}
- PSR (주가매출비율): {_fmt(v.get('PSR'))}

[수익성]
- ROE (자기자본이익률): {_fmt(p.get('ROE'), '%')}
- ROA (총자산이익률): {_fmt(p.get('ROA'), '%')}
- 영업이익률: {_fmt(p.get('operating_margin'), '%')}
- 순이익률: {_fmt(p.get('net_margin'), '%')}
- 매출총이익률: {_fmt(p.get('gross_margin'), '%')}

[성장성]
- 매출 YoY 성장률: {_fmt(g.get('revenue_yoy'), '%')}
- 순이익 YoY 성장률: {_fmt(g.get('earnings_yoy'), '%')}

[안정성]
- 부채비율 (부채/자본): {_fmt(s.get('debt_to_equity'))}
- 유동비율: {_fmt(s.get('current_ratio'))}
- 당좌비율: {_fmt(s.get('quick_ratio'))}

[규모]
- 시가총액: {_fmt(sz.get('market_cap'), 'B')}
- 총매출: {_fmt(sz.get('total_revenue'), 'B' if data.get('market') == 'US' else '억원')}
- 잉여현금흐름: {_fmt(sz.get('free_cashflow'), 'B')}
- 영업현금흐름: {_fmt(sz.get('operating_cashflow'), 'B' if data.get('market') == 'US' else '억원')}
==============================================

위 데이터를 바탕으로 아래 형식으로 분석해줘.

1. 종합 평가: 매수 / 중립 / 매도 중 하나로 판단하고 이유 2~3줄
2. 핵심 강점: 3가지 이내
3. 주요 리스크: 3가지 이내
4. 투자 판단 근거: 수치 기반으로 구체적으로
5. 주의사항: 이 분석의 한계나 추가로 확인해야 할 것들

한국어로 작성해줘.
"""
    return prompt.strip()


# -----------------------------------------------------------------------
# Gemini 호출
# -----------------------------------------------------------------------

def analyze(data: dict) -> str:
    """
    preprocess 결과물을 받아 Gemini로 분석 후 리포트 반환

    data: preprocess() 결과 dict
    """
    prompt = _build_prompt(data)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[{data.get('ticker')}] 분석 실패: {e}"