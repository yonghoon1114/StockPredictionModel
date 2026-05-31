import os
import json
from datetime import datetime
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash")


# -----------------------------------------------------------------------
# 프롬프트 구성
# -----------------------------------------------------------------------

def _build_prompt(data: dict) -> str:
    def _fmt(val, suffix=""):
        return f"{val}{suffix}" if val is not None else "데이터 없음"

    v = data.get("valuation", {})
    p = data.get("profitability", {})
    g = data.get("growth", {})
    s = data.get("stability", {})
    sz = data.get("size", {})

    prompt = f"""
너는 피터 린치, 워렌 버핏, 벤저민 그레이엄의 분석 방식을 결합한 투자 분석가다.

목표:
현재 주가가 저평가인지 판단하라.

특히 다음을 중점적으로 분석하라.

- 앞으로 10년 뒤에도 존재할 사업인가
- 경쟁우위가 있는가
- 현재 실적이 일시적 호황인가
- 순이익 증가가 지속 가능한가
- 현금흐름이 순이익을 뒷받침하는가
- 현재 주가가 과열 상태인가

최종적으로:

1. 적정주가 범위
2. 현재 주가 대비 할인율
3. 투자 매력도(0~100)
4. 투자하지 말아야 할 이유
5. 가장 우려되는 위험 요소

를 제시하라.

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
반드시 JSON 형식으로만 응답하고 다른 텍스트는 절대 포함하지 마.

{{
  "overall": "매수 / 중립 / 매도 중 하나",
  "summary": "종합 평가 2~3줄",
  "strengths": ["강점1", "강점2", "강점3"],
  "risks": ["리스크1", "리스크2", "리스크3"],
  "rationale": "수치 기반 투자 판단 근거 구체적으로",
  "caution": "이 분석의 한계나 추가 확인 사항"
}}

한국어로 작성해줘.
"""
    return prompt.strip()


# -----------------------------------------------------------------------
# Gemini 호출
# -----------------------------------------------------------------------

def analyze(data: dict) -> dict:
    """
    preprocess 결과물을 받아 Gemini로 분석 후 결과 dict 반환
    """
    prompt = _build_prompt(data)

    try:
        response = model.generate_content(prompt)
        text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"error": "JSON 파싱 실패", "raw": response.text}
    except Exception as e:
        result = {"error": str(e)}

    return result


# -----------------------------------------------------------------------
# 저장
# -----------------------------------------------------------------------

def save_report(ticker: str, data: dict, result: dict, base_dir: str = "data/reports"):
    """
    분석 결과를 JSON 파일로 저장
    - 종목별로 파일 하나
    - 실행할 때마다 날짜별로 누적 저장
    """
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{ticker}.json")

    # 기존 파일 있으면 불러오기
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # 새 레코드 추가
    record = {
        "date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "market": data.get("market"),
        "current_price": data.get("current_price"),
        "analysis": result
    }
    existing.append(record)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[{ticker}] 분석 결과 저장 완료: {file_path}")