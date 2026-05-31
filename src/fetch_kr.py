import os
import time
import requests
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
from datetime import date, timedelta
from config import DART_API_KEY


# -----------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------

def _get_corp_code(stock_code: str) -> str | None:
    """종목코드 → DART 고유 기업코드 변환"""
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": DART_API_KEY}
    resp = requests.get(url, params=params)

    import zipfile, io
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    for corp in tree.getroot().findall("list"):
        if corp.findtext("stock_code") == stock_code:
            return corp.findtext("corp_code")

    return None


def _safe_int(val):
    try:
        return int(str(val).replace(",", "").strip())
    except:
        return None


# -----------------------------------------------------------------------
# 주가 데이터
# -----------------------------------------------------------------------

def fetch_stock_price_kr(stock_code: str, save_dir: str) -> pd.DataFrame:
    """FinanceDataReader로 한국 주가 수집"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{stock_code}_stock.csv")
    end = date.today().strftime("%Y-%m-%d")

    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, parse_dates=["Date"])
        last_date = existing["Date"].max().date()
        new_start = last_date + timedelta(days=1)

        if pd.Timestamp(new_start) >= pd.Timestamp(end):
            print(f"[{stock_code}] 주가 최신 상태")
            return existing

        print(f"[{stock_code}] 주가 업데이트: {new_start} → {end}")
        new_data = fdr.DataReader(stock_code, start=str(new_start), end=end)

        if new_data.empty:
            print(f"[{stock_code}] 새 주가 데이터 없음")
            return existing

        new_data = new_data.reset_index().rename(columns={"index": "Date"})
        new_data = new_data[["Date", "Close", "High", "Low", "Open", "Volume"]]

        merged = pd.concat([existing, new_data], ignore_index=True)
        merged.drop_duplicates(subset="Date", keep="last", inplace=True)
        merged.sort_values("Date", inplace=True)
        merged.to_csv(file_path, index=False)
        return merged

    else:
        print(f"[{stock_code}] 전체 주가 이력 다운로드")
        data = fdr.DataReader(stock_code, start="2000-01-01", end=end)
        data = data.reset_index().rename(columns={"index": "Date"})
        data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
        data.to_csv(file_path, index=False)
        print(f"[{stock_code}] 주가 저장 완료: {file_path}")
        return data


# -----------------------------------------------------------------------
# 재무제표 (DART)
# -----------------------------------------------------------------------

def fetch_financial_statements(corp_code: str, stock_code: str, save_dir: str) -> pd.DataFrame:
    """
    DART 단일회사 전체 재무제표 수집
    - 최근 4년치 연간 + 반기 데이터
    - 연결재무제표 우선, 없으면 개별재무제표
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{stock_code}_financials.csv")

    all_records = []
    current_year = date.today().year

    for year in range(current_year - 4, current_year):
        for report_code, report_name in [("11011", "사업보고서"), ("11012", "반기보고서")]:
            url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
            params = {
                "crtfc_key": DART_API_KEY,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
                "fs_div": "CFS"  # 연결재무제표 우선
            }

            try:
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()

                if data.get("status") != "000":
                    # 연결재무제표 없으면 개별재무제표 시도
                    params["fs_div"] = "OFS"
                    resp = requests.get(url, params=params, timeout=10)
                    data = resp.json()

                if data.get("status") == "000":
                    for item in data.get("list", []):
                        all_records.append({
                            "year": year,
                            "report": report_name,
                            "account": item.get("account_nm"),
                            "fs_type": item.get("fs_nm"),
                            "current": _safe_int(item.get("thstrm_amount")),
                            "previous": _safe_int(item.get("frmtrm_amount")),
                        })
                    print(f"[{stock_code}] {year} {report_name} 수집 완료")
                else:
                    print(f"[{stock_code}] {year} {report_name} 없음: {data.get('message')}")

                time.sleep(0.3)  # API 호출 제한 방지

            except Exception as e:
                print(f"[{stock_code}] {year} {report_name} 오류: {e}")

    if not all_records:
        print(f"[{stock_code}] 재무제표 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.to_csv(file_path, index=False)
    print(f"[{stock_code}] 재무제표 저장 완료: {file_path}")
    return df


def fetch_info_kr(corp_code: str, stock_code: str, save_dir: str) -> dict:
    """DART 기업 개황 수집"""
    url = "https://opendart.fss.or.kr/api/company.json"
    params = {"crtfc_key": DART_API_KEY, "corp_code": corp_code}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") == "000":
            info = {
                "name": data.get("corp_name"),
                "industry": data.get("induty_code"),
                "ceo": data.get("ceo_nm"),
                "stock_code": stock_code,
                "listing_date": data.get("list_date"),
                "capital": data.get("cap_stock"),
            }
            info_df = pd.DataFrame([info])
            path = os.path.join(save_dir, f"{stock_code}_info.csv")
            info_df.to_csv(path, index=False)
            print(f"[{stock_code}] 기업 개황 저장")
            return info
        else:
            print(f"[{stock_code}] 기업 개황 실패: {data.get('message')}")
            return {}

    except Exception as e:
        print(f"[{stock_code}] 기업 개황 오류: {e}")
        return {}


# -----------------------------------------------------------------------
# yfinance 보완 (시가총액, 밸류에이션)
# -----------------------------------------------------------------------

def fetch_yf_supplement(stock_code: str, save_dir: str) -> dict:
    """
    yfinance로 한국 주식 보완 데이터 수집
    - DART에 없는 시가총액, PER, PBR 등
    - 종목코드 뒤에 .KS (코스피) 또는 .KQ (코스닥) 붙여야 함
    - 실패해도 괜찮음 (DART 데이터로 대체)
    """
    suffixes = [".KS", ".KQ"]

    for suffix in suffixes:
        try:
            ticker = f"{stock_code}{suffix}"
            info = yf.Ticker(ticker).info

            # 유효한 데이터인지 확인 (시가총액이 있으면 유효)
            if not info.get("marketCap"):
                continue

            supplement = {
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "PER": info.get("trailingPE"),
                "forward_PER": info.get("forwardPE"),
                "PBR": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "long_name": info.get("longName"),
            }

            path = os.path.join(save_dir, f"{stock_code}_yf_supplement.csv")
            pd.DataFrame([supplement]).to_csv(path, index=False)
            print(f"[{stock_code}] yfinance 보완 데이터 저장 ({ticker})")
            return supplement

        except Exception as e:
            print(f"[{stock_code}] yfinance {suffix} 실패: {e}")

    print(f"[{stock_code}] yfinance 보완 데이터 없음")
    return {}


# -----------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------

def fetch_kr(stock_code: str, base_dir: str = "data/raw/kr") -> dict:
    """
    한국 종목 전체 데이터 수집 진입점

    stock_code: 6자리 종목코드 (예: "005930" 삼성전자)
    """
    save_dir = os.path.join(base_dir, stock_code)

    # 1. 주가
    stock_df = fetch_stock_price_kr(stock_code, save_dir)

    # 2. DART 기업코드 조회
    print(f"[{stock_code}] DART 기업코드 조회 중...")
    corp_code = _get_corp_code(stock_code)

    if corp_code is None:
        print(f"[{stock_code}] DART 기업코드를 찾을 수 없음")
        return {"stock_code": stock_code, "stock": stock_df}

    # 3. 재무제표
    financials_df = fetch_financial_statements(corp_code, stock_code, save_dir)

    # 4. 기업 개황
    info = fetch_info_kr(corp_code, stock_code, save_dir)

    # 5. yfinance 보완 (시가총액, PER, PBR 등)
    yf_supplement = fetch_yf_supplement(stock_code, save_dir)

    return {
        "stock_code": stock_code,
        "stock": stock_df,
        "financials": financials_df,
        "info": info,
        "yf_supplement": yf_supplement
    }


if __name__ == "__main__":
    # 삼성전자, SK하이닉스
    stocks = ["005930", "000660"]
    for code in stocks:
        fetch_kr(code)