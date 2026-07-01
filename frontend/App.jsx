import { useState, useEffect, useCallback } from "react";
import {
  TrendingUp, TrendingDown, RefreshCw, Plus, X, ChevronRight,
  AlertTriangle, CheckCircle2, Circle, Loader2, BarChart3,
  Wallet, Target, Layers
} from "lucide-react";

// -----------------------------------------------------------------------
// API 베이스 URL — Codespaces 등 포트포워딩 환경 대응
// -----------------------------------------------------------------------
const API_BASE = (() => {
  const { protocol, hostname } = window.location;
  if (hostname.includes("-")) {
    // Codespaces 형태: xxxx-3000.app.github.dev → xxxx-8000.app.github.dev
    const swapped = hostname.replace(/-\d+\./, "-8000.");
    return `${protocol}//${swapped}`;
  }
  return "https://scaling-space-guide-x5pqq57x9pvcpv96-8000.app.github.dev";
})();

// -----------------------------------------------------------------------
// 유틸
// -----------------------------------------------------------------------

function formatNumber(n, currency) {
  if (n === null || n === undefined) return "—";
  const fixed = Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(2);
  return Number(fixed).toLocaleString();
}

function formatCurrency(n, currency) {
  if (n === null || n === undefined) return "—";
  const symbol = currency === "USD" ? "$" : currency === "KRW" ? "₩" : "";
  return `${symbol}${formatNumber(n)}`;
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "요청 실패");
  }
  return res.json();
}

// -----------------------------------------------------------------------
// 작은 컴포넌트들
// -----------------------------------------------------------------------

function ProfitBadge({ pct }) {
  if (pct === null || pct === undefined) {
    return <span className="badge badge-neutral">—</span>;
  }
  const positive = pct >= 0;
  return (
    <span className={`badge ${positive ? "badge-up" : "badge-down"}`}>
      {positive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
      {positive ? "+" : ""}{pct.toFixed(2)}%
    </span>
  );
}

function OpinionPill({ opinion }) {
  if (!opinion) return null;
  const map = {
    "강력 매수": "pill-strong-buy", "매수": "pill-buy",
    "보유": "pill-hold", "관망": "pill-watch", "매도": "pill-sell",
    "추가매수": "pill-buy",
  };
  const cls = map[opinion] || "pill-hold";
  return <span className={`pill ${cls}`}>{opinion}</span>;
}

function SectionLabel({ children }) {
  return <div className="section-label">{children}</div>;
}

// -----------------------------------------------------------------------
// 종목 추가 모달
// -----------------------------------------------------------------------

function AddHoldingModal({ onClose, onAdded }) {
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("");
  const [avgPrice, setAvgPrice] = useState("");
  const [market, setMarket] = useState("US");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await api("/portfolio", {
        method: "POST",
        body: JSON.stringify({
          ticker: ticker.trim().toUpperCase(),
          quantity: parseFloat(quantity),
          avg_price: parseFloat(avgPrice),
          market,
        }),
      });
      onAdded();
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>종목 추가</h3>
          <button className="icon-btn" onClick={onClose}><X size={18} /></button>
        </div>
        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-row">
            <label>시장</label>
            <div className="market-toggle">
              <button type="button" className={market === "US" ? "active" : ""} onClick={() => setMarket("US")}>미국</button>
              <button type="button" className={market === "KR" ? "active" : ""} onClick={() => setMarket("KR")}>한국</button>
            </div>
          </div>
          <div className="form-row">
            <label>티커 / 종목코드</label>
            <input value={ticker} onChange={(e) => setTicker(e.target.value)}
              placeholder={market === "US" ? "AAPL" : "005930"} required />
          </div>
          <div className="form-row">
            <label>수량</label>
            <input type="number" step="any" value={quantity} onChange={(e) => setQuantity(e.target.value)}
              placeholder="10" required />
          </div>
          <div className="form-row">
            <label>평균 매입가</label>
            <input type="number" step="any" value={avgPrice} onChange={(e) => setAvgPrice(e.target.value)}
              placeholder={market === "US" ? "180.00" : "62000"} required />
          </div>
          {error && <div className="form-error">{error}</div>}
          <button type="submit" className="btn-primary" disabled={submitting}>
            {submitting ? <Loader2 size={16} className="spin" /> : "추가하기"}
          </button>
        </form>
      </div>
    </div>
  );
}

function AddAnalyzeModal({
  onClose,
  setSelectedStock,
}) {
  const [ticker, setTicker] = useState("");
  const [market, setMarket] = useState("US");

  const handleSubmit = (e) => {
    e.preventDefault();

    const stockTicker = ticker.trim().toUpperCase();

    if (!stockTicker) return;

    setSelectedStock({
      ticker: stockTicker,
      market,
    });

    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>AI 종목 분석</h3>

          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">

          <div className="form-row">
            <label>시장</label>

            <div className="market-toggle">
              <button
                type="button"
                className={market === "US" ? "active" : ""}
                onClick={() => setMarket("US")}
              >
                미국
              </button>

              <button
                type="button"
                className={market === "KR" ? "active" : ""}
                onClick={() => setMarket("KR")}
              >
                한국
              </button>
            </div>
          </div>

          <div className="form-row">
            <label>티커 / 종목코드</label>

            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder={market === "US" ? "AAPL" : "005930"}
              autoFocus
              required
            />
          </div>

          <button
            type="submit"
            className="btn-primary"
          >
            분석하기
          </button>
        </form>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------
// 종목별 분석 상세 패널
// -----------------------------------------------------------------------

function StockAnalysisPanel({ ticker, market, onClose }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api("/analyze/stock", {
      method: "POST",
      body: JSON.stringify({ ticker, market }),
    })
      .then((res) => { if (!cancelled) setData(res); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ticker, market]);

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <div className="drawer-ticker">{ticker}</div>
            <div className="drawer-sub">{market === "US" ? "미국" : "한국"} 증시</div>
          </div>
          <button className="icon-btn" onClick={onClose}><X size={20} /></button>
        </div>

        {loading && (
          <div className="drawer-loading">
            <Loader2 size={28} className="spin" />
            <p>재무 데이터를 분석하는 중입니다...</p>
          </div>
        )}

        {error && (
          <div className="drawer-error">
            <AlertTriangle size={20} />
            <p>{error}</p>
          </div>
        )}

        {data && !loading && (
          <div className="drawer-body">
            <div className="drawer-top-row">
              <div className="drawer-name">{data.data?.name || ticker}</div>
              <OpinionPill opinion={data.analysis?.conclusion?.final_opinion} />
            </div>
            <div className="drawer-price">
              {formatCurrency(data.data?.current_price, data.data?.currency)}
              <span className="drawer-sector">{data.data?.sector}</span>
            </div>

            {data.analysis?.conclusion && (
              <div className="score-grid">
                <div className="score-card">
                  <div className="score-label">투자 매력도</div>
                  <div className="score-value">{data.analysis.conclusion.investment_score ?? "—"}</div>
                </div>
                <div className="score-card">
                  <div className="score-label">성장 점수</div>
                  <div className="score-value">{data.analysis.conclusion.growth_score ?? "—"}</div>
                </div>
                <div className="score-card">
                  <div className="score-label">가치 점수</div>
                  <div className="score-value">{data.analysis.conclusion.value_score ?? "—"}</div>
                </div>
              </div>
            )}

            {data.analysis?.company_type && (
              <div className="tag-row">
                {data.analysis.company_type.map((t, i) => (
                  <span key={i} className="tag">{t}</span>
                ))}
              </div>
            )}

            {data.analysis?.conclusion?.price_evaluation && (
              <div className="callout">
                <Target size={16} />
                <div>
                  <div className="callout-title">현재 주가 평가</div>
                  <div className="callout-text">{data.analysis.conclusion.price_evaluation}
                    {data.analysis.conclusion.fair_value_range && ` · 적정가 범위 ${data.analysis.conclusion.fair_value_range}`}
                  </div>
                </div>
              </div>
            )}

            {data.analysis?.action && (
              <div className="callout callout-action">
                <CheckCircle2 size={16} />
                <div>
                  <div className="callout-title">보유 중 — 추천 액션: {data.analysis.action}</div>
                  <div className="callout-text">{data.analysis.action_reason}</div>
                </div>
              </div>
            )}

            {data.analysis?.conclusion?.strengths && (
              <div className="list-section">
                <SectionLabel>강점</SectionLabel>
                <ul className="bullet-list bullet-up">
                  {data.analysis.conclusion.strengths.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}

            {data.analysis?.conclusion?.key_risks && (
              <div className="list-section">
                <SectionLabel>리스크</SectionLabel>
                <ul className="bullet-list bullet-down">
                  {data.analysis.conclusion.key_risks.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}

            {data.analysis?.conclusion?.five_year_outlook && (
              <div className="list-section">
                <SectionLabel>5년 전망</SectionLabel>
                <p className="prose">{data.analysis.conclusion.five_year_outlook}</p>
              </div>
            )}

            {data.analysis?.caution && (
              <div className="caution-box">
                <AlertTriangle size={14} />
                {data.analysis.caution}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------
// 포트폴리오 진단 패널
// -----------------------------------------------------------------------

function PortfolioDiagnosisPanel({ result }) {
  if (!result) return null;
  const { summary, risk_level, sector_analysis, strengths, risks,
          best_pick, worst_pick, rebalancing, recommendations,
          profile_fit, asset_allocation, caution } = result;

  return (
    <div className="diagnosis-panel">
      <div className="diagnosis-header">
        <Layers size={18} />
        <h2>포트폴리오 진단</h2>
        {risk_level && (
          <span className={`risk-tag risk-${risk_level === "낮음" ? "low" : risk_level === "높음" ? "high" : "mid"}`}>
            리스크 {risk_level}
          </span>
        )}
      </div>

      {summary && <p className="diagnosis-summary">{summary}</p>}

      <div className="diagnosis-grid">
        {best_pick && (
          <div className="pick-card pick-best">
            <div className="pick-label">베스트 픽</div>
            <div className="pick-ticker">{best_pick.ticker}</div>
            <div className="pick-reason">{best_pick.reason}</div>
          </div>
        )}
        {worst_pick && (
          <div className="pick-card pick-worst">
            <div className="pick-label">주의 종목</div>
            <div className="pick-ticker">{worst_pick.ticker}</div>
            <div className="pick-reason">{worst_pick.reason}</div>
          </div>
        )}
      </div>

      {(profile_fit || asset_allocation) && (
        <div className="list-section">
          <SectionLabel>내 상황과의 적합도</SectionLabel>
          {profile_fit && <p className="prose">{profile_fit}</p>}
          {asset_allocation && <p className="prose">{asset_allocation}</p>}
        </div>
      )}

      {sector_analysis && (
        <div className="list-section">
          <SectionLabel>섹터 분석</SectionLabel>
          <p className="prose">{sector_analysis.concentration}</p>
          <p className="prose">{sector_analysis.market_cycle_fit}</p>
          <p className="prose prose-accent">→ {sector_analysis.suggestion}</p>
        </div>
      )}

      <div className="diagnosis-grid-2">
        {strengths && (
          <div className="list-section">
            <SectionLabel>강점</SectionLabel>
            <ul className="bullet-list bullet-up">
              {strengths.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}
        {risks && (
          <div className="list-section">
            <SectionLabel>리스크</SectionLabel>
            <ul className="bullet-list bullet-down">
              {risks.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}
      </div>

      {rebalancing && rebalancing.length > 0 && (
        <div className="list-section">
          <SectionLabel>리밸런싱 제안</SectionLabel>
          <div className="rebalance-list">
            {rebalancing.map((r, i) => (
              <div key={i} className="rebalance-item">
                <span className={`pill pill-${r.action?.includes("매도") || r.action?.includes("축소") ? "sell" : "buy"}`}>
                  {r.action}
                </span>
                <strong>{r.ticker}</strong>
                <span className="rebalance-detail">{r.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {recommendations && recommendations.length > 0 && (
        <div className="list-section">
          <SectionLabel>추천 종목</SectionLabel>
          <div className="rec-list">
            {recommendations.map((r, i) => (
              <div key={i} className="rec-item">
                <strong>{r.ticker}</strong>
                <span className="rec-market">{r.market}</span>
                <span className="rec-reason">{r.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {caution && <div className="caution-box"><AlertTriangle size={14} />{caution}</div>}
    </div>
  );
}

// -----------------------------------------------------------------------
// 메인 앱
// -----------------------------------------------------------------------

export default function App() {
  const [holdings, setHoldings] = useState([]);
  const [profile, setProfile] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showAnalyzeModal, setShowAnalyzeModal] = useState(false);
  const [selectedStock, setSelectedStock] = useState(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [diagnosis, setDiagnosis] = useState(null);

  const loadPortfolio = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api("/portfolio");
      setHoldings(data.holdings || []);
      setProfile(data.profile || {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadPortfolio(); }, [loadPortfolio]);

  const handleDiagnose = async () => {
    setDiagnosing(true);
    try {
      const res = await api("/analyze/portfolio", { method: "POST" });
      setDiagnosis(res.analysis);
    } catch (err) {
      setError(err.message);
    } finally {
      setDiagnosing(false);
    }
  };

  const handleRemove = async (ticker) => {
    await api(`/portfolio/${ticker}`, { method: "DELETE" });
    loadPortfolio();
  };

  const totalValue = holdings.reduce((totals, h) => {
    const currency =
      h.currency ||
      (h.market === "US" ? "USD" : "KRW");

    const value = (h.current_price || 0) * (h.quantity || 0);

    totals[currency] = (totals[currency] || 0) + value;

    return totals;
  }, {});
  const totalCost = holdings.reduce((sum, h) => sum + (h.avg_price || 0) * (h.quantity || 0), 0);
  const totalProfit = totalValue - totalCost;
  const totalProfitPct = totalCost ? (totalProfit / totalCost) * 100 : 0;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <BarChart3 size={22} />
          <span>포트폴리오 분석</span>
        </div>
        <button className="btn-ghost" onClick={loadPortfolio}>
          <RefreshCw size={15} className={loading ? "spin" : ""} />
          새로고침
        </button>
      </header>

      <main className="main-grid">
        {/* 좌측: 포트폴리오 현황 */}
        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <Wallet size={17} />
              보유 현황
            </div>
            <button className="btn-icon-add" onClick={() => setShowAddModal(true)}>
              <Plus size={15} /> 종목 추가
            </button>
          </div>

          <div className="total-summary">
           {Object.entries(totalValue).map(([currency, total]) => (
            <div key={currency} className="total-value">
              {currency}: {formatNumber(total)}
            </div>
          ))}
          </div>

          {error && (
            <div className="inline-error">
              <AlertTriangle size={14} /> {error}
            </div>
          )}

          {loading ? (
            <div className="loading-block"><Loader2 size={24} className="spin" /></div>
          ) : holdings.length === 0 ? (
            <div className="empty-block">
              <Circle size={28} strokeWidth={1.2} />
              <p>보유 종목이 없습니다</p>
              <span>종목을 추가하고 분석을 시작해보세요</span>
            </div>
          ) : (
            <div className="holdings-list">
              {holdings.map((h) => (
                <div key={h.ticker} className="holding-row" onClick={() => setSelectedStock(h)}>
                  <div className="holding-main">
                    <div className="holding-ticker">{h.ticker}</div>
                    <div className="holding-market">{h.market === "US" ? "미국" : "한국"}</div>
                  </div>
                  <div className="holding-mid">
                    <div className="holding-qty">{h.quantity}주 · 평단 {formatNumber(h.avg_price)}</div>
                    <div className="holding-price">{formatCurrency(h.current_price)}</div>
                  </div>
                  <div className="holding-right">
                    <ProfitBadge pct={h.profit_pct} />
                    <button
                      className="icon-btn-sm"
                      onClick={(e) => { e.stopPropagation(); handleRemove(h.ticker); }}
                    >
                      <X size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {profile && Object.keys(profile).length > 0 && (
            <div className="profile-strip">
              <SectionLabel>내 투자 프로필</SectionLabel>
              <div className="profile-tags">
                {profile.age && <span className="profile-tag">{profile.age}세</span>}
                {profile.risk_tolerance && <span className="profile-tag">리스크 허용 {profile.risk_tolerance}</span>}
                {profile.investment_horizon && <span className="profile-tag">{profile.investment_horizon}</span>}
              </div>
            </div>
          )}

          <button
            className="btn-primary btn-diagnose"
            onClick={handleDiagnose}
            disabled={diagnosing || holdings.length === 0}
          >
            {diagnosing ? (
              <><Loader2 size={16} className="spin" /> 포트폴리오 분석 중...</>
            ) : (
              <>AI 포트폴리오 진단 실행 <ChevronRight size={16} /></>
            )}
          </button>
            <button className="btn-icon-add"  style={{ marginTop: "10px" }} onClick={() => setShowAnalyzeModal(true)}>
            <Plus size={15} /> 개별 종목 분석
            </button>
        </section>

        {/* 우측: 분석 결과 */}
        <section className="panel panel-results">
          {diagnosing && (
            <div className="loading-block tall">
              <Loader2 size={32} className="spin" />
              <p>전체 포트폴리오를 분석하는 중입니다...</p>
              <span className="hint">재무 데이터 수집과 AI 진단에 시간이 걸릴 수 있어요</span>
            </div>
          )}

          {!diagnosing && diagnosis && <PortfolioDiagnosisPanel result={diagnosis} />}

          {!diagnosing && !diagnosis && (
            <div className="empty-block tall">
              <Target size={32} strokeWidth={1.2} />
              <p>아직 분석 결과가 없습니다</p>
              <span>좌측에서 진단을 실행하거나 종목을 클릭해 개별 분석을 확인하세요</span>
            </div>
          )}
        </section>
      </main>

      {showAddModal && (
        <AddHoldingModal onClose={() => setShowAddModal(false)} onAdded={loadPortfolio} />
      )}
      {showAnalyzeModal && (
        <AddAnalyzeModal
          onClose={() => setShowAnalyzeModal(false)}
          setSelectedStock={setSelectedStock}
        />
      )}


      {selectedStock && (
        <StockAnalysisPanel
          ticker={selectedStock.ticker}
          market={selectedStock.market}
          onClose={() => setSelectedStock(null)}
        />
      )}
    </div>
  );
}
