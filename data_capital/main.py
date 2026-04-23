"""
DATA CAPITAL — main.py (Robust Automation Version)
==========================================
"기계만 켜두면 돌아가는 자율 주행 트레이더"
"""

import time as time_module
import json
import logging
import traceback
from datetime import datetime, time, timedelta, timezone
UTC = timezone.utc
from typing import Optional
import pandas as pd
import sys
import os

# .env 파일 자동 로딩 (python-dotenv 없이 직접 파싱)
def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from core.harness    import MarketData, MarketState, BuyFilters, SellRules, Position
from agents          import create_all_agents
from meta_agents     import CIO, Guardian, Oracle
from failure_db_backtest import FailureLearningDB

# ════════════════════════════════════════════
#  로깅 시스템 (Console + File 동시 출력)
# ════════════════════════════════════════════
class Logger:
    def __init__(self, filename="trading.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 즉시 파일에 기록

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 로그 파일 위치 설정 및 리다이렉션 (stderr는 터미널만 — 경고 로그 오염 방지)
LOG_FILE = "trading.log"
sys.stdout = Logger(LOG_FILE)

# 표준 logging을 파일+콘솔 양쪽에 연결
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

# ════════════════════════════════════════════
#  Firebase 연동 (생략 가능, 오프라인 모드 지원)
# ════════════════════════════════════════════
class FirebaseSync:
    def __init__(self, service_account_path: str = "serviceAccount.json"):
        self.db = None
        self._init_firebase(service_account_path)

    def _init_firebase(self, path: str):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred_json = os.environ.get("FIREBASE_CREDENTIALS", "").strip()
                if cred_json:
                    cred = credentials.Certificate(json.loads(cred_json))
                elif os.path.exists(path):
                    cred = credentials.Certificate(path)
                else:
                    print("[Firebase] 자격증명 없음 — 오프라인 모드")
                    return
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("[Firebase] 연결 성공")
        except Exception as e:
            print(f"[Firebase] 연결 실패: {e}")

    def _safe_write(self, collection: str, doc_id: Optional[str], data: dict):
        if self.db is None:
            return
        try:
            from firebase_admin import firestore
            data["synced_at"] = firestore.SERVER_TIMESTAMP
            if doc_id:
                self.db.collection(collection).document(doc_id).set(data)
            else:
                self.db.collection(collection).add(data)
        except Exception as e:
            print(f"[Firebase] 쓰기 실패 ({collection}): {e}")

    def save_trade(self, trade: dict): self._safe_write("trades", None, trade)
    def update_portfolio(self, portfolio: dict): self._safe_write("portfolio", "live", portfolio)
    def save_alert(self, message: str, severity: str = "INFO"):
        print(f"[{severity}] {message}")
        self._safe_write("alerts", None, {"message": message, "severity": severity, "timestamp": datetime.now().isoformat()})

# ════════════════════════════════════════════
#  DATA CAPITAL 메인 엔진 (완전 자동화 버전)
# ════════════════════════════════════════════
class DataCapital:
    TOTAL_CAPITAL  = 3_000_000   
    DAILY_MDD_HALT = -0.01       

    def __init__(self, service_account_path: str = "serviceAccount.json"):
        print("\n" + "═"*55)
        print("  DATA CAPITAL v3.6 [LOGGING ENABLED]")
        print(f"  로그 파일: {os.path.abspath(LOG_FILE)}")
        print("  '변화가 곧 안정이다 - 시스템 가동'")
        print("═"*55 + "\n")

        self.agents     = create_all_agents(self.TOTAL_CAPITAL)
        self.cio        = CIO(self.TOTAL_CAPITAL)
        self.guardian   = Guardian()
        self.oracle     = Oracle()
        self.failure_db = FailureLearningDB()
        self.firebase   = FirebaseSync(service_account_path)

        self.open_positions:    dict  = {}
        self.daily_pnl_pct:     float = 0.0
        self.is_halted:         bool  = False
        # 누적 자본 추적 (Guardian MDD 계산용)
        self.current_capital:   float = float(self.TOTAL_CAPITAL)
        self.peak_capital:      float = float(self.TOTAL_CAPITAL)
        self.last_api_latency:  float = 0.0   # ms

    def execute_order(self, agent_id: str, ticker: str, signal, amount: float):
        """BUY 주문 실행. TP/SL은 에이전트의 Signal에서 가져온다 (하드코딩 금지)."""
        kst_now = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=9)
        price = signal.entry_price

        print(
            f"\n>>> [ORDER] {kst_now.strftime('%H:%M:%S')} | {agent_id} | 매수 | "
            f"{price:,.0f}원 | {amount:,.0f}원치 | "
            f"TP:{signal.target_price:,.0f} SL:{signal.stop_price:,.0f}"
        )

        shares = int(amount / price)
        if shares > 0:
            self.open_positions[agent_id] = {
                "agent_id":    agent_id,
                "ticker":      ticker,
                "entry_price": price,
                "entry_time":  kst_now,
                "shares":      shares,
                "position_obj": Position(
                    agent_id=agent_id, ticker=ticker, entry_price=price,
                    entry_time=kst_now, shares=shares,
                    target_price=signal.target_price,
                    stop_price=signal.stop_price,
                ),
            }
            self.firebase.save_trade({"side": "BUY", "agent": agent_id, "price": price, "shares": shares})

    def run_iteration(self, market_data_list: list):
        if not market_data_list:
            return
        md = market_data_list[0]
        kst_now = md.current_time

        # ── 포지션 청산 체크 ────────────────────────────
        for agent_id, pos in list(self.open_positions.items()):
            res = SellRules.check(pos["position_obj"], md.close, kst_now)
            if res:
                trade_pnl_pct = (md.close - pos["entry_price"]) / pos["entry_price"]
                trade_amt     = pos["position_obj"].shares * pos["entry_price"]
                portfolio_pnl = trade_pnl_pct * (trade_amt / self.TOTAL_CAPITAL)
                self.daily_pnl_pct  += portfolio_pnl
                self.current_capital += trade_amt * trade_pnl_pct
                self.peak_capital    = max(self.peak_capital, self.current_capital)
                pnl_won = trade_pnl_pct * trade_amt
                print(f"\n<<< [EXIT] {agent_id} | {res} | {trade_pnl_pct:+.2%} | {pnl_won:+,.0f}원 | 일일PnL: {self.daily_pnl_pct:+.3%}")
                del self.open_positions[agent_id]
                self.firebase.save_trade({
                    "side": "SELL", "agent": agent_id, "price": md.close,
                    "pnl": trade_pnl_pct, "portfolio_pnl": portfolio_pnl,
                })

        # ── Guardian 시스템 감시 ────────────────────────
        mdd_pct = (self.current_capital - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        data_ok = (md.close > 0 and md.volume > 0)
        health  = self.guardian.health_check(
            daily_pnl_pct   = self.daily_pnl_pct,
            mdd_pct         = mdd_pct,
            api_latency_ms  = self.last_api_latency,
            data_quality_ok = data_ok,
            vkospi          = md.vkospi,
            avg_correlation = 0.3,
        )
        if health["halt"] and not self.is_halted:
            self.is_halted = True
            for alert in health["alerts"]:
                print(f"\n[GUARDIAN] {alert}")
            print("[GUARDIAN] 거래 중단 선언 — 청산 대기")
        elif health["alerts"]:
            for alert in health["alerts"]:
                print(f"\n[GUARDIAN] {alert}")

        if self.is_halted or not (time(9, 5) <= kst_now.time() <= time(15, 0)):
            return

        # ── 에이전트 신호 수집 ────────────────────────
        all_signals: dict = {}
        buy_signals:  dict = {}
        for aid, agent in self.agents.items():
            if aid not in self.open_positions:
                sig = agent.generate_signal(md, daily_pnl_pct=self.daily_pnl_pct)
                all_signals[aid] = sig
                if sig.signal == "BUY":
                    buy_signals[aid] = sig

        # ── Guardian 충돌 감지 ────────────────────────
        if all_signals:
            conflicts = self.guardian.check_conflict(all_signals)
            for c in conflicts:
                print(f"  [CONFLICT] BUY:{c['buy']} vs SELL:{c['sell']} → {c['resolution']}")

        if not buy_signals:
            return

        # ── Oracle 합의 → CIO 배분 → 주문 ───────────
        consensus = self.oracle.form_consensus(all_signals)
        print(f"  [Oracle] {consensus.decision} | {consensus.reasoning[:70]}")
        if consensus.decision in ("BUY", "SPLIT"):
            split_factor = 0.4 if consensus.decision == "SPLIT" else 1.0
            allocations, vetos = self.cio.allocate(self.agents, buy_signals)
            if vetos:
                print(f"  [VETO] {vetos}")
            for aid, amt in allocations.items():
                adj_amt = amt * split_factor
                if adj_amt > 100_000:
                    self.execute_order(aid, md.ticker, buy_signals[aid], adj_amt)

# ════════════════════════════════════════════
#  완전 자동화 실행부
# ════════════════════════════════════════════
def fetch_and_calculate(ticker="069500"):
    from pykrx import stock
    kst_now = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=9)
    today = kst_now.strftime("%Y%m%d")
    start = (kst_now - timedelta(days=60)).strftime("%Y%m%d")  # 지표 계산용 60일

    try:
        df = stock.get_market_ohlcv(start, today, ticker)
        if df.empty or len(df) < 21: return None

        close  = df['종가']
        volume = df['거래량']

        ma20    = close.rolling(20).mean().iloc[-1]
        bb_std  = close.rolling(20).std().iloc[-1]
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        vol_ma5 = float(volume.rolling(5).mean().iloc[-1])
        rsi     = calculate_rsi(close).iloc[-1]

        # KOSPI 등락률 실시간 조회 (실패 시 기본값 사용)
        kospi_change = 0.0
        try:
            # pykrx 내부 출력 억제
            import contextlib
            with contextlib.redirect_stderr(None), contextlib.redirect_stdout(None):
                kdf = stock.get_index_ohlcv(start, today, "1001")  # KOSPI
            if not kdf.empty and len(kdf) >= 2:
                kospi_change = float(kdf['등락률'].iloc[-1])
        except Exception:
            kospi_change = 0.0  # 실패 시 0.0 유지

        md = MarketData(
            ticker=ticker, current_time=kst_now,
            open=float(df['시가'].iloc[-1]), high=float(df['고가'].iloc[-1]),
            low=float(df['저가'].iloc[-1]), close=float(close.iloc[-1]),
            volume=int(volume.iloc[-1]), prev_close=float(close.iloc[-2]),
            ma20=ma20, rsi14=rsi,
            bb_upper=bb_upper, bb_middle=ma20, bb_lower=bb_lower,
            vol_ma5=vol_ma5,
            vkospi=20.0, kospi_change=kospi_change,
        )
        return [md]
    except Exception as e:
        print(f"\n[fetch] 데이터 수집 오류 ({kst_now.strftime('%H:%M:%S')}): {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    system = DataCapital()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 자동 매매 엔진 가동 중...", flush=True)

    while True:
        try:
            kst_now = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=9)
            now_time = kst_now.time()

            if time(9, 0) <= now_time <= time(15, 40):
                t0 = time_module.monotonic()
                data = fetch_and_calculate()
                system.last_api_latency = (time_module.monotonic() - t0) * 1000  # ms
                if data:
                    system.run_iteration(data)
                    print(f"  [LIVE] {kst_now.strftime('%H:%M:%S')} | 종가:{data[0].close:,.0f} | RSI:{data[0].rsi14:.1f} | 지연:{system.last_api_latency:.0f}ms", flush=True)
                else:
                    print(f"  [WAIT] {kst_now.strftime('%H:%M:%S')} | 데이터 수집 대기 중...", flush=True)
            
            elif now_time > time(15, 40):
                cum_pct = (system.current_capital - system.TOTAL_CAPITAL) / system.TOTAL_CAPITAL
                print(f"\n[{kst_now.strftime('%H:%M:%S')}] 금일 장 종료. 일일PnL:{system.daily_pnl_pct:+.3%} | 누적PnL:{cum_pct:+.3%}", flush=True)
                system.daily_pnl_pct = 0.0
                system.is_halted     = False  # 다음날 재개
                time_module.sleep(3600 * 5)

            else:
                print(f"  [SLEEP] {kst_now.strftime('%H:%M:%S')} | 개장 대기 중...             ", end='\r', flush=True)

            time_module.sleep(60)

        except KeyboardInterrupt:
            print("\n사용자에 의해 시스템이 종료되었습니다.")
            sys.exit()
        except Exception as e:
            print(f"\n[ERROR] 시스템 오류 발생: {e}")
            traceback.print_exc()
            time_module.sleep(60)
