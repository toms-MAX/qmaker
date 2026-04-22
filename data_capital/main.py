"""
DATA CAPITAL — main.py (Robust Automation Version)
==========================================
"기계만 켜두면 돌아가는 자율 주행 트레이더"
"""

import time as time_module
import json
from datetime import datetime, time, timedelta, timezone
UTC = timezone.utc
from typing import Optional
import traceback
import pandas as pd
import sys
import os

from core.harness    import MarketData, MarketState, BuyFilters, SellRules, Position
from agents          import create_all_agents
from meta_agents     import CIO, Guardian, Oracle, Coach
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
            if os.path.exists(path):
                if not firebase_admin._apps:
                    cred = credentials.Certificate(path)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print("[Firebase] 연결 성공")
            else:
                print("[Firebase] 설정 파일 없음 - 오프라인 모드")
        except Exception as e:
            print(f"[Firebase] 연결 실패: {e}")

    def _safe_write(self, collection: str, doc_id: Optional[str], data: dict):
        if self.db is None: return
        try:
            from firebase_admin import firestore
            data["synced_at"] = firestore.SERVER_TIMESTAMP
            if doc_id: self.db.collection(collection).document(doc_id).set(data)
            else: self.db.collection(collection).add(data)
        except: pass

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

        self.open_positions:  dict  = {}
        self.daily_pnl_pct:   float = 0.0
        self.is_halted:       bool  = False

    def execute_order(self, agent_id: str, ticker: str, side: str, price: float, amount: float):
        kst_now = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=9)
        side_kor = "매수" if side == "BUY" else "매도"
        
        print(f"\n>>> [ORDER] {kst_now.strftime('%H:%M:%S')} | {agent_id} | {side_kor} | {price:,.0f}원 | {amount:,.0f}원치")
        
        if side == "BUY":
            shares = int(amount / price)
            if shares > 0:
                self.open_positions[agent_id] = {
                    "agent_id": agent_id,
                    "ticker": ticker,
                    "entry_price": price,
                    "entry_time": kst_now,
                    "shares": shares,
                    "position_obj": Position(
                        agent_id=agent_id, ticker=ticker, entry_price=price,
                        entry_time=kst_now, shares=shares,
                        target_price=price * 1.01,
                        stop_price=price * 0.995
                    )
                }
                self.firebase.save_trade({"side": "BUY", "agent": agent_id, "price": price, "shares": shares})

    def run_iteration(self, market_data_list: list):
        if not market_data_list: return
        md = market_data_list[0]
        kst_now = md.current_time

        for agent_id, pos in list(self.open_positions.items()):
            res = SellRules.check(pos["position_obj"], md.close, kst_now)
            if res:
                trade_pnl_pct = (md.close - pos["entry_price"]) / pos["entry_price"]
                trade_amt     = pos["position_obj"].shares * pos["entry_price"]
                # 포트폴리오 기준 손익 반영 (개별 거래 수익률 × 거래 비중)
                portfolio_pnl = trade_pnl_pct * (trade_amt / self.TOTAL_CAPITAL)
                self.daily_pnl_pct += portfolio_pnl
                pnl_won = trade_pnl_pct * trade_amt
                print(f"\n<<< [EXIT] {agent_id} | {res} | {trade_pnl_pct:+.2%} | {pnl_won:+,.0f}원 | 일일PnL: {self.daily_pnl_pct:+.3%}")
                del self.open_positions[agent_id]
                self.firebase.save_trade({"side": "SELL", "agent": agent_id, "price": md.close,
                                          "pnl": trade_pnl_pct, "portfolio_pnl": portfolio_pnl})

        if time(9, 5) <= kst_now.time() <= time(15, 0) and not self.is_halted:
            # 모든 에이전트 신호 수집 (포지션 없는 에이전트만)
            all_signals = {}
            buy_signals  = {}
            for aid, agent in self.agents.items():
                if aid not in self.open_positions:
                    sig = agent.generate_signal(md, daily_pnl_pct=self.daily_pnl_pct)
                    all_signals[aid] = sig          # Oracle: 전체 신호 (HOLD 포함)
                    if sig.signal == "BUY":
                        buy_signals[aid] = sig      # CIO: BUY만

            if buy_signals:
                consensus = self.oracle.form_consensus(all_signals)  # 전체 신호로 합의
                print(f"  [Oracle] {consensus.decision} | {consensus.reasoning[:70]}")
                if consensus.decision in ("BUY", "SPLIT"):
                    split_factor = 0.4 if consensus.decision == "SPLIT" else 1.0
                    allocations, vetos = self.cio.allocate(self.agents, buy_signals)
                    if vetos:
                        print(f"  [VETO] {vetos}")
                    for aid, amt in allocations.items():
                        adj_amt = amt * split_factor
                        if adj_amt > 100_000:
                            self.execute_order(aid, md.ticker, "BUY", md.close, adj_amt)

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
            kdf = stock.get_index_ohlcv(start, today, "1001")  # KOSPI
            if not kdf.empty and len(kdf) >= 2:
                kospi_change = float(kdf['등락률'].iloc[-1])
        except Exception:
            pass

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
        print(f"[fetch] 오류: {e}")
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
                data = fetch_and_calculate()
                if data:
                    system.run_iteration(data)
                    print(f"  [LIVE] {kst_now.strftime('%H:%M:%S')} | 종가:{data[0].close:,.0f} | RSI:{data[0].rsi14:.1f} | 감시 중...", flush=True)
                else:
                    print(f"  [WAIT] {kst_now.strftime('%H:%M:%S')} | 데이터 수집 대기 중...", flush=True)
            
            elif now_time > time(15, 40):
                print(f"\n[{kst_now.strftime('%H:%M:%S')}] 금일 장 종료. 수면 모드 진입.", flush=True)
                system.daily_pnl_pct = 0.0 
                time_module.sleep(3600 * 5) 

            else:
                print(f"  [SLEEP] {kst_now.strftime('%H:%M:%S')} | 개장 대기 중...             ", end='\r', flush=True)

            time_module.sleep(60)

        except KeyboardInterrupt:
            print("\n사용자에 의해 시스템이 종료되었습니다.")
            sys.exit()
        except Exception as e:
            print(f"\n[ERROR] 시스템 오류 발생: {e}")
            time_module.sleep(60)
