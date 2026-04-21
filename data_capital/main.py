"""
DATA CAPITAL — main.py
==========================================
FirebaseSync: Firestore 실시간 동기화
DataCapital:  전체 시스템 메인 엔진
"""

import time as time_module
import json
from datetime import datetime, time
from typing import Optional
import traceback

from core.harness    import MarketData, MarketState, BuyFilters, SellRules
from agents          import create_all_agents
from meta_agents     import CIO, Guardian, Oracle, Coach
from failure_db_backtest import FailureLearningDB


# ════════════════════════════════════════════
#  Firebase 연동
# ════════════════════════════════════════════
class FirebaseSync:
    """
    Python 매매 엔진 ↔ Firestore 실시간 동기화.
    매매 결과, 에이전트 상태, 알림을 자동 저장.
    """

    def __init__(self, service_account_path: str = "serviceAccount.json"):
        self.db = None
        self._init_firebase(service_account_path)

    def _init_firebase(self, path: str):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred = credentials.Certificate(path)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("[Firebase] 연결 성공")
        except Exception as e:
            print(f"[Firebase] 연결 실패 (오프라인 모드): {e}")
            self.db = None

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
            print(f"[Firebase] 쓰기 실패 {collection}/{doc_id}: {e}")

    def save_trade(self, trade: dict):
        """거래 기록 저장"""
        self._safe_write("trades", None, trade)

    def save_signal(self, signal_data: dict):
        """에이전트 신호 저장"""
        self._safe_write("signals", None, signal_data)

    def update_portfolio(self, portfolio: dict):
        """포트폴리오 실시간 업데이트"""
        self._safe_write("portfolio", "live", portfolio)

    def save_consensus(self, consensus_data: dict):
        """오라클 합의 결과 저장"""
        self._safe_write("consensus", None, consensus_data)

    def save_alert(self, message: str, severity: str = "INFO"):
        """가디언 경고 저장 → PWA 푸시 + 텔레그램"""
        self._safe_write("alerts", None, {
            "message":   message,
            "severity":  severity,
            "timestamp": datetime.now().isoformat(),
        })
        self._send_telegram(message, severity)

    def save_failure(self, failure_data: dict):
        """실패 학습 DB 저장"""
        self._safe_write("failure_db", None, failure_data)

    def save_review(self, date_str: str, review: dict):
        """일일/주간 복기 저장"""
        self._safe_write("reviews", date_str, review)

    def save_agent_stats(self, agent_id: str, stats: dict):
        """에이전트 성과 통계 저장"""
        self._safe_write("agents", agent_id, stats)

    def _send_telegram(self, message: str, severity: str):
        """텔레그램 알림 발송"""
        import os, urllib.request, urllib.parse
        token   = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not (token and chat_id):
            return
        try:
            emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "INFO": "📊"}.get(severity, "📌")
            text  = f"{emoji} DATA CAPITAL\n{message}"
            url   = f"https://api.telegram.org/bot{token}/sendMessage"
            data  = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
            urllib.request.urlopen(url, data, timeout=5)
        except Exception as e:
            print(f"[Telegram] 발송 실패: {e}")

    def load_system_prompt(self) -> str:
        """Firestore에서 시스템 프롬프트 불러오기"""
        if self.db is None:
            return ""
        try:
            doc = self.db.collection("config").document("system_prompt").get()
            return doc.to_dict().get("content", "") if doc.exists else ""
        except Exception:
            return ""

    def call_claude(self, user_message: str, max_tokens: int = 800) -> str:
        """Claude API 호출 (Firebase Functions 우회 직접 호출)"""
        import os, urllib.request
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "[Claude API 키 없음]"
        try:
            system_prompt = self.load_system_prompt()
            payload = json.dumps({
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_message}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except Exception as e:
            return f"[Claude 호출 오류: {e}]"


# ════════════════════════════════════════════
#  DATA CAPITAL 메인 엔진
# ════════════════════════════════════════════
class DataCapital:
    """
    DATA CAPITAL 전체 시스템.

    4중 아키텍처:
    L1 운용계층: 7개 에이전트
    L2 감시계층: 가디언
    L3 합의계층: 오라클 + 실패DB
    L4 진화계층: 코치

    철학: 변화가 곧 안정. 꼬꾸라지지 않고 방법을 찾는다.
    """

    TOTAL_CAPITAL  = 3_000_000   # 300만원
    DAILY_MDD_HALT = -0.008      # -0.8% 일일 MDD 한도
    MIN_EV_TRADE   = 0.003       # EV 최소 0.3% 이상 거래만

    def __init__(self, service_account_path: str = "serviceAccount.json"):
        print("\n" + "="*55)
        print("  DATA CAPITAL v3.0")
        print("  '변화가 곧 안정이다'")
        print("="*55 + "\n")

        # 에이전트 초기화
        self.agents     = create_all_agents(self.TOTAL_CAPITAL)
        self.cio        = CIO(self.TOTAL_CAPITAL)
        self.guardian   = Guardian()
        self.oracle     = Oracle()
        self.coach      = Coach(target_annual_return=0.18)
        self.failure_db = FailureLearningDB()
        self.firebase   = FirebaseSync(service_account_path)

        # 상태
        self.open_positions:  dict  = {}
        self.daily_pnl_pct:   float = 0.0
        self.total_pnl_pct:   float = 0.0
        self.portfolio_exposure: float = 0.0
        self.consecutive_losses: int = 0
        self.is_halted:       bool  = False
        self.today:           str   = datetime.now().strftime("%Y%m%d")

    def morning_run(self, market_data_list: list) -> dict:
        """
        장 시작 — 오전 9시 루틴.
        가디언 헬스체크 → 에이전트 신호 수집 → 오라클 합의 → CIO 배분.
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🌅 모닝 루틴 시작")

        # ── 가디언 헬스체크 ──
        if market_data_list:
            md = market_data_list[0]
            health = self.guardian.health_check(
                daily_pnl_pct=self.daily_pnl_pct,
                mdd_pct=self.daily_pnl_pct,
                api_latency_ms=50,
                data_quality_ok=True,
                vkospi=md.vkospi,
                avg_correlation=0.3,
            )
            if health["halt"]:
                self.is_halted = True
                self.firebase.save_alert("\n".join(health["alerts"]), "CRITICAL")
                print(f"[Guardian] 🚨 전면 거래 중단: {health['alerts']}")
                return {"status": "HALTED", "alerts": health["alerts"]}

        # ── 에이전트별 신호 수집 ──
        context = {
            "daily_loss_pct":       self.daily_pnl_pct,
            "portfolio_exposure":   self.portfolio_exposure,
            "consecutive_losses":   self.consecutive_losses,
        }

        all_signals = {}
        for agent_id, agent in self.agents.items():
            for md in market_data_list:
                try:
                    same_dir = sum(
                        1 for pos in self.open_positions.values()
                        if pos.get("direction") == "BUY"
                    )
                    signal = agent.generate_signal(
                        md,
                        same_direction_agents=same_dir,
                        **context,
                    )

                    # EV 최소 기준 필터
                    if signal.signal != "NO_SIGNAL" and signal.ev < self.MIN_EV_TRADE:
                        print(f"  [{agent_id}] EV {signal.ev:.4%} < 0.3% 기준 — 패스")
                        continue

                    all_signals[agent_id] = signal
                    print(f"  [{agent_id}] {signal.signal} | 신뢰도 {signal.confidence:.0%} | EV {signal.ev:.4%}")

                except Exception as e:
                    print(f"  [{agent_id}] 신호 생성 오류: {e}")

        # ── 실패 DB 경고 조회 ──
        failure_warning = None
        if market_data_list:
            md = market_data_list[0]
            warning = self.failure_db.get_warning({
                "market_state": md.market_state.value,
                "vkospi_bucket": int(md.vkospi // 5) * 5,
                "kospi_direction": "UP" if md.kospi_change > 0 else "DOWN",
            })
            if warning and warning["type"] == "DANGER":
                failure_warning = warning["message"]
                print(f"  [FailureDB] ⚠️ {failure_warning}")

        # ── 오라클 합의 ──
        consensus = self.oracle.form_consensus(all_signals, failure_warning)
        print(f"\n[Oracle] 결론: {consensus.decision} | 신뢰도 {consensus.confidence:.0%}")
        print(f"         {consensus.reasoning}")

        if consensus.minority_opinion:
            print(f"[Oracle] 소수 의견 추적: {consensus.minority_opinion['agents']}")

        # ── CIO 자금 배분 ──
        active_signals = {k: v for k, v in all_signals.items() if v.signal != "NO_SIGNAL"}
        allocations, veto_reasons = self.cio.allocate(self.agents, active_signals)

        for agent_id, amount in allocations.items():
            if amount > 0:
                print(f"[CIO] {agent_id}: {amount:,.0f}원 배분")
        for agent_id, reason in veto_reasons.items():
            print(f"[CIO] VETO {agent_id}: {reason}")

        # Firebase 저장
        self.firebase.save_consensus({
            "decision":      consensus.decision,
            "confidence":    consensus.confidence,
            "reasoning":     consensus.reasoning,
            "buy_agents":    consensus.buy_agents,
            "hold_agents":   consensus.hold_agents,
            "sell_agents":   consensus.sell_agents,
            "minority":      consensus.minority_opinion,
            "allocations":   allocations,
            "timestamp":     datetime.now().isoformat(),
        })

        return {
            "status":      "OK",
            "consensus":   consensus.decision,
            "allocations": allocations,
            "signals":     {k: v.to_dict() for k, v in all_signals.items()},
        }

    def intraday_check(self, market_data_list: list):
        """
        장 중 포지션 관리.
        열린 포지션 매도 조건 체크.
        """
        for pos_id, pos in list(self.open_positions.items()):
            from core.harness import Position
            position = pos["position_obj"]
            for md in market_data_list:
                result = SellRules.check(position, md.close, md.current_time)
                if result:
                    pnl_pct = (md.close - position.entry_price) / position.entry_price
                    print(f"[{result}] {pos_id} @ {md.close:,.0f}원 (PnL: {pnl_pct:+.3%})")

                    self.daily_pnl_pct += pnl_pct
                    if pnl_pct < 0:
                        self.consecutive_losses += 1
                        # 실패 DB 기록
                        self.firebase.save_failure({
                            "agent_id":     pos["agent_id"],
                            "pnl_pct":      pnl_pct,
                            "reason":       result,
                            "market_state": md.market_state.value,
                            "vkospi":       md.vkospi,
                            "timestamp":    datetime.now().isoformat(),
                        })
                    else:
                        self.consecutive_losses = 0

                    del self.open_positions[pos_id]

                    self.firebase.update_portfolio({
                        "daily_pnl_pct":     self.daily_pnl_pct,
                        "total_pnl_pct":     self.total_pnl_pct,
                        "open_positions":    len(self.open_positions),
                        "consecutive_losses": self.consecutive_losses,
                        "timestamp":         datetime.now().isoformat(),
                    })

                    # MDD 체크
                    if self.daily_pnl_pct <= self.DAILY_MDD_HALT:
                        self.is_halted = True
                        msg = f"🚨 MDD -0.8% 도달 ({self.daily_pnl_pct:.2%}) — 전면 거래 중단"
                        self.firebase.save_alert(msg, "CRITICAL")
                        print(f"[Guardian] {msg}")

    def closing_run(self):
        """
        장 마감 후 루틴 (15:30~).
        성과 기록 → Claude 일일 복기 요청 → 텔레그램 발송.
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🌙 마감 루틴 시작")

        # 에이전트 통계 저장
        for agent_id, agent in self.agents.items():
            stats = agent.get_stats()
            self.firebase.save_agent_stats(agent_id, stats)

        # Claude 일일 복기 요청
        review_prompt = f"""[일일복기] {self.today}

포트폴리오: 오늘 {self.daily_pnl_pct:.2%} / 누적 {self.total_pnl_pct:.2%}
열린 포지션: {len(self.open_positions)}개
연속 손실: {self.consecutive_losses}회
시스템 상태: {'중단' if self.is_halted else '정상'}

에이전트 성과:
{json.dumps({k: v.get_stats() for k, v in self.agents.items()}, ensure_ascii=False, indent=2)}

## 총평 (1줄)
## 매수 타이밍 평가
## 코드 수정 필요: YES/NO
## 내일 주의사항 (1줄)
"""
        analysis = self.firebase.call_claude(review_prompt)

        today_str = datetime.now().strftime("%Y-%m-%d")
        self.firebase.save_review(today_str, {
            "analysis":        analysis,
            "daily_pnl_pct":   self.daily_pnl_pct,
            "total_pnl_pct":   self.total_pnl_pct,
            "trades_today":    sum(a.win_count + a.loss_count for a in self.agents.values()),
            "timestamp":       datetime.now().isoformat(),
        })

        print(f"\n[Claude 복기]\n{analysis}")

        # 일일 초기화
        self.daily_pnl_pct = 0.0
        self.is_halted     = False
        self.today         = datetime.now().strftime("%Y%m%d")

    def run_backtest_mode(self):
        """
        백테스트 모드 실행.
        실전 투입 전 Walk-Forward 검증.
        """
        from failure_db_backtest import WalkForwardBacktest, BacktestConfig
        from agents import (
            GapTradingAgent, MeanRevAgent, MomentumAgent,
            PairsAgent, EODAgent, VolatilityAgent, LevDecayAgent,
        )

        print("\n📊 Walk-Forward 백테스트 모드\n")

        config = BacktestConfig(
            ticker="069500",
            start_date="20200101",
            end_date="20241231",
        )
        bt = WalkForwardBacktest(config)
        bt.load_data()

        agent_classes = [
            GapTradingAgent, MeanRevAgent, MomentumAgent,
            PairsAgent, EODAgent, VolatilityAgent, LevDecayAgent,
        ]

        results = bt.run_all_parallel(agent_classes)
        bt.print_summary(results)

        # 통과 에이전트만 저장
        passed = {k: v for k, v in results.items() if v.passed}
        print(f"\n✅ 실전 투입 승인: {list(passed.keys())}")
        return results


# ─────────────────────────────────────────────
#  실행 엔트리포인트 (페이퍼 트레이딩 지원)
# ─────────────────────────────────────────────
def fetch_realtime_market_data(ticker="069500") -> list:
    """실시간에 가까운 데이터를 pykrx로 수집 (페이퍼 트레이딩용)"""
    from pykrx import stock
    import pandas as pd
    from datetime import datetime
    
    today = datetime.now().strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv(today, today, ticker)
        if df.empty:
            # 장 시작 전이거나 데이터가 없으면 어제 데이터 참고 (테스트용)
            return []
            
        row = df.iloc[-1]
        # 임의의 지표 계산 (실전 시에는 더 정교하게 필요)
        md = MarketData(
            ticker=ticker,
            current_time=datetime.now(),
            open=float(row['시가']),
            high=float(row['고가']),
            low=float(row['저가']),
            close=float(row['종가']),
            volume=int(row['거래량']),
            prev_close=float(row['종가'] / (1 + row['등락률']/100)) if '등락률' in row else float(row['종가']),
            vkospi=20.0,  # 실시간 VKOSPI API 연동 전까지 기본값
            kospi_change=0.5, # 기본값
        )
        # 지표 강제 주입 (테스트용)
        md.rsi14 = 50.0
        md.ma20 = md.close * 0.99
        md.vol_ma5 = md.volume
        
        return [md]
    except Exception as e:
        print(f"[Fetch] 오류: {e}")
        return []

def is_trading_time():
    now = datetime.now().time()
    return time(9, 0) <= now <= time(15, 30)

if __name__ == "__main__":
    import sys

    system = DataCapital(service_account_path="serviceAccount.json")

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        # 백테스트 모드
        system.run_backtest_mode()

    else:
        # 페이퍼 트레이딩 모드
        print("\n🚀 DATA CAPITAL 페이퍼 트레이딩 시작")
        print("   - 종목: KODEX 200 (069500)")
        print("   - 모드: 실시간 데이터 기반 가상 매매")
        
        has_run_morning = False
        has_run_closing = False

        while True:
            now = datetime.now()
            
            if is_trading_time():
                market_data = fetch_realtime_market_data()
                
                if market_data:
                    # 1. 모닝 런 (하루 한 번)
                    if now.time() >= time(9, 5) and not has_run_morning:
                        system.morning_run(market_data)
                        has_run_morning = True
                    
                    # 2. 인트라데이 체크 (상시)
                    system.intraday_check(market_data)
                
                has_run_closing = False # 다음 장 마감을 위해 리셋
            
            else:
                # 장 마감 루틴 (하루 한 번)
                if now.time() >= time(15, 30) and not has_run_closing:
                    system.closing_run()
                    has_run_closing = True
                    has_run_morning = False # 다음 날 장 시작을 위해 리셋
                
                if now.time() > time(16, 0):
                    print(f"[{now.strftime('%H:%M:%S')}] 장외 시간 — 대기 중...")
                    time_module.sleep(3600) # 1시간 대기
                    continue

            time_module.sleep(60) # 1분 단위 루프
