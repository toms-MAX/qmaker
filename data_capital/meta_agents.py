"""
DATA CAPITAL — meta_agents.py
==============================
CIO:      자금 배분 + 5대 VETO
Guardian: 시스템 감시 + 충돌 중재
Oracle:   7개 에이전트 합의 + 집단지성
Coach:    이사장 멘탈 + 알파 붕괴 검사
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

from core.harness import AgentHarness, MarketData, MarketState, SignalResult


# ─────────────────────────────────────────────
#  CIO — 자금 배분 엔진
# ─────────────────────────────────────────────
class CIO:
    """
    7개 에이전트에 자금을 배분하는 최고투자책임자.

    점수 계산:
    score = EV(0.30) + market_fit(0.25) + bayes_trust(0.20)
            + risk_adj(0.15) + streak_penalty(0.10)
    """

    MAX_AGENT_ALLOC = 0.35   # 에이전트당 최대 35%
    CASH_RESERVE    = 0.25   # 현금 유보 25%
    MIN_MARKET_FIT  = 0.40   # VETO 기준 시장 적합도

    def __init__(self, total_capital: float):
        self.total_capital     = total_capital
        self.deployable        = total_capital * (1 - self.CASH_RESERVE)
        self.daily_pnl_pct     = 0.0
        self.portfolio_exposure = 0.0
        self.agent_positions    = {}   # agent_id → 현재 포지션 금액
        self.correlation_matrix = {}   # 에이전트 간 상관관계

    def score_agent(self, agent: AgentHarness, signal: SignalResult) -> float:
        """에이전트 점수 계산"""
        # EV 정규화 (0~1)
        ev_norm = min(1.0, max(0.0, signal.ev / 0.005))

        # 베이지안 신뢰도
        bayes = agent.bayesian_win_rate if (agent.win_count + agent.loss_count) > 10 else 0.5

        # 리스크 조정 (샤프 개념)
        total_trades = agent.win_count + agent.loss_count
        if total_trades > 0 and len(agent.trade_history) >= 5:
            recent_pnls = [t["pnl"] for t in agent.trade_history[-20:]]
            pnl_std = np.std(recent_pnls) if len(recent_pnls) > 1 else 0.001
            sharpe  = np.mean(recent_pnls) / pnl_std if pnl_std > 0 else 0
            risk_adj = min(1.0, max(0.0, sharpe / 2))
        else:
            risk_adj = 0.5

        # 연속 손실 패널티
        consecutive_losses = 0
        for t in reversed(agent.trade_history[-5:]):
            if t["pnl"] < 0:
                consecutive_losses += 1
            else:
                break
        streak_penalty = max(0.0, 1.0 - consecutive_losses * 0.15)

        score = (
            ev_norm            * 0.30 +
            signal.market_fit  * 0.25 +
            bayes              * 0.20 +
            risk_adj           * 0.15 +
            streak_penalty     * 0.10
        )
        return score

    def check_veto(
        self,
        agent: AgentHarness,
        signal: SignalResult,
        same_direction_exposure: float,
        avg_correlation: float,
    ) -> Optional[str]:
        """
        5대 VETO 조건 확인.
        VETO 이유 반환 (없으면 None)
        """
        # V1: 시장 적합도 < 0.40
        if signal.market_fit < self.MIN_MARKET_FIT:
            return f"V1: 시장 적합도 {signal.market_fit:.2f} < 0.40"

        # V2: 동방향 쏠림 > 70%
        if same_direction_exposure > self.deployable * 0.70:
            return f"V2: 동방향 쏠림 {same_direction_exposure/self.deployable:.0%} > 70%"

        # V3: 상관관계 > 0.65
        if avg_correlation > 0.65:
            return f"V3: 에이전트 평균 상관관계 {avg_correlation:.2f} > 0.65"

        # V4: 베이지안 승률 하한 < 50%
        if (agent.win_count + agent.loss_count) > 30 and agent.bayesian_win_rate < 0.50:
            return f"V4: 베이지안 승률 하한 {agent.bayesian_win_rate:.1%} < 50% — 강제 휴식"

        # V5: 일일 손실 한도 소진
        if self.daily_pnl_pct <= -0.008:
            return f"V5: 일일 MDD {self.daily_pnl_pct:.2%} 도달 — 전체 거래 중단"

        return None

    def allocate(
        self,
        agents: dict,
        signals: dict,
        avg_correlation: float = 0.3,
    ) -> dict:
        """
        에이전트별 자금 배분 결정.
        반환: {agent_id: allocated_capital}
        """
        # V5: 전체 MDD 도달 시 즉시 전체 0
        if self.daily_pnl_pct <= -0.008:
            return {k: 0.0 for k in agents}

        # 긴급 프로토콜: 상관관계 급등
        if avg_correlation > 0.60:
            emergency_ratio = 0.50 if avg_correlation > 0.60 else 0.75
        else:
            emergency_ratio = 1.0

        scores       = {}
        veto_reasons = {}

        for agent_id, signal in signals.items():
            if signal.signal == "NO_SIGNAL":
                continue
            agent = agents[agent_id]

            # VETO 확인
            same_dir = sum(
                amt for aid, amt in self.agent_positions.items()
                if signals.get(aid, signal).signal == signal.signal
            )
            veto = self.check_veto(agent, signal, same_dir, avg_correlation)
            if veto:
                veto_reasons[agent_id] = veto
                continue

            scores[agent_id] = self.score_agent(agent, signal)

        # 배분 계산
        total_score = sum(scores.values()) or 1.0
        allocations = {}

        for agent_id, score in scores.items():
            raw_alloc = self.deployable * (score / total_score)
            capped    = min(raw_alloc, self.total_capital * self.MAX_AGENT_ALLOC)
            final     = capped * emergency_ratio
            allocations[agent_id] = round(final, 0)

        # VETO된 에이전트 0원
        for agent_id in veto_reasons:
            allocations[agent_id] = 0.0

        return allocations, veto_reasons


# ─────────────────────────────────────────────
#  Guardian — 시스템 감시자
# ─────────────────────────────────────────────
class Guardian:
    """
    다른 에이전트들을 감시하고, 충돌을 중재하며,
    시스템 이상을 감지하는 메타 에이전트.
    코치는 이사장을 돕고, 가디언은 시스템을 지킨다.
    """

    def __init__(self):
        self.alert_log:    list = []
        self.conflict_log: list = []
        self.health_score: float = 1.0

    def check_conflict(self, signals: dict) -> list:
        """
        에이전트 간 신호 충돌 감지.
        동일 종목에 BUY + SELL 동시 신호 등.
        """
        conflicts = []
        buy_agents  = [aid for aid, sig in signals.items() if sig.signal == "BUY"]
        sell_agents = [aid for aid, sig in signals.items() if "SELL" in sig.signal]

        if buy_agents and sell_agents:
            conflict = {
                "type": "DIRECTION_CONFLICT",
                "buy":  buy_agents,
                "sell": sell_agents,
                "resolution": "REDUCE_BOTH",
                "timestamp": datetime.now().isoformat(),
            }
            conflicts.append(conflict)
            self.conflict_log.append(conflict)

        return conflicts

    def health_check(
        self,
        daily_pnl_pct: float,
        mdd_pct: float,
        api_latency_ms: float,
        data_quality_ok: bool,
        vkospi: float,
        avg_correlation: float,
    ) -> dict:
        """
        시스템 건강 상태 체크.
        전체 운영 가능 여부 판단.
        """
        alerts = []
        halt   = False

        # MDD 체크
        if daily_pnl_pct <= -0.008:
            alerts.append("🚨 일일 MDD -0.8% 도달 — 전면 거래 중단")
            halt = True
        elif daily_pnl_pct <= -0.006:
            alerts.append("⚠️ 일일 손실 -0.6% — 포지션 50% 축소")

        # API 응답 체크
        if api_latency_ms > 3000:
            alerts.append(f"⚠️ API 응답 {api_latency_ms:.0f}ms — 장애 의심")

        # 데이터 품질
        if not data_quality_ok:
            alerts.append("⚠️ 데이터 이상 감지 — 신호 생성 중단")
            halt = True

        # VKOSPI 공황
        if vkospi >= 35:
            alerts.append(f"🚨 VKOSPI {vkospi:.1f} >= 35 — 공황 선언")
            halt = True

        # 상관관계 급등
        if avg_correlation >= 0.60:
            alerts.append(f"⚠️ 에이전트 평균 상관관계 {avg_correlation:.2f} >= 0.60 — 리스크 집중")

        self.health_score = 0.0 if halt else max(0.0, 1.0 - len(alerts) * 0.1)

        result = {
            "healthy": not halt,
            "halt":    halt,
            "score":   self.health_score,
            "alerts":  alerts,
            "timestamp": datetime.now().isoformat(),
        }

        if alerts:
            self.alert_log.extend(alerts)

        return result

    def validate_patch(self, old_sharpe: float, new_sharpe: float) -> bool:
        """
        코드 패치가 성능을 유지하는지 검증.
        샤프 비율 30% 이상 하락 시 패치 거부.
        """
        if old_sharpe <= 0:
            return True
        degradation = (old_sharpe - new_sharpe) / old_sharpe
        return degradation < 0.30


# ─────────────────────────────────────────────
#  Oracle — 집단지성 합의 시스템
# ─────────────────────────────────────────────
@dataclass
class ConsensusResult:
    decision:      str       # BUY / SELL / HOLD / SPLIT
    confidence:    float
    capital_ratio: float     # 총 deployable 대비 비율
    buy_agents:    list
    hold_agents:   list
    sell_agents:   list
    minority_opinion: Optional[dict]   # 소수 의견 추적
    reasoning:     str
    timestamp:     datetime = field(default_factory=datetime.now)


class Oracle:
    """
    7개 에이전트의 의견을 종합해서
    1개의 메타 결론을 도출하는 합의 시스템.
    활성 에이전트 수에 따라 임계값을 동적으로 조정.
    """

    TOTAL_AGENTS = 7          # 목표 에이전트 수 (확장 시 조정)

    def __init__(self):
        self.consensus_history: list = []
        self.minority_accuracy: dict = {}   # 소수 의견 적중 추적

    def _cross_validate(self, signals: dict) -> dict:
        """에이전트 이유 간 교차 검증"""
        buy_reasons  = [s.reason for s in signals.values() if s.signal == "BUY"]
        sell_reasons = [s.reason for s in signals.values() if "SELL" in s.signal]

        # 서로 다른 근거로 같은 결론 → 독립적 확증 (신뢰도 +20%)
        independent_confirmation = len(set(buy_reasons)) >= 2 if buy_reasons else False

        return {
            "independent_confirmation": independent_confirmation,
            "buy_reasons": buy_reasons,
            "sell_reasons": sell_reasons,
        }

    def form_consensus(
        self,
        signals: dict,
        failure_db_warning: Optional[str] = None,
    ) -> ConsensusResult:
        """
        7개 에이전트 신호 → 1개 합의 결론

        Args:
            signals: {agent_id: SignalResult}
            failure_db_warning: 실패 DB에서 온 경고 (있으면 반영)
        """
        # 전체 신호 수 기준으로 임계값 계산 (HOLD/NO_SIGNAL도 "반대" 표로 취급)
        n_total = max(len(signals), self.TOTAL_AGENTS)   # 최소 TOTAL_AGENTS 기준 유지
        active = {k: v for k, v in signals.items() if v.signal not in ("NO_SIGNAL", "HOLD")}

        buy_agents  = [k for k, v in signals.items() if v.signal == "BUY"]
        sell_agents = [k for k, v in signals.items() if "SELL" in v.signal]
        hold_agents = [k for k, v in signals.items() if v.signal in ("HOLD", "NO_SIGNAL")]

        n_buy   = len(buy_agents)
        n_sell  = len(sell_agents)

        # n_total 기준 동적 임계값 — 7명 중 몇 명이 BUY를 외쳤는가
        unanimity_threshold = max(1, round(n_total * 0.43))   # 3/7=43%+ → 만장일치(현재 3명)
        majority_threshold  = max(1, round(n_total * 0.28))   # 2/7=28%+ → 과반(현재 2명)

        cross_val = self._cross_validate(active)

        # ── 소수 의견 추적 ──
        minority_opinion = None
        if n_buy >= majority_threshold and n_sell >= 1:  # type: ignore[operator]
            minority_opinion = {
                "agents": sell_agents,
                "reasons": [active[a].reason for a in sell_agents],
                "note": "소수 반대 의견 기록 — 향후 적중 시 가중치 상향",
            }

        # ── 실패 DB 경고 반영 ──
        if failure_db_warning:
            return ConsensusResult(
                decision="HOLD",
                confidence=0.3,
                capital_ratio=0.0,
                buy_agents=buy_agents,
                hold_agents=hold_agents,
                sell_agents=sell_agents,
                minority_opinion=minority_opinion,
                reasoning=f"실패 DB 경고: {failure_db_warning}",
            )

        # ── 만장일치: 전체 에이전트 43%+ BUY (현재 3명 기준: 3명) ──
        if n_buy >= unanimity_threshold:
            conf_bonus = 0.20 if cross_val["independent_confirmation"] else 0.0
            confidence = min(0.95, 0.75 + conf_bonus)
            return ConsensusResult(
                decision="BUY",
                confidence=confidence,
                capital_ratio=0.60,
                buy_agents=buy_agents,
                hold_agents=hold_agents,
                sell_agents=sell_agents,
                minority_opinion=minority_opinion,
                reasoning=(
                    f"활성 에이전트 BUY {n_buy}/{n_total} | "
                    f"독립 확증: {cross_val['independent_confirmation']} | "
                    f"신뢰도 {confidence:.0%}"
                ),
            )

        # ── 과반: 28%+ BUY (현재 기준: 2명) ──
        if n_buy >= majority_threshold:
            confidence = 0.55 + (n_buy - majority_threshold) * 0.05
            return ConsensusResult(
                decision="BUY",
                confidence=confidence,
                capital_ratio=0.35,
                buy_agents=buy_agents,
                hold_agents=hold_agents,
                sell_agents=sell_agents,
                minority_opinion=minority_opinion,
                reasoning=f"과반 BUY {n_buy}/{n_total} | 신뢰도 {confidence:.0%}",
            )

        # ── 소수 단독 신호 ──
        if n_buy == 1 and n_sell == 0:
            return ConsensusResult(
                decision="SPLIT",
                confidence=0.35,
                capital_ratio=0.08,
                buy_agents=buy_agents,
                hold_agents=hold_agents,
                sell_agents=sell_agents,
                minority_opinion=None,
                reasoning=f"단독 신호 [{buy_agents[0]}] — 소규모 진입 (40% 사이즈)",
            )

        # ── 관망 ──
        return ConsensusResult(
            decision="HOLD",
            confidence=0.30,
            capital_ratio=0.0,
            buy_agents=buy_agents,
            hold_agents=hold_agents,
            sell_agents=sell_agents,
            minority_opinion=None,
            reasoning="충분한 합의 미형성 — 관망",
        )

    def track_minority_accuracy(self, past_consensus: ConsensusResult, actual_result: str):
        """소수 의견이 맞았는지 사후 추적"""
        if not past_consensus.minority_opinion:
            return
        if actual_result != past_consensus.decision:
            for agent_id in past_consensus.minority_opinion.get("agents", []):
                self.minority_accuracy[agent_id] = self.minority_accuracy.get(agent_id, 0) + 1


# ─────────────────────────────────────────────
#  Coach — 이사장 자문 + 알파 붕괴 검사
# ─────────────────────────────────────────────
class Coach:
    """
    이사장의 멘탈 관리, 목표 진행률, 알파 붕괴 검사를 담당.
    Claude API와 연동되어 주간 코칭 리포트를 자동 생성.
    """

    ALPHA_DECAY_CRITICAL = 0.50   # 샤프 50% 하락 시 긴급 경고
    ALPHA_DECAY_WARNING  = 0.70   # 샤프 30% 하락 시 경고

    def __init__(self, target_annual_return: float = 0.18):
        self.target_annual  = target_annual_return
        self.target_daily   = target_annual_return / 252
        self.session_count  = 0
        self.coach_log:list = []

    def check_alpha_decay(
        self,
        agent_id: str,
        sharpe_30d: float,
        sharpe_90d: float,
        win_rate_30d: float,
        win_rate_90d: float,
    ) -> dict:
        """
        알파 붕괴 검사.
        최근 30일 vs 90일 샤프 비율 비교.
        """
        if sharpe_90d <= 0:
            return {"status": "INSUFFICIENT_DATA", "agent": agent_id}

        degradation = (sharpe_90d - sharpe_30d) / sharpe_90d

        if degradation >= (1 - self.ALPHA_DECAY_CRITICAL):
            status = "CRITICAL"
            action = f"{agent_id}: 배분 0% 즉시 축소 + 전략 재설계 필요"
        elif degradation >= (1 - self.ALPHA_DECAY_WARNING):
            status = "WARNING"
            action = f"{agent_id}: 신호 조건 강화 또는 진입 빈도 축소 검토"
        elif win_rate_30d < win_rate_90d - 0.10:
            status = "WARNING"
            action = f"{agent_id}: 승률 {win_rate_30d:.0%} (30d) vs {win_rate_90d:.0%} (90d) — 시장 환경 변화 확인"
        else:
            status = "SAFE"
            action = "이상 없음"

        result = {
            "agent":        agent_id,
            "status":       status,
            "sharpe_30d":   round(sharpe_30d, 2),
            "sharpe_90d":   round(sharpe_90d, 2),
            "degradation":  round(degradation, 2),
            "win_rate_30d": round(win_rate_30d, 4),
            "win_rate_90d": round(win_rate_90d, 4),
            "action":       action,
            "timestamp":    datetime.now().isoformat(),
        }
        self.coach_log.append(result)
        return result

    def weekly_brief(
        self,
        weekly_pnl_pct: float,
        cumulative_pnl_pct: float,
        trade_count: int,
        win_rate: float,
        mdd_pct: float,
        manual_override_count: int = 0,
        balance_check_count: int = 0,
    ) -> dict:
        """
        주간 코칭 세션 데이터 준비.
        Claude API로 전달할 구조화 데이터.
        """
        days_elapsed = self.session_count * 7
        progress_pct = min(100, (cumulative_pnl_pct / self.target_annual) * 100)

        # 멘탈 위험 신호
        mental_risks = []
        if balance_check_count > 3:
            mental_risks.append(f"잔고 조회 {balance_check_count}회/주 (목표 1회) — 과잉 반응 위험")
        if manual_override_count > 0:
            mental_risks.append(f"수동 개입 {manual_override_count}회 — 시스템 신뢰 저하")
        if win_rate < 0.45:
            mental_risks.append(f"승률 {win_rate:.0%} — Month 2~3 위험 구간")

        self.session_count += 1

        return {
            "session":               self.session_count,
            "days_elapsed":          days_elapsed,
            "target_annual":         self.target_annual,
            "progress_pct":          round(progress_pct, 1),
            "weekly_pnl_pct":        round(weekly_pnl_pct, 4),
            "cumulative_pnl_pct":    round(cumulative_pnl_pct, 4),
            "trade_count":           trade_count,
            "win_rate":              round(win_rate, 4),
            "mdd_pct":               round(mdd_pct, 4),
            "mental_risks":          mental_risks,
            "manual_overrides":      manual_override_count,
            "balance_check_count":   balance_check_count,
            "verdict": (
                "정상 진행" if not mental_risks and weekly_pnl_pct > 0
                else "주의 필요" if mental_risks
                else "손실 구간 — 버텨야 할 시기"
            ),
        }

    def build_claude_prompt(self, brief: dict, alpha_results: list) -> str:
        """Claude API로 보낼 주간 코칭 프롬프트 생성"""
        alpha_summary = "\n".join([
            f"{r['agent']}: [{r['status']}] 샤프 {r['sharpe_30d']} (30d) vs {r['sharpe_90d']} (90d)"
            for r in alpha_results
        ])

        return f"""[주간 코칭 세션] Week {brief['session']}

목표 진행률: {brief['progress_pct']}%
이번 주 수익: {brief['weekly_pnl_pct']:.2%}
누적 수익: {brief['cumulative_pnl_pct']:.2%}
거래: {brief['trade_count']}건 | 승률: {brief['win_rate']:.0%} | MDD: {brief['mdd_pct']:.2%}

이사장 행동:
- 잔고 조회: {brief['balance_check_count']}회/주
- 수동 개입: {brief['manual_overrides']}회

멘탈 위험 신호:
{chr(10).join(brief['mental_risks']) if brief['mental_risks'] else '없음'}

알파 붕괴 검사:
{alpha_summary if alpha_summary else '데이터 부족'}

답변 형식:
## 목표 진행 평가 (3줄)
## 멘탈 점검 (위험 신호 있으면 구체적 행동 지침)
## 알파 붕괴 대응 (CRITICAL/WARNING 에이전트 있으면 수정안)
## 이번 주 딱 1가지 할 일
"""
