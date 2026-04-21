"""CIO — 자금 배분 엔진."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.core.harness import LiveAgentHarness, MarketState, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """CIO 배분 결과 1회분."""
    allocations:    dict[str, float]   # agent_id → 배분 금액
    vetoed:         list[str]          # VETO된 에이전트 ID 목록
    total_deployed: float              # 실제 투입 자금
    cash_reserve:   float              # 유보 현금
    reason:         dict[str, str]     # agent_id → 배분/거절 이유


class CIO:
    """
    7개 에이전트에 자금을 배분하는 최고투자책임자.

    점수 = EV(0.30) + market_fit(0.25) + bayes_trust(0.20)
           + risk_adj(0.15) + streak_penalty(0.10)

    5대 VETO 조건:
    V1. 시장 적합도 < 0.40
    V2. 일일 MDD 한도 초과
    V3. 포트폴리오 노출도 75% 초과
    V4. 동일 방향 에이전트 2개 이상 중복
    V5. 연속 손실 5회 이상 → 강제 휴식
    """

    MAX_AGENT_ALLOC = 0.35
    CASH_RESERVE    = 0.25
    MIN_MARKET_FIT  = 0.40
    MAX_DAILY_LOSS  = -0.008
    MAX_EXPOSURE    = 0.75
    MAX_STREAK_LOSS = 5

    def __init__(self, total_capital: float):
        self.total_capital      = total_capital
        self.deployable         = total_capital * (1 - self.CASH_RESERVE)
        self.daily_pnl_pct      = 0.0
        self.portfolio_exposure = 0.0
        self.agent_positions: dict[str, float] = {}
        self.consecutive_losses: dict[str, int] = {}

    def allocate(
        self,
        agents:  dict[str, "LiveAgentHarness"],
        signals: dict[str, "SignalResult"],
        market_state: "MarketState",
    ) -> AllocationResult:
        """에이전트별 신호와 시장 상태를 받아 자금 배분을 결정한다."""
        allocations = {}
        vetoed = []
        reasons = {}
        total_deployed = 0.0
        
        # 1. 스코어링 및 VETO 체크
        scored_agents = []
        for aid, sig in signals.items():
            if sig.signal == "NO_SIGNAL":
                continue
                
            agent = agents[aid]
            is_vetoed, reason = self.veto_check(agent, sig, market_state)
            
            if is_vetoed:
                vetoed.append(aid)
                reasons[aid] = f"VETO: {reason}"
                continue
                
            score = self.score_agent(agent, sig)
            scored_agents.append((aid, score, sig))
            
        # 2. 점수 순 정렬
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 배분
        remaining = self.deployable
        for aid, score, sig in scored_agents:
            # 켈리 기반 요청 금액
            request = sig.capital_request
            
            # 포트폴리오 제약: 최대 노출도
            if total_deployed + request > self.total_capital * self.MAX_EXPOSURE:
                request = max(0, self.total_capital * self.MAX_EXPOSURE - total_deployed)
                reasons[aid] = f"EXPOSURE LIMIT REACHED. Reduced to {request:,.0f}"
            else:
                reasons[aid] = f"Score {score:.2f} passed."
                
            if request > 0:
                allocations[aid] = request
                total_deployed += request
                remaining -= request
            else:
                if aid not in reasons:
                    reasons[aid] = "NO REMAINING EXPOSURE"

        return AllocationResult(
            allocations=allocations,
            vetoed=vetoed,
            total_deployed=total_deployed,
            cash_reserve=self.total_capital - total_deployed,
            reason=reasons,
        )

    def score_agent(
        self,
        agent:  "LiveAgentHarness",
        signal: "SignalResult",
    ) -> float:
        """에이전트 점수 계산 (0.0 ~ 1.0)."""
        ev_score = min(max(signal.ev * 100, 0), 1)  # EV 1% 면 1점
        market_score = signal.market_fit
        bayes_score = agent.bayesian_win_rate
        
        # Streak penalty
        streak = self.consecutive_losses.get(agent.agent_id, 0)
        streak_score = max(0, 1 - (streak / self.MAX_STREAK_LOSS))
        
        # Risk adjustment (MDD 기반 등 - 여기선 단순화)
        risk_score = 1.0
        
        score = (
            ev_score * 0.30 +
            market_score * 0.25 +
            bayes_score * 0.20 +
            risk_score * 0.15 +
            streak_score * 0.10
        )
        return score

    def veto_check(
        self,
        agent:  "LiveAgentHarness",
        signal: "SignalResult",
        market_state: "MarketState",
    ) -> tuple[bool, str]:
        """VETO 여부 확인. (is_vetoed, reason) 반환."""
        # V1. 시장 적합도
        if signal.market_fit < self.MIN_MARKET_FIT:
            return True, f"Low Market Fit ({signal.market_fit:.2f} < {self.MIN_MARKET_FIT})"
            
        # V2. 일일 MDD 한도
        if self.daily_pnl_pct <= self.MAX_DAILY_LOSS:
            return True, f"Daily MDD Limit Exceeded ({self.daily_pnl_pct:.2%})"
            
        # V3. 포트폴리오 노출도
        if self.portfolio_exposure >= self.MAX_EXPOSURE:
            return True, "Max Exposure Reached"
            
        # V4. 동일 방향 에이전트 중복 (여기서는 agent_positions로 체크 가능하지만 단순화)
        # signals를 다 봐야 하므로 allocate에서 처리하는게 나을 수도 있음
        
        # V5. 연속 손실
        streak = self.consecutive_losses.get(agent.agent_id, 0)
        if streak >= self.MAX_STREAK_LOSS:
            return True, f"Consecutive Losses ({streak})"
            
        return False, ""

    def reset_daily(self):
        """일일 리셋 (장 시작 전 호출)."""
        self.daily_pnl_pct      = 0.0
        self.portfolio_exposure = 0.0

    def update_pnl(self, pnl_pct: float):
        self.daily_pnl_pct += pnl_pct
