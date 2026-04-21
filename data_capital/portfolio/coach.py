"""Coach — 이사장 멘탈 관리 + 알파 붕괴 검사."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.core.harness import LiveAgentHarness


@dataclass
class AgentHealth:
    """에이전트 건강 상태."""
    agent_id:        str
    alpha_decaying:  bool    # 알파 붕괴 감지
    forced_rest:     bool    # 강제 휴식 상태
    win_rate:        float   # 승률
    recommendation:  str     # "ACTIVE" | "REDUCE" | "PAUSE" | "RETIRE"


class Coach:
    """
    에이전트 장기 건강 관리자.

    역할:
    1. 알파 붕괴 감지: 30일 승률이 과거 대비 15%↓ 지속
    2. 레짐 변화 감지: 시장 구조 변화 시 파라미터 재검토 신호
    3. 강제 휴식: 연속 손실 5회 → 3일 쿨다운
    4. 은퇴 판정: 90일 누적 수익 음수 + 알파 붕괴 → 비활성화
    """

    ALPHA_DECAY_THRESHOLD = 0.15   # 승률 15%p 하락
    COOLDOWN_DAYS         = 3

    def __init__(self):
        self.agent_health: dict[str, AgentHealth] = {}
        self.resting: dict[str, int] = {}   # agent_id → 남은 휴식일

    def evaluate(self, agents: dict[str, "LiveAgentHarness"]) -> dict[str, AgentHealth]:
        """모든 에이전트 건강 상태를 평가한다."""
        healths = {}
        for aid, agent in agents.items():
            decaying = self.check_alpha_decay(agent)
            
            # 연속 손실 체크 (LiveAgentHarness에 streak 정보가 있다고 가정하거나 직접 추적)
            # 여기서는 단순화
            forced_rest = aid in self.resting
            
            if decaying:
                recommendation = "REDUCE"
            elif forced_rest:
                recommendation = "PAUSE"
            else:
                recommendation = "ACTIVE"
                
            healths[aid] = AgentHealth(
                agent_id=aid,
                alpha_decaying=decaying,
                forced_rest=forced_rest,
                win_rate=agent.win_rate,
                recommendation=recommendation
            )
        self.agent_health = healths
        return healths

    def check_alpha_decay(self, agent: "LiveAgentHarness") -> bool:
        """알파 붕괴 여부 판단."""
        # 베이지안 승률이 0.3 이하면 붕괴로 간주 (단순 예시)
        if agent.win_count + agent.loss_count > 10 and agent.bayesian_win_rate < 0.3:
            return True
        return False

    def tick_rest(self):
        """매일 장 시작 전 휴식 카운트다운."""
        self.resting = {k: v - 1 for k, v in self.resting.items() if v > 1}
