"""Coach — 이사장 멘탈 관리 + 알파 붕괴 검사."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.core.harness import LiveAgentHarness

logger = logging.getLogger(__name__)


@dataclass
class AgentHealth:
    """에이전트 건강 상태."""
    agent_id:        str
    alpha_decaying:  bool    
    forced_rest:     bool    
    win_rate:        float   
    bayes_lower:     float   
    recommendation:  str     # "ACTIVE" | "REDUCE" | "PAUSE" | "RETIRE"


class Coach:
    """
    에이전트 장기 건강 관리자.
    """

    ALPHA_DECAY_THRESHOLD = 0.35   # 베이지안 하한 승률이 35% 미만이면 위험
    COOLDOWN_DAYS         = 3

    def __init__(self):
        self.agent_health: dict[str, AgentHealth] = {}
        self.resting: dict[str, int] = {}   # agent_id → 남은 휴식일

    def evaluate(self, agents: dict[str, "LiveAgentHarness"]) -> dict[str, AgentHealth]:
        """모든 에이전트 건강 상태를 평가한다."""
        healths = {}
        for aid, agent in agents.items():
            bayes_lower = agent.bayesian_win_rate
            decaying = bayes_lower < self.ALPHA_DECAY_THRESHOLD and (agent.win_count + agent.loss_count) >= 10
            
            # 강제 휴식 관리
            if aid in self.resting:
                recommendation = "PAUSE"
            elif decaying:
                recommendation = "REDUCE"
            else:
                recommendation = "ACTIVE"
                
            healths[aid] = AgentHealth(
                agent_id=aid,
                alpha_decaying=decaying,
                forced_rest=aid in self.resting,
                win_rate=agent.win_rate,
                bayes_lower=bayes_lower,
                recommendation=recommendation
            )
        self.agent_health = healths
        return healths

    def check_and_assign_rest(self, agent_id: str, consecutive_losses: int):
        """연속 손실 발생 시 휴식 부여."""
        if consecutive_losses >= 5:
            self.resting[agent_id] = self.COOLDOWN_DAYS
            logger.warning(f"Agent {agent_id} assigned {self.COOLDOWN_DAYS} days rest due to streak loss.")

    def tick_rest(self):
        """매일 장 시작 전 휴식 카운트다운."""
        new_resting = {}
        for aid, days in self.resting.items():
            if days > 1:
                new_resting[aid] = days - 1
            else:
                logger.info(f"Agent {aid} rest period finished.")
        self.resting = new_resting
