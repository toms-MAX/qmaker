"""Oracle — 7개 에이전트 합의 + 집단지성."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.core.harness import SignalResult


@dataclass
class Consensus:
    """Oracle 합의 결과."""
    final_signal:   str            # BUY / SELL / HOLD / CONFLICT
    confidence:     float          # 집단 신뢰도
    agree_count:    int            # 동의 에이전트 수
    disagree_count: int            # 반대 에이전트 수
    reason:         str
    contributors:   list[str]      # 신호에 기여한 에이전트 ID


class Oracle:
    """
    7개 에이전트 신호를 합산해 집단지성 최종 신호를 만든다.

    합의 규칙:
    - 동일 방향 3개↑ + 가중 신뢰도 0.6↑  → 합의 신호
    - 방향 충돌 (롱 vs 숏 동시)           → CONFLICT → Guardian 중재
    - 1~2개 신호만 있을 때                → 단독 에이전트 신호 그대로
    - 모두 NO_SIGNAL                      → HOLD
    """

    QUORUM       = 3    # 합의에 필요한 최소 에이전트 수
    MIN_CONF     = 0.60 # 합의 신뢰도 하한

    def __init__(self):
        self.consensus_history: list[Consensus] = []

    def aggregate(self, signals: dict[str, "SignalResult"]) -> Consensus:
        """에이전트 신호 dict를 받아 합의 결과를 반환한다."""
        buys = [s for s in signals.values() if s.signal == "BUY"]
        sells = [s for s in signals.values() if s.signal == "SELL"]
        
        if not buys and not sells:
            return Consensus("HOLD", 0.0, 0, 0, "No active signals", [])
            
        if buys and sells:
            return Consensus(
                "CONFLICT", 
                0.5, 
                len(buys), 
                len(sells), 
                f"Conflict: {len(buys)} BUY vs {len(sells)} SELL", 
                list(signals.keys())
            )
            
        # 단방향 신호들만 있음
        active_sigs = buys if buys else sells
        direction = "BUY" if buys else "SELL"
        
        avg_conf = sum(s.confidence for s in active_sigs) / len(active_sigs)
        contributors = [s.agent_id for s in active_sigs]
        
        if len(active_sigs) >= self.QUORUM and avg_conf >= self.MIN_CONF:
            reason = f"Consensus reached by {len(active_sigs)} agents"
        else:
            reason = f"Minority signal(s): {len(active_sigs)} agents"
            
        return Consensus(direction, avg_conf, len(active_sigs), 0, reason, contributors)

    def detect_conflict(self, signals: dict[str, "SignalResult"]) -> bool:
        """롱/숏 충돌 여부 확인."""
        has_buy = any(s.signal == "BUY" for s in signals.values())
        has_sell = any(s.signal == "SELL" for s in signals.values())
        return has_buy and has_sell
