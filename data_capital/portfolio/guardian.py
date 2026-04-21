"""Guardian — 시스템 감시 + 충돌 중재."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.portfolio.oracle import Consensus
    from data_capital.core.harness import MarketData


class AlertLevel(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
    SHUTDOWN = "SHUTDOWN"   # 전체 매매 중단


@dataclass
class GuardianAlert:
    level:   AlertLevel
    message: str
    action:  str   # 취해야 할 조치


class Guardian:
    """
    시스템 전체를 감시하는 수호자.

    역할:
    1. Oracle CONFLICT 중재 → 에이전트 신뢰도 기반 최종 결정
    2. 이상 거래 탐지 (급등락, VI, 슬리피지 과다)
    3. 일일 MDD 3단계 경고 (-0.5% / -0.8% / -1.0%)
    4. 시스템 장애 감지 및 긴급 청산 명령
    """

    MDD_WARN1    = -0.005   # -0.5%: 경고
    MDD_WARN2    = -0.008   # -0.8%: 신규 진입 금지
    MDD_SHUTDOWN = -0.010   # -1.0%: 전체 청산

    def __init__(self):
        self.alerts:        list[GuardianAlert] = []
        self.trading_halted: bool = False

    def watch(self, md: "MarketData", daily_pnl_pct: float) -> list[GuardianAlert]:
        """매 틱마다 호출. 발생한 경고 리스트를 반환한다."""
        current_alerts = []
        
        if daily_pnl_pct <= self.MDD_SHUTDOWN:
            self.trading_halted = True
            alert = GuardianAlert(AlertLevel.SHUTDOWN, f"Daily MDD {daily_pnl_pct:.2%} hit threshold {self.MDD_SHUTDOWN:.2%}", "LIQUIDATE_ALL")
            current_alerts.append(alert)
        elif daily_pnl_pct <= self.MDD_WARN2:
            alert = GuardianAlert(AlertLevel.CRITICAL, f"Daily MDD {daily_pnl_pct:.2%} deep. Entry prohibited.", "STOP_ENTRY")
            current_alerts.append(alert)
        elif daily_pnl_pct <= self.MDD_WARN1:
            alert = GuardianAlert(AlertLevel.WARNING, f"Daily MDD {daily_pnl_pct:.2%} warning.", "MONITOR_CLOSELY")
            current_alerts.append(alert)
            
        # 시장 변동성 감시 (예시)
        if md.vkospi >= 35:
            alert = GuardianAlert(AlertLevel.CRITICAL, f"Market extreme volatility (VKOSPI {md.vkospi})", "REDUCE_SIZE")
            current_alerts.append(alert)
            
        self.alerts.extend(current_alerts)
        return current_alerts

    def mediate_conflict(self, consensus: "Consensus", agent_stats: dict) -> str:
        """Oracle CONFLICT 시 최종 신호 결정. BUY/SELL/HOLD 반환."""
        # 각 에이전트의 베이지안 승률 등을 고려하여 가중 합산
        # 여기서는 단순화하여 더 많은 기여를 한 쪽을 선택하거나, 
        # 승률이 가장 높은 에이전트의 손을 들어줌
        return "HOLD" # 보수적으로 충돌 시 HOLD

    def emergency_halt(self, reason: str):
        """전체 매매 즉시 중단."""
        self.trading_halted = True
        self.alerts.append(GuardianAlert(
            level=AlertLevel.SHUTDOWN,
            message=reason,
            action="LIQUIDATE_ALL",
        ))
