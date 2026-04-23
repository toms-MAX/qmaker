"""Guardian — 시스템 감시 + 충돌 중재."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_capital.portfolio.oracle import Consensus
    from data_capital.core.harness import MarketData

logger = logging.getLogger(__name__)


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
    """

    MDD_WARN1    = -0.005   
    MDD_WARN2    = -0.008   
    MDD_SHUTDOWN = -0.010   
    MAX_SLIPPAGE = 0.002    # 0.2% 이상 슬리피지 발생 시 경고

    def __init__(self):
        self.alerts:        list[GuardianAlert] = []
        self.trading_halted: bool = False

    def watch(self, md: "MarketData", daily_pnl_pct: float, last_execution: dict | None = None) -> list[GuardianAlert]:
        """매 틱마다 호출. 발생한 경고 리스트를 반환한다."""
        current_alerts = []
        
        # 1. 일일 MDD 감시
        if daily_pnl_pct <= self.MDD_SHUTDOWN:
            self.trading_halted = True
            current_alerts.append(GuardianAlert(AlertLevel.SHUTDOWN, f"Daily MDD {daily_pnl_pct:.2%} hit SHUTDOWN", "LIQUIDATE_ALL"))
        elif daily_pnl_pct <= self.MDD_WARN2:
            current_alerts.append(GuardianAlert(AlertLevel.CRITICAL, f"Daily MDD {daily_pnl_pct:.2%} hit CRITICAL", "STOP_ENTRY"))
        elif daily_pnl_pct <= self.MDD_WARN1:
            current_alerts.append(GuardianAlert(AlertLevel.WARNING, f"Daily MDD {daily_pnl_pct:.2%} hit WARNING", "REDUCE_SIZE_50"))

        # 2. 시장 변동성(Panic) 감시
        if md.vkospi >= 30:
            current_alerts.append(GuardianAlert(AlertLevel.WARNING, f"Extreme Volatility (VKOSPI {md.vkospi:.1f})", "TIGHTEN_STOP"))

        # 3. 슬리피지 감시 (체결 데이터가 있을 경우)
        if last_execution:
            expected = last_execution.get("expected_price", 0)
            actual = last_execution.get("actual_price", 0)
            if expected > 0:
                slippage = abs(actual - expected) / expected
                if slippage > self.MAX_SLIPPAGE:
                    current_alerts.append(GuardianAlert(AlertLevel.CRITICAL, f"High Slippage Detected: {slippage:.2%}", "SWITCH_TO_LIMIT_ORDER"))

        self.alerts.extend(current_alerts)
        return current_alerts

    def mediate_conflict(self, consensus: "Consensus", agents: dict) -> str:
        """
        Oracle CONFLICT 시 최종 신호 결정.
        BUY 진영과 SELL 진영의 베이지안 승률 합산을 비교하여
        우세한 진영의 신호를 채택한다. 팽팽하면 보수적 HOLD.
        """
        buy_score  = 0.0
        sell_score = 0.0

        for aid in consensus.buy_agents:
            agent = agents.get(aid)
            if agent is not None:
                buy_score += getattr(agent, "bayesian_win_rate", 0.5)

        for aid in consensus.sell_agents:
            agent = agents.get(aid)
            if agent is not None:
                sell_score += getattr(agent, "bayesian_win_rate", 0.5)

        MARGIN = 0.15  # 15% 이상 차이가 있어야 진영 채택
        if buy_score > sell_score + MARGIN:
            logger.info("Guardian 중재: BUY 채택 (buy=%.2f sell=%.2f)", buy_score, sell_score)
            return "BUY"
        if sell_score > buy_score + MARGIN:
            logger.info("Guardian 중재: SELL 채택 (buy=%.2f sell=%.2f)", buy_score, sell_score)
            return "SELL"

        logger.info("Guardian 중재: HOLD 유지 (buy=%.2f sell=%.2f — 팽팽)", buy_score, sell_score)
        return "HOLD"

    def emergency_halt(self, reason: str):
        self.trading_halted = True
        self.alerts.append(GuardianAlert(AlertLevel.SHUTDOWN, reason, "LIQUIDATE_ALL"))
