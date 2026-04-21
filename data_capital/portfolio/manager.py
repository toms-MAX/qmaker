"""PortfolioManager — CIO, Oracle, Guardian, Coach를 총괄하는 지휘소."""

from __future__ import annotations

from typing import TYPE_CHECKING
from data_capital.portfolio.cio import CIO
from data_capital.portfolio.oracle import Oracle
from data_capital.portfolio.guardian import Guardian
from data_capital.portfolio.coach import Coach

if TYPE_CHECKING:
    from data_capital.core.harness import LiveAgentHarness, MarketData, SignalResult


class PortfolioManager:
    """
    포트폴리오 운영의 총 책임자.
    
    워크플로우:
    1. Coach가 에이전트들의 건강 상태를 점검하여 활동 여부 결정.
    2. 에이전트들이 생성한 신호를 Oracle이 수집하여 집단지성 합의 도출.
    3. Guardian이 시장 상황과 합의 신호의 충돌 여부를 감시.
    4. CIO가 최종적으로 가용 자산을 배분하여 주문 실행 가이드 생성.
    """

    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.cio = CIO(total_capital)
        self.oracle = Oracle()
        self.guardian = Guardian()
        self.coach = Coach()

    def run_cycle(
        self, 
        md: MarketData, 
        agents: dict[str, LiveAgentHarness], 
        signals: dict[str, SignalResult]
    ):
        """매매 사이클 실행 (매 틱 또는 봉 완성 시)"""
        
        # 1. 건강 검진 (Coach)
        healths = self.coach.evaluate(agents)
        
        # 2. 신호 합의 (Oracle)
        consensus = self.oracle.aggregate(signals)
        
        # 3. 리스크 감시 (Guardian)
        alerts = self.guardian.watch(md, self.cio.daily_pnl_pct)
        
        # 4. 자금 배분 (CIO)
        # Guardian이 SHUTDOWN이면 배분 금지
        if self.guardian.trading_halted:
            return None
            
        allocation = self.cio.allocate(agents, signals, md.market_state)
        
        return {
            "consensus": consensus,
            "alerts": alerts,
            "allocation": allocation,
            "healths": healths
        }

    def reset_daily(self):
        """장 시작 전 초기화"""
        self.cio.reset_daily()
        self.coach.tick_rest()
