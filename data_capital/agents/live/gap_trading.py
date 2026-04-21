"""실시간 갭트레이딩 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class GapTradingLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="gap",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # L1~L5 필터 통합 실행
        f = BuyFilters.run_all(
            md, 
            l3_time_override="GAP",
            l4_custom=[f"GAP {md.gap_pct:.2%}"]
        )
        
        # 갭다운 조건 (시가 기준 -0.3% ~ -2.5%)
        gap_ok = -0.025 <= md.gap_pct <= -0.003
        
        if f["all_passed"] and gap_ok:
            entry_price = md.close
            confidence = 0.65 # 과거 승률 기반 설정
            size = self.kelly_size(confidence)
            
            return SignalResult(
                agent_id=self.agent_id,
                signal="BUY",
                confidence=confidence,
                market_fit=self.get_market_fit(md.market_state),
                expected_return=self.take_profit_pct,
                max_loss=self.stop_loss_pct,
                capital_request=size,
                entry_price=entry_price,
                target_price=self.calc_target_price(entry_price),
                stop_price=self.calc_stop_price(entry_price),
                reason=f"L1-L5 Pass + GapDown {md.gap_pct:.2%}",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id, reason=str(f))

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
