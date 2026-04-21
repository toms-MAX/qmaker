"""실시간 변동성(Panic Buy) 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class VolatilityLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="volatility",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.010,
            take_profit_pct=0.020
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # L1 override (위기 상황 진입 전략이므로 L1 시장 필터 무시)
        f = BuyFilters.run_all(
            md, 
            l1_override=True,
            l2_bearish_ok=True, # 급락 중 매수
            l4_required=1,      # 급락 자체가 강력한 신호
            l4_custom=[f"PanicRet {md.gap_pct:.2%}"]
        )
        
        # 장중 급락 조건 (-2% 이하)
        panic_ok = (md.close - md.open) / md.open <= -0.02
        
        if f["all_passed"] and panic_ok:
            entry_price = md.close
            confidence = 0.70
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
                reason=f"Panic Buy Triggered ({md.gap_pct:.2%})",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
