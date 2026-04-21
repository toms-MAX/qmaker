"""실시간 페어트레이딩(Pairs) 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class PairsLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="pairs",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # 시장 중립 지향 전략이므로 L1(공황) 오버라이드 고려
        f = BuyFilters.run_all(
            md, 
            l1_override=True, # 시장이 나빠도 가격 왜곡은 발생
            l4_required=1,
            l4_custom=[f"BB_Lower_Dist {md.close - md.bb_lower:.1f}"]
        )
        
        # 실시간 조건: 볼린저 하단 이탈 (Z-Score -2.0 수준)
        # 백테스트의 Pairs 로직을 실시간 데이터의 BB 하단 터치로 단순화
        spread_breach = md.close <= md.bb_lower
        
        if f["all_passed"] and spread_breach:
            entry_price = md.close
            confidence = 0.50
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
                target_price=md.bb_middle,
                stop_price=self.calc_stop_price(entry_price),
                reason="Price-BB Spread Breach (Oversold Distortion)",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
