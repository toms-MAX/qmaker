"""실시간 평균회귀(Mean Reversion) 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class MeanRevLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="mean_rev",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.015
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # L2_bearish_ok=True: 하락장에서의 반등을 노리는 전략
        f = BuyFilters.run_all(
            md, 
            l2_bearish_ok=True,
            l4_required=2,
            l4_custom=[f"RSI {md.rsi14:.1f}", f"BB_Dist {(md.close - md.bb_lower) / md.bb_lower:.2%}"]
        )
        
        # 실시간 조건: RSI 30 이하 AND 볼린저 하단 이탈
        oversold = md.rsi14 <= 30 and md.close <= md.bb_lower
        
        if f["all_passed"] and oversold:
            entry_price = md.close
            confidence = 0.60
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
                target_price=md.bb_middle if md.bb_middle > entry_price else self.calc_target_price(entry_price),
                stop_price=self.calc_stop_price(entry_price),
                reason=f"Oversold (RSI {md.rsi14:.1f}) + BB Lower hit",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
