"""실시간 레버리지 디케이 방어 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class LevDecayLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="lev",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.007,
            take_profit_pct=0.015
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        f = BuyFilters.run_all(
            md, 
            l4_required=1,
            l4_custom=[f"VKOSPI {md.vkospi:.1f}"]
        )
        
        # 실시간 조건: 저변동성 (VKOSPI 18 이하) AND 60일 이평선 위 안정적 추세
        low_vol = md.vkospi <= 18
        stable_trend = md.close >= md.ma20 * 1.01 # 20일선 대비 약간 이격 (강세 확인)
        
        if f["all_passed"] and low_vol and stable_trend:
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
                target_price=self.calc_target_price(entry_price),
                stop_price=self.calc_stop_price(entry_price),
                reason="Low Volatility Stable Trend (Avoiding Decay)",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
