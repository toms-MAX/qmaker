"""실시간 모멘텀(Breakout) 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class MomentumLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="momentum",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.015
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # L1-L5 필터
        f = BuyFilters.run_all(
            md, 
            l4_required=2,
            l4_custom=[f"VolRatio {md.vol_ratio:.1f}x"]
        )
        
        # 실시간 조건: 정배열 (5 > 20) AND 거래량 동반한 가격 상승
        trend_up = md.ma20 > 0 and md.ma200 > 0 and md.close > md.ma20
        breakout = md.vol_ratio >= 1.3 and (md.close - md.open) / md.open >= 0.01
        
        if f["all_passed"] and trend_up and breakout:
            entry_price = md.close
            confidence = 0.55
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
                reason="Trend Alignment + Volume Spike Breakout",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
