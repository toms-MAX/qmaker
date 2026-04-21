"""실시간 종가베팅(EOD) 에이전트."""

from __future__ import annotations

from data_capital.core.harness import (
    LiveAgentHarness, MarketData, MarketState, SignalResult, BuyFilters, NO_SIGNAL
)


class EODLive(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="eod",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # L3_time_override="EOD": 15:10~15:20 시간대 체크
        f = BuyFilters.run_all(
            md, 
            l3_time_override="EOD",
            l4_required=1,
            l4_custom=[f"DayRet {(md.close - md.open) / md.open:.2%}"]
        )
        
        # 실시간 조건: 당일 양봉 모멘텀 유지 AND 20일 이평선 위
        strong_close = (md.close - md.open) / md.open >= 0.012
        above_ma = md.close >= md.ma20
        
        if f["all_passed"] and strong_close and above_ma:
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
                reason="Strong Daily Closing Momentum Above 20MA",
                filter_results=f
            )
            
        return NO_SIGNAL(self.agent_id)

    def get_market_fit(self, market_state: MarketState) -> float:
        return super().get_market_fit(market_state)
