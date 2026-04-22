"""
DATA CAPITAL — agents/__init__.py
=========================
7개 매매 에이전트 전체 구현 (오류 수정 및 공격적 튜닝)
"""

from datetime import time
from typing import Optional
from core.harness import (
    LiveAgentHarness as AgentHarness, MarketData, MarketState,
    SignalResult, BuyFilters, NO_SIGNAL,
)


# ─────────────────────────────────────────────
#  1. 갭트레이딩 에이전트
# ─────────────────────────────────────────────
class GapTradingAgent(AgentHarness):
    GAP_MIN = 0.002   
    GAP_MAX = 0.030   
    
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="gap_trading",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.002,     
            take_profit_pct=0.005,    
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        gap = abs(md.gap_pct)
        if not (self.GAP_MIN <= gap <= self.GAP_MAX):
            return NO_SIGNAL(self.agent_id, f"갭 부족: {gap:.3%}")

        # 인자 맵핑: daily_pnl_pct -> daily_loss_pct
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l2_override=True, l3_time_override="GAP", l4_required=1
        )

        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        # 갭다운만 BUY (갭업 숏은 ETF 특성상 스킵)
        if md.gap_pct >= 0:
            return NO_SIGNAL(self.agent_id, f"갭업 {md.gap_pct:.3%} — 갭다운 전략 미해당")

        confidence = min(0.9, gap / 0.01)

        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.8, expected_return=0.005, max_loss=0.002,
            capital_request=self.kelly_size(confidence),
            entry_price=md.open, target_price=md.open * 1.005, stop_price=md.open * 0.998,
            reason=f"갭다운 {md.gap_pct:.3%} 되돌림 매수"
        )


# ─────────────────────────────────────────────
#  2. 평균회귀 에이전트
# ─────────────────────────────────────────────
class MeanRevAgent(AgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="mean_rev",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        if md.rsi14 >= 45:
            return NO_SIGNAL(self.agent_id, f"RSI {md.rsi14:.1f} (기준 45 미만)")

        filters = BuyFilters.run_all(
            md, 
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l2_bearish_ok=True, l4_required=1
        )

        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        confidence = (50 - md.rsi14) / 50
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.7, expected_return=0.01, max_loss=0.005,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close, target_price=md.close * 1.01, stop_price=md.close * 0.995,
            reason=f"RSI {md.rsi14:.1f} 눌림목 매수"
        )


# ─────────────────────────────────────────────
#  3. 모멘텀 에이전트
# ─────────────────────────────────────────────
class MomentumAgent(AgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="momentum",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.007,
            take_profit_pct=0.020,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        is_strong = md.rsi14 > 65 and md.close > md.ma20
        
        if not is_strong:
            return NO_SIGNAL(self.agent_id, f"모멘텀 약화 (RSI:{md.rsi14:.1f})")

        filters = BuyFilters.run_all(
            md, 
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l4_required=1
        )

        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        confidence = min(0.8, md.rsi14 / 100)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.9, expected_return=0.02, max_loss=0.007,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close, target_price=md.close * 1.02, stop_price=md.close * 0.993,
            reason=f"강세장 모멘텀 추격 (RSI:{md.rsi14:.1f})"
        )

class PairsAgent(AgentHarness):
    def __init__(self, c: float): super().__init__("pairs", c)
    def generate_signal(self, md, **ctx): return NO_SIGNAL(self.agent_id, "준비 중")

class EODAgent(AgentHarness):
    def __init__(self, c: float): super().__init__("eod", c)
    def generate_signal(self, md, **ctx): return NO_SIGNAL(self.agent_id, "준비 중")

class VolatilityAgent(AgentHarness):
    def __init__(self, c: float): super().__init__("volatility", c)
    def generate_signal(self, md, **ctx): return NO_SIGNAL(self.agent_id, "준비 중")

class LevDecayAgent(AgentHarness):
    def __init__(self, c: float): super().__init__("lev_decay", c)
    def generate_signal(self, md, **ctx): return NO_SIGNAL(self.agent_id, "준비 중")


def create_all_agents(total_capital: float) -> dict:
    base = total_capital / 7
    return {
        "gap_trading": GapTradingAgent(base),
        "mean_rev":    MeanRevAgent(base),
        "momentum":    MomentumAgent(base),
        "pairs":       PairsAgent(base),
        "eod":         EODAgent(base),
        "volatility":  VolatilityAgent(base),
        "lev_decay":   LevDecayAgent(base),
    }
