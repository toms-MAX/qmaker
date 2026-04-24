"""
DATA CAPITAL — agents/__init__.py
7개 라이브 트레이딩 에이전트 구현
"""

from datetime import time
from core.harness import (
    LiveAgentHarness, MarketData, MarketState,
    SignalResult, BuyFilters, NO_SIGNAL,
)


# ─────────────────────────────────────────────
#  1. 갭트레이딩 에이전트
#  전략: 갭다운(0.2~3%) 후 당일 되돌림 매수
# ─────────────────────────────────────────────
class GapTradingAgent(LiveAgentHarness):
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
        # 갭다운 방향 먼저 확인 (abs 계산 전에 방향 체크)
        if md.gap_pct >= 0:
            return NO_SIGNAL(self.agent_id, f"갭업 {md.gap_pct:.3%} — 갭다운 전략 미해당")

        gap = abs(md.gap_pct)
        if not (self.GAP_MIN <= gap <= self.GAP_MAX):
            return NO_SIGNAL(self.agent_id, f"갭 범위 이탈: {gap:.3%} (기준 0.2~3%)")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l2_override=True,           # 갭다운은 MA20 아래에서도 진입
            l3_time_override="GAP",     # 09:05~09:30 창
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        confidence = min(0.9, gap / 0.01)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.8, expected_return=0.005, max_loss=0.002,
            capital_request=self.kelly_size(confidence),
            entry_price=md.open, target_price=md.open * 1.005, stop_price=md.open * 0.998,
            reason=f"갭다운 {md.gap_pct:.3%} 되돌림 매수",
        )


# ─────────────────────────────────────────────
#  2. 평균회귀 에이전트
#  전략: RSI 과매도 + 볼린저 하단 이탈 시 매수
# ─────────────────────────────────────────────
class MeanRevAgent(LiveAgentHarness):
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
            l2_bearish_ok=True,
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        confidence = (50 - md.rsi14) / 50
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.7, expected_return=0.01, max_loss=0.005,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close, target_price=md.close * 1.01, stop_price=md.close * 0.995,
            reason=f"RSI {md.rsi14:.1f} 눌림목 매수",
        )


# ─────────────────────────────────────────────
#  3. 모멘텀 에이전트
#  전략: RSI 65+ + MA20 상승추세 돌파 추격
# ─────────────────────────────────────────────
class MomentumAgent(LiveAgentHarness):
    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="momentum",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.007,
            take_profit_pct=0.020,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        if not (md.rsi14 > 65 and md.close > md.ma20):
            return NO_SIGNAL(self.agent_id, f"모멘텀 약화 (RSI:{md.rsi14:.1f})")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        confidence = min(0.8, md.rsi14 / 100)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.9, expected_return=0.02, max_loss=0.007,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close, target_price=md.close * 1.02, stop_price=md.close * 0.993,
            reason=f"강세장 모멘텀 추격 (RSI:{md.rsi14:.1f})",
        )


# ─────────────────────────────────────────────
#  4. 페어/스프레드 회귀 에이전트
#  전략: 볼린저 하단 이탈(z-score ~-2σ) 시 평균 회귀 매수
#  단일 ETF에서 BB를 스프레드 대리 지표로 활용
# ─────────────────────────────────────────────
class PairsAgent(LiveAgentHarness):
    BB_ENTRY_BUFFER = 0.001   # 밴드 하단 대비 0.1% 여유

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="pairs",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # BB 하단 이탈 = z-score 약 -2σ (pairs 스프레드 논리 차용)
        if md.bb_lower <= 0:
            return NO_SIGNAL(self.agent_id, "BB 데이터 없음")

        entry_threshold = md.bb_lower * (1 + self.BB_ENTRY_BUFFER)
        if md.close > entry_threshold:
            return NO_SIGNAL(self.agent_id, f"BB 하단 미이탈 (종가:{md.close:.0f} > 하단:{md.bb_lower:.0f})")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l2_bearish_ok=True,    # 하단 이탈은 MA20 아래 가능
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        # 이격도 기반 신뢰도 (하단 대비 얼마나 더 빠졌는가)
        band_width = md.bb_upper - md.bb_lower
        if band_width <= 0:
            return NO_SIGNAL(self.agent_id, "BB 밴드 폭 이상")
        deviation = (entry_threshold - md.close) / band_width
        confidence = min(0.85, 0.5 + deviation * 2)

        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.75, expected_return=0.01, max_loss=0.005,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close,
            target_price=md.bb_middle,        # 목표: BB 중심선 회귀
            stop_price=md.close * 0.995,
            reason=f"BB 하단 이탈 회귀 매수 (종가:{md.close:.0f} < 하단:{md.bb_lower:.0f})",
        )


# ─────────────────────────────────────────────
#  5. 종가베팅(EOD) 에이전트
#  전략: 장중 1.2%+ 강한 상승 + 거래량 동반 시 종가 매수
#        익일 시가에 모멘텀 이어짐을 기대
# ─────────────────────────────────────────────
class EODAgent(LiveAgentHarness):
    RETURN_THRESHOLD = 0.012   # 당일 수익률 1.2% 이상
    VOL_RATIO_MIN    = 1.3     # 거래량 1.3배 이상

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="eod",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.005,
            take_profit_pct=0.010,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # 종가베팅 시간 창: 15:10~15:20
        t = md.current_time.time()
        if not (time(15, 10) <= t <= time(15, 20)):
            return NO_SIGNAL(self.agent_id, f"종가베팅 시간 아님 ({t})")

        # 당일 장중 수익률 (시가 대비 현재가)
        if md.open <= 0:
            return NO_SIGNAL(self.agent_id, "시가 데이터 없음")
        intraday_ret = (md.close - md.open) / md.open
        if intraday_ret < self.RETURN_THRESHOLD:
            return NO_SIGNAL(self.agent_id, f"장중 수익률 부족 ({intraday_ret:.2%} < {self.RETURN_THRESHOLD:.1%})")

        # 거래량 동반 확인
        if md.vol_ma5 > 0 and md.vol_ratio < self.VOL_RATIO_MIN:
            return NO_SIGNAL(self.agent_id, f"거래량 부족 ({md.vol_ratio:.1f}배 < {self.VOL_RATIO_MIN}배)")

        # MA20 위 추세 확인 (안전 장치)
        if md.ma20 > 0 and md.close < md.ma20:
            return NO_SIGNAL(self.agent_id, f"MA20 아래 — 종가베팅 제외 ({md.close:.0f} < {md.ma20:.0f})")

        confidence = min(0.80, 0.50 + intraday_ret * 10)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.85, expected_return=0.010, max_loss=0.005,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close,
            target_price=md.close * 1.010,
            stop_price=md.close * 0.995,
            reason=f"강한 종가 마감 {intraday_ret:+.2%} — 익일 모멘텀 기대",
        )


# ─────────────────────────────────────────────
#  6. 변동성/패닉 매수 에이전트
#  전략: 장중 -2% 이상 급락 시 역베팅 (공황 매수)
#        과도한 하락 후 기술적 반등 포착
# ─────────────────────────────────────────────
class VolatilityAgent(LiveAgentHarness):
    PANIC_THRESHOLD  = -0.020   # 장중 -2% 이하
    VKOSPI_MIN       = 22.0     # 공포 지수 최소 기준 (너무 낮으면 패닉 아님)

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="volatility",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.010,
            take_profit_pct=0.020,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        if md.open <= 0:
            return NO_SIGNAL(self.agent_id, "시가 데이터 없음")

        # 장중 급락 확인 (시가 대비 현재가)
        intraday_ret = (md.close - md.open) / md.open
        if intraday_ret > self.PANIC_THRESHOLD:
            return NO_SIGNAL(self.agent_id, f"패닉 기준 미달 ({intraday_ret:.2%} > {self.PANIC_THRESHOLD:.1%})")

        # VKOSPI 공포 지수 확인 (너무 잔잔하면 패닉 매수 전략 미해당)
        if md.vkospi < self.VKOSPI_MIN:
            return NO_SIGNAL(self.agent_id, f"VKOSPI {md.vkospi:.1f} < {self.VKOSPI_MIN} — 패닉 아님")

        # CRISIS 상태에서는 VKOSPI >= 30 → L1 필터가 막음
        # 여기서는 l1_override=True 사용 (패닉 매수는 의도적 역행)
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l1_override=True,       # 패닉 구간 진입 허용
            l2_bearish_ok=True,     # 하락장에서 진입
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        # 낙폭이 클수록 높은 신뢰도
        confidence = min(0.75, abs(intraday_ret) * 15)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.65, expected_return=0.020, max_loss=0.010,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close,
            target_price=md.close * 1.020,
            stop_price=md.close * 0.990,
            reason=f"패닉 매수 — 장중 {intraday_ret:.2%} 급락 (VKOSPI {md.vkospi:.1f})",
        )


# ─────────────────────────────────────────────
#  7. 레버리지 디케이 방어 에이전트
#  전략: 저변동성 구간 + 상승추세 + 강한 마감 시 진입
#        고변동성 구간 배제로 레버리지 ETF 디케이 손실 최소화
# ─────────────────────────────────────────────
class LevDecayAgent(LiveAgentHarness):
    VKOSPI_MAX        = 20.0   # 저변동성 기준
    CANDLE_CLOSE_RATIO = 0.70  # 고가 대비 종가 위치 (강한 마감)

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="lev_decay",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.007,
            take_profit_pct=0.015,
        )

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        # 저변동성 구간만 진입 (고변동성 → 디케이 위험)
        if md.vkospi >= self.VKOSPI_MAX:
            return NO_SIGNAL(self.agent_id, f"고변동성 구간 배제 (VKOSPI {md.vkospi:.1f} >= {self.VKOSPI_MAX})")

        # 중기 상승추세 확인 (MA20)
        if md.ma20 > 0 and md.close < md.ma20:
            return NO_SIGNAL(self.agent_id, f"MA20 아래 — 추세 미확인 ({md.close:.0f} < {md.ma20:.0f})")

        # 강한 마감 확인: 종가가 당일 고가 대비 상위 30% 이내
        candle_range = md.high - md.low
        if candle_range > 0:
            close_ratio = (md.close - md.low) / candle_range
            if close_ratio < self.CANDLE_CLOSE_RATIO:
                return NO_SIGNAL(self.agent_id, f"약한 마감 ({close_ratio:.0%} < {self.CANDLE_CLOSE_RATIO:.0%})")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_pnl_pct", 0.0),
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 미통과")

        # 낮은 변동성 = 높은 신뢰도 (디케이 없음)
        vkospi_score = max(0, (self.VKOSPI_MAX - md.vkospi) / self.VKOSPI_MAX)
        confidence = min(0.75, 0.50 + vkospi_score * 0.25)
        return SignalResult(
            agent_id=self.agent_id, signal="BUY", confidence=confidence,
            market_fit=0.80, expected_return=0.015, max_loss=0.007,
            capital_request=self.kelly_size(confidence),
            entry_price=md.close,
            target_price=md.close * 1.015,
            stop_price=md.close * 0.993,
            reason=f"저변동성 추세 추종 (VKOSPI {md.vkospi:.1f} — 디케이 위험 낮음)",
        )


# ─────────────────────────────────────────────
#  팩토리 함수
# ─────────────────────────────────────────────

# v1.5 Walk-Forward 재검증 + mean_rev 튜닝 결과 채택된 4개 에이전트.
# gap_trading / momentum / lev_decay 는 ETF 유니버스에서 엣지 없음 — 라이브 제외.
ADOPTED_AGENTS = ("mean_rev", "volatility", "eod", "pairs")


def create_all_agents(total_capital: float) -> dict:
    """7개 전체 에이전트 — 레거시/백테스트용."""
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


def create_adopted_agents(total_capital: float) -> dict:
    """v1.5 채택 4개 에이전트만 — 라이브/페이퍼 트레이딩용."""
    base = total_capital / len(ADOPTED_AGENTS)
    return {
        "mean_rev":   MeanRevAgent(base),
        "volatility": VolatilityAgent(base),
        "eod":        EODAgent(base),
        "pairs":      PairsAgent(base),
    }
