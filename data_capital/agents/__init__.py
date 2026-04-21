"""
DATA CAPITAL — agents/__init__.py
=========================
7개 매매 에이전트 전체 구현
1. GapTradingAgent    — 갭트레이딩
2. MeanRevAgent       — 평균회귀
3. MomentumAgent      — 모멘텀
4. PairsAgent         — 페어트레이딩
5. EODAgent           — 종가베팅
6. VolatilityAgent    — 변동성
7. LevDecayAgent      — 레버리지감쇠
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
    """
    전략: 시가갭 되돌림 (09:05~09:30 집중)
    승률: 70~74%
    손절: -0.15%
    타임컷: 10:00 강제 청산
    """

    GAP_MIN = 0.003   # 0.3% 최소 갭
    GAP_MAX = 0.020   # 2.0% 최대 갭 (뉴스갭 배제)
    NEWS_KEYWORDS = [
        "어닝서프라이즈", "실적발표", "공시", "상한가", "하한가",
        "유상증자", "워크아웃", "상장폐지", "합병", "분할",
        "감자", "자사주", "전환사채", "신주인수권", "스팩",
    ]

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="gap_trading",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.0015,     # -0.15% (좁은 손절)
            take_profit_pct=0.003,    # +0.3%
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.82,
            MarketState.BEAR_TREND:    0.65,
            MarketState.LOW_VOL_SIDE:  0.55,
            MarketState.HIGH_VOL_SIDE: 0.78,
            MarketState.REBOUND:       0.70,
            MarketState.CRISIS:        0.20,  # 공황 시 갭 예측 불가
        }
        return fit_map.get(market_state, 0.5)

    def _is_news_gap(self, news_text: str = "") -> bool:
        """뉴스갭 여부 확인"""
        return any(kw in news_text for kw in self.NEWS_KEYWORDS)

    def generate_signal(self, md: MarketData, news_text: str = "", **context) -> SignalResult:
        # ── 고유 조건 확인 먼저 ──
        gap = abs(md.gap_pct)

        if not (self.GAP_MIN <= gap <= self.GAP_MAX):
            return NO_SIGNAL(self.agent_id, f"갭 크기 {gap:.3%} — 범위 밖 ({self.GAP_MIN:.1%}~{self.GAP_MAX:.1%})")

        if self._is_news_gap(news_text):
            return NO_SIGNAL(self.agent_id, "뉴스갭 감지 — 진입 금지")

        if md.vkospi >= 25:
            return NO_SIGNAL(self.agent_id, f"VKOSPI {md.vkospi} >= 25 — 공포장 갭 불안정")

        # ── 5단계 필터 (L2, L3 특수 적용) ──
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l2_override=True,       # 갭트레이딩은 역추세 — L2 면제
            l3_time_override="GAP", # 09:05~09:30 전용 윈도우
            l4_required=1,          # 갭 자체가 신호
            l4_custom=[f"갭 {md.gap_pct:.3%}"],
        )

        if not filters["all_passed"]:
            failed = [k for k, v in filters.items() if isinstance(v, type(BuyFilters.L1_market(md))) and not v.passed]
            return NO_SIGNAL(self.agent_id, f"필터 실패: {failed}")

        # ── 신호 생성 ──
        direction = "BUY" if md.gap_pct < 0 else "SELL"
        entry     = md.open
        confidence = min(0.85, gap / 0.015)
        market_fit = self.get_market_fit(md.market_state)
        size       = self.kelly_size(confidence)

        return SignalResult(
            agent_id=self.agent_id,
            signal=direction,
            confidence=confidence,
            market_fit=market_fit,
            expected_return=self.take_profit_pct,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=entry,
            target_price=self.calc_target_price(entry),
            stop_price=self.calc_stop_price(entry),
            reason=(
                f"갭 {md.gap_pct:.3%} 되돌림 기대 | "
                f"VKOSPI {md.vkospi} | 신뢰도 {confidence:.0%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  2. 평균회귀 에이전트
# ─────────────────────────────────────────────
class MeanRevAgent(AgentHarness):
    """
    전략: RSI+볼린저 과매도 역매수 (3단계 분할)
    승률: 63~67%
    손절: -0.2%
    타임컷: 3일
    """

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="mean_rev",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.002,
            take_profit_pct=0.004,
        )
        self.split_stage = 1  # 현재 분할 단계

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.60,
            MarketState.BEAR_TREND:    0.50,
            MarketState.LOW_VOL_SIDE:  0.80,
            MarketState.HIGH_VOL_SIDE: 0.65,
            MarketState.REBOUND:       0.90,
            MarketState.CRISIS:        0.30,
        }
        return fit_map.get(market_state, 0.5)

    def generate_signal(self, md: MarketData, high_52w: float = 0, low_52w: float = 0, **context) -> SignalResult:
        # ── 고유 조건 ──
        if md.rsi14 >= 30:
            return NO_SIGNAL(self.agent_id, f"RSI {md.rsi14:.1f} >= 30 — 과매도 아님")

        if md.ma200 > 0 and md.close < md.ma200 * 0.95:
            return NO_SIGNAL(self.agent_id, "종가 < 200일MA × 0.95 — 장기 하락추세")

        if low_52w > 0 and md.close <= low_52w * 1.05:
            return NO_SIGNAL(self.agent_id, "52주 최저가 근처 — 바닥 불확실")

        if md.vol_ratio < 1.3:
            return NO_SIGNAL(self.agent_id, f"거래량 {md.vol_ratio:.1f}배 — 130% 미달")

        # ── 5단계 필터 (L2 완화) ──
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l2_bearish_ok=True,   # 평균회귀는 20일MA 하방도 OK
            l4_required=2,
            l4_custom=[f"RSI {md.rsi14:.1f}", "볼린저 하단" if md.close <= md.bb_lower else ""],
        )

        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 실패")

        # ── 분할 단계 결정 ──
        if md.rsi14 <= 20:
            stage = 3
            size_pct = 0.34
        elif md.rsi14 <= 25:
            stage = 2
            size_pct = 0.33
        else:
            stage = 1
            size_pct = 0.33

        confidence = (30 - md.rsi14) / 30
        entry      = md.close
        size       = self.allocated_capital * size_pct * self.kelly_fraction

        return SignalResult(
            agent_id=self.agent_id,
            signal="BUY",
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=self.take_profit_pct,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=entry,
            target_price=md.bb_middle,   # 볼린저 중심선 회귀 시 익절
            stop_price=self.calc_stop_price(entry),
            reason=(
                f"분할 {stage}단계 | RSI {md.rsi14:.1f} 과매도 | "
                f"볼린저 하단 {md.close <= md.bb_lower} | 신뢰도 {confidence:.0%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  3. 모멘텀 에이전트
# ─────────────────────────────────────────────
class MomentumAgent(AgentHarness):
    """
    전략: 저항선 돌파 추격 + ATR 포지션 사이징
    승률: 48~54% (손익비 3:1)
    손절: ATR × 1
    익절: ATR × 3
    """

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="momentum",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.003,
            take_profit_pct=0.010,
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.90,
            MarketState.BEAR_TREND:    0.15,
            MarketState.LOW_VOL_SIDE:  0.30,
            MarketState.HIGH_VOL_SIDE: 0.40,
            MarketState.REBOUND:       0.55,
            MarketState.CRISIS:        0.05,
        }
        return fit_map.get(market_state, 0.5)

    def generate_signal(self, md: MarketData, high_20d: float = 0, **context) -> SignalResult:
        # 모멘텀은 상승장에서만
        if md.vkospi >= 22:
            return NO_SIGNAL(self.agent_id, f"VKOSPI {md.vkospi} >= 22 — 모멘텀 취약")
        if md.close < md.ma20:
            return NO_SIGNAL(self.agent_id, "종가 < 20일MA — 상승추세 아님")
        if not high_20d:
            return NO_SIGNAL(self.agent_id, "20일 최고가 데이터 없음")

        # 돌파 확인
        breakout = md.close > high_20d * 1.002
        if not breakout:
            return NO_SIGNAL(self.agent_id, f"저항선 돌파 미확인 (20일 고가: {high_20d:.0f})")

        # 거래량 200% 이상
        if md.vol_ratio < 2.0:
            return NO_SIGNAL(self.agent_id, f"거래량 {md.vol_ratio:.1f}배 — 200% 미달")

        # 프로그램 매매 조기 급등 제외
        t = md.current_time.time()
        if t < time(9, 30) and md.vol_ratio > 5.0:
            return NO_SIGNAL(self.agent_id, "장초 프로그램 매매 의심 — 진입 금지")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l4_required=2,
            l4_custom=["저항선 돌파", f"거래량 {md.vol_ratio:.1f}배"],
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 실패")

        # ATR 기반 포지션 사이징
        atr = md.atr14 if md.atr14 > 0 else md.close * 0.005
        risk_amount  = self.allocated_capital * 0.01
        shares       = int(risk_amount / atr)
        size         = shares * md.close
        confidence   = min(0.85, md.vol_ratio / 4)

        return SignalResult(
            agent_id=self.agent_id,
            signal="BUY",
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=atr * 3 / md.close,   # ATR × 3
            max_loss=atr / md.close,               # ATR × 1
            capital_request=size,
            entry_price=md.close,
            target_price=md.close + atr * 3,
            stop_price=md.close - atr,
            reason=(
                f"20일 저항선 {high_20d:.0f} 돌파 | "
                f"거래량 {md.vol_ratio:.1f}배 | ATR {atr:.0f} | 신뢰도 {confidence:.0%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  4. 페어트레이딩 에이전트
# ─────────────────────────────────────────────
class PairsAgent(AgentHarness):
    """
    전략: 현물-선물 베이시스 차익거래 (시장중립)
    승률: 68~72%
    손절: -0.15%
    타임컷: 당일 15:15
    """

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="pairs",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.0015,
            take_profit_pct=0.004,
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        # 시장중립 전략 — 모든 장세에서 일정
        fit_map = {
            MarketState.BULL_TREND:    0.85,
            MarketState.BEAR_TREND:    0.85,
            MarketState.LOW_VOL_SIDE:  0.90,
            MarketState.HIGH_VOL_SIDE: 0.80,
            MarketState.REBOUND:       0.80,
            MarketState.CRISIS:        0.50,
        }
        return fit_map.get(market_state, 0.8)

    def generate_signal(
        self,
        md: MarketData,
        basis_pct: float = 0.0,       # 현물-선물 베이시스
        halflife_days: float = 0.0,   # OU 반감기
        coint_pvalue: float = 1.0,    # 공적분 p-value
        **context
    ) -> SignalResult:
        # ── 고유 조건 ──
        if coint_pvalue >= 0.05:
            return NO_SIGNAL(self.agent_id, f"공적분 p-value {coint_pvalue:.3f} >= 0.05")
        if not (1.0 <= halflife_days <= 3.0):
            return NO_SIGNAL(self.agent_id, f"반감기 {halflife_days:.1f}일 — 범위 밖 (1~3일)")
        if abs(basis_pct) < 0.0008:
            return NO_SIGNAL(self.agent_id, f"베이시스 {basis_pct:.4%} < 0.08% — 수익성 부족")

        # L1만 적용 (시장중립이라 L2 면제)
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l1_override=False,
            l2_override=True,   # 시장중립 — L2 면제
            l3_time_override=None,
            l4_required=1,
            l4_custom=[f"베이시스 {basis_pct:.4%}"],
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 실패")

        confidence = min(0.90, abs(basis_pct) / 0.002 * (3 / halflife_days))
        size       = self.kelly_size(confidence)
        direction  = "BUY" if basis_pct < 0 else "SELL"  # 스프레드 수렴 방향

        return SignalResult(
            agent_id=self.agent_id,
            signal=direction,
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=abs(basis_pct) * 0.5,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=md.close,
            target_price=md.close * (1 + abs(basis_pct) * 0.5) if direction == "BUY" else md.close * (1 - abs(basis_pct) * 0.5),
            stop_price=self.calc_stop_price(md.close),
            reason=(
                f"베이시스 {basis_pct:.4%} | 반감기 {halflife_days:.1f}일 | "
                f"공적분 p={coint_pvalue:.3f} | 신뢰도 {confidence:.0%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  5. 종가베팅 에이전트
# ─────────────────────────────────────────────
class EODAgent(AgentHarness):
    """
    전략: 15:10 기관 리밸런싱 포착
    승률: 62~66%
    손절: -0.2%
    타임컷: 15:20 강제 청산
    """

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="eod",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.002,
            take_profit_pct=0.0025,
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.85,
            MarketState.BEAR_TREND:    0.45,
            MarketState.LOW_VOL_SIDE:  0.75,
            MarketState.HIGH_VOL_SIDE: 0.65,
            MarketState.REBOUND:       0.70,
            MarketState.CRISIS:        0.10,
        }
        return fit_map.get(market_state, 0.6)

    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l3_time_override="EOD",   # 15:10~15:20 전용
            l4_required=1,
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 실패 (시간 또는 조건)")

        # 기관 + 프로그램 방향 일치
        kospi_up       = md.kospi_change > 0
        inst_buying    = md.institutional_net > 0
        prog_buying    = md.program_trade == "BUY"

        if not (inst_buying and prog_buying and kospi_up):
            return NO_SIGNAL(
                self.agent_id,
                f"기관순매수={inst_buying}, 프로그램BUY={prog_buying}, 코스피상승={kospi_up}"
            )

        confidence = 0.65
        size       = self.kelly_size(confidence)

        return SignalResult(
            agent_id=self.agent_id,
            signal="BUY",
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=self.take_profit_pct,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=md.close,
            target_price=self.calc_target_price(md.close),
            stop_price=self.calc_stop_price(md.close),
            reason=(
                f"기관순매수 {md.institutional_net:.0f}억 | "
                f"프로그램 {md.program_trade} | 코스피 {md.kospi_change:+.2f}%"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  6. 변동성 에이전트
# ─────────────────────────────────────────────
class VolatilityAgent(AgentHarness):
    """
    전략: VKOSPI 25+ 공황 역베팅 + 분할매수
    승률: 60~65%
    손절: -0.3%
    타임컷: 40일
    """

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="volatility",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.003,
            take_profit_pct=0.007,
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.20,
            MarketState.BEAR_TREND:    0.80,
            MarketState.LOW_VOL_SIDE:  0.10,
            MarketState.HIGH_VOL_SIDE: 0.60,
            MarketState.REBOUND:       0.50,
            MarketState.CRISIS:        0.92,
        }
        return fit_map.get(market_state, 0.4)

    def generate_signal(
        self,
        md: MarketData,
        vkospi_3d: Optional[list] = None,  # 최근 3일 VKOSPI
        split_stage: int = 1,              # 현재 분할 단계 (1~3)
        **context
    ) -> SignalResult:
        if md.vkospi <= 25:
            return NO_SIGNAL(self.agent_id, f"VKOSPI {md.vkospi} <= 25")

        # 3일 연속 상승 확인
        if vkospi_3d and len(vkospi_3d) >= 3:
            rising_3d = all(vkospi_3d[i] < vkospi_3d[i+1] for i in range(len(vkospi_3d)-1))
            if not rising_3d:
                return NO_SIGNAL(self.agent_id, "VKOSPI 3일 연속 상승 미확인")

            # 속도 둔화 확인 (고점 근접)
            speed_now  = vkospi_3d[-1] - vkospi_3d[-2]
            speed_prev = vkospi_3d[-2] - vkospi_3d[-3]
            if speed_now >= speed_prev:
                return NO_SIGNAL(self.agent_id, "VKOSPI 상승 속도 아직 가속 중 — 고점 미확인")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l1_override=True,   # 공황 역베팅이므로 L1 완화
            l2_override=True,   # 하락장 역추세이므로 L2 면제
            l4_required=1,
            l4_custom=[f"VKOSPI {md.vkospi:.1f} 공황 구간"],
        )
        if not filters["L5"]:
            return NO_SIGNAL(self.agent_id, "L5 리스크 필터 실패")

        confidence = min(0.85, (md.vkospi - 25) / 15)
        size       = self.kelly_size(confidence) * 0.33   # 분할 1/3씩

        return SignalResult(
            agent_id=self.agent_id,
            signal="BUY",
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=self.take_profit_pct,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=md.close,
            target_price=self.calc_target_price(md.close),
            stop_price=self.calc_stop_price(md.close),
            reason=(
                f"VKOSPI {md.vkospi:.1f} 공황 역베팅 | "
                f"분할 {split_stage}단계 | 신뢰도 {confidence:.0%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  7. 레버리지감쇠 에이전트
# ─────────────────────────────────────────────
class LevDecayAgent(AgentHarness):
    """
    전략: 레버리지ETF 고변동 구간 감쇠 포착
    승률: 60~64%
    손절: -0.2%
    타임컷: 3일
    세금: 배당소득세 15.4% 반영
    """

    TAX_RATE = 0.154  # 파생형 ETF 배당소득세

    def __init__(self, allocated_capital: float):
        super().__init__(
            agent_id="lev_decay",
            allocated_capital=allocated_capital,
            stop_loss_pct=0.002,
            take_profit_pct=0.004 * (1 - 0.154),  # 세후 수익 기준
        )

    def get_market_fit(self, market_state: MarketState) -> float:
        fit_map = {
            MarketState.BULL_TREND:    0.40,
            MarketState.BEAR_TREND:    0.40,
            MarketState.LOW_VOL_SIDE:  0.20,
            MarketState.HIGH_VOL_SIDE: 0.85,
            MarketState.REBOUND:       0.50,
            MarketState.CRISIS:        0.30,
        }
        return fit_map.get(market_state, 0.4)

    def generate_signal(
        self,
        md: MarketData,
        lev_decay_pct: float = 0.0,    # 레버리지 감쇠율
        lev_gap_ratio: float = 0.0,    # KODEX200 대비 레버리지 괴리율
        **context
    ) -> SignalResult:
        if not (20 <= md.vkospi <= 30):
            return NO_SIGNAL(self.agent_id, f"VKOSPI {md.vkospi} — 고변동횡보 범위 아님 (20~30)")

        if lev_decay_pct < 0.005:
            return NO_SIGNAL(self.agent_id, f"레버리지 감쇠율 {lev_decay_pct:.3%} — 0.5% 미만")

        filters = BuyFilters.run_all(
            md,
            daily_loss_pct=context.get("daily_loss_pct", 0),
            portfolio_exposure=context.get("portfolio_exposure", 0),
            same_direction_agents=context.get("same_direction_agents", 0),
            consecutive_losses=context.get("consecutive_losses", 0),
            l2_override=True,   # 방향성 무관 전략
            l4_required=1,
            l4_custom=[f"감쇠율 {lev_decay_pct:.3%}"],
        )
        if not filters["all_passed"]:
            return NO_SIGNAL(self.agent_id, "필터 실패")

        confidence = min(0.75, lev_decay_pct / 0.02)
        # 세후 수익률 계산
        gross_return = 0.004
        net_return   = gross_return * (1 - self.TAX_RATE)
        size         = self.kelly_size(confidence)

        return SignalResult(
            agent_id=self.agent_id,
            signal="SELL_LEV",   # 레버리지 ETF 매도 포지션
            confidence=confidence,
            market_fit=self.get_market_fit(md.market_state),
            expected_return=net_return,
            max_loss=self.stop_loss_pct,
            capital_request=size,
            entry_price=md.close,
            target_price=self.calc_target_price(md.close, net_return),
            stop_price=self.calc_stop_price(md.close),
            reason=(
                f"레버리지 감쇠율 {lev_decay_pct:.3%} | "
                f"VKOSPI {md.vkospi:.1f} 고변동횡보 | "
                f"세후 예상 {net_return:.3%}"
            ),
            filter_results=filters,
        )


# ─────────────────────────────────────────────
#  에이전트 팩토리
# ─────────────────────────────────────────────
def create_all_agents(total_capital: float) -> dict:
    """
    7개 에이전트 생성.
    초기 자금은 균등 배분 (CIO가 나중에 재배분).
    """
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
