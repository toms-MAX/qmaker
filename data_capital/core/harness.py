"""
DATA CAPITAL — core/harness.py
==============================
AgentHarness: 모든 에이전트의 기반 클래스
Ontology: 시장 상태 6종 분류
BuyFilters: 매수 5단계 공통 필터 (L1~L5)
SellRules: 매도 3원칙 + 트레일링 스탑
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  시장 상태 온톨로지 (6종)
# ─────────────────────────────────────────────
class MarketState(Enum):
    BULL_TREND   = "강세추세"
    BEAR_TREND   = "약세추세"
    LOW_VOL_SIDE = "저변동횡보"
    HIGH_VOL_SIDE= "고변동횡보"
    REBOUND      = "반등구간"
    CRISIS       = "위기공황"


MARKET_AGENT_FIT = {
    MarketState.BULL_TREND:    {"best": ["momentum","gap","eod"],         "avoid": ["mean_rev","volatility"]},
    MarketState.BEAR_TREND:    {"best": ["volatility","mean_rev","lev"],   "avoid": ["momentum"]},
    MarketState.LOW_VOL_SIDE:  {"best": ["pairs","eod"],                   "avoid": ["momentum","volatility"]},
    MarketState.HIGH_VOL_SIDE: {"best": ["gap","lev","pairs"],             "avoid": []},
    MarketState.REBOUND:       {"best": ["mean_rev","pairs"],              "avoid": []},
    MarketState.CRISIS:        {"best": ["volatility"],                    "avoid": ["momentum","gap","eod"],
                                "emergency": True},
}


def classify_market(
    kospi_change: float,
    vkospi: float,
    volume_ratio: float,
    ma20_dist: float,
    bb_width_ratio: float,
    foreign_net: float,
) -> MarketState:
    """
    6종 시장 상태 분류
    """
    # 위기/공황
    if vkospi > 30 or kospi_change < -3.0:
        return MarketState.CRISIS

    # 강세추세
    if kospi_change > 0 and ma20_dist > 0 and vkospi < 18 and volume_ratio > 1.2:
        return MarketState.BULL_TREND

    # 약세추세
    if kospi_change < 0 and ma20_dist < 0 and vkospi > 20:
        return MarketState.BEAR_TREND

    # 반등구간
    if ma20_dist < -0.02 and vkospi < 25 and foreign_net > 0:
        return MarketState.REBOUND

    # 고변동횡보
    if bb_width_ratio > 1.3 and 20 <= vkospi <= 30:
        return MarketState.HIGH_VOL_SIDE

    # 저변동횡보 (기본값)
    return MarketState.LOW_VOL_SIDE


# ─────────────────────────────────────────────
#  시장 데이터 컨테이너
# ─────────────────────────────────────────────
@dataclass
class MarketData:
    ticker:         str
    current_time:   datetime
    open:           float
    high:           float
    low:            float
    close:          float
    volume:         int
    prev_close:     float

    # 지표
    rsi14:          float = 0.0
    ma20:           float = 0.0
    ma200:          float = 0.0
    bb_upper:       float = 0.0
    bb_middle:      float = 0.0
    bb_lower:       float = 0.0
    atr14:          float = 0.0
    vol_ma5:        float = 0.0
    vol_ma20:       float = 0.0

    # 시장 전체
    vkospi:         float = 0.0
    kospi_change:   float = 0.0
    foreign_net:    float = 0.0
    institutional_net: float = 0.0
    program_trade:  str   = "NEUTRAL"  # BUY / SELL / NEUTRAL
    vi_status:      str   = "NORMAL"   # NORMAL / ACTIVE / RELEASED
    vi_elapsed_sec: int   = 9999

    # 계산값
    gap_pct:        float = field(init=False)
    vol_ratio:      float = field(init=False)
    market_state:   MarketState = field(init=False)

    def __post_init__(self):
        self.gap_pct   = (self.open - self.prev_close) / self.prev_close if self.prev_close else 0
        self.vol_ratio = self.volume / self.vol_ma5 if self.vol_ma5 else 1.0
        self.market_state = classify_market(
            self.kospi_change, self.vkospi, self.vol_ratio,
            (self.close - self.ma20) / self.ma20 if self.ma20 else 0,
            (self.bb_upper - self.bb_lower) / self.bb_middle if self.bb_middle else 0,
            self.foreign_net,
        )


# ─────────────────────────────────────────────
#  매수 5단계 필터 (모든 에이전트 공통)
# ─────────────────────────────────────────────
@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    layer:  str = ""

    def __bool__(self):
        return self.passed


class BuyFilters:
    """
    L1 시장 필터
    L2 추세 필터
    L3 타이밍 필터
    L4 신호 필터
    L5 리스크 필터
    """

    @staticmethod
    def L1_market(md: MarketData, override: bool = False) -> FilterResult:
        """L1: 오늘 싸울 만한 날인가?"""
        if override:
            return FilterResult(True, "L1 override (시장중립 전략)", "L1")

        if md.vkospi >= 30:
            return FilterResult(False, f"VKOSPI {md.vkospi} >= 30 (공황 구간)", "L1")
        if md.kospi_change <= -2.0:
            return FilterResult(False, f"코스피 {md.kospi_change}% 급락 중", "L1")
        if md.vi_status == "ACTIVE":
            return FilterResult(False, "VI 발동 중 — 진입 금지", "L1")
        return FilterResult(True, "L1 통과", "L1")

    @staticmethod
    def L2_trend(md: MarketData, override: bool = False, bearish_ok: bool = False) -> FilterResult:
        """L2: 방향이 맞는가?"""
        if override:
            return FilterResult(True, "L2 override (역추세 전략)", "L2")
        if bearish_ok:
            return FilterResult(True, "L2 bearish_ok (평균회귀 전략)", "L2")

        if md.close < md.ma20:
            return FilterResult(False, f"종가 {md.close} < 20일MA {md.ma20:.0f}", "L2")
        return FilterResult(True, "L2 통과 — 20일MA 위", "L2")

    @staticmethod
    def L3_timing(md: MarketData, time_override: Optional[str] = None) -> FilterResult:
        """L3: 지금 이 순간이 맞는가?"""
        t = md.current_time.time()

        if time_override == "GAP":
            if not (time(9, 5) <= t <= time(9, 30)):
                return FilterResult(False, f"{t} — 갭 시간 윈도우 아님 (09:05~09:30)", "L3")
            if md.vi_status == "ACTIVE":
                return FilterResult(False, "VI 발동 중", "L3")
            if md.vi_status == "RELEASED" and md.vi_elapsed_sec < 180:
                return FilterResult(False, f"VI 해제 후 {md.vi_elapsed_sec}초 — 3분 대기", "L3")
            return FilterResult(True, "L3 통과 (갭)", "L3")

        if time_override == "EOD":
            if not (time(15, 10) <= t <= time(15, 20)):
                return FilterResult(False, f"{t} — 종가베팅 시간 아님 (15:10~15:20)", "L3")
            return FilterResult(True, "L3 통과 (종가베팅)", "L3")

        # 기본 타이밍 필터
        if t < time(9, 5):
            return FilterResult(False, "시초가 혼돈 구간 (09:05 이전)", "L3")
        if time(11, 30) <= t <= time(12, 30):
            return FilterResult(False, "점심 유동성 저하 구간", "L3")
        if t >= time(15, 0):
            return FilterResult(False, "장 마감 직전 — 종가베팅 외 금지", "L3")
        return FilterResult(True, "L3 통과", "L3")

    @staticmethod
    def L4_signal(
        md: MarketData,
        required_signals: int = 2,
        custom_signals: Optional[list] = None,
    ) -> FilterResult:
        """L4: 통계적 우위가 있는가?"""
        signals = []

        # RSI 과매도 또는 추세 확인
        if md.rsi14 < 30:
            signals.append(f"RSI 과매도 {md.rsi14:.1f}")
        elif md.rsi14 > 50:
            signals.append(f"RSI 추세 확인 {md.rsi14:.1f}")

        # 거래량
        if md.vol_ratio >= 1.3:
            signals.append(f"거래량 {md.vol_ratio:.1f}배")

        # 볼린저밴드
        if md.close <= md.bb_lower:
            signals.append("볼린저 하단 이탈")
        elif md.close >= md.bb_middle and md.close <= md.bb_upper:
            signals.append("볼린저 중심선 상단 구간")

        # 커스텀 신호 (에이전트별 추가)
        if custom_signals:
            signals.extend(custom_signals)

        if len(signals) >= required_signals:
            return FilterResult(True, f"L4 통과 — {', '.join(signals)}", "L4")
        return FilterResult(
            False,
            f"L4 실패 — 신호 {len(signals)}개 (필요: {required_signals}개). 감지된 신호: {signals}",
            "L4",
        )

    @staticmethod
    def L5_risk(
        md: MarketData,
        daily_loss_pct: float,
        portfolio_exposure: float,
        same_direction_agents: int,
        consecutive_losses: int,
        max_daily_loss: float = -0.008,
        max_exposure: float = 0.75,
    ) -> FilterResult:
        """L5: 잃어도 괜찮은 구조인가?"""
        if daily_loss_pct <= max_daily_loss:
            return FilterResult(False, f"일일 MDD 한도 소진 ({daily_loss_pct:.2%})", "L5")
        if portfolio_exposure >= max_exposure:
            return FilterResult(False, f"포트폴리오 노출도 {portfolio_exposure:.0%} >= 75%", "L5")
        if same_direction_agents >= 2:
            return FilterResult(False, f"동일 방향 에이전트 {same_direction_agents}개 이미 포지션", "L5")
        if consecutive_losses >= 3:
            return FilterResult(True, f"연속 손실 {consecutive_losses}회 — 사이즈 50% 축소 필요", "L5")
        return FilterResult(True, "L5 통과", "L5")

    @classmethod
    def run_all(
        cls,
        md: MarketData,
        daily_loss_pct: float = 0.0,
        portfolio_exposure: float = 0.0,
        same_direction_agents: int = 0,
        consecutive_losses: int = 0,
        l1_override: bool = False,
        l2_override: bool = False,
        l2_bearish_ok: bool = False,
        l3_time_override: Optional[str] = None,
        l4_required: int = 2,
        l4_custom: Optional[list] = None,
    ) -> dict:
        """5단계 필터 전체 실행. 결과 dict 반환."""
        results = {
            "L1": cls.L1_market(md, l1_override),
            "L2": cls.L2_trend(md, l2_override, l2_bearish_ok),
            "L3": cls.L3_timing(md, l3_time_override),
            "L4": cls.L4_signal(md, l4_required, l4_custom),
            "L5": cls.L5_risk(
                md, daily_loss_pct, portfolio_exposure,
                same_direction_agents, consecutive_losses,
            ),
        }
        results["all_passed"] = all(r.passed for r in results.values() if isinstance(r, FilterResult))
        return results


# ─────────────────────────────────────────────
#  매도 3원칙 + 트레일링 스탑
# ─────────────────────────────────────────────
@dataclass
class Position:
    agent_id:       str
    ticker:         str
    entry_price:    float
    entry_time:     datetime
    shares:         int
    target_price:   float
    stop_price:     float
    max_hold_time:  time = time(15, 15)  # 타임컷 기본 15:15
    trailing_stop:  bool = True

    def update_trailing_stop(self, current_price: float) -> float:
        """트레일링 스탑 업데이트. 새 손절가 반환."""
        if not self.trailing_stop:
            return self.stop_price

        profit_pct = (current_price - self.entry_price) / self.entry_price

        if profit_pct >= 0.003:      # +0.3% 달성 → +0.2% 확보
            new_stop = self.entry_price * 1.002
        elif profit_pct >= 0.002:    # +0.2% 달성 → +0.1% 확보
            new_stop = self.entry_price * 1.001
        elif profit_pct >= 0.001:    # +0.1% 달성 → 본전 보호
            new_stop = self.entry_price
        else:
            new_stop = self.stop_price  # 초기 손절선 유지

        self.stop_price = max(self.stop_price, new_stop)
        return self.stop_price


class SellRules:
    """매도 3원칙: 익절 / 손절 / 타임컷"""

    @staticmethod
    def check(position: Position, current_price: float, current_time: datetime) -> Optional[str]:
        """
        매도 조건 확인.
        반환값: 'TAKE_PROFIT' | 'STOP_LOSS' | 'TIME_CUT' | None
        """
        # 트레일링 스탑 업데이트
        position.update_trailing_stop(current_price)

        # S1: 익절
        if current_price >= position.target_price:
            return "TAKE_PROFIT"

        # S2: 손절
        if current_price <= position.stop_price:
            return "STOP_LOSS"

        # S3: 타임컷
        if current_time.time() >= position.max_hold_time:
            return "TIME_CUT"

        return None


# ─────────────────────────────────────────────
#  백테스트 공통 Signal / AgentConfig / AgentHarness
# ─────────────────────────────────────────────

TRANSACTION_COST = 0.0008  # 편도 0.08%


@dataclass
class AgentConfig:
    name: str
    stop_loss_pct:   float = 0.002
    take_profit_pct: float = 0.004
    size:            float = 0.02   # 자본 대비 투입 비율


@dataclass
class Signal:
    """백테스트 엔진에서 소비되는 표준 시그널."""
    date:        "pd.Timestamp"
    direction:   int    # +1 롱 / -1 숏
    entry_price: float
    stop_loss:   float
    take_profit: float
    size:        float = 0.02
    exit_mode:   str   = "next_open"  # next_open | same_day | pure_next_open


class AgentHarness(ABC):
    """백테스트용 에이전트 기반 클래스."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._signals: list[Signal] = []

    @abstractmethod
    def generate_signals(self, df: "pd.DataFrame") -> list[Signal]:
        pass

    def run(self, df: "pd.DataFrame") -> list[Signal]:
        self._signals = self.generate_signals(df)
        return self._signals

    def signals_to_df(self) -> "pd.DataFrame":
        if not self._signals:
            return pd.DataFrame()
        rows = [
            {
                "date":        s.date,
                "direction":   s.direction,
                "entry_price": s.entry_price,
                "stop_loss":   s.stop_loss,
                "take_profit": s.take_profit,
                "size":        s.size,
                "exit_mode":   s.exit_mode,
            }
            for s in self._signals
        ]
        return pd.DataFrame(rows).set_index("date")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


# ─────────────────────────────────────────────
#  LiveAgentHarness — 실시간 에이전트의 기반 클래스
# ─────────────────────────────────────────────
@dataclass
class SignalResult:
    """에이전트 신호 표준 출력 형식"""
    agent_id:        str
    signal:          str            # BUY / SELL / HOLD / NO_SIGNAL
    confidence:      float          # 0.0 ~ 1.0
    market_fit:      float          # 0.0 ~ 1.0 (현재 시장 적합도)
    expected_return: float          # 예상 수익률
    max_loss:        float          # 최대 손실률
    capital_request: float          # 요청 자금 (원)
    entry_price:     float          # 제안 진입가
    target_price:    float          # 목표가
    stop_price:      float          # 손절가
    reason:          str            # 진입 이유 (합의 시스템용)
    filter_results:  dict = field(default_factory=dict)
    timestamp:       datetime = field(default_factory=datetime.now)

    @property
    def ev(self) -> float:
        """기대값 계산"""
        return (self.confidence * self.expected_return) - ((1 - self.confidence) * self.max_loss)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "market_fit": round(self.market_fit, 4),
            "expected_return": round(self.expected_return, 4),
            "max_loss": round(self.max_loss, 4),
            "ev": round(self.ev, 6),
            "capital_request": round(self.capital_request, 0),
            "entry_price": round(self.entry_price, 0),
            "target_price": round(self.target_price, 0),
            "stop_price": round(self.stop_price, 0),
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


def NO_SIGNAL(agent_id: str, reason: str) -> SignalResult:
    return SignalResult(
        agent_id=agent_id,
        signal="NO_SIGNAL",
        confidence=0.0,
        market_fit=0.0,
        expected_return=0.0,
        max_loss=0.0,
        capital_request=0.0,
        entry_price=0.0,
        target_price=0.0,
        stop_price=0.0,
        reason=reason,
    )


class LiveAgentHarness(ABC):
    """
    실시간 7개 에이전트의 기반 클래스.
    """

    def __init__(
        self,
        agent_id: str,
        allocated_capital: float,
        stop_loss_pct: float   = 0.005,
        take_profit_pct: float = 0.015,
        kelly_fraction: float  = 0.25,
    ):
        self.agent_id         = agent_id
        self.allocated_capital = allocated_capital
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self.kelly_fraction   = kelly_fraction

        self.trade_history: list  = []
        self.win_count:     int   = 0
        self.loss_count:    int   = 0
        self.total_pnl:     float = 0.0

    @abstractmethod
    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        pass

    def get_market_fit(self, market_state: MarketState) -> float:
        """
        온톨로지(MARKET_AGENT_FIT)를 기반으로 적합도 자동 계산.
        최적화: 하드코딩 대신 온톨로지 맵 참조.
        """
        fit_info = MARKET_AGENT_FIT.get(market_state, {})
        best_agents = fit_info.get("best", [])
        avoid_agents = fit_info.get("avoid", [])

        if self.agent_id in best_agents:
            return 1.0
        if self.agent_id in avoid_agents:
            return 0.2  # 회피 구역이어도 신호 강하면 진입 가능하도록 0.2 부여
        return 0.5      # 중립

    def kelly_size(self, confidence: float, win_loss_ratio: float = 2.0) -> float:
        q = 1 - confidence
        kelly_full = (confidence * win_loss_ratio - q) / win_loss_ratio
        kelly_safe = kelly_full * self.kelly_fraction
        return self.allocated_capital * max(0.0, min(kelly_safe, 0.35))

    def calc_target_price(self, entry_price: float, profit_pct: Optional[float] = None) -> float:
        pct = profit_pct if profit_pct is not None else self.take_profit_pct
        return entry_price * (1 + pct)

    def calc_stop_price(self, entry_price: float, loss_pct: Optional[float] = None) -> float:
        pct = loss_pct if loss_pct is not None else self.stop_loss_pct
        return entry_price * (1 - pct)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def bayesian_win_rate(self) -> float:
        from scipy.stats import beta as beta_dist
        alpha = self.win_count + 1
        beta  = self.loss_count + 1
        return float(beta_dist.ppf(0.05, alpha, beta))
