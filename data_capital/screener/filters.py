"""스크리너 필터 3종: Quality, Liquidity, Affordability.

입력 계약:
    ohlcv_map: Dict[ticker, pd.DataFrame] — index=DatetimeIndex, 컬럼 [open, high, low, close, volume].
              스냅샷일(date) 전일까지의 데이터만 포함되어야 한다 (lookahead 방지).

각 필터는 정적 메서드 `apply(ticker, df, **kwargs) -> FilterDecision` 제공.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterDecision:
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # noqa: D401
        return self.passed


# ─────────────────────────────────────────────
#  Quality (근사 필터, 선택지 B)
# ─────────────────────────────────────────────
class QualityFilter:
    """구조적 리스크 차단 필터 (중도형 B).

    "품질" 심사가 아니라 **"못 빠져나올 위험"** 만 막는다.
    - 거래정지 근사: 최근 30일 중 거래량 0 1회 이상 → 제외
    - 관리종목 근사: 90일 내 -50% 이상 폭락 → 제외

    단기과열 2회 / 상한가 2연 필터는 고의로 제거했다 — 변동성 자체는
    갭·변동성 에이전트의 알파 원천이며, VI·슬리피지 리스크는 L3 타이밍
    필터와 Liquidity surge_cap에서 방어한다.
    """

    STOPPED_LOOKBACK = 30
    CRASH_LOOKBACK = 90
    CRASH_DD_THRESHOLD = -0.5

    @classmethod
    def apply(cls, ticker: str, df: pd.DataFrame) -> FilterDecision:
        if df is None or len(df) < cls.STOPPED_LOOKBACK:
            return FilterDecision(False, f"데이터 부족 (<{cls.STOPPED_LOOKBACK}행)")

        recent_stop = df.tail(cls.STOPPED_LOOKBACK)
        zero_days = int((recent_stop["volume"] == 0).sum())
        if zero_days >= 1:
            return FilterDecision(False, f"최근 30일 거래정지 {zero_days}일")

        if len(df) >= cls.CRASH_LOOKBACK:
            recent = df.tail(cls.CRASH_LOOKBACK)
            peak = recent["close"].cummax()
            drawdown = float((recent["close"] / peak - 1.0).min())
            if drawdown <= cls.CRASH_DD_THRESHOLD:
                return FilterDecision(False, f"관리종목 근사 (90일 DD {drawdown:.1%})")

        return FilterDecision(True, "Quality 통과")


# ─────────────────────────────────────────────
#  Liquidity
# ─────────────────────────────────────────────
class LiquidityFilter:
    """거래대금 기반 유동성 필터.

    - 기준: 직전 영업일의 거래대금 = close × volume (일봉 근사)
    - 통과 조건: 20일 평균 거래대금 ≥ `min_avg_value` (기본 100억)
                AND 최근 거래대금 / 20일 평균 ≤ `surge_cap` (기본 5배)
    """

    DEFAULT_WINDOW = 20
    DEFAULT_MIN_AVG_VALUE = 10_000_000_000  # 100억원
    # 중도형 B: VI 반복 발동·호가 스프레드 급확장만 방어 (5배 → 20배로 완화)
    DEFAULT_SURGE_CAP = 20.0

    @classmethod
    def compute_trading_value(cls, df: pd.DataFrame) -> pd.Series:
        return df["close"] * df["volume"]

    @classmethod
    def apply(
        cls,
        ticker: str,
        df: pd.DataFrame,
        window: int = DEFAULT_WINDOW,
        min_avg_value: float = DEFAULT_MIN_AVG_VALUE,
        surge_cap: float = DEFAULT_SURGE_CAP,
    ) -> Tuple[FilterDecision, float]:
        """결정 + 최신 거래대금(정렬용) 반환."""
        if df is None or len(df) < window:
            return FilterDecision(False, f"데이터 부족 (<{window}행)"), 0.0

        tv = cls.compute_trading_value(df)
        recent = float(tv.iloc[-1])
        # 최근일은 제외한 직전 window일 평균 — 급증 비교 기준이 희석되지 않도록
        if len(tv) > window:
            avg20 = float(tv.iloc[-(window + 1):-1].mean())
        else:
            avg20 = float(tv.iloc[:-1].mean()) if len(tv) > 1 else 0.0

        if avg20 < min_avg_value:
            return (
                FilterDecision(False, f"20일 평균 거래대금 {avg20/1e8:.1f}억 < {min_avg_value/1e8:.0f}억"),
                recent,
            )

        ratio = recent / avg20 if avg20 > 0 else np.inf
        if ratio > surge_cap:
            return (
                FilterDecision(False, f"거래대금 급증 ({ratio:.1f}배 > {surge_cap}배)"),
                recent,
            )

        return FilterDecision(True, f"Liquidity 통과 (avg20={avg20/1e8:.1f}억, ratio={ratio:.2f})"), recent


# ─────────────────────────────────────────────
#  Affordability
# ─────────────────────────────────────────────
class AffordabilityFilter:
    """에이전트 잔고로 최소 1주 살 수 있는지.

    스냅샷 가격 기준:
      (A) 기본: 직전 영업일 종가 (lookahead 없는 안전한 선택)
      (B) 라이브 모드: 당일 시가 — 호출 측에서 price_override로 주입
    """

    @classmethod
    def apply(
        cls,
        ticker: str,
        df: pd.DataFrame,
        budget: float,
        price_override: float | None = None,
    ) -> FilterDecision:
        if price_override is not None:
            price = float(price_override)
        else:
            if df is None or df.empty:
                return FilterDecision(False, "데이터 없음")
            price = float(df["close"].iloc[-1])

        if price <= 0:
            return FilterDecision(False, f"유효하지 않은 가격 ({price})")
        if budget < price:
            return FilterDecision(False, f"잔고 {budget:,.0f} < 1주 가격 {price:,.0f}")
        return FilterDecision(True, f"Affordability 통과 (1주 {price:,.0f})")
