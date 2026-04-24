"""Screener 파이프라인 — 09:05 1회 고정 스냅샷.

순서:
    1. 유니버스 로드 (KOSPI 200 스냅샷)
    2. 품질 필터 (거래정지·관리종목 근사)
    3. 유동성 필터 (20일 평균 ≥ 100억, surge cap ≤ 20배)
        → 거래대금 내림차순 상위 N 선별 (기본 20)
    4. 상위 N → affordability 통과 → 최종 상위 K (기본 10) 채택

스냅샷 원칙 (A안):
    - 전일 종가까지의 데이터만 입력 (lookahead 없음)
    - 라이브 모드에선 호출 측에서 09:05 시점 호가로 price_override 주입 가능
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .filters import (
    AffordabilityFilter,
    FilterDecision,
    LiquidityFilter,
    QualityFilter,
)
from .universe import load_universe

logger = logging.getLogger(__name__)


@dataclass
class ScreenResult:
    date: pd.Timestamp
    candidates: List[str]
    rejected: Dict[str, str] = field(default_factory=dict)
    trading_values: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"ScreenResult({self.date.date()}): {len(self.candidates)} candidates"]
        for stage, count in self.stats.items():
            lines.append(f"  {stage}: {count}")
        lines.append(f"  최종: {self.candidates}")
        return "\n".join(lines)


class Screener:
    """KOSPI200 → 품질 → 유동성 → affordability 파이프라인."""

    def __init__(
        self,
        liquidity_top_n: int = 20,
        final_top_k: int = 10,
        min_avg_value: float = LiquidityFilter.DEFAULT_MIN_AVG_VALUE,
        surge_cap: float = LiquidityFilter.DEFAULT_SURGE_CAP,
        window: int = LiquidityFilter.DEFAULT_WINDOW,
    ):
        self.liquidity_top_n = liquidity_top_n
        self.final_top_k = final_top_k
        self.min_avg_value = min_avg_value
        self.surge_cap = surge_cap
        self.window = window

    def run(
        self,
        date: pd.Timestamp | str,
        ohlcv_map: Dict[str, pd.DataFrame],
        budget_per_agent: float,
        universe: Optional[List[str]] = None,
        price_overrides: Optional[Dict[str, float]] = None,
    ) -> ScreenResult:
        """스크리너 실행.

        Args:
            date:              스냅샷 기준일
            ohlcv_map:         {ticker: df} — df는 date 이전(<=date-1) 데이터만 포함
            budget_per_agent:  에이전트 잔고 (affordability)
            universe:          None이면 load_universe(date) 사용
            price_overrides:   라이브 모드에서 09:05 시점 호가 주입용

        Returns:
            ScreenResult
        """
        ts = pd.Timestamp(date)
        if universe is None:
            universe = load_universe(ts)
        price_overrides = price_overrides or {}

        rejected: Dict[str, str] = {}
        stats: Dict[str, int] = {"universe": len(universe)}

        # 1. OHLCV 존재 확인
        with_data = []
        for t in universe:
            if t in ohlcv_map and ohlcv_map[t] is not None and not ohlcv_map[t].empty:
                with_data.append(t)
            else:
                rejected[t] = "OHLCV 없음"
        stats["with_data"] = len(with_data)

        # 2. Quality
        quality_pass = []
        for t in with_data:
            decision = QualityFilter.apply(t, ohlcv_map[t])
            if decision:
                quality_pass.append(t)
            else:
                rejected[t] = f"quality: {decision.reason}"
        stats["quality_pass"] = len(quality_pass)

        # 3. Liquidity (통과 종목을 거래대금 기준 정렬)
        liq_pass: List[tuple[str, float]] = []
        for t in quality_pass:
            decision, trading_value = LiquidityFilter.apply(
                t,
                ohlcv_map[t],
                window=self.window,
                min_avg_value=self.min_avg_value,
                surge_cap=self.surge_cap,
            )
            if decision:
                liq_pass.append((t, trading_value))
            else:
                rejected[t] = f"liquidity: {decision.reason}"

        liq_pass.sort(key=lambda x: x[1], reverse=True)
        top_n = liq_pass[: self.liquidity_top_n]
        stats["liquidity_pass"] = len(liq_pass)
        stats["liquidity_top_n"] = len(top_n)

        # 4. Affordability
        candidates: List[str] = []
        trading_values: Dict[str, float] = {}
        for t, tv in top_n:
            decision = AffordabilityFilter.apply(
                t,
                ohlcv_map[t],
                budget=budget_per_agent,
                price_override=price_overrides.get(t),
            )
            if decision:
                candidates.append(t)
                trading_values[t] = tv
            else:
                rejected[t] = f"afford: {decision.reason}"
            if len(candidates) >= self.final_top_k:
                break
        stats["final"] = len(candidates)

        result = ScreenResult(
            date=ts,
            candidates=candidates,
            rejected=rejected,
            trading_values=trading_values,
            stats=stats,
        )
        logger.info("Screener %s: %d → %d", ts.date(), stats["universe"], len(candidates))
        return result
