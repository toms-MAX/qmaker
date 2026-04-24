"""Screener 단위 테스트 — 중도형 B 정책 기준."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_capital.screener import (
    AffordabilityFilter,
    LiquidityFilter,
    QualityFilter,
    ScreenResult,
    Screener,
    load_universe,
)


def _make_ohlcv(
    n: int = 60,
    start: str = "2024-01-02",
    price: float = 50_000.0,
    volume: int = 500_000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n, name="date")
    rets = rng.normal(0.0, 0.01, size=n)
    closes = price * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "open":   closes * (1 + rng.normal(0, 0.001, size=n)),
            "high":   closes * (1 + np.abs(rng.normal(0, 0.003, size=n))),
            "low":    closes * (1 - np.abs(rng.normal(0, 0.003, size=n))),
            "close":  closes,
            "volume": np.full(n, volume, dtype=int),
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_static_fallback_loads(self):
        tickers = load_universe(None)
        assert len(tickers) >= 10
        assert all(isinstance(t, str) and len(t) == 6 for t in tickers)
        assert "005930" in tickers  # 삼성전자

    def test_universe_with_date_falls_back_when_pykrx_fails(self):
        # KRX 로그인 없는 환경에서도 정적 CSV로 폴백되어야 함
        tickers = load_universe("2024-01-02")
        assert len(tickers) >= 10


# ---------------------------------------------------------------------------
# QualityFilter (중도형 B)
# ---------------------------------------------------------------------------

class TestQualityFilter:
    def test_normal_stock_passes(self):
        df = _make_ohlcv(n=60)
        assert QualityFilter.apply("TEST", df).passed

    def test_trading_halt_rejected(self):
        df = _make_ohlcv(n=60)
        # 최근 30일 중 한 날 거래량 0
        df.iloc[-5, df.columns.get_loc("volume")] = 0
        decision = QualityFilter.apply("TEST", df)
        assert not decision.passed
        assert "거래정지" in decision.reason

    def test_insufficient_data_rejected(self):
        df = _make_ohlcv(n=20)
        assert not QualityFilter.apply("TEST", df).passed

    def test_50pct_crash_rejected(self):
        # 90일 전 고점 대비 -60% 하락
        df = _make_ohlcv(n=100, price=100_000.0)
        # 마지막 종가를 고점 절반 이하로
        df.iloc[-1, df.columns.get_loc("close")] = 30_000.0
        decision = QualityFilter.apply("TEST", df)
        assert not decision.passed
        assert "관리종목" in decision.reason or "DD" in decision.reason

    def test_upper_limit_streak_now_allowed(self):
        """B안: 상한가 연속·단기과열은 더 이상 제외 사유 아님."""
        df = _make_ohlcv(n=60)
        # 최근 2일 연속 +29%
        close = df["close"].values.copy()
        close[-2] = close[-3] * 1.29
        close[-1] = close[-2] * 1.29
        df["close"] = close
        assert QualityFilter.apply("TEST", df).passed


# ---------------------------------------------------------------------------
# LiquidityFilter
# ---------------------------------------------------------------------------

class TestLiquidityFilter:
    def test_passes_when_avg_above_threshold(self):
        # close 50000 * volume 500000 = 250억원 → 100억 초과
        df = _make_ohlcv(n=30, price=50_000, volume=500_000)
        decision, tv = LiquidityFilter.apply("TEST", df)
        assert decision.passed
        assert tv > 0

    def test_rejects_when_avg_below_threshold(self):
        # close 50000 * volume 10000 = 5억원 → 100억 미달
        df = _make_ohlcv(n=30, price=50_000, volume=10_000)
        decision, _ = LiquidityFilter.apply("TEST", df)
        assert not decision.passed
        assert "평균 거래대금" in decision.reason

    def test_surge_cap_20x_is_lenient(self):
        """B안: surge cap 20배 — 정상 변동은 통과."""
        df = _make_ohlcv(n=30, price=50_000, volume=500_000)
        # 최신일 거래량 10배 (5배 기존값 대비 훨씬 초과하지만 20배 미만)
        df.iloc[-1, df.columns.get_loc("volume")] = 5_000_000
        decision, _ = LiquidityFilter.apply("TEST", df)
        assert decision.passed

    def test_surge_cap_blocks_extreme_spike(self):
        """초대형 급증(~100배)은 여전히 제외."""
        df = _make_ohlcv(n=30, price=50_000, volume=500_000)
        df.iloc[-1, df.columns.get_loc("volume")] = 100_000_000
        decision, _ = LiquidityFilter.apply("TEST", df)
        assert not decision.passed
        assert "급증" in decision.reason


# ---------------------------------------------------------------------------
# AffordabilityFilter
# ---------------------------------------------------------------------------

class TestAffordabilityFilter:
    def test_affordable_passes(self):
        df = _make_ohlcv(n=30, price=50_000)
        decision = AffordabilityFilter.apply("TEST", df, budget=100_000)
        assert decision.passed

    def test_unaffordable_rejected(self):
        df = _make_ohlcv(n=30, price=500_000)
        decision = AffordabilityFilter.apply("TEST", df, budget=100_000)
        assert not decision.passed
        assert "1주 가격" in decision.reason

    def test_price_override(self):
        df = _make_ohlcv(n=30, price=50_000)
        # override 가격이 잔고 초과
        decision = AffordabilityFilter.apply(
            "TEST", df, budget=100_000, price_override=200_000
        )
        assert not decision.passed


# ---------------------------------------------------------------------------
# Screener end-to-end
# ---------------------------------------------------------------------------

class TestScreenerPipeline:
    def _make_universe(self) -> dict:
        """10종목: 8개 정상, 1개 거래정지, 1개 저유동."""
        ohlcv = {}
        for i in range(8):
            ohlcv[f"0000{i:02d}"] = _make_ohlcv(
                n=40, price=50_000 + i * 1000, volume=400_000 + i * 10_000, seed=i
            )

        halted = _make_ohlcv(n=40, seed=100)
        halted.iloc[-3, halted.columns.get_loc("volume")] = 0
        ohlcv["000008"] = halted

        low_liq = _make_ohlcv(n=40, volume=5_000, seed=101)
        ohlcv["000009"] = low_liq
        return ohlcv

    def test_pipeline_returns_valid_candidates(self):
        ohlcv = self._make_universe()
        screener = Screener(liquidity_top_n=10, final_top_k=5)
        result = screener.run(
            date="2024-03-01",
            ohlcv_map=ohlcv,
            budget_per_agent=10_000_000,
            universe=list(ohlcv.keys()),
        )
        assert isinstance(result, ScreenResult)
        assert len(result.candidates) <= 5
        assert result.stats["universe"] == 10
        assert "000008" in result.rejected
        assert "000009" in result.rejected

    def test_budget_constraint_filters_high_priced(self):
        ohlcv = self._make_universe()
        # 비싼 종목 하나 추가
        ohlcv["999999"] = _make_ohlcv(n=40, price=500_000, volume=400_000, seed=200)
        screener = Screener(final_top_k=10)
        result = screener.run(
            date="2024-03-01",
            ohlcv_map=ohlcv,
            budget_per_agent=100_000,  # 50000 이상만 가능
            universe=list(ohlcv.keys()),
        )
        assert "999999" in result.rejected
        assert "afford" in result.rejected["999999"]

    def test_final_top_k_limit(self):
        ohlcv = {f"0000{i:02d}": _make_ohlcv(n=40, seed=i) for i in range(15)}
        screener = Screener(liquidity_top_n=15, final_top_k=10)
        result = screener.run(
            date="2024-03-01",
            ohlcv_map=ohlcv,
            budget_per_agent=10_000_000,
            universe=list(ohlcv.keys()),
        )
        assert len(result.candidates) == 10

    def test_missing_ohlcv_rejected(self):
        ohlcv = {f"0000{i:02d}": _make_ohlcv(n=40, seed=i) for i in range(3)}
        screener = Screener(final_top_k=10)
        result = screener.run(
            date="2024-03-01",
            ohlcv_map=ohlcv,
            budget_per_agent=10_000_000,
            universe=["000000", "000001", "000002", "999998", "999999"],
        )
        assert "999998" in result.rejected
        assert "999999" in result.rejected
        assert "OHLCV 없음" in result.rejected["999999"]

    def test_trading_value_sorted_desc(self):
        """후보는 거래대금 내림차순이어야 함."""
        ohlcv = {}
        for i in range(5):
            # i가 클수록 거래대금이 커지도록
            ohlcv[f"TK{i:04d}"] = _make_ohlcv(
                n=40, price=50_000, volume=100_000 * (i + 1), seed=i
            )
        screener = Screener(liquidity_top_n=5, final_top_k=5)
        result = screener.run(
            date="2024-03-01",
            ohlcv_map=ohlcv,
            budget_per_agent=10_000_000,
            universe=list(ohlcv.keys()),
        )
        vals = [result.trading_values[t] for t in result.candidates]
        assert vals == sorted(vals, reverse=True)
