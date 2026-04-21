"""AgentHarness 및 공통 모듈 단위 테스트."""

import pandas as pd
import numpy as np
import pytest

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.core.splitter import split_data, DataSplit
from data_capital.indicators.common import rsi, atr, overnight_gap


# ---------------------------------------------------------------------------
# 테스트용 더미 에이전트
# ---------------------------------------------------------------------------

class DummyAgent(AgentHarness):
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        return [
            Signal(
                date=df.index[0],
                direction=+1,
                entry_price=float(df["close"].iloc[0]),
                stop_loss=float(df["close"].iloc[0]) * 0.98,
                take_profit=float(df["close"].iloc[0]) * 1.02,
            )
        ]


def _make_ohlcv(n: int = 100, start: str = "2020-01-02") -> pd.DataFrame:
    """재현 가능한 더미 OHLCV 데이터."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=n)
    close = 30_000 + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.002, n)),
        "high":   close * (1 + rng.uniform(0, 0.01, n)),
        "low":    close * (1 - rng.uniform(0, 0.01, n)),
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# AgentHarness
# ---------------------------------------------------------------------------

class TestAgentHarness:
    def test_run_returns_signals(self):
        agent = DummyAgent(AgentConfig(name="dummy"))
        df = _make_ohlcv()
        signals = agent.run(df)
        assert len(signals) == 1
        assert signals[0].direction == +1

    def test_signals_to_df(self):
        agent = DummyAgent(AgentConfig(name="dummy"))
        df = _make_ohlcv()
        agent.run(df)
        result = agent.signals_to_df()
        assert isinstance(result, pd.DataFrame)
        assert "direction" in result.columns

    def test_repr(self):
        agent = DummyAgent(AgentConfig(name="dummy"))
        assert "DummyAgent" in repr(agent)


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------

class TestSplitter:
    def setup_method(self):
        # 2020~2024 영업일 데이터 생성
        self.df = _make_ohlcv(n=1250, start="2020-01-02")

    def test_split_returns_datasplit(self):
        result = split_data(self.df)
        assert isinstance(result, DataSplit)

    def test_no_data_leakage(self):
        result = split_data(self.df)
        if len(result.train) and len(result.valid):
            assert result.train.index.max() < result.valid.index.min()
        if len(result.valid) and len(result.test):
            assert result.valid.index.max() < result.test.index.min()

    def test_summary_runs(self):
        result = split_data(self.df)
        s = result.summary()
        assert "train" in s


# ---------------------------------------------------------------------------
# 지표 함수
# ---------------------------------------------------------------------------

class TestIndicators:
    def setup_method(self):
        self.df = _make_ohlcv(200)

    def test_rsi_range(self):
        r = rsi(self.df["close"], 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_atr_positive(self):
        a = atr(self.df, 14).dropna()
        assert (a > 0).all()

    def test_overnight_gap_length(self):
        g = overnight_gap(self.df)
        assert len(g) == len(self.df)
        assert pd.isna(g.iloc[0])  # 첫 행은 NaN

    def test_rsi_period_sensitivity(self):
        r14 = rsi(self.df["close"], 14).dropna()
        r5  = rsi(self.df["close"], 5).dropna()
        # 짧은 기간 RSI가 더 많이 변동해야 함
        assert r5.std() >= r14.std()
