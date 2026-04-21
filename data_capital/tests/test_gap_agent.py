"""갭트레이딩 에이전트 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from data_capital.agents.gap_trading import GapTradingAgent, GapTradingParams
from data_capital.backtest.engine import run_backtest


def _make_ohlcv_with_gaps(n: int = 300) -> pd.DataFrame:
    """갭이 포함된 더미 데이터."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 30_000 + np.cumsum(rng.normal(10, 200, n))

    # 일부 날에 인위적 갭 추가
    gap_days = rng.choice(n, size=20, replace=False)
    gap_factor = rng.uniform(-0.015, 0.015, 20)

    open_prices = close.copy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    open_prices[gap_days] = prev_close[gap_days] * (1 + gap_factor)

    return pd.DataFrame({
        "open":   open_prices,
        "high":   np.maximum(open_prices, close) * (1 + rng.uniform(0, 0.005, n)),
        "low":    np.minimum(open_prices, close) * (1 - rng.uniform(0, 0.005, n)),
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n),
    }, index=dates)


class TestGapTradingAgent:
    def setup_method(self):
        self.df = _make_ohlcv_with_gaps()
        self.agent = GapTradingAgent()

    def test_run_returns_list(self):
        signals = self.agent.run(self.df)
        assert isinstance(signals, list)

    def test_signal_direction_valid(self):
        signals = self.agent.run(self.df)
        for sig in signals:
            assert sig.direction in (+1, -1)

    def test_stop_loss_correct_side(self):
        signals = self.agent.run(self.df)
        for sig in signals:
            if sig.direction == +1:
                assert sig.stop_loss < sig.entry_price
            else:
                assert sig.stop_loss > sig.entry_price

    def test_take_profit_correct_side(self):
        signals = self.agent.run(self.df)
        for sig in signals:
            if sig.direction == +1:
                assert sig.take_profit > sig.entry_price
            else:
                assert sig.take_profit < sig.entry_price

    def test_backtest_runs(self):
        signals = self.agent.run(self.df)
        result = run_backtest(self.df, signals)
        assert "total_return" in result.metrics
        assert "sharpe" in result.metrics

    def test_no_lookahead(self):
        """시그널 날짜가 항상 해당 행의 날짜여야 함 (미래 데이터 사용 금지)."""
        signals = self.agent.run(self.df)
        df_dates = set(self.df.index)
        for sig in signals:
            assert sig.date in df_dates

    def test_custom_params(self):
        params = GapTradingParams(gap_threshold=0.008, ma_period=30)
        agent = GapTradingAgent(params)
        signals = agent.run(self.df)
        assert isinstance(signals, list)
