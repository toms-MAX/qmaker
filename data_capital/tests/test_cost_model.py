"""CostModel 및 비용 반영 백테스트 회귀 테스트."""

import numpy as np
import pandas as pd
import pytest

from data_capital.core.harness import (
    CostModel,
    KOSPI_STOCK,
    KOSPI_ETF,
    KOSDAQ_STOCK,
    LiveAgentHarness,
    MarketData,
    Signal,
    SignalResult,
    NO_SIGNAL,
)
from data_capital.backtest.engine import run_backtest


# ---------------------------------------------------------------------------
# CostModel 단위 테스트
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_default_is_kospi_stock(self):
        cm = CostModel()
        assert cm.commission    == pytest.approx(0.00015)
        assert cm.transfer_tax  == pytest.approx(0.0015)
        assert cm.education_tax == pytest.approx(0.0015)
        assert cm.slippage      == pytest.approx(0.0015)

    def test_buy_rate_formula(self):
        cm = KOSPI_STOCK
        assert cm.buy_rate == pytest.approx(cm.commission + cm.slippage)

    def test_sell_rate_includes_taxes(self):
        cm = KOSPI_STOCK
        expected = cm.commission + cm.transfer_tax + cm.education_tax + cm.slippage
        assert cm.sell_rate == pytest.approx(expected)

    def test_sell_rate_greater_than_buy_rate(self):
        """매도 비용이 매수 비용보다 커야 한다 (세금 때문에)."""
        assert KOSPI_STOCK.sell_rate > KOSPI_STOCK.buy_rate

    def test_round_trip_is_sum(self):
        assert KOSPI_STOCK.round_trip == pytest.approx(KOSPI_STOCK.buy_rate + KOSPI_STOCK.sell_rate)

    def test_round_trip_approx_065_percent(self):
        """KOSPI 개별주 왕복 비용이 약 0.66% 근처여야 한다."""
        rt = KOSPI_STOCK.round_trip
        assert 0.0060 < rt < 0.0070, f"예상 범위 벗어남: {rt:.4%}"

    def test_etf_has_no_transfer_tax(self):
        assert KOSPI_ETF.transfer_tax == 0.0
        assert KOSPI_ETF.education_tax == 0.0

    def test_etf_round_trip_cheaper_than_stock(self):
        assert KOSPI_ETF.round_trip < KOSPI_STOCK.round_trip

    def test_kosdaq_no_education_tax(self):
        assert KOSDAQ_STOCK.education_tax == 0.0
        assert KOSDAQ_STOCK.transfer_tax == pytest.approx(0.0015)

    def test_kosdaq_cheaper_than_kospi(self):
        assert KOSDAQ_STOCK.round_trip < KOSPI_STOCK.round_trip

    def test_frozen_dataclass(self):
        """CostModel은 frozen이어야 한다."""
        cm = CostModel()
        with pytest.raises(Exception):
            cm.commission = 0.999  # type: ignore


# ---------------------------------------------------------------------------
# run_backtest 비용 반영 테스트
# ---------------------------------------------------------------------------

def _ohlcv_flat(n: int = 20, start: str = "2024-01-02", price: float = 10_000.0) -> pd.DataFrame:
    """가격이 완전히 고정된 OHLCV. 비용만이 수익에 영향."""
    dates = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame({
        "open":   [price] * n,
        "high":   [price] * n,
        "low":    [price] * n,
        "close":  [price] * n,
        "volume": [1_000_000] * n,
    }, index=dates)


class TestBacktestWithCostModel:
    def test_flat_market_loses_exactly_round_trip(self):
        """가격이 고정일 때 pnl_pct ≈ -round_trip (비용만큼 손해)."""
        df = _ohlcv_flat()
        price = float(df["close"].iloc[0])
        sig = Signal(
            date=df.index[0],
            direction=+1,
            entry_price=price,
            stop_loss=price * 0.90,
            take_profit=price * 1.10,
            size=0.02,
            exit_mode="next_open",
        )
        result = run_backtest(df, [sig], cost_model=KOSPI_STOCK, initial_capital=1_000_000)
        assert len(result.trades) == 1
        pnl_pct = float(result.trades["pnl_pct"].iloc[0])
        # 편도 비용이 양쪽에서 제거되므로 pnl_pct ≈ -round_trip (근사)
        assert pnl_pct < 0
        assert abs(pnl_pct + KOSPI_STOCK.round_trip) < 0.0005  # 5bp 오차 허용

    def test_etf_loses_less_than_stock(self):
        """동일 시그널에서 ETF가 개별주보다 손실이 적어야 한다."""
        df = _ohlcv_flat()
        price = float(df["close"].iloc[0])
        sig = Signal(
            date=df.index[0],
            direction=+1,
            entry_price=price,
            stop_loss=price * 0.90,
            take_profit=price * 1.10,
            size=0.02,
            exit_mode="next_open",
        )
        stock_result = run_backtest(df, [sig], cost_model=KOSPI_STOCK, initial_capital=1_000_000)
        etf_result   = run_backtest(df, [sig], cost_model=KOSPI_ETF,   initial_capital=1_000_000)
        assert etf_result.trades["pnl_pct"].iloc[0] > stock_result.trades["pnl_pct"].iloc[0]

    def test_backward_compat_cost_rate(self):
        """구버전 cost_rate 파라미터 사용 시 대칭 비용으로 동작."""
        df = _ohlcv_flat()
        price = float(df["close"].iloc[0])
        sig = Signal(
            date=df.index[0],
            direction=+1,
            entry_price=price,
            stop_loss=price * 0.90,
            take_profit=price * 1.10,
            size=0.02,
            exit_mode="next_open",
        )
        # 구버전 경로: cost_rate=0.001 (편도) → 왕복 0.2%
        legacy_result = run_backtest(df, [sig], cost_rate=0.001, initial_capital=1_000_000)
        pnl_pct = float(legacy_result.trades["pnl_pct"].iloc[0])
        assert abs(pnl_pct + 0.002) < 0.0005  # 왕복 0.2% 근사

    def test_default_cost_model_is_kospi_stock(self):
        """cost_model/cost_rate 둘 다 생략 시 KOSPI_STOCK 기본값 적용."""
        df = _ohlcv_flat()
        price = float(df["close"].iloc[0])
        sig = Signal(
            date=df.index[0],
            direction=+1,
            entry_price=price,
            stop_loss=price * 0.90,
            take_profit=price * 1.10,
            size=0.02,
            exit_mode="next_open",
        )
        default_result  = run_backtest(df, [sig], initial_capital=1_000_000)
        explicit_result = run_backtest(df, [sig], cost_model=KOSPI_STOCK, initial_capital=1_000_000)
        assert default_result.trades["pnl_pct"].iloc[0] == pytest.approx(
            explicit_result.trades["pnl_pct"].iloc[0]
        )


# ---------------------------------------------------------------------------
# LiveAgentHarness.calc_net_ev 테스트
# ---------------------------------------------------------------------------

class _DummyLiveAgent(LiveAgentHarness):
    def generate_signal(self, md: MarketData, **context) -> SignalResult:
        return NO_SIGNAL(self.agent_id, "dummy")


class TestLiveAgentNetEV:
    def setup_method(self):
        self.agent = _DummyLiveAgent(agent_id="dummy", allocated_capital=1_000_000)

    def test_positive_ev_when_edge_beats_cost(self):
        """승률 60%, 기대수익 2%, 최대손실 1% → EV > 0 (비용 0.66% 차감 후에도)."""
        ev = self.agent.calc_net_ev(win_prob=0.60, expected_return=0.020, max_loss=0.010)
        assert ev > 0

    def test_negative_ev_when_edge_is_thin(self):
        """승률 50%, 기대수익 0.5%, 최대손실 0.5% → 비용 때문에 EV < 0."""
        ev = self.agent.calc_net_ev(win_prob=0.50, expected_return=0.005, max_loss=0.005)
        assert ev < 0

    def test_ev_decreases_with_higher_cost(self):
        """COST_MODEL이 무거울수록 EV는 감소한다."""
        class StockAgent(LiveAgentHarness):
            COST_MODEL = KOSPI_STOCK
            def generate_signal(self, md, **ctx):
                return NO_SIGNAL("x", "")

        class ETFAgent(LiveAgentHarness):
            COST_MODEL = KOSPI_ETF
            def generate_signal(self, md, **ctx):
                return NO_SIGNAL("x", "")

        stock = StockAgent("s", 1_000_000)
        etf   = ETFAgent("e", 1_000_000)
        ev_stock = stock.calc_net_ev(0.55, 0.015, 0.010)
        ev_etf   = etf.calc_net_ev(0.55, 0.015, 0.010)
        assert ev_etf > ev_stock
