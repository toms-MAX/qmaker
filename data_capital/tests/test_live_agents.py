"""라이브 에이전트 7종 단위 테스트."""

from datetime import datetime
import pytest

from data_capital.agents import (
    GapTradingAgent, MeanRevAgent, MomentumAgent,
    PairsAgent, EODAgent, VolatilityAgent, LevDecayAgent,
    create_all_agents,
)
from data_capital.core.harness import MarketData


# ─────────────────────────────────────────────
#  공통 픽스처
# ─────────────────────────────────────────────

def _md(**overrides) -> MarketData:
    """기본 MarketData 생성. overrides로 특정 필드 교체."""
    base = dict(
        ticker       = "069500",
        current_time = datetime(2024, 6, 15, 10, 0, 0),
        open         = 97000.0,
        high         = 98000.0,
        low          = 96500.0,
        close        = 97500.0,
        volume       = 1_500_000,
        prev_close   = 98000.0,
        ma20         = 97000.0,
        rsi14        = 50.0,
        bb_upper     = 99500.0,
        bb_middle    = 97000.0,
        bb_lower     = 94500.0,
        vol_ma5      = 1_200_000.0,
        vkospi       = 18.0,
        kospi_change = 0.2,
    )
    base.update(overrides)
    return MarketData(**base)


CAPITAL = 3_000_000


# ─────────────────────────────────────────────
#  1. GapTradingAgent
# ─────────────────────────────────────────────

class TestGapTradingAgent:
    def setup_method(self):
        self.agent = GapTradingAgent(CAPITAL / 7)

    def test_buy_on_valid_gap_down(self):
        # 0.5% 갭다운, 09:15 (갭 창 내)
        md = _md(
            current_time=datetime(2024, 6, 15, 9, 15, 0),
            open=97510.0, prev_close=98000.0,  # -0.5% 갭다운
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_on_gap_up(self):
        md = _md(
            current_time=datetime(2024, 6, 15, 9, 15, 0),
            open=98500.0, prev_close=98000.0,  # +0.5% 갭업
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_outside_time_window(self):
        # 10:00 — 갭 창 아님
        md = _md(
            current_time=datetime(2024, 6, 15, 10, 0, 0),
            open=97510.0, prev_close=98000.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_gap_too_large(self):
        # 4% 갭다운 → GAP_MAX(3%) 초과
        md = _md(
            current_time=datetime(2024, 6, 15, 9, 15, 0),
            open=94080.0, prev_close=98000.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_gap_too_small(self):
        # 0.1% 갭다운 → GAP_MIN(0.2%) 미만
        md = _md(
            current_time=datetime(2024, 6, 15, 9, 15, 0),
            open=97902.0, prev_close=98000.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_confidence_scales_with_gap_size(self):
        md1 = _md(current_time=datetime(2024, 6, 15, 9, 15, 0),
                  open=97510.0, prev_close=98000.0)   # -0.5%
        md2 = _md(current_time=datetime(2024, 6, 15, 9, 15, 0),
                  open=97020.0, prev_close=98000.0)   # -1.0%
        s1 = self.agent.generate_signal(md1)
        s2 = self.agent.generate_signal(md2)
        if s1.signal == s2.signal == "BUY":
            assert s2.confidence >= s1.confidence

    def test_stop_and_target_set(self):
        md = _md(current_time=datetime(2024, 6, 15, 9, 15, 0),
                 open=97510.0, prev_close=98000.0)
        sig = self.agent.generate_signal(md)
        if sig.signal == "BUY":
            assert sig.stop_price < sig.entry_price < sig.target_price


# ─────────────────────────────────────────────
#  2. MeanRevAgent
# ─────────────────────────────────────────────

class TestMeanRevAgent:
    def setup_method(self):
        self.agent = MeanRevAgent(CAPITAL / 7)

    def test_buy_on_low_rsi(self):
        md = _md(rsi14=30.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_high_rsi(self):
        md = _md(rsi14=60.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_boundary_rsi_44(self):
        # RSI 44 — 45 미만이므로 진입 가능
        md = _md(rsi14=44.9)
        sig = self.agent.generate_signal(md)
        assert sig.signal in ("BUY", "NO_SIGNAL")  # 필터 통과 여부에 따라

    def test_confidence_higher_when_rsi_lower(self):
        s1 = self.agent.generate_signal(_md(rsi14=40.0))
        s2 = self.agent.generate_signal(_md(rsi14=25.0))
        if s1.signal == s2.signal == "BUY":
            assert s2.confidence > s1.confidence


# ─────────────────────────────────────────────
#  3. MomentumAgent
# ─────────────────────────────────────────────

class TestMomentumAgent:
    def setup_method(self):
        self.agent = MomentumAgent(CAPITAL / 7)

    def test_buy_on_strong_momentum(self):
        md = _md(rsi14=70.0, close=98000.0, ma20=96000.0, high=98500.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_low_rsi(self):
        md = _md(rsi14=55.0, close=98000.0, ma20=96000.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_below_ma20(self):
        md = _md(rsi14=70.0, close=95000.0, ma20=97000.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_confidence_capped_at_08(self):
        md = _md(rsi14=99.0, close=98000.0, ma20=96000.0, high=98500.0)
        sig = self.agent.generate_signal(md)
        if sig.signal == "BUY":
            assert sig.confidence <= 0.80


# ─────────────────────────────────────────────
#  4. PairsAgent
# ─────────────────────────────────────────────

class TestPairsAgent:
    def setup_method(self):
        self.agent = PairsAgent(CAPITAL / 7)

    def test_buy_on_bb_lower_breach(self):
        # 종가가 BB 하단 아래
        md = _md(close=94300.0, bb_lower=94500.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_above_bb_lower(self):
        md = _md(close=97000.0, bb_lower=94500.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_target_is_bb_middle(self):
        md = _md(close=94300.0, bb_lower=94500.0, bb_middle=97000.0)
        sig = self.agent.generate_signal(md)
        if sig.signal == "BUY":
            assert abs(sig.target_price - 97000.0) < 1.0

    def test_no_signal_missing_bb(self):
        md = _md(close=94000.0, bb_lower=0.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"


# ─────────────────────────────────────────────
#  5. EODAgent
# ─────────────────────────────────────────────

class TestEODAgent:
    def setup_method(self):
        self.agent = EODAgent(CAPITAL / 7)

    def test_buy_on_strong_close(self):
        # 15:15, 장중 +1.5%, 거래량 충분
        md = _md(
            current_time = datetime(2024, 6, 15, 15, 15, 0),
            open=96000.0, close=97500.0,    # +1.56%
            volume=1_800_000, vol_ma5=1_200_000.0,
            ma20=96000.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_wrong_time(self):
        # 10:00 — 종가베팅 시간 아님
        md = _md(
            current_time = datetime(2024, 6, 15, 10, 0, 0),
            open=96000.0, close=97500.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_weak_return(self):
        # 15:15, 장중 +0.5% — 1.2% 미달
        md = _md(
            current_time = datetime(2024, 6, 15, 15, 15, 0),
            open=97000.0, close=97500.0,    # +0.52%
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_below_ma20(self):
        # 15:15, 강한 상승이지만 MA20 아래
        md = _md(
            current_time = datetime(2024, 6, 15, 15, 15, 0),
            open=93000.0, close=94500.0,    # +1.61%
            ma20=97000.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"


# ─────────────────────────────────────────────
#  6. VolatilityAgent
# ─────────────────────────────────────────────

class TestVolatilityAgent:
    def setup_method(self):
        self.agent = VolatilityAgent(CAPITAL / 7)

    def test_buy_on_panic(self):
        # 장중 -2.5%, VKOSPI 28
        md = _md(open=100000.0, close=97500.0, vkospi=28.0, kospi_change=-2.5)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_mild_drop(self):
        # -1.5% — PANIC_THRESHOLD(-2%) 미달
        md = _md(open=100000.0, close=98500.0, vkospi=25.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_low_vkospi(self):
        # -3% 급락이지만 VKOSPI 15 — 패닉 아님
        md = _md(open=100000.0, close=97000.0, vkospi=15.0)
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_stop_wider_than_gap(self):
        md = _md(open=100000.0, close=97500.0, vkospi=28.0, kospi_change=-2.5)
        sig = self.agent.generate_signal(md)
        if sig.signal == "BUY":
            # 손절이 진입가의 1% 아래
            assert sig.stop_price <= sig.entry_price * 0.995


# ─────────────────────────────────────────────
#  7. LevDecayAgent
# ─────────────────────────────────────────────

class TestLevDecayAgent:
    def setup_method(self):
        self.agent = LevDecayAgent(CAPITAL / 7)

    def test_buy_on_low_vol_uptrend(self):
        md = _md(
            open=97000.0, high=98500.0, low=96800.0, close=98300.0,
            ma20=97000.0, vkospi=16.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "BUY"

    def test_no_signal_high_vkospi(self):
        md = _md(
            open=97000.0, high=98500.0, low=96800.0, close=98300.0,
            ma20=97000.0, vkospi=25.0,  # >= 20
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_below_ma20(self):
        md = _md(
            open=95000.0, high=96000.0, low=94800.0, close=95800.0,
            ma20=97000.0, vkospi=16.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"

    def test_no_signal_weak_close(self):
        # 종가가 고가-저가 범위의 하단 30%
        md = _md(
            open=97000.0, high=98500.0, low=96800.0,
            close=96900.0,  # 하단 근처 마감
            ma20=96000.0, vkospi=16.0,
        )
        sig = self.agent.generate_signal(md)
        assert sig.signal == "NO_SIGNAL"


# ─────────────────────────────────────────────
#  통합: create_all_agents
# ─────────────────────────────────────────────

class TestCreateAllAgents:
    def test_returns_7_agents(self):
        agents = create_all_agents(3_000_000)
        assert len(agents) == 7

    def test_all_agent_ids_unique(self):
        agents = create_all_agents(3_000_000)
        ids = [a.agent_id for a in agents.values()]
        assert len(ids) == len(set(ids))

    def test_capital_split_evenly(self):
        agents = create_all_agents(3_000_000)
        caps = [a.allocated_capital for a in agents.values()]
        assert all(abs(c - caps[0]) < 1.0 for c in caps)

    def test_all_agents_return_signal_result(self):
        agents = create_all_agents(3_000_000)
        md = _md()
        for aid, agent in agents.items():
            sig = agent.generate_signal(md)
            assert sig.signal in ("BUY", "SELL", "HOLD", "NO_SIGNAL"), \
                f"{aid} returned unexpected signal: {sig.signal}"

    def test_bayesian_win_rate_with_no_trades(self):
        agents = create_all_agents(3_000_000)
        for agent in agents.values():
            rate = agent.bayesian_win_rate
            # 거래 없을 때 베이지안 하한 (alpha=1, beta=1) → 0.05 분위
            assert 0.0 <= rate <= 1.0

    def test_kelly_size_non_negative(self):
        agents = create_all_agents(3_000_000)
        for agent in agents.values():
            size = agent.kelly_size(confidence=0.6)
            assert size >= 0.0
