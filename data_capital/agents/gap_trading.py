"""백테스트용 갭트레이딩 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import sma


@dataclass
class GapTradingParams:
    gap_threshold: float = 0.003   # 0.3% 이상 갭다운
    ma_period:     int   = 20
    gap_max:       float = 0.025   # 과도한 낙폭(패닉) 제외
    stop_pct:      float = 0.005
    take_pct:      float = 0.010
    size:          float = 0.20    # 에이전트당 20% — walk-forward 최적값


class GapTradingAgent(AgentHarness):
    """
    전략: 갭다운 되돌림 (익일 시가 청산)
    개선: 비중 확대 및 과도한 패닉 갭 제외
    """

    def __init__(self, params: GapTradingParams | None = None):
        self.params = params or GapTradingParams()
        super().__init__(AgentConfig(
            name="gap_trading",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        ma = sma(df["close"], p.ma_period)
        signals: list[Signal] = []

        for i in range(p.ma_period, len(df)):
            row = df.iloc[i]
            prev_close = df["close"].iloc[i - 1]
            if prev_close == 0:
                continue

            gap = (row["open"] - prev_close) / prev_close

            # 갭다운 필터 (음수값이므로 -p.gap_max <= gap <= -p.gap_threshold)
            if not (-p.gap_max <= gap <= -p.gap_threshold):
                continue

            # 이평선 필터: 갭 전날 종가가 MA 위 = 상승추세에서의 갭다운
            if df["close"].iloc[i - 1] < ma.iloc[i - 1]:
                continue

            entry = row["open"]
            signals.append(Signal(
                date=df.index[i],
                direction=+1,
                entry_price=entry,
                stop_loss=entry * (1 - p.stop_pct),
                take_profit=entry * (1 + p.take_pct),
                size=p.size,
                exit_mode="pure_next_open",
            ))

        return signals
