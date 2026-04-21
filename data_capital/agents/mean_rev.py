"""백테스트용 평균회귀 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import rsi, bollinger_bands, sma


@dataclass
class MeanRevParams:
    rsi_period:     int   = 14
    rsi_entry:      float = 30.0   # RSI 이하일 때 매수
    bb_period:      int   = 20
    bb_std:         float = 2.0
    vol_ratio_min:  float = 1.3    # 최소 거래량 배율
    stop_pct:       float = 0.002
    take_pct:       float = 0.004
    size:           float = 0.02


class MeanRevAgent(AgentHarness):
    """
    전략: RSI + 볼린저밴드 과매도 역매수
    청산: 볼린저 중심선 회귀 or next_open
    """

    def __init__(self, params: MeanRevParams | None = None):
        self.params = params or MeanRevParams()
        super().__init__(AgentConfig(
            name="mean_rev",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        r = rsi(df["close"], p.rsi_period)
        upper, middle, lower = bollinger_bands(df["close"], p.bb_period, p.bb_std)
        vol_ma = sma(df["volume"], 20)
        
        signals: list[Signal] = []
        
        for i in range(max(p.rsi_period, p.bb_period), len(df)):
            row = df.iloc[i]
            
            # 조건 1: RSI 과매도
            if r.iloc[i] > p.rsi_entry:
                continue
            
            # 조건 2: 볼린저 밴드 하단 터치/이탈
            if row["close"] > lower.iloc[i]:
                continue
                
            # 조건 3: 거래량 확인 (옵션이나 파라미터에 있음)
            if row["volume"] < vol_ma.iloc[i] * p.vol_ratio_min:
                continue
            
            entry = row["close"]
            signals.append(Signal(
                date=df.index[i],
                direction=+1,
                entry_price=entry,
                stop_loss=entry * (1 - p.stop_pct),
                take_profit=middle.iloc[i] if middle.iloc[i] > entry else entry * (1 + p.take_pct),
                size=p.size,
                exit_mode="next_open",
            ))

        return signals
