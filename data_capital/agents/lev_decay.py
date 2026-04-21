"""백테스트용 레버리지 디케이 방어 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import historical_vol, sma


@dataclass
class LevDecayParams:
    vol_lookback:   int   = 20
    vol_max:        float = 0.20   # 연율화 변동성 20% 초과 시 디케이 위험으로 진입 금지
    ma_period:      int   = 60     # 중기 추세 확인
    stop_pct:       float = 0.007
    take_pct:       float = 0.015
    size:           float = 0.05


class LevDecayAgent(AgentHarness):
    """
    전략: 저변동성 구간에서의 안정적 추세 추종
    레버리지 ETF 특유의 변동성 잠식(Decay)을 피하기 위해 고변동성 구간 진입을 제한함
    """

    def __init__(self, params: LevDecayParams | None = None):
        self.params = params or LevDecayParams()
        super().__init__(AgentConfig(
            name="lev_decay",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        vol = historical_vol(df["close"], p.vol_lookback)
        ma_long = sma(df["close"], p.ma_period)
        
        signals: list[Signal] = []
        
        for i in range(max(p.vol_lookback, p.ma_period), len(df)):
            row = df.iloc[i]
            
            # 조건 1: 저변동성 구간 (Decay 방어)
            if vol.iloc[i] > p.vol_max:
                continue
                
            # 조건 2: 중기 이평선 위 (추세 확인)
            if row["close"] < ma_long.iloc[i]:
                continue
            
            # 조건 3: 당일 종가가 고가 부근 (강한 마감)
            if (row["close"] - row["low"]) / (row["high"] - row["low"] + 1e-6) < 0.7:
                continue

            entry = row["close"]
            signals.append(Signal(
                date=df.index[i],
                direction=+1,
                entry_price=entry,
                stop_loss=entry * (1 - p.stop_pct),
                take_profit=entry * (1 + p.take_pct),
                size=p.size,
                exit_mode="next_open",
            ))

        return signals
