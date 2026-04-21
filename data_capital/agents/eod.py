"""백테스트용 종가베팅 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import sma


@dataclass
class EODParams:
    return_threshold: float = 0.012   # 기준 상향 (1.0% -> 1.2%)
    vol_ratio_min:    float = 1.3     # 거래량 기준 상향 (1.2 -> 1.3)
    ma_period:        int   = 20
    stop_pct:         float = 0.005
    take_pct:         float = 0.010
    size:             float = 0.05


class EODAgent(AgentHarness):
    """
    전략: 강한 종가 마감 시 베팅
    개선: 20일 이평선 위에서만 진입하여 '눌림목 후 돌파' 또는 '추세 지속' 상황만 포착
    """

    def __init__(self, params: EODParams | None = None):
        self.params = params or EODParams()
        super().__init__(AgentConfig(
            name="eod",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        daily_ret = (df["close"] - df["open"]) / df["open"]
        vol_ma = sma(df["volume"], 20)
        ma20 = sma(df["close"], p.ma_period)
        
        signals: list[Signal] = []
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # 조건 1: 20일 이평선 위 (안전 장치)
            if row["close"] < ma20.iloc[i]:
                continue
                
            # 조건 2: 당일 강한 상승
            if daily_ret.iloc[i] < p.return_threshold:
                continue
                
            # 조건 3: 거래량 동반
            if row["volume"] < vol_ma.iloc[i] * p.vol_ratio_min:
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
