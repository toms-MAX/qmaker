"""백테스트용 페어트레이딩 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import sma


@dataclass
class PairsParams:
    window:           int   = 20
    zscore_entry:     float = 2.0   # 2표준편차 이탈 시 진입
    stop_pct:         float = 0.005
    take_pct:         float = 0.010
    size:             float = 0.05


class PairsAgent(AgentHarness):
    """
    전략: 가격 스프레드 회귀
    백테스트용: 단일 종목 내에서 가격의 급격한 왜곡(볼린저 밴드와 유사하나 스프레드 관점) 포착
    """

    def __init__(self, params: PairsParams | None = None):
        self.params = params or PairsParams()
        super().__init__(AgentConfig(
            name="pairs",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        # 스프레드 대신 종가의 이동평균 대비 이격도 사용 (Pairs 논리 차용)
        ma = sma(df["close"], p.window)
        std = df["close"].rolling(p.window).std()
        zscore = (df["close"] - ma) / (std + 1e-6)
        
        signals: list[Signal] = []
        
        for i in range(p.window, len(df)):
            row = df.iloc[i]
            z = zscore.iloc[i]
            
            # 하방 왜곡 (Underpriced) -> 매수
            if z <= -p.zscore_entry:
                direction = 1
            # 상방 왜곡 (Overpriced) -> 매도 (백테스트상 롱만 일단 처리)
            elif z >= p.zscore_entry:
                continue 
            else:
                continue

            entry = row["close"]
            signals.append(Signal(
                date=df.index[i],
                direction=direction,
                entry_price=entry,
                stop_loss=entry * (1 - p.stop_pct),
                take_profit=entry * (1 + p.take_pct),
                size=p.size,
                exit_mode="next_open",
            ))

        return signals
