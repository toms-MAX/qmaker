"""백테스트용 변동성 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_capital.core.harness import AgentConfig, AgentHarness, Signal


@dataclass
class VolatilityParams:
    panic_threshold: float = -0.025  # -2.5% 이하 — 더 강한 패닉만 포착 (최적값)
    stop_pct:        float = 0.010
    take_pct:        float = 0.025   # 목표가 2.5% (최적값)
    size:            float = 0.10    # 에이전트당 10% — walk-forward 최적값


class VolatilityAgent(AgentHarness):
    """
    전략: 공항 구매 (Panic Buy) — 급락 시 역베팅
    """

    def __init__(self, params: VolatilityParams | None = None):
        self.params = params or VolatilityParams()
        super().__init__(AgentConfig(
            name="volatility",
            stop_loss_pct=self.params.stop_pct,
            take_profit_pct=self.params.take_pct,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        daily_ret = (df["close"] - df["open"]) / df["open"]
        
        signals: list[Signal] = []
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            
            # 조건: 장중 급락 (시가 대비 종가가 -2% 이하)
            if daily_ret.iloc[i] > p.panic_threshold:
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
