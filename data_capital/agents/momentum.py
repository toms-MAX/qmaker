"""백테스트용 모멘텀 에이전트."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_capital.core.harness import AgentConfig, AgentHarness, Signal
from data_capital.indicators.common import atr, sma


@dataclass
class MomentumParams:
    lookback:       int   = 10     
    breakout_pct:   float = 0.001  
    vol_ratio_min:  float = 1.3    
    ma_short:       int   = 5      # 5일 이평선
    ma_long:        int   = 20     # 20일 이평선
    atr_period:     int   = 14
    sl_atr_mult:    float = 1.5    
    tp_atr_mult:    float = 4.0    
    size:           float = 0.05   


class MomentumAgent(AgentHarness):
    """
    전략: N일 저항선 돌파 + 정배열 + 거래량
    개선: 5/20 정배열 구간에서만 진입하여 Bull Trap 방어
    """

    def __init__(self, params: MomentumParams | None = None):
        self.params = params or MomentumParams()
        super().__init__(AgentConfig(
            name="momentum",
            stop_loss_pct=0.005,
            take_profit_pct=0.015,
            size=self.params.size,
        ))

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        high_n = df["high"].rolling(p.lookback).max().shift(1)
        vol_ma = sma(df["volume"], 20)
        ma5 = sma(df["close"], p.ma_short)
        ma20 = sma(df["close"], p.ma_long)
        a = atr(df, p.atr_period)
        
        signals: list[Signal] = []
        
        for i in range(max(p.lookback, 20), len(df)):
            row = df.iloc[i]
            
            # 조건 1: 전고점 돌파
            if row["close"] < high_n.iloc[i] * (1 + p.breakout_pct):
                continue
                
            # 조건 2: 정배열 (5 > 20) — 추세 강화 필터
            if ma5.iloc[i] < ma20.iloc[i]:
                continue
                
            # 조건 3: 거래량 동반
            if row["volume"] < vol_ma.iloc[i] * p.vol_ratio_min:
                continue

            # 조건 4: 장대양봉 확인 (꼬리가 너무 길면 제외)
            body = abs(row["close"] - row["open"])
            candle_range = row["high"] - row["low"]
            if candle_range > 0 and (body / candle_range) < 0.6:
                continue
                
            entry = row["close"]
            current_atr = a.iloc[i]
            
            signals.append(Signal(
                date=df.index[i],
                direction=+1,
                entry_price=entry,
                stop_loss=entry - (current_atr * p.sl_atr_mult),
                take_profit=entry + (current_atr * p.tp_atr_mult),
                size=p.size,
                exit_mode="next_open",
            ))
            
        return signals
