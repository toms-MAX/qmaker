"""공통 기술 지표 함수 모음 (RSI, ATR, EMA, VWAP 등)."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 추세 지표
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """지수 이동평균."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """단순 이동평균."""
    return series.rolling(period).mean()


# ---------------------------------------------------------------------------
# 모멘텀 지표
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder 방식 RSI.

    Args:
        series: 종가 시리즈
        period: 룩백 기간 (기본 14)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 변동성 지표
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Args:
        df: 'high', 'low', 'close' 컬럼 필요
        period: 룩백 기간
    """
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def historical_vol(series: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """로그 수익률 기반 역사적 변동성."""
    log_ret = np.log(series / series.shift(1))
    vol = log_ret.rolling(period).std()
    return vol * (252 ** 0.5) if annualize else vol


# ---------------------------------------------------------------------------
# 갭 관련 유틸 (갭트레이딩 에이전트 전용)
# ---------------------------------------------------------------------------

def overnight_gap(df: pd.DataFrame) -> pd.Series:
    """
    오버나이트 갭 비율 = (당일 시가 - 전일 종가) / 전일 종가.

    Returns:
        갭 비율 시리즈 (양수=갭업, 음수=갭다운)
    """
    return (df["open"] - df["close"].shift(1)) / df["close"].shift(1)


def gap_fill_ratio(df: pd.DataFrame) -> pd.Series:
    """당일 갭이 얼마나 메워졌는지 비율 (0~1, 1=완전 메움)."""
    gap = df["open"] - df["close"].shift(1)
    # 갭이 0이면 NaN 반환
    gap = gap.replace(0, np.nan)

    # 갭업: 저가가 전일 종가까지 내려왔으면 1
    fill_up   = ((df["open"] - df["low"])  / gap.abs()).clip(0, 1)
    fill_down = ((df["high"] - df["open"]) / gap.abs()).clip(0, 1)

    direction = np.sign(gap)
    filled = pd.Series(np.where(direction > 0, fill_up, fill_down), index=df.index)
    return filled


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    볼린저 밴드 (상단, 중단, 하단).
    """
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


# ---------------------------------------------------------------------------
# 필터 헬퍼 (L1~L5 공통 빌딩 블록)
# ---------------------------------------------------------------------------

def is_above_ma(series: pd.Series, period: int) -> pd.Series:
    """종가 > SMA(period) 이면 True."""
    return series > sma(series, period)


def is_volume_spike(volume: pd.Series, period: int = 20, multiplier: float = 1.5) -> pd.Series:
    """거래량이 이동평균의 multiplier배 초과하면 True."""
    return volume > sma(volume, period) * multiplier
