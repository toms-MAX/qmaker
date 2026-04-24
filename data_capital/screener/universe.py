"""KOSPI 200 유니버스 로더.

1순위: pykrx `get_index_portfolio_deposit_file("1028", date)` 실시간 조회.
2순위: `data/kospi200_static.csv` 정적 폴백.

pykrx 집계 API는 KRX 환경변수(KRX_ID/KRX_PW) 요구로 실패할 수 있다.
실패 시 로거 경고 후 정적 CSV로 폴백한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

KOSPI200_INDEX = "1028"  # KRX 기준 KOSPI 200 인덱스 코드
STATIC_CSV = Path(__file__).parent / "data" / "kospi200_static.csv"
ETF_CSV    = Path(__file__).parent / "data" / "kospi_etf_static.csv"


def _load_static() -> List[str]:
    df = pd.read_csv(STATIC_CSV, dtype={"ticker": str})
    return df["ticker"].tolist()


def load_etf_universe() -> List[str]:
    """KOSPI/KOSDAQ 주요 ETF 유니버스 (비용 0.09% 전제).

    v1.5 피벗 이후 기본 유니버스. 개별주 대비 왕복 비용이
    0.66% → 0.09%로 7배 낮아 알파 보존에 유리.
    """
    df = pd.read_csv(ETF_CSV, dtype={"ticker": str})
    return df["ticker"].tolist()


def _load_pykrx(date_yyyymmdd: str) -> Optional[List[str]]:
    """pykrx 시도. 빈 리스트나 예외는 None 반환."""
    try:
        from pykrx import stock
    except ImportError:
        return None

    try:
        tickers = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX, date_yyyymmdd)
    except Exception as e:  # noqa: BLE001 — pykrx/KRX 레이어의 다양한 예외 포괄
        logger.warning("pykrx KOSPI200 조회 실패 (%s): %s", date_yyyymmdd, e)
        return None

    if tickers is None or len(tickers) == 0:
        return None

    return [str(t).zfill(6) for t in tickers]


def load_universe(date: pd.Timestamp | str | None = None) -> List[str]:
    """KOSPI 200 유니버스 티커 리스트 반환.

    Args:
        date: 스냅샷 기준일. None이면 정적 CSV만 사용.

    Returns:
        6자리 티커 문자열 리스트.
    """
    if date is not None:
        ts = pd.Timestamp(date)
        ymd = ts.strftime("%Y%m%d")
        tickers = _load_pykrx(ymd)
        if tickers:
            logger.info("pykrx 유니버스 로드: %d 종목 (%s)", len(tickers), ymd)
            return tickers
        logger.warning("pykrx 유니버스 실패 — 정적 CSV 폴백")

    tickers = _load_static()
    logger.info("정적 유니버스 로드: %d 종목", len(tickers))
    return tickers
