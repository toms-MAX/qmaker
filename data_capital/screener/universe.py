"""KOSPI 200 유니버스 로더 (KRX 정적 스냅샷 기반).

KRX 정보데이터시스템에서 연초 3개 스냅샷을 수동 다운로드:
    `kospi200_2022.csv`, `kospi200_2023.csv`, `kospi200_2024.csv`

해석:
    - `load_universe(None)`  → 3 스냅샷 union (223종목). 데이터 수집 풀.
    - `load_universe(date)`  → 해당 날짜 이전 가장 가까운 스냅샷 멤버십.
      Screener 호출 시 point-in-time 편향을 피하기 위함.

pykrx 실시간 조회는 2026년 현재 KRX 로그인 강제라 사용 불가.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).parent
ETF_CSV    = _PKG_DIR / "data" / "kospi_etf_static.csv"
UNION_CSV  = _PKG_DIR / "data" / "kospi200_union.csv"
STATIC_CSV = _PKG_DIR / "data" / "kospi200_static.csv"  # 레거시 50종목 폴백

# 스냅샷 기준일 → CSV. 기준일은 해당 연도 첫 영업일로 맞춤.
SNAPSHOTS: Dict[pd.Timestamp, Path] = {
    pd.Timestamp("2022-01-03"): _PKG_DIR / "kospi200_2022.csv",
    pd.Timestamp("2023-01-02"): _PKG_DIR / "kospi200_2023.csv",
    pd.Timestamp("2024-01-02"): _PKG_DIR / "kospi200_2024.csv",
}


def _read_krx_snapshot(path: Path) -> List[str]:
    df = pd.read_csv(path, dtype={"종목코드": str})
    return df["종목코드"].str.zfill(6).tolist()


def load_etf_universe() -> List[str]:
    """KOSPI/KOSDAQ 주요 ETF 유니버스."""
    df = pd.read_csv(ETF_CSV, dtype={"ticker": str})
    return df["ticker"].tolist()


def _load_union() -> List[str]:
    """3 스냅샷 union — 캐시 파일 우선, 없으면 스냅샷에서 재빌드."""
    if UNION_CSV.exists():
        df = pd.read_csv(UNION_CSV, dtype={"ticker": str})
        return df["ticker"].str.zfill(6).tolist()

    tickers: set[str] = set()
    for path in SNAPSHOTS.values():
        if path.exists():
            tickers |= set(_read_krx_snapshot(path))
    if tickers:
        return sorted(tickers)

    # 최후 폴백: 50종목 정적 CSV
    logger.warning("KOSPI200 스냅샷 파일 없음 — %s 폴백", STATIC_CSV)
    df = pd.read_csv(STATIC_CSV, dtype={"ticker": str})
    return df["ticker"].tolist()


def load_universe(date: pd.Timestamp | str | None = None) -> List[str]:
    """
    Args:
        date: None이면 전체 union (데이터 수집용).
              값이 있으면 해당 날짜 이전 가장 가까운 스냅샷 멤버십 반환.

    Returns:
        6자리 티커 문자열 리스트.
    """
    if date is None:
        tickers = _load_union()
        logger.info("유니버스 union 로드: %d 종목", len(tickers))
        return tickers

    ts = pd.Timestamp(date)
    prior = sorted(d for d in SNAPSHOTS if d <= ts)
    if prior:
        snap_date = prior[-1]
    else:
        snap_date = min(SNAPSHOTS)
        logger.warning("요청일 %s < 최초 스냅샷 %s — 최초 스냅샷 사용",
                       ts.date(), snap_date.date())

    path = SNAPSHOTS[snap_date]
    if not path.exists():
        logger.warning("스냅샷 파일 없음 %s — union 폴백", path)
        return _load_union()

    tickers = _read_krx_snapshot(path)
    logger.info("유니버스 asof %s → 스냅샷 %s: %d 종목",
                ts.date(), snap_date.date(), len(tickers))
    return tickers
