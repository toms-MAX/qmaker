"""pykrx를 이용한 OHLCV 데이터 수집 및 Train/Valid/Test 저장 스크립트."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pykrx import stock

from data_capital.core.splitter import split_data

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent

# 수집 기간 (테스트셋 포함 전체 커버)
FETCH_START = "20200101"
FETCH_END   = "20241231"

# v1.5 멀티티커 기본 기간 (Train:2022 / Valid:2023 / Test:2024)
UNIVERSE_FETCH_START = "20210601"   # 2022-01-02 첫 영업일의 MA200·100일 DD 계산용 여유분
UNIVERSE_FETCH_END   = "20241231"

# pykrx 레이트리밋 대응: 종목 간 sleep
PYKRX_SLEEP_SEC = 0.25


def fetch_ohlcv(ticker: str, start: str = FETCH_START, end: str = FETCH_END) -> pd.DataFrame:
    """
    pykrx로 ETF/주식 OHLCV 데이터를 수집한다.

    Args:
        ticker: 종목코드 (예: "069500")
        start:  시작일 "YYYYMMDD"
        end:    종료일 "YYYYMMDD"

    Returns:
        DatetimeIndex, 컬럼 [open, high, low, close, volume] DataFrame
    """
    logger.info("데이터 수집: ticker=%s %s~%s", ticker, start, end)
    df = stock.get_market_ohlcv_by_date(start, end, ticker)

    # pykrx 컬럼명 → 영문 통일
    col_map = {
        "시가": "open", "고가": "high", "저가": "low",
        "종가": "close", "거래량": "volume",
    }
    df = df.rename(columns=col_map)
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # 거래 없는 날(거래량=0) 제거
    df = df[df["volume"] > 0]

    logger.info("수집 완료: %d rows", len(df))
    return df


def save_splits(ticker: str, df: pd.DataFrame | None = None) -> None:
    """
    OHLCV 데이터를 Train/Valid/Test로 분할하여 CSV로 저장한다.

    Args:
        ticker: 종목코드
        df:     이미 수집된 DataFrame. None이면 내부에서 수집.
    """
    if df is None:
        df = fetch_ohlcv(ticker)

    splits = split_data(df)

    raw_dir = DATA_DIR / "raw" / ticker
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 전체 원본 저장
    full_path = raw_dir / "full.csv"
    df.to_csv(full_path)
    logger.info("전체 데이터 저장: %s", full_path)

    # 분할 저장
    processed_dir = DATA_DIR / "processed" / ticker
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        split_df = getattr(splits, split_name)
        out_path = processed_dir / f"{split_name}.csv"
        split_df.to_csv(out_path)
        logger.info(
            "%s 저장: %s (%d rows, %s~%s)",
            split_name, out_path, len(split_df),
            split_df.index.min().date(), split_df.index.max().date(),
        )

    print(splits.summary())


def load_split(ticker: str, split: str) -> pd.DataFrame:
    """
    저장된 분할 데이터를 로드한다.

    Args:
        ticker: 종목코드
        split:  "train" | "valid" | "test"

    Returns:
        DatetimeIndex DataFrame
    """
    path = DATA_DIR / "processed" / ticker / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"데이터 없음: {path}\n`save_splits('{ticker}')` 먼저 실행하세요.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def load_processed(ticker: str) -> pd.DataFrame:
    """
    저장된 전체 데이터를 로드한다 (raw/ticker/full.csv).
    """
    path = DATA_DIR / "raw" / ticker / "full.csv"
    if not path.exists():
        # raw에 없으면 processed에서 합쳐서라도 가져옴 (혹은 에러)
        raise FileNotFoundError(f"데이터 없음: {path}")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def fetch_universe_ohlcv(
    tickers: Iterable[str],
    start: str = UNIVERSE_FETCH_START,
    end: str = UNIVERSE_FETCH_END,
    force: bool = False,
    sleep_sec: float = PYKRX_SLEEP_SEC,
) -> dict[str, pd.DataFrame]:
    """KOSPI 200 유니버스 OHLCV 배치 수집 + 증분 캐싱.

    - `data/raw/{ticker}/full.csv` 가 존재하면 로드 (force=True 시 강제 재수집).
    - pykrx 호출 사이 sleep_sec 대기.
    - 실패 종목은 로그만 남기고 빈 결과로 처리(스킵).

    Args:
        tickers:   수집할 티커 iterable
        start/end: "YYYYMMDD"
        force:     True면 캐시 무시하고 재수집
        sleep_sec: pykrx 호출 간격

    Returns:
        {ticker: OHLCV DataFrame} — 수집 실패 종목은 누락
    """
    result: dict[str, pd.DataFrame] = {}
    failures: List[str] = []
    tickers = list(tickers)
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        ticker = str(ticker).zfill(6)
        raw_dir = DATA_DIR / "raw" / ticker
        cache_path = raw_dir / "full.csv"

        if not force and cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
                if len(df) > 0:
                    result[ticker] = df
                    logger.debug("[%d/%d] %s cache hit (%d rows)", idx, total, ticker, len(df))
                    continue
            except Exception as e:  # noqa: BLE001
                logger.warning("[%d/%d] %s 캐시 손상, 재수집: %s", idx, total, ticker, e)

        try:
            df = fetch_ohlcv(ticker, start=start, end=end)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%d/%d] %s 수집 실패: %s", idx, total, ticker, e)
            failures.append(ticker)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue

        if df.empty:
            logger.warning("[%d/%d] %s 데이터 없음", idx, total, ticker)
            failures.append(ticker)
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
            result[ticker] = df
            logger.info("[%d/%d] %s 저장 (%d rows)", idx, total, ticker, len(df))

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    logger.info("유니버스 수집 완료: 성공=%d, 실패=%d", len(result), len(failures))
    if failures:
        logger.info("실패 종목: %s", failures)
    return result


def load_universe_ohlcv(
    tickers: Iterable[str],
    min_rows: int = 100,
) -> dict[str, pd.DataFrame]:
    """이미 캐시된 유니버스 OHLCV만 로드. 네트워크 호출 없음.

    Args:
        tickers:  티커 iterable
        min_rows: 이 행수 미만은 제외 (지표 계산용 여유분)

    Returns:
        {ticker: DataFrame}
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker = str(ticker).zfill(6)
        cache_path = DATA_DIR / "raw" / ticker / "full.csv"
        if not cache_path.exists():
            continue
        try:
            df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 캐시 로드 실패: %s", ticker, e)
            continue
        if len(df) >= min_rows:
            result[ticker] = df
    return result


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KOSPI200 OHLCV 수집기")
    parser.add_argument("--mode", choices=["single", "universe"], default="universe")
    parser.add_argument("--ticker", default="069500")
    parser.add_argument("--start", default=UNIVERSE_FETCH_START)
    parser.add_argument("--end",   default=UNIVERSE_FETCH_END)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="유니버스 중 N개만 수집")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode == "single":
        save_splits(args.ticker)
        return

    from data_capital.screener.universe import load_universe

    tickers = load_universe(None)
    if args.limit:
        tickers = tickers[: args.limit]
    logger.info("유니버스 수집 시작: %d 종목, %s~%s", len(tickers), args.start, args.end)
    fetch_universe_ohlcv(tickers, start=args.start, end=args.end, force=args.force)


if __name__ == "__main__":
    _cli()
