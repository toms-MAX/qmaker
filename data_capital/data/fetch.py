"""pykrx를 이용한 OHLCV 데이터 수집 및 Train/Valid/Test 저장 스크립트."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pykrx import stock

from data_capital.core.splitter import split_data

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent

# 수집 기간 (테스트셋 포함 전체 커버)
FETCH_START = "20200101"
FETCH_END   = "20241231"


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    save_splits("069500")
