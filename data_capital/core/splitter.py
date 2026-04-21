"""Train / Valid / Test 데이터 분할 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 공식 분할 경계 (봉인된 테스트셋 보호)
SPLIT_TRAIN_START = "2020-01-01"
SPLIT_TRAIN_END   = "2022-12-31"
SPLIT_VALID_START = "2023-01-01"
SPLIT_VALID_END   = "2023-12-31"
SPLIT_TEST_START  = "2024-01-01"
SPLIT_TEST_END    = "2024-12-31"


@dataclass
class DataSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        for split_name in ("train", "valid", "test"):
            df = getattr(self, split_name)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(f"{split_name} index must be DatetimeIndex")

    def summary(self) -> str:
        lines = []
        for split_name in ("train", "valid", "test"):
            df = getattr(self, split_name)
            lines.append(
                f"  {split_name:5s}: {len(df):>4d} rows  "
                f"[{df.index.min().date()} ~ {df.index.max().date()}]"
            )
        return "DataSplit\n" + "\n".join(lines)


def split_data(df: pd.DataFrame) -> DataSplit:
    """
    DatetimeIndex를 가진 DataFrame을 공식 분할 기준으로 나눈다.

    Args:
        df: 전체 기간 OHLCV 데이터 (index=DatetimeIndex)

    Returns:
        DataSplit(train, valid, test)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    train = df.loc[SPLIT_TRAIN_START:SPLIT_TRAIN_END].copy()
    valid = df.loc[SPLIT_VALID_START:SPLIT_VALID_END].copy()
    test  = df.loc[SPLIT_TEST_START:SPLIT_TEST_END].copy()

    return DataSplit(train=train, valid=valid, test=test)
