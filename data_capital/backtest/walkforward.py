"""Walk-Forward 검증 — 과적합 방지."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from backtest.engine import BacktestResult, run_backtest

if TYPE_CHECKING:
    from core.harness import AgentHarness


@dataclass
class WFWindow:
    """Walk-Forward 단일 창."""
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    train_result: BacktestResult | None = None
    test_result:  BacktestResult | None = None

    @property
    def efficiency(self) -> float:
        """테스트 수익률 / 학습 수익률. 1.0에 가까울수록 일반화 잘 됨."""
        if self.train_result is None or self.test_result is None:
            return 0.0
        train_r = self.train_result.metrics.get("total_return", 0)
        test_r  = self.test_result.metrics.get("total_return", 0)
        return test_r / train_r if train_r != 0 else 0.0


@dataclass
class WalkForwardResult:
    windows:   list[WFWindow]
    summary:   dict = field(default_factory=dict)

    def print_summary(self):
        print(f"\n{'─'*55}")
        print(f"  Walk-Forward 검증 ({len(self.windows)}창)")
        print(f"{'─'*55}")
        for i, w in enumerate(self.windows, 1):
            tr = w.train_result.metrics if w.train_result else {}
            te = w.test_result.metrics  if w.test_result  else {}
            print(
                f"  창 {i}: Train {w.train_start[:4]}~{w.train_end[:4]}"
                f"  ret={tr.get('total_return', 0):.2%}"
                f"  │  Test {w.test_start[:4]}~{w.test_end[:4]}"
                f"  ret={te.get('total_return', 0):.2%}"
                f"  eff={w.efficiency:.2f}"
            )
        print(f"{'─'*55}")
        avg_eff = sum(w.efficiency for w in self.windows) / len(self.windows)
        print(f"  평균 효율: {avg_eff:.2f}  (목표: 0.5 이상)")


# 공식 Walk-Forward 창 정의
DEFAULT_WINDOWS = [
    WFWindow("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    WFWindow("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]


def run_walk_forward(
    agent_cls,
    df:       pd.DataFrame,
    params=None,
    windows:  list[WFWindow] | None = None,
) -> WalkForwardResult:
    """
    Walk-Forward 검증 실행.

    Args:
        agent_cls: AgentHarness 서브클래스 (인스턴스화 전)
        df:        전체 기간 OHLCV
        params:    에이전트 파라미터 (선택)
        windows:   WFWindow 리스트 (기본: DEFAULT_WINDOWS)

    Returns:
        WalkForwardResult
    """
    windows = windows or DEFAULT_WINDOWS

    for w in windows:
        agent = agent_cls(params) if params is not None else agent_cls()

        train_df = df.loc[w.train_start:w.train_end]
        test_df  = df.loc[w.test_start:w.test_end]

        train_signals = agent.run(train_df)
        w.train_result = run_backtest(train_df, train_signals)

        # 테스트 창은 학습 없이 동일 파라미터 그대로 적용
        agent2 = agent_cls(params) if params is not None else agent_cls()
        test_signals = agent2.run(test_df)
        w.test_result = run_backtest(test_df, test_signals)

    summary = {
        "n_windows":   len(windows),
        "avg_efficiency": sum(w.efficiency for w in windows) / len(windows),
        "all_positive": all(
            (w.test_result.metrics.get("total_return", 0) > 0)
            for w in windows if w.test_result
        ),
    }
    return WalkForwardResult(windows=windows, summary=summary)
