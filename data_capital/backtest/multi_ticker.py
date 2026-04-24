"""멀티티커 백테스트 — 균등배분 포트폴리오 시뮬.

v1.5 Step 5: 각 에이전트를 N개 종목에서 독립 백테스트한 뒤,
종목별 1/N 자본으로 균등 배분한 합산 equity curve로 통합 성과를 산출한다.

이 구조는 "이 에이전트가 KOSPI 대표 종목군에서 알파를 유지하는가?"
라는 질문에 답하는 것이 목적이다. 실제 라이브 운용의 포지션 제한·
screener 회전은 별도 레이어(main.py)에서 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_capital.backtest.engine import BacktestResult, _calc_metrics, _empty_metrics, run_backtest
from data_capital.core.harness import CostModel, KOSPI_STOCK


@dataclass
class MultiTickerResult:
    per_ticker: Dict[str, BacktestResult]
    aggregate_equity: pd.Series
    aggregate_trades: pd.DataFrame
    metrics: dict
    adopted: bool
    reason: str

    def summary(self) -> str:
        m = self.metrics
        flag = "ADOPT" if self.adopted else "REJECT"
        lines = [
            f"  Tickers        : {len(self.per_ticker)}",
            f"  Total Return   : {m['total_return']:>8.2%}",
            f"  CAGR           : {m['cagr']:>8.2%}",
            f"  Sharpe Ratio   : {m['sharpe']:>8.2f}",
            f"  Max Drawdown   : {m['max_drawdown']:>8.2%}",
            f"  Win Rate       : {m['win_rate']:>8.2%}",
            f"  Profit Factor  : {m['profit_factor']:>8.2f}",
            f"  Total Trades   : {m['total_trades']:>8d}",
            f"  Decision       : {flag} — {self.reason}",
        ]
        return "MultiTickerResult\n" + "\n".join(lines)


def run_multi_ticker(
    agent_cls,
    ohlcv_map: Dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
    params=None,
    cost_model: Optional[CostModel] = None,
    total_capital: float = 100_000_000,
    cagr_threshold: float = 0.02,
    sharpe_threshold: float = 0.5,
) -> MultiTickerResult:
    """N 종목 균등배분 멀티티커 백테스트.

    Args:
        agent_cls:        AgentHarness 서브클래스
        ohlcv_map:        {ticker: OHLCV DataFrame}
        period_start/end: 백테스트 기간 (YYYY-MM-DD)
        params:           에이전트 파라미터
        cost_model:       CostModel. 기본 KOSPI_STOCK.
        total_capital:    총 자본금
        cagr_threshold:   채택 최소 연환산
        sharpe_threshold: 채택 최소 Sharpe

    Returns:
        MultiTickerResult
    """
    cost_model = cost_model or KOSPI_STOCK
    usable = {t: df.loc[period_start:period_end] for t, df in ohlcv_map.items()}
    usable = {t: df for t, df in usable.items() if len(df) >= 30}
    if not usable:
        return MultiTickerResult(
            per_ticker={}, aggregate_equity=pd.Series(dtype=float),
            aggregate_trades=pd.DataFrame(), metrics=_empty_metrics(),
            adopted=False, reason="데이터 없음",
        )

    n = len(usable)
    per_ticker_capital = total_capital / n

    per_ticker: Dict[str, BacktestResult] = {}
    all_trades: List[pd.DataFrame] = []

    for ticker, df in usable.items():
        agent = agent_cls(params) if params is not None else agent_cls()
        signals = agent.run(df)
        res = run_backtest(
            df, signals,
            cost_model=cost_model,
            initial_capital=per_ticker_capital,
        )
        per_ticker[ticker] = res
        if not res.trades.empty:
            trades_t = res.trades.copy()
            trades_t["ticker"] = ticker
            all_trades.append(trades_t)

    # 일별 equity 합산
    aggregate_equity = _aggregate_equity_curves(per_ticker, total_capital)
    aggregate_trades = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades else pd.DataFrame()
    )

    metrics = (
        _calc_metrics(aggregate_equity, aggregate_trades, total_capital)
        if not aggregate_equity.empty else _empty_metrics()
    )

    adopted, reason = _decision(metrics, cagr_threshold, sharpe_threshold)
    return MultiTickerResult(
        per_ticker=per_ticker,
        aggregate_equity=aggregate_equity,
        aggregate_trades=aggregate_trades,
        metrics=metrics,
        adopted=adopted,
        reason=reason,
    )


def _aggregate_equity_curves(
    per_ticker: Dict[str, BacktestResult],
    total_initial: float,
) -> pd.Series:
    """종목별 equity curve를 동일 날짜축에서 합산.

    각 종목은 자기 자본을 받아 거래한다. 거래가 없는 날은 직전 equity 유지.
    일별 union 인덱스에 forward-fill 후 합산하면 전체 포트폴리오 equity.
    """
    if not per_ticker:
        return pd.Series(dtype=float)

    n = len(per_ticker)
    per_ticker_initial = total_initial / n

    all_dates = pd.DatetimeIndex([])
    for res in per_ticker.values():
        if not res.equity_curve.empty:
            all_dates = all_dates.union(res.equity_curve.index)
    if len(all_dates) == 0:
        return pd.Series(dtype=float)
    all_dates = all_dates.sort_values()

    aggregate = pd.Series(0.0, index=all_dates)
    for res in per_ticker.values():
        if res.equity_curve.empty:
            reindexed = pd.Series(per_ticker_initial, index=all_dates)
        else:
            reindexed = res.equity_curve.reindex(all_dates).ffill()
            reindexed = reindexed.fillna(per_ticker_initial)
        aggregate = aggregate + reindexed
    return aggregate


def _decision(
    metrics: dict, cagr_th: float, sharpe_th: float,
) -> Tuple[bool, str]:
    cagr = metrics.get("cagr", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    if cagr >= cagr_th and sharpe >= sharpe_th:
        return True, f"CAGR {cagr:.2%} ≥ {cagr_th:.0%}, Sharpe {sharpe:.2f} ≥ {sharpe_th:.1f}"
    fails = []
    if cagr < cagr_th:
        fails.append(f"CAGR {cagr:.2%} < {cagr_th:.0%}")
    if sharpe < sharpe_th:
        fails.append(f"Sharpe {sharpe:.2f} < {sharpe_th:.1f}")
    return False, ", ".join(fails)


# ─────────────────────────────────────────────
# Walk-Forward (multi-ticker)
# ─────────────────────────────────────────────

DEFAULT_WF_WINDOWS = [
    ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-07-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]


@dataclass
class WalkForwardReport:
    agent_name: str
    windows: List[dict] = field(default_factory=list)
    adopted: bool = False
    reason: str = ""

    def print_summary(self) -> None:
        print(f"\n{'─'*90}")
        print(f"  Walk-Forward (multi-ticker) — {self.agent_name}")
        print(f"{'─'*90}")
        for i, w in enumerate(self.windows, 1):
            te = w["test"]["metrics"]
            print(
                f"  창 {i} Test [{w['test_range']}]: "
                f"ret={te.get('total_return', 0):>+7.2%} "
                f"CAGR={te.get('cagr', 0):>+7.2%} "
                f"Sharpe={te.get('sharpe', 0):>+6.2f} "
                f"MDD={te.get('max_drawdown', 0):>+6.2%} "
                f"trades={te.get('total_trades', 0):>4d} "
                f"WR={te.get('win_rate', 0):>5.1%} "
                f"PF={te.get('profit_factor', 0):>5.2f} "
                f"({'PASS' if w['test_pass'] else 'FAIL'})"
            )
        flag = "ADOPT" if self.adopted else "REJECT"
        print(f"  → {flag}: {self.reason}")


def run_walk_forward_multi(
    agent_name: str,
    agent_cls,
    ohlcv_map: Dict[str, pd.DataFrame],
    params=None,
    cost_model: Optional[CostModel] = None,
    windows: Optional[List[Tuple[str, str, str, str]]] = None,
    cagr_threshold: float = 0.02,
    sharpe_threshold: float = 0.5,
    require_all_pass: bool = True,
) -> WalkForwardReport:
    """멀티티커 Walk-Forward. 3창 모두 Test 기준 통과 시 ADOPT."""
    windows = windows or DEFAULT_WF_WINDOWS

    results = []
    passes = []

    for (tr_s, tr_e, te_s, te_e) in windows:
        train = run_multi_ticker(
            agent_cls, ohlcv_map, tr_s, tr_e,
            params=params, cost_model=cost_model,
            cagr_threshold=cagr_threshold, sharpe_threshold=sharpe_threshold,
        )
        test = run_multi_ticker(
            agent_cls, ohlcv_map, te_s, te_e,
            params=params, cost_model=cost_model,
            cagr_threshold=cagr_threshold, sharpe_threshold=sharpe_threshold,
        )
        results.append({
            "train_range": f"{tr_s}~{tr_e}",
            "test_range":  f"{te_s}~{te_e}",
            "train": {"metrics": train.metrics},
            "test":  {"metrics": test.metrics},
            "test_pass": test.adopted,
        })
        passes.append(test.adopted)

    if require_all_pass:
        adopted = all(passes)
        reason = f"{sum(passes)}/{len(passes)} 창 통과 (전창 통과 필요)"
    else:
        adopted = sum(passes) >= (len(passes) + 1) // 2  # 과반
        reason = f"{sum(passes)}/{len(passes)} 창 통과 (과반 통과 기준)"

    return WalkForwardReport(
        agent_name=agent_name,
        windows=results,
        adopted=adopted,
        reason=reason,
    )
