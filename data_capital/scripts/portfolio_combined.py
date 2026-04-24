"""Step 5b: 채택 4 에이전트 통합 백테스트.

각 에이전트에 총 자본의 25%씩 독립 할당하고, equity curve를 합산해
포트폴리오 레벨 메트릭을 계산한다. CIO/Guardian 동적 배분은 적용하지 않는다
(스켈레톤 상태라 잡음만 유발).

검증 기간: DEFAULT_WF_WINDOWS 의 3번째 test 창 직전까지의 전체(2022-01~2024-12).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_capital.agents.eod import EODAgent
from data_capital.agents.mean_rev import MeanRevAgent
from data_capital.agents.pairs import PairsAgent
from data_capital.agents.volatility import VolatilityAgent
from data_capital.backtest.screened_backtest import (
    _precompute_screener_by_date,
    run_screened_backtest,
)
from data_capital.core.harness import KOSPI_ETF
from data_capital.data.fetch import load_universe_ohlcv
from data_capital.screener.screener import Screener
from data_capital.screener.universe import load_etf_universe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ADOPTED = [
    ("mean_rev",   MeanRevAgent),
    ("eod",        EODAgent),
    ("volatility", VolatilityAgent),
    ("pairs",      PairsAgent),
]


def _year_fraction(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 0.0
    return (idx[-1] - idx[0]).days / 365.25


def _portfolio_metrics(equity: pd.Series, initial: float) -> dict:
    if equity.empty:
        return {}
    returns = equity.pct_change().dropna()
    total_ret = equity.iloc[-1] / initial - 1.0
    yrs = _year_fraction(equity.index)
    cagr = (1 + total_ret) ** (1 / yrs) - 1.0 if yrs > 0 else 0.0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    peak = equity.cummax()
    mdd = float((equity / peak - 1).min())
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-start", default="2022-01-01")
    parser.add_argument("--period-end",   default="2024-12-31")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--total-capital", type=float, default=100_000_000)
    args = parser.parse_args()

    tickers = load_etf_universe()
    ohlcv_map = load_universe_ohlcv(tickers, min_rows=200)
    print(f"ETF 유니버스: {len(ohlcv_map)}/{len(tickers)} 종목")
    print(f"기간: {args.period_start} ~ {args.period_end}")
    print(f"총 자본: {args.total_capital:,.0f}, 에이전트당: {args.total_capital/len(ADOPTED):,.0f}")

    # 1) Screener pre-compute (에이전트 간 공유)
    screener = Screener(
        liquidity_top_n=args.max_positions * 2,
        final_top_k=args.max_positions,
    )
    period_dates = pd.date_range(args.period_start, args.period_end, freq="B")
    # 실 거래일만 — ohlcv 인덱스에서 추출
    all_dates = sorted(set().union(*(df.index for df in ohlcv_map.values())))
    trading_dates = [d for d in all_dates if pd.Timestamp(args.period_start) <= d <= pd.Timestamp(args.period_end)]
    print(f"거래일 수: {len(trading_dates)}")

    agent_capital = args.total_capital / len(ADOPTED)

    # 2) 에이전트별 독립 실행
    results: Dict[str, object] = {}
    equity_series: List[pd.Series] = []
    for name, cls in ADOPTED:
        res = run_screened_backtest(
            cls, ohlcv_map,
            args.period_start, args.period_end,
            max_positions=args.max_positions,
            cost_model=KOSPI_ETF,
            total_capital=agent_capital,
            screener=screener,
        )
        results[name] = res
        m = res.metrics
        print(f"\n[{name}]")
        print(f"  total_return={m.get('total_return',0):+.2%}  "
              f"CAGR={m.get('cagr',0):+.2%}  "
              f"Sharpe={m.get('sharpe',0):+.2f}  "
              f"MDD={m.get('max_drawdown',0):+.2%}  "
              f"trades={int(m.get('total_trades',0))}  "
              f"WR={m.get('win_rate',0):.1%}  "
              f"PF={m.get('profit_factor',0):.2f}")
        if not res.equity_curve.empty:
            equity_series.append(res.equity_curve.rename(name))

    if not equity_series:
        print("equity 데이터 없음")
        return 1

    # 3) 포트폴리오 equity curve = 각 에이전트 equity 합산 (일별)
    combined = pd.concat(equity_series, axis=1).ffill()
    # 각 에이전트 첫 equity 없음인 경우 초기 자본으로 채움
    for col in combined.columns:
        combined[col] = combined[col].fillna(agent_capital)
    portfolio_equity = combined.sum(axis=1)
    portfolio_equity.name = "portfolio"

    # 4) 포트폴리오 메트릭
    port_metrics = _portfolio_metrics(portfolio_equity, args.total_capital)
    print("\n" + "=" * 70)
    print("  포트폴리오 통합 (equal-weight 25%)")
    print("=" * 70)
    print(f"  total_return = {port_metrics.get('total_return',0):+.2%}")
    print(f"  CAGR         = {port_metrics.get('cagr',0):+.2%}")
    print(f"  Sharpe       = {port_metrics.get('sharpe',0):+.2f}")
    print(f"  MDD          = {port_metrics.get('max_drawdown',0):+.2%}")

    # 5) 상관관계 (에이전트 일간 수익률)
    daily_returns = combined.pct_change().dropna(how="all")
    corr = daily_returns.corr()
    print("\n  에이전트 수익률 상관계수:")
    print(corr.round(2).to_string())
    avg_corr = corr.values[np.triu_indices_from(corr, k=1)].mean()
    print(f"\n  평균 상호 상관: {avg_corr:+.2f}  (낮을수록 분산효과 ↑)")

    # 6) 최소 성공 기준 비교
    print("\n  v1.5 기준 대비:")
    print(f"    연환산(CAGR)  {port_metrics.get('cagr',0):+.2%}  (최소 ≥ 0%, 목표 ≥ 8%)")
    print(f"    MDD          {port_metrics.get('max_drawdown',0):+.2%}  (최소 ≥ -10%, 목표 ≥ -7%)")
    print(f"    Sharpe       {port_metrics.get('sharpe',0):+.2f}  (최소 ≥ 0.5, 목표 ≥ 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
