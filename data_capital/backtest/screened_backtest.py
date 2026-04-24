"""Screener-driven backtest — 매일 top K 선정 + 선정된 티커만 매매.

v1.5 업데이트: 20종목 1/N 분산 대신, 매일 screener가 상위 K(=3) 티커를
선정하고 그 티커들에 해당하는 신호만 실행한다. 1 포지션 = equity/K 자본.

이 모델은 라이브 운용과 훨씬 가깝다 — 매일 현황 기준 상위만 진입,
나머지는 기다림. Sharpe·PF가 의미 있는 수치가 된다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_capital.backtest.engine import BacktestResult, _calc_metrics, _empty_metrics
from data_capital.core.harness import CostModel, KOSPI_ETF, Signal
from data_capital.screener.screener import Screener


@dataclass
class ScreenedResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    screener_stats: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"  Total Return   : {m['total_return']:>8.2%}",
            f"  CAGR           : {m['cagr']:>8.2%}",
            f"  Sharpe Ratio   : {m['sharpe']:>8.2f}",
            f"  Max Drawdown   : {m['max_drawdown']:>8.2%}",
            f"  Win Rate       : {m['win_rate']:>8.2%}",
            f"  Profit Factor  : {m['profit_factor']:>8.2f}",
            f"  Total Trades   : {m['total_trades']:>8d}",
            f"  Signals Filtered: {self.screener_stats.get('signals_filtered', 0)}",
            f"  Signals Executed: {self.screener_stats.get('signals_executed', 0)}",
        ]
        return "ScreenedResult\n" + "\n".join(lines)


def _precompute_screener_by_date(
    ohlcv_map: Dict[str, pd.DataFrame],
    trading_dates: List[pd.Timestamp],
    screener: Screener,
    budget: float,
) -> Dict[pd.Timestamp, List[str]]:
    """각 거래일의 상위 K 티커를 미리 계산. OHLCV는 D-1까지만 사용."""
    result: Dict[pd.Timestamp, List[str]] = {}
    universe = list(ohlcv_map.keys())

    for date in trading_dates:
        # D 이전 데이터만 사용 (lookahead 방지)
        snapshot = {}
        for t, df in ohlcv_map.items():
            mask = df.index < date
            sliced = df.loc[mask]
            if len(sliced) >= 20:  # 유동성 계산 최소 window
                snapshot[t] = sliced
        if len(snapshot) < screener.final_top_k:
            result[date] = []
            continue
        sr = screener.run(
            date=date,
            ohlcv_map=snapshot,
            budget_per_agent=budget,
            universe=universe,
        )
        result[date] = sr.candidates
    return result


def _collect_signals_by_date(
    agent_cls,
    ohlcv_map: Dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
    params,
) -> Tuple[Dict[pd.Timestamp, List[Tuple[str, Signal]]], int]:
    """각 에이전트를 각 티커에 대해 돌려 신호를 수집, 날짜별로 index.

    Returns:
        by_date:       {date: [(ticker, signal), ...]}
        total_signals: 전체 신호 수
    """
    by_date: Dict[pd.Timestamp, List[Tuple[str, Signal]]] = defaultdict(list)
    total = 0
    for ticker, df in ohlcv_map.items():
        sliced = df.loc[period_start:period_end]
        if len(sliced) < 30:
            continue
        agent = agent_cls(params) if params is not None else agent_cls()
        signals = agent.run(sliced)
        for s in signals:
            by_date[s.date].append((ticker, s))
            total += 1
    return by_date, total


def _execute_trade(
    df: pd.DataFrame,
    sig: Signal,
    cost_model: CostModel,
    position_capital: float,
) -> Optional[dict]:
    """engine.py의 실행 로직과 동일. 실패 시 None."""
    date_idx = {d: i for i, d in enumerate(df.index)}
    if sig.date not in date_idx:
        return None
    i = date_idx[sig.date]

    exit_mode = getattr(sig, "exit_mode", "next_open")
    buy_cost  = cost_model.buy_rate
    sell_cost = cost_model.sell_rate

    if exit_mode == "pure_next_open":
        if i + 1 >= len(df):
            return None
        exit_price  = df.iloc[i + 1]["open"]
        exit_reason = "next_open"
        exit_date   = df.index[i + 1]
    elif exit_mode == "same_day":
        check_row   = df.iloc[i]
        fallback_px = check_row["close"]
        exit_date   = sig.date
    else:  # "next_open"
        if i + 1 >= len(df):
            return None
        check_row   = df.iloc[i + 1]
        fallback_px = check_row["open"]
        exit_date   = df.index[i + 1]

    entry = sig.entry_price * (1 + buy_cost * sig.direction)

    if exit_mode == "pure_next_open":
        pass  # already set
    elif sig.direction == +1:
        if check_row["low"] <= sig.stop_loss:
            exit_price, exit_reason = sig.stop_loss, "stop_loss"
        elif check_row["high"] >= sig.take_profit:
            exit_price, exit_reason = sig.take_profit, "take_profit"
        else:
            exit_price, exit_reason = fallback_px, exit_mode
    else:
        if check_row["high"] >= sig.stop_loss:
            exit_price, exit_reason = sig.stop_loss, "stop_loss"
        elif check_row["low"] <= sig.take_profit:
            exit_price, exit_reason = sig.take_profit, "take_profit"
        else:
            exit_price, exit_reason = fallback_px, exit_mode

    exit_net = exit_price * (1 - sell_cost * sig.direction)
    pnl_pct  = (exit_net - entry) / entry * sig.direction
    pnl_abs  = position_capital * pnl_pct
    return {
        "entry_date":  sig.date,
        "exit_date":   exit_date,
        "direction":   sig.direction,
        "entry_price": entry,
        "exit_price":  exit_net,
        "pnl_pct":     pnl_pct,
        "pnl_abs":     pnl_abs,
        "exit_reason": exit_reason,
    }


def run_screened_backtest(
    agent_cls,
    ohlcv_map: Dict[str, pd.DataFrame],
    period_start: str,
    period_end: str,
    params=None,
    max_positions: int = 3,
    cost_model: Optional[CostModel] = None,
    total_capital: float = 100_000_000,
    screener: Optional[Screener] = None,
    screener_by_date: Optional[Dict[pd.Timestamp, List[str]]] = None,
) -> ScreenedResult:
    """매일 screener 상위 K개 선정 후 해당 티커 신호만 실행하는 백테스트.

    Args:
        agent_cls:      AgentHarness 서브클래스
        ohlcv_map:      {ticker: OHLCV DataFrame} (전체 기간)
        period_start/end: 백테스트 기간
        max_positions: 동시 보유 최대 포지션 수 (screener top K와 동일)
        cost_model:    기본 KOSPI_ETF
        total_capital: 초기 자본
        screener:      커스텀 스크리너. None이면 기본 설정으로 생성.
    """
    cost_model = cost_model or KOSPI_ETF
    screener = screener or Screener(
        liquidity_top_n=max_positions * 2,
        final_top_k=max_positions,
    )

    # 1) 에이전트 신호 수집 (날짜별 index)
    by_date, total_signals = _collect_signals_by_date(
        agent_cls, ohlcv_map, period_start, period_end, params,
    )

    # 2) 거래일 목록 (정렬)
    trading_dates = sorted(by_date.keys())
    if not trading_dates:
        return ScreenedResult(
            equity_curve=pd.Series(dtype=float),
            trades=pd.DataFrame(),
            metrics=_empty_metrics(),
            screener_stats={"signals_total": 0, "signals_filtered": 0, "signals_executed": 0},
        )

    # 3) 각 거래일의 screener 상위 K 미리 계산 (외부 캐시 허용)
    if screener_by_date is None:
        screener_by_date = _precompute_screener_by_date(
            ohlcv_map, trading_dates, screener, budget=total_capital,
        )

    # 4) 일별 시뮬레이션
    equity = float(total_capital)
    equity_history: List[Tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(period_start), equity)
    ]
    trade_records: List[dict] = []
    filtered = 0
    executed = 0

    for date in trading_dates:
        top_k = screener_by_date.get(date, [])
        if not top_k:
            continue

        # 신호 중 top_k 에 속하는 티커만 선택, 티커당 1개 (첫 신호)
        seen_tickers: set = set()
        picked: List[Tuple[str, Signal]] = []
        for (ticker, sig) in by_date[date]:
            if ticker not in top_k:
                filtered += 1
                continue
            if ticker in seen_tickers:
                continue  # 같은 티커 중복 제거
            picked.append((ticker, sig))
            seen_tickers.add(ticker)
            if len(picked) >= max_positions:
                break

        if not picked:
            continue

        # 포지션당 capital = equity / max_positions
        position_capital = equity / max_positions

        day_pnl = 0.0
        for (ticker, sig) in picked:
            df = ohlcv_map[ticker]
            rec = _execute_trade(df, sig, cost_model, position_capital)
            if rec is None:
                continue
            rec["agent"] = agent_cls.__name__
            rec["ticker"] = ticker
            trade_records.append(rec)
            day_pnl += rec["pnl_abs"]
            executed += 1

        equity += day_pnl
        equity_history.append((date, equity))

    # 5) 메트릭
    trades = pd.DataFrame(trade_records)
    if not equity_history:
        return ScreenedResult(
            equity_curve=pd.Series(dtype=float),
            trades=trades,
            metrics=_empty_metrics(),
            screener_stats={"signals_total": total_signals, "signals_filtered": filtered, "signals_executed": executed},
        )

    equity_curve = pd.Series(
        [v for _, v in equity_history],
        index=[d for d, _ in equity_history],
        name="equity",
    )

    if trades.empty:
        metrics = _empty_metrics()
    else:
        metrics = _calc_metrics(equity_curve, trades, total_capital)

    return ScreenedResult(
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        screener_stats={
            "signals_total": total_signals,
            "signals_filtered": filtered,
            "signals_executed": executed,
        },
    )


# ─────────────────────────────────────────────
# Screened Walk-Forward
# ─────────────────────────────────────────────

DEFAULT_WF_WINDOWS = [
    ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-07-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]


@dataclass
class ScreenedWFReport:
    agent_name: str
    windows: List[dict]
    adopted: bool
    reason: str

    def print_summary(self) -> None:
        print(f"\n{'─'*92}")
        print(f"  Screened WF — {self.agent_name}")
        print(f"{'─'*92}")
        for i, w in enumerate(self.windows, 1):
            te = w["test"]["metrics"]
            ss = w["test"].get("screener_stats", {})
            print(
                f"  창 {i} Test [{w['test_range']}]: "
                f"ret={te.get('total_return', 0):>+7.2%} "
                f"CAGR={te.get('cagr', 0):>+7.2%} "
                f"Sharpe={te.get('sharpe', 0):>+6.2f} "
                f"MDD={te.get('max_drawdown', 0):>+6.2%} "
                f"trades={int(te.get('total_trades', 0)):>4d} "
                f"WR={te.get('win_rate', 0):>5.1%} "
                f"PF={te.get('profit_factor', 0):>5.2f} "
                f"({'PASS' if w['test_pass'] else 'FAIL'})"
            )
        flag = "ADOPT" if self.adopted else "REJECT"
        print(f"  → {flag}: {self.reason}")


def _passes(metrics: dict, sharpe_th: float, pf_th: float) -> bool:
    return (
        metrics.get("sharpe", 0.0) >= sharpe_th
        and metrics.get("profit_factor", 0.0) >= pf_th
        and metrics.get("total_trades", 0) >= 10  # 최소 샘플
    )


def run_screened_walk_forward(
    agent_name: str,
    agent_cls,
    ohlcv_map: Dict[str, pd.DataFrame],
    params=None,
    cost_model: Optional[CostModel] = None,
    max_positions: int = 3,
    windows: Optional[List[Tuple[str, str, str, str]]] = None,
    sharpe_threshold: float = 0.5,
    pf_threshold: float = 1.1,
    require_all_pass: bool = False,
    screener_by_date: Optional[Dict[pd.Timestamp, List[str]]] = None,
) -> ScreenedWFReport:
    """Screener-driven walk-forward. 기본: 과반(2/3) 창 통과 시 ADOPT."""
    windows = windows or DEFAULT_WF_WINDOWS
    results = []
    passes = []

    for (tr_s, tr_e, te_s, te_e) in windows:
        train = run_screened_backtest(
            agent_cls, ohlcv_map, tr_s, tr_e,
            params=params, max_positions=max_positions, cost_model=cost_model,
            screener_by_date=screener_by_date,
        )
        test = run_screened_backtest(
            agent_cls, ohlcv_map, te_s, te_e,
            params=params, max_positions=max_positions, cost_model=cost_model,
            screener_by_date=screener_by_date,
        )
        ok = _passes(test.metrics, sharpe_threshold, pf_threshold)
        results.append({
            "train_range": f"{tr_s}~{tr_e}",
            "test_range":  f"{te_s}~{te_e}",
            "train": {"metrics": train.metrics, "screener_stats": train.screener_stats},
            "test":  {"metrics": test.metrics,  "screener_stats": test.screener_stats},
            "test_pass": ok,
        })
        passes.append(ok)

    n = len(passes)
    passed = sum(passes)
    if require_all_pass:
        adopted = (passed == n)
        reason = f"{passed}/{n} 창 통과 (전창 필수)"
    else:
        threshold_n = (n + 1) // 2  # 과반
        adopted = passed >= threshold_n
        reason = f"{passed}/{n} 창 통과 (과반 {threshold_n}개 필요)"

    return ScreenedWFReport(
        agent_name=agent_name,
        windows=results,
        adopted=adopted,
        reason=reason,
    )
