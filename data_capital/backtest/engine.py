"""백테스트 엔진: Signal 리스트 → 성과 지표 계산."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_capital.core.harness import TRANSACTION_COST, Signal


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict

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
        ]
        return "BacktestResult\n" + "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    signals: list[Signal],
    cost_rate: float = TRANSACTION_COST,
    initial_capital: float = 100_000_000,  # 1억원
) -> BacktestResult:
    """
    시그널 리스트를 시뮬레이션하여 성과를 계산한다.

    포지션: 시그널 당일 시가 진입, 익절/손절 or 다음날 시가 청산.
    비용:   편도 cost_rate (기본 0.08%)

    Args:
        df:              OHLCV 데이터프레임
        signals:         AgentHarness.run() 결과
        cost_rate:       편도 거래 비용
        initial_capital: 초기 자본금

    Returns:
        BacktestResult
    """
    if not signals:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            equity_curve=empty,
            trades=pd.DataFrame(),
            metrics=_empty_metrics(),
        )

    date_idx = {d: i for i, d in enumerate(df.index)}
    trade_records = []
    capital = float(initial_capital)

    for sig in signals:
        if sig.date not in date_idx:
            continue
        i = date_idx[sig.date]

        exit_mode = getattr(sig, "exit_mode", "next_open")

        if exit_mode == "pure_next_open":
            # SL/TP 체크 없이 다음날 시가 청산
            if i + 1 >= len(df):
                continue
            exit_price  = df.iloc[i + 1]["open"]
            exit_reason = "next_open"
            exit_date   = df.index[i + 1]
        elif exit_mode == "same_day":
            check_row   = df.iloc[i]
            fallback_px = check_row["close"]
            exit_date   = sig.date
        else:  # "next_open" (SL/TP 체크 포함)
            if i + 1 >= len(df):
                continue
            check_row   = df.iloc[i + 1]
            fallback_px = check_row["open"]
            exit_date   = df.index[i + 1]

        entry = sig.entry_price * (1 + cost_rate * sig.direction)

        if exit_mode == "pure_next_open":
            pass  # exit_price already set
        elif sig.direction == +1:
            if check_row["low"] <= sig.stop_loss:
                exit_price = sig.stop_loss
                exit_reason = "stop_loss"
            elif check_row["high"] >= sig.take_profit:
                exit_price = sig.take_profit
                exit_reason = "take_profit"
            else:
                exit_price = fallback_px
                exit_reason = exit_mode
        else:  # 숏
            if check_row["high"] >= sig.stop_loss:
                exit_price = sig.stop_loss
                exit_reason = "stop_loss"
            elif check_row["low"] <= sig.take_profit:
                exit_price = sig.take_profit
                exit_reason = "take_profit"
            else:
                exit_price = fallback_px
                exit_reason = exit_mode

        exit_net = exit_price * (1 - cost_rate * sig.direction)
        pnl_pct  = (exit_net - entry) / entry * sig.direction
        pnl_abs  = capital * sig.size * pnl_pct
        capital += pnl_abs

        trade_records.append({
            "entry_date":  sig.date,
            "exit_date":   exit_date,
            "direction":   sig.direction,
            "entry_price": entry,
            "exit_price":  exit_net,
            "pnl_pct":     pnl_pct,
            "pnl_abs":     pnl_abs,
            "capital":     capital,
            "exit_reason": exit_reason,
        })

    trades = pd.DataFrame(trade_records)
    if trades.empty:
        return BacktestResult(pd.Series(dtype=float), trades, _empty_metrics())

    # 자본금 곡선
    equity_curve = pd.Series(
        trades["capital"].values,
        index=trades["exit_date"],
        name="equity",
    )
    equity_curve = pd.concat([
        pd.Series([float(initial_capital)], index=[df.index[0]]),
        equity_curve,
    ])

    metrics = _calc_metrics(equity_curve, trades, initial_capital)
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _calc_metrics(equity: pd.Series, trades: pd.DataFrame, initial: float) -> dict:
    total_return = (equity.iloc[-1] - initial) / initial
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = (1 + total_return) ** (1 / n_years) - 1

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    winners = trades[trades["pnl_abs"] > 0]["pnl_abs"]
    losers  = trades[trades["pnl_abs"] <= 0]["pnl_abs"].abs()
    win_rate = len(winners) / len(trades) if len(trades) else 0.0
    profit_factor = (winners.sum() / losers.sum()) if losers.sum() > 0 else float("inf")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": len(trades),
    }


def _empty_metrics() -> dict:
    return {k: 0.0 for k in (
        "total_return", "cagr", "sharpe", "max_drawdown",
        "win_rate", "profit_factor", "total_trades",
    )}
