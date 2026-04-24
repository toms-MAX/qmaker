"""비용 감도 분석 — 같은 50종목·2024 Test창에서 cost만 바꿔 비교."""

from __future__ import annotations

import logging

from data_capital.agents.gap_trading import GapTradingAgent
from data_capital.agents.mean_rev import MeanRevAgent
from data_capital.agents.momentum import MomentumAgent
from data_capital.agents.eod import EODAgent
from data_capital.agents.volatility import VolatilityAgent
from data_capital.agents.lev_decay import LevDecayAgent
from data_capital.agents.pairs import PairsAgent
from data_capital.backtest.multi_ticker import run_multi_ticker
from data_capital.core.harness import CostModel, KOSPI_ETF, KOSPI_STOCK
from data_capital.data.fetch import load_universe_ohlcv
from data_capital.screener.universe import load_universe

logging.basicConfig(level=logging.WARNING)

AGENTS = [
    ("gap_trading", GapTradingAgent),
    ("mean_rev",    MeanRevAgent),
    ("momentum",    MomentumAgent),
    ("eod",         EODAgent),
    ("volatility",  VolatilityAgent),
    ("lev_decay",   LevDecayAgent),
    ("pairs",       PairsAgent),
]

ZERO_COST = CostModel(commission=0, transfer_tax=0, education_tax=0, slippage=0)


def main() -> int:
    tickers = load_universe(None)
    ohlcv = load_universe_ohlcv(tickers, min_rows=200)
    print(f"유니버스: {len(ohlcv)} 종목")

    period = ("2024-01-01", "2024-12-31")

    print(f"\n{'='*80}")
    print(f"  비용 감도 (2024 Test): ZERO vs ETF(~0.09%) vs STOCK(~0.66%)")
    print(f"{'='*80}")
    print(f"  {'Agent':<13} {'Cost':<10} {'CAGR':>8} {'Sharpe':>7} {'PF':>6} {'WR':>6} {'Trades':>7}")
    print(f"  {'-'*13} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")

    for name, cls in AGENTS:
        for cost_name, cost_model in (("zero", ZERO_COST), ("etf", KOSPI_ETF), ("stock", KOSPI_STOCK)):
            try:
                res = run_multi_ticker(
                    cls, ohlcv, period[0], period[1],
                    cost_model=cost_model,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<13} {cost_name:<10} ERROR: {e}")
                continue
            m = res.metrics
            print(
                f"  {name:<13} {cost_name:<10} "
                f"{m['cagr']:>+7.2%} "
                f"{m['sharpe']:>+6.2f} "
                f"{m['profit_factor']:>5.2f} "
                f"{m['win_rate']:>5.1%} "
                f"{m['total_trades']:>7d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
