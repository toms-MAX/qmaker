"""Step 5: 7개 에이전트 멀티티커 Walk-Forward 재검증.

사용법:
    PYTHONPATH=/home/user/qmaker .venv/bin/python -m data_capital.scripts.revalidate_agents
    PYTHONPATH=/home/user/qmaker .venv/bin/python -m data_capital.scripts.revalidate_agents --universe etf
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict

import pandas as pd

from data_capital.agents.gap_trading import GapTradingAgent
from data_capital.agents.mean_rev import MeanRevAgent
from data_capital.agents.momentum import MomentumAgent
from data_capital.agents.eod import EODAgent
from data_capital.agents.volatility import VolatilityAgent
from data_capital.agents.lev_decay import LevDecayAgent
from data_capital.agents.pairs import PairsAgent
from data_capital.backtest.multi_ticker import WalkForwardReport, run_walk_forward_multi
from data_capital.backtest.screened_backtest import (
    ScreenedWFReport,
    run_screened_walk_forward,
)
from data_capital.core.harness import KOSPI_ETF, KOSPI_STOCK
from data_capital.data.fetch import load_universe_ohlcv
from data_capital.screener.universe import load_etf_universe, load_universe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

AGENTS = [
    ("gap_trading", GapTradingAgent),
    ("mean_rev",    MeanRevAgent),
    ("momentum",    MomentumAgent),
    ("eod",         EODAgent),
    ("volatility",  VolatilityAgent),
    ("lev_decay",   LevDecayAgent),
    ("pairs",       PairsAgent),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe", choices=["stock", "etf"], default="etf",
        help="stock=KOSPI 200 스냅샷 union(223종), etf=KOSPI/KOSDAQ ETF 20종 (기본)",
    )
    parser.add_argument(
        "--mode", choices=["dispersion", "screened"], default="screened",
        help="dispersion=1/N 분산, screened=매일 top K 선정 (기본)",
    )
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    if args.universe == "etf":
        tickers = load_etf_universe()
        cost_model = KOSPI_ETF
        label = "ETF"
    else:
        tickers = load_universe(None)
        cost_model = KOSPI_STOCK
        label = "STOCK"

    ohlcv_map: Dict[str, pd.DataFrame] = load_universe_ohlcv(tickers, min_rows=args.min_rows)
    print(f"유니버스({label}): {len(ohlcv_map)} 종목 (요청 {len(tickers)})")
    print(f"비용 모델: {cost_model} (왕복 {cost_model.round_trip:.3%})")

    if len(ohlcv_map) < 5:
        print("수집된 데이터 부족 — 먼저 fetch 실행")
        return 1

    reports = []
    for name, cls in AGENTS:
        try:
            if args.mode == "screened":
                rep = run_screened_walk_forward(
                    name, cls, ohlcv_map,
                    cost_model=cost_model,
                    max_positions=args.max_positions,
                )
            else:
                rep = run_walk_forward_multi(name, cls, ohlcv_map, cost_model=cost_model)
        except Exception as e:  # noqa: BLE001
            print(f"\n[{name}] 실행 실패: {e}")
            continue
        rep.print_summary()
        reports.append(rep)

    adopted = [r for r in reports if r.adopted]
    rejected = [r for r in reports if not r.adopted]

    print("\n" + "=" * 70)
    print(f"  채택: {len(adopted)}  폐기 후보: {len(rejected)}")
    print("=" * 70)
    for r in adopted:
        print(f"  ✓ {r.agent_name}")
    for r in rejected:
        print(f"  ✗ {r.agent_name}: {r.reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
