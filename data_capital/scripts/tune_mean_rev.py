"""Step 6: mean_rev 에이전트 파라미터 그리드 튜닝 1회.

ETF 유니버스 + screener top 3 + 비용 0.09% 기준.
목표: 과반 창(2/3) 통과 + 각 창 최소 15 거래.
"""

from __future__ import annotations

import itertools
import logging
import sys
from dataclasses import asdict
from typing import Dict

import pandas as pd

from data_capital.agents.mean_rev import MeanRevAgent, MeanRevParams
from data_capital.backtest.screened_backtest import (
    ScreenedWFReport,
    _precompute_screener_by_date,
    run_screened_walk_forward,
)
from data_capital.core.harness import KOSPI_ETF
from data_capital.data.fetch import load_universe_ohlcv
from data_capital.screener.screener import Screener
from data_capital.screener.universe import load_etf_universe

logging.basicConfig(level=logging.WARNING)

# 16 조합 그리드
GRID = {
    "rsi_entry": [30.0, 40.0],
    "bb_std":    [1.5, 2.0],
    "stop_pct":  [0.003, 0.007],
    "take_pct":  [0.010, 0.020],
}


def score(rep: ScreenedWFReport) -> tuple:
    """순위 키: (통과창수, 평균 Sharpe, 평균 PF)."""
    passed = sum(1 for w in rep.windows if w["test_pass"])
    test_metrics = [w["test"]["metrics"] for w in rep.windows]
    avg_sharpe = sum(m.get("sharpe", 0.0) for m in test_metrics) / len(test_metrics)
    avg_pf = sum(m.get("profit_factor", 0.0) for m in test_metrics) / len(test_metrics)
    avg_trades = sum(m.get("total_trades", 0) for m in test_metrics) / len(test_metrics)
    return (passed, avg_sharpe, avg_pf, avg_trades)


def main() -> int:
    tickers = load_etf_universe()
    ohlcv = load_universe_ohlcv(tickers, min_rows=200)
    print(f"유니버스: {len(ohlcv)} ETF")

    # screener 출력은 파라미터 불문 동일 → 1회만 precompute, 16조합 공유
    print("screener top-3 precompute…")
    screener = Screener(liquidity_top_n=6, final_top_k=3)
    # 전체 2022-01 ~ 2024-12 범위 일자 집합
    all_dates = sorted({d for df in ohlcv.values() for d in df.index})
    all_dates = [d for d in all_dates if pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2024-12-31")]
    screener_cache = _precompute_screener_by_date(
        ohlcv, all_dates, screener, budget=100_000_000,
    )
    print(f"  → {len(screener_cache)} 일자 캐싱 완료")

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"그리드: {len(combos)} 조합")

    results = []
    for idx, values in enumerate(combos, 1):
        params = MeanRevParams(**dict(zip(keys, values)))
        rep = run_screened_walk_forward(
            f"mean_rev_v{idx}",
            MeanRevAgent,
            ohlcv,
            params=params,
            cost_model=KOSPI_ETF,
            max_positions=3,
            screener_by_date=screener_cache,
        )
        s = score(rep)
        passed = s[0]
        avg_sharpe = s[1]
        avg_pf = s[2]
        avg_trades = s[3]
        print(
            f"[{idx:2d}/{len(combos)}] "
            f"rsi={params.rsi_entry:.0f} bb={params.bb_std} "
            f"sl={params.stop_pct:.3f} tp={params.take_pct:.3f} "
            f"│ pass={passed}/3 Sharpe̅={avg_sharpe:>+5.2f} PF̅={avg_pf:>4.2f} trades̅={avg_trades:>4.1f} "
            f"{'★' if rep.adopted else ' '}"
        )
        results.append((params, rep, s))

    # 상위 5개
    results.sort(key=lambda x: x[2], reverse=True)
    print("\n" + "=" * 80)
    print("  Top 5 (통과창수 → 평균Sharpe → 평균PF 순)")
    print("=" * 80)
    for i, (params, rep, s) in enumerate(results[:5], 1):
        print(f"\n[{i}] params: {asdict(params)}")
        rep.print_summary()

    print("\n" + "=" * 80)
    best_params, best_rep, best_score = results[0]
    if best_rep.adopted:
        print(f"  최종 선택 (ADOPT): {asdict(best_params)}")
    else:
        print(f"  최고 성적도 채택 기준 미달 — mean_rev 폐기 권장")
        print(f"  best: pass {best_score[0]}/3, Sharpe̅ {best_score[1]:.2f}, PF̅ {best_score[2]:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
