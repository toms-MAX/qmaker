"""병렬 백테스트 러너 — concurrent.futures 기반."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from data_capital.backtest.engine import BacktestResult, run_backtest

if TYPE_CHECKING:
    from data_capital.core.harness import AgentHarness


@dataclass
class RunSpec:
    """에이전트 1개 실행 명세."""
    name:      str
    agent_cls: type
    params:    object | None
    df:        pd.DataFrame


def _run_one(spec: RunSpec) -> tuple[str, BacktestResult]:
    agent   = spec.agent_cls(spec.params) if spec.params is not None else spec.agent_cls()
    signals = agent.run(spec.df)
    result  = run_backtest(spec.df, signals)
    return spec.name, result


def run_parallel(
    specs:   list[RunSpec],
    workers: int = 4,
) -> dict[str, BacktestResult]:
    """
    여러 에이전트를 병렬로 백테스트한다.

    Args:
        specs:   RunSpec 리스트
        workers: 최대 병렬 워커 수

    Returns:
        {name: BacktestResult} dict
    """
    results: dict[str, BacktestResult] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, spec): spec.name for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result
    return results


def run_all_agents(
    df:      pd.DataFrame,
    workers: int = 4,
) -> dict[str, BacktestResult]:
    """
    7개 에이전트 전체를 병렬 백테스트한다.
    """
    from data_capital.agents.gap_trading import GapTradingAgent
    from data_capital.agents.mean_rev import MeanRevAgent
    from data_capital.agents.momentum import MomentumAgent
    from data_capital.agents.eod import EODAgent
    from data_capital.agents.volatility import VolatilityAgent
    from data_capital.agents.lev_decay import LevDecayAgent
    from data_capital.agents.pairs import PairsAgent

    specs = [
        RunSpec("gap_trading", GapTradingAgent, None, df),
        RunSpec("mean_rev",    MeanRevAgent,    None, df),
        RunSpec("momentum",    MomentumAgent,   None, df),
        RunSpec("eod",         EODAgent,        None, df),
        RunSpec("volatility",  VolatilityAgent, None, df),
        RunSpec("lev_decay",   LevDecayAgent,   None, df),
        RunSpec("pairs",       PairsAgent,      None, df),
    ]
    return run_parallel(specs, workers)
