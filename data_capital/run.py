"""
DATA CAPITAL — 진입점.

사용법:
    python run.py backtest          # 전체 에이전트 병렬 백테스트
    python run.py walkforward       # 갭트레이딩 Walk-Forward 검증
    python run.py live              # 실시간 매매 루프 (미완성)
"""

import sys

import pandas as pd


def cmd_backtest():
    from data_capital.data.fetch import load_processed
    from data_capital.backtest.runner import run_all_agents

    df = load_processed("069500")
    results = run_all_agents(df)
    for name, r in results.items():
        print(f"\n[{name}]")
        print(r.summary())


def cmd_walkforward():
    from data_capital.data.fetch import load_processed
    from data_capital.agents.gap_trading import GapTradingAgent
    from data_capital.backtest.walkforward import run_walk_forward

    df = load_processed("069500")
    result = run_walk_forward(GapTradingAgent, df)
    result.print_summary()


def cmd_live():
    raise NotImplementedError("실시간 매매는 아직 준비 중입니다.")


COMMANDS = {
    "backtest":    cmd_backtest,
    "walkforward": cmd_walkforward,
    "live":        cmd_live,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "walkforward"
    if cmd not in COMMANDS:
        print(f"사용법: python run.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[cmd]()
