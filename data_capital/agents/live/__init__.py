# live/ 패키지: 실시간 신호 생성용 7개 에이전트
# 백테스트용 에이전트는 agents/*.py 에 위치
from agents.live.gap_trading import LiveGapTradingAgent
from agents.live.mean_rev    import LiveMeanRevAgent
from agents.live.momentum    import LiveMomentumAgent
from agents.live.pairs       import LivePairsAgent
from agents.live.eod         import LiveEODAgent
from agents.live.volatility  import LiveVolatilityAgent
from agents.live.lev_decay   import LiveLevDecayAgent


def create_all_live_agents(total_capital: float) -> dict:
    base = total_capital / 7
    return {
        "gap_trading": LiveGapTradingAgent(base),
        "mean_rev":    LiveMeanRevAgent(base),
        "momentum":    LiveMomentumAgent(base),
        "pairs":       LivePairsAgent(base),
        "eod":         LiveEODAgent(base),
        "volatility":  LiveVolatilityAgent(base),
        "lev_decay":   LiveLevDecayAgent(base),
    }
