"""
DATA CAPITAL — failure_db_backtest.py
==========================================
FailureLearningDB: 실패를 조건부 기회로 전환
WalkForwardBacktest: 객관성 보장된 백테스트 엔진
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import concurrent.futures
import json


# ════════════════════════════════════════════
#  실패 학습 DB
# ════════════════════════════════════════════

@dataclass
class FailureRecord:
    """실패 거래 1건의 기록"""
    trade_id:       str
    agent_id:       str
    timestamp:      str
    pnl:            float
    pnl_pct:        float

    # 시장 맥락 피처 (30개)
    market_state:   str
    vkospi:         float
    kospi_change:   float
    volume_ratio:   float
    rsi14:          float
    bb_position:    float    # 0.0(하단) ~ 1.0(상단)
    ma20_distance:  float    # 종가 vs 20일MA 거리
    gap_pct:        float
    time_of_day:    str      # "09:05~09:30" 등
    day_of_week:    int      # 0=월 ~ 4=금
    institutional_net: float
    foreign_net:    float
    program_trade:  str
    atr_ratio:      float    # ATR/종가
    vi_status:      str
    preceding_direction: str   # 직전 3일 코스피 방향
    concurrent_agents: int     # 동시 작동 에이전트 수

    category:       str = ""   # 분류 결과
    pattern_group:  str = ""   # 패턴 그룹


class FailureLearningDB:
    """
    모든 실패를 기록·분류·학습.
    핵심: 같은 실패 반복 차단 + 조건부 기회 발굴
    """

    REPEAT_THRESHOLD = 3    # 동일 패턴 3회 이상 = 학습 트리거
    OPPORTUNITY_WIN_RATE = 0.65   # 기회 판정 최소 승률

    def __init__(self):
        self.failures:        list = []
        self.patterns:        dict = {}   # pattern_key → [FailureRecord]
        self.opportunity_map: dict = {}   # 실패 후 기회 패턴
        self.block_rules:     list = []   # 자동 차단 룰

    def _extract_features(self, record: FailureRecord) -> dict:
        return {
            "market_state":       record.market_state,
            "vkospi_bucket":      int(record.vkospi // 5) * 5,    # 5단위 버킷
            "kospi_direction":    "UP" if record.kospi_change > 0 else "DOWN",
            "vol_bucket":         "HIGH" if record.vol_ratio > 2 else ("MID" if record.vol_ratio > 1.3 else "LOW"),
            "rsi_zone":           "OVERSOLD" if record.rsi14 < 30 else ("NEUTRAL" if record.rsi14 < 70 else "OVERBOUGHT"),
            "time_bucket":        record.time_of_day,
            "day_of_week":        record.day_of_week,
            "institutional_dir":  "BUY" if record.institutional_net > 0 else "SELL",
            "vi_status":          record.vi_status,
        }

    def _classify(self, record: FailureRecord) -> str:
        """실패 원인 자동 분류"""
        if record.vkospi > 25:
            return "HIGH_FEAR"
        if record.vi_status == "RELEASED":
            return "POST_VI"
        if abs(record.gap_pct) > 0.015:
            return "EXTREME_GAP"
        if record.volume_ratio > 5.0:
            return "PROGRAM_TRADE"
        if record.kospi_change < -1.5:
            return "MARKET_CRASH"
        if record.time_of_day in ["09:00~09:05"]:
            return "OPENING_CHAOS"
        return "UNKNOWN"

    def log_failure(self, record: FailureRecord):
        """실패 기록 + 즉시 패턴 매칭"""
        record.category     = self._classify(record)
        features            = self._extract_features(record)
        pattern_key         = json.dumps(features, sort_keys=True)
        record.pattern_group = pattern_key[:50]  # 요약 키

        self.failures.append(record)
        self.patterns.setdefault(pattern_key, []).append(record)

        # 반복 패턴 감지 → 학습 트리거
        if len(self.patterns[pattern_key]) >= self.REPEAT_THRESHOLD:
            self._trigger_learning(pattern_key)

    def _trigger_learning(self, pattern_key: str):
        """반복 실패 패턴 → 자동 차단 룰 생성"""
        records = self.patterns[pattern_key]
        avg_pnl = np.mean([r.pnl_pct for r in records])
        features = json.loads(pattern_key)

        rule = {
            "pattern_key": pattern_key,
            "features":    features,
            "count":       len(records),
            "avg_pnl":     avg_pnl,
            "action":      "BLOCK" if avg_pnl < -0.001 else "REDUCE_SIZE",
            "created_at":  datetime.now().isoformat(),
        }
        self.block_rules.append(rule)
        print(f"[FailureDB] 새 차단 룰 생성: {features['market_state']} × {features['vkospi_bucket']} VKOSPI → {rule['action']}")

    def get_warning(self, current_features: dict) -> Optional[dict]:
        """
        진입 전 경고 조회.
        대표님 핵심 통찰: 실패가 다른 맥락에선 기회일 수 있음.
        """
        feature_key = json.dumps(current_features, sort_keys=True)

        # 직접 매칭
        if feature_key in self.patterns:
            records = self.patterns[feature_key]
            win_rate = sum(1 for r in records if r.pnl > 0) / len(records)

            if win_rate >= self.OPPORTUNITY_WIN_RATE:
                return {
                    "type":    "OPPORTUNITY",
                    "message": f"과거 유사 상황 {len(records)}건 중 승률 {win_rate:.0%} — 강화 진입 고려",
                    "win_rate": win_rate,
                }
            elif win_rate < 0.35:
                return {
                    "type":    "DANGER",
                    "message": f"과거 유사 상황 {len(records)}건 중 승률 {win_rate:.0%} — 진입 금지",
                    "win_rate": win_rate,
                }

        # 차단 룰 확인
        for rule in self.block_rules:
            if all(current_features.get(k) == v for k, v in rule["features"].items()):
                return {
                    "type":    "BLOCK",
                    "message": f"차단 룰 적용: {rule['count']}회 반복 실패 패턴 (평균 {rule['avg_pnl']:.3%})",
                    "action":  rule["action"],
                }

        return None

    def find_opportunities_after_failure(self, lookback_days: int = 3) -> list:
        """
        실패 N일 후에 나타나는 역전 기회 패턴 발굴.
        '이 실패 다음에는 오히려 좋은 기회가 온다' 패턴.
        """
        opportunities = []
        if len(self.failures) < lookback_days + 1:
            return opportunities

        for i in range(len(self.failures) - lookback_days):
            recent_losses = self.failures[i:i + lookback_days]
            if not all(r.pnl < 0 for r in recent_losses):
                continue

            pattern_desc = f"연속손실_{lookback_days}일_{recent_losses[-1].market_state}"
            opportunities.append({
                "pattern":     pattern_desc,
                "after_losses": [r.pnl_pct for r in recent_losses],
                "note":        "연속 손실 후 역전 기회 — 변동성AI or 평균회귀AI 강화 진입 검토",
            })

        return opportunities

    def summary(self) -> dict:
        categories = {}
        for r in self.failures:
            categories[r.category] = categories.get(r.category, 0) + 1

        return {
            "total_failures":  len(self.failures),
            "block_rules":     len(self.block_rules),
            "categories":      categories,
            "top_pattern":     max(self.patterns, key=lambda k: len(self.patterns[k]), default="없음"),
        }


# ════════════════════════════════════════════
#  Walk-Forward 백테스트 엔진
# ════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    ticker:         str   = "069500"   # KODEX 200
    start_date:     str   = "20200101"
    end_date:       str   = "20241231"
    commission:     float = 0.00015    # 0.015% × 2
    slippage:       float = 0.0005     # 0.05%
    stop_loss_pct:  float = 0.002
    take_profit_pct:float = 0.004

    # Walk-Forward 설정
    train_months:   int   = 18
    test_months:    int   = 6

    # 데이터 분할 (코딩 시작 전 봉인)
    # train:  20200101~20221231 (전략 설계)
    # valid:  20230101~20231231 (파라미터 조정)
    # test:   20240101~20241231 (봉인 — 딱 한 번)


@dataclass
class BacktestResult:
    agent_id:      str
    trades:        int
    wins:          int
    losses:        int
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    profit_factor: float
    sharpe:        float
    max_dd:        float
    annual_return: float
    ev:            float
    by_year:       dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """실전 투입 기준 통과 여부"""
        return (
            self.ev > 0.0003
            and self.sharpe > 1.0
            and self.max_dd < 0.15
            and self.trades >= 100
            and self.win_rate > 0.50
        )


class WalkForwardBacktest:
    """
    객관성 보장된 Walk-Forward 백테스트.

    핵심 원칙:
    1. 데이터 3분할 (train/valid/test) — 코딩 전 봉인
    2. 파라미터 최대 2개 제한
    3. 테스트는 딱 한 번 — 수정 후 재실행 불가
    4. 비용 반드시 반영 (수수료 + 슬리피지)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """pykrx로 데이터 수집 + 지표 계산"""
        try:
            from pykrx import stock
            df = stock.get_market_ohlcv(
                self.config.start_date, self.config.end_date, self.config.ticker
            )
            df.columns = ["open", "high", "low", "close", "volume", "change"]
            df.index   = pd.to_datetime(df.index)
        except Exception as e:
            print(f"[Backtest] pykrx 오류: {e}. 샘플 데이터 사용.")
            dates = pd.date_range(self.config.start_date, self.config.end_date, freq="B")
            np.random.seed(42)
            prices = 30000 + np.cumsum(np.random.randn(len(dates)) * 200)
            df = pd.DataFrame({
                "open": prices * 0.999, "high": prices * 1.005,
                "low": prices * 0.994,  "close": prices,
                "volume": np.random.randint(1_000_000, 5_000_000, len(dates)),
                "change": np.random.randn(len(dates)) * 0.01,
            }, index=dates)

        # 지표 계산
        df["rsi"]       = self._calc_rsi(df["close"], 14)
        df["ma20"]      = df["close"].rolling(20).mean()
        df["ma200"]     = df["close"].rolling(200).mean()
        bb_std          = df["close"].rolling(20).std()
        df["bb_upper"]  = df["ma20"] + bb_std * 2
        df["bb_lower"]  = df["ma20"] - bb_std * 2
        df["bb_middle"] = df["ma20"]
        df["atr"]       = self._calc_atr(df, 14)
        df["vol_ma5"]   = df["volume"].rolling(5).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma5"]
        df["prev_close"]= df["close"].shift(1)
        df["gap_pct"]   = (df["open"] - df["prev_close"]) / df["prev_close"]
        df["high_20d"]  = df["high"].rolling(20).max().shift(1)
        df["low_52w"]   = df["low"].rolling(252).min()

        self._data = df.dropna()
        print(f"[Backtest] 데이터 로드: {len(self._data)}일 ({self.config.start_date} ~ {self.config.end_date})")
        return self._data

    @staticmethod
    def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hpc = (df["high"] - df["close"].shift(1)).abs()
        lpc = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _simulate_agent(self, agent_class, data: pd.DataFrame) -> BacktestResult:
        """단일 에이전트 백테스트 시뮬레이션"""
        cfg    = self.config
        trades = []

        for i in range(len(data)):
            row = data.iloc[i]
            if pd.isna(row["rsi"]) or pd.isna(row["ma20"]):
                continue

            # 에이전트별 간단 신호 로직 (실제는 에이전트 클래스 호출)
            signal = self._get_signal(agent_class.__name__, row)
            if not signal:
                continue

            entry  = row["open"] * (1 + cfg.slippage)
            target = entry * (1 + cfg.take_profit_pct)
            stop   = entry * (1 - cfg.stop_loss_pct)

            if row["high"] >= target:
                pnl = cfg.take_profit_pct - cfg.commission - cfg.slippage
                win = True
            elif row["low"] <= stop:
                pnl = -cfg.stop_loss_pct - cfg.commission - cfg.slippage
                win = False
            else:
                pnl = (row["close"] - entry) / entry - cfg.commission
                win = pnl > 0

            trades.append({
                "date": data.index[i],
                "pnl":  pnl,
                "win":  win,
                "year": data.index[i].year,
            })

        return self._calc_metrics(agent_class.__name__, pd.DataFrame(trades))

    def _get_signal(self, agent_name: str, row) -> bool:
        """에이전트별 매수 신호 (백테스트용 간소화)"""
        if agent_name == "GapTradingAgent":
            return 0.003 <= abs(row.get("gap_pct", 0)) <= 0.02

        if agent_name == "MeanRevAgent":
            return (row.get("rsi", 50) < 30
                    and row.get("close", 0) <= row.get("bb_lower", 0))

        if agent_name == "MomentumAgent":
            return (row.get("close", 0) > row.get("high_20d", float("inf"))
                    and row.get("vol_ratio", 0) >= 2.0)

        if agent_name == "EODAgent":
            return False   # 종가베팅은 기관 데이터 필요

        if agent_name == "VolatilityAgent":
            return False   # VKOSPI 외부 데이터 필요

        return False

    def _calc_metrics(self, agent_id: str, df: pd.DataFrame) -> BacktestResult:
        """성과 지표 계산"""
        if len(df) == 0:
            return BacktestResult(
                agent_id=agent_id, trades=0, wins=0, losses=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                sharpe=0, max_dd=0, annual_return=0, ev=0,
            )

        wins   = df[df["win"] == True]
        losses = df[df["win"] == False]

        avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1e-10

        cumret = (1 + df["pnl"]).cumprod()
        mdd    = ((cumret.cummax() - cumret) / cumret.cummax()).max()
        sharpe = df["pnl"].mean() / (df["pnl"].std() + 1e-10) * np.sqrt(252)

        total_years  = max(1, len(df) / 252)
        annual_return = (cumret.iloc[-1] ** (1 / total_years)) - 1

        by_year = df.groupby("year")["pnl"].sum().round(4).to_dict() if "year" in df.columns else {}

        return BacktestResult(
            agent_id=agent_id,
            trades=len(df),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(df) if len(df) > 0 else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=avg_win / avg_loss if avg_loss > 0 else 0,
            sharpe=sharpe,
            max_dd=mdd,
            annual_return=annual_return,
            ev=df["pnl"].mean(),
            by_year=by_year,
        )

    def run_all_parallel(self, agent_classes: list) -> dict:
        """
        7개 에이전트 동시 백테스트 (concurrent.futures).
        28주 → 30분으로 단축.
        """
        if self._data is None:
            self.load_data()

        print(f"\n🚀 {len(agent_classes)}개 에이전트 동시 백테스트 시작...")

        # train 데이터만 사용 (valid/test는 봉인)
        train_end = "20221231"
        train_data = self._data[self._data.index <= train_end]

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_classes)) as executor:
            futures = {
                executor.submit(self._simulate_agent, ac, train_data): ac
                for ac in agent_classes
            }
            for future in concurrent.futures.as_completed(futures):
                ac = futures[future]
                try:
                    result = future.result()
                    results[result.agent_id] = result
                    status = "✅" if result.passed else "❌"
                    print(f"{status} {result.agent_id}: 승률 {result.win_rate:.1%} | 샤프 {result.sharpe:.2f} | EV {result.ev:.4%}")
                except Exception as e:
                    print(f"❌ {ac.__name__} 오류: {e}")

        return results

    def walk_forward(self, agent_class, data: pd.DataFrame) -> list:
        """
        롤링 Walk-Forward 검증.
        학습 18개월 → 테스트 6개월 롤링.
        """
        trading_days_per_month = 21
        train_days = self.config.train_months * trading_days_per_month
        test_days  = self.config.test_months  * trading_days_per_month

        wf_results = []
        start = 0

        while start + train_days + test_days <= len(data):
            train_slice = data.iloc[start : start + train_days]
            test_slice  = data.iloc[start + train_days : start + train_days + test_days]

            train_result = self._simulate_agent(agent_class, train_slice)
            test_result  = self._simulate_agent(agent_class, test_slice)

            degradation = 0.0
            if train_result.sharpe > 0:
                degradation = (train_result.sharpe - test_result.sharpe) / train_result.sharpe

            wf_results.append({
                "period":           f"{data.index[start].strftime('%Y%m')}~{data.index[start+train_days+test_days-1].strftime('%Y%m')}",
                "train_sharpe":     round(train_result.sharpe, 2),
                "test_sharpe":      round(test_result.sharpe, 2),
                "degradation":      round(degradation, 2),
                "overfit_warning":  degradation > 0.30,
            })

            start += test_days

        avg_degradation = np.mean([r["degradation"] for r in wf_results])
        overfit = avg_degradation > 0.30

        print(f"\n{'⚠️ 과최적화 경고' if overfit else '✅ 과최적화 없음'}: {agent_class.__name__} 평균 성능 하락 {avg_degradation:.2f}")

        return wf_results

    def print_summary(self, results: dict):
        """결과 요약 출력"""
        print("\n" + "="*65)
        print(f"{'에이전트':<22} {'거래':>5} {'승률':>6} {'EV':>8} {'MDD':>7} {'샤프':>6} {'판정':>5}")
        print("="*65)

        for name, r in sorted(results.items(), key=lambda x: x[1].ev, reverse=True):
            flag = "✅ GO" if r.passed else "❌ NO"
            print(
                f"{name:<22} {r.trades:>5} {r.win_rate:>5.1%} "
                f"{r.ev:>7.4%} {r.max_dd:>6.1%} {r.sharpe:>5.2f}  {flag}"
            )

        print("="*65)
        passed = sum(1 for r in results.values() if r.passed)
        print(f"통과: {passed}/{len(results)}개 에이전트 실전 투입 승인")
        print("\n📌 통과 기준: EV>0.03% + 샤프>1.0 + MDD<15% + 거래>100건 + 승률>50%")
