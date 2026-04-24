"""안전장치 — 타임스탬프 검증 + 세션 상태 영속화.

v1.5 Step 7:
    - validate_timestamp: 데이터가 최신인지 확인 (stale 데이터로 매매 방지)
    - StateManager:       open_positions / capital 스냅샷을 JSON에 저장·복원
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Timestamp 검증
# ─────────────────────────────────────────────

def validate_timestamp(
    data_time: datetime,
    now: Optional[datetime] = None,
    max_age_seconds: int = 600,
) -> tuple[bool, str]:
    """데이터 타임스탬프가 `max_age_seconds` 이내면 OK.

    장중 실시간 루프에서 fetch 결과의 최신성을 확인한다.
    반환: (ok, 사유)
    """
    now = now or datetime.now()
    # timezone 제거 (naive 비교)
    if data_time.tzinfo is not None:
        data_time = data_time.replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)

    if data_time > now + timedelta(seconds=60):
        return False, f"미래 timestamp ({data_time} > {now})"

    age = (now - data_time).total_seconds()
    if age > max_age_seconds:
        return False, f"stale 데이터 ({age:.0f}s 경과, 한도 {max_age_seconds}s)"
    return True, f"ok ({age:.0f}s 경과)"


# ─────────────────────────────────────────────
# 상태 영속화
# ─────────────────────────────────────────────

class StateManager:
    """프로세스 재시작 후 open_positions·capital을 복원하기 위한 영속 저장소.

    파일 포맷: JSON. 원자적 쓰기(tempfile + os.replace).
    스키마 버전(`version`) 포함해서 향후 변경 대응.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = "state.json"):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, state: Dict[str, Any]) -> None:
        """원자적으로 state를 저장한다."""
        payload = {
            "version":   self.SCHEMA_VERSION,
            "saved_at":  datetime.now().isoformat(),
            "state":     _serialize(state),
        }
        # 원자적 쓰기
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load(self) -> Optional[Dict[str, Any]]:
        """존재하면 load, 없거나 손상되면 None.

        스키마 버전 불일치 시 경고 로그 후 None (새 출발).
        """
        if not self.path.exists():
            return None
        try:
            with self.path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("상태 파일 로드 실패 (%s): %s", self.path, e)
            return None

        version = payload.get("version")
        if version != self.SCHEMA_VERSION:
            logger.warning(
                "상태 스키마 불일치 (현재=%s, 파일=%s) — 무시하고 새로 시작",
                self.SCHEMA_VERSION, version,
            )
            return None

        return payload.get("state")

    def clear(self) -> None:
        """상태 파일 삭제. 수동 리셋용."""
        if self.path.exists():
            self.path.unlink()


def _serialize(obj: Any) -> Any:
    """JSON 직렬화 가능한 형태로 변환. datetime은 ISO8601."""
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize(asdict(obj))
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # fallback: repr
    return repr(obj)


def deserialize_datetime(value: Any) -> Any:
    """_serialize가 싸둔 datetime 마커를 복원한다."""
    if isinstance(value, dict):
        if value.get("__type__") == "datetime":
            return datetime.fromisoformat(value["value"])
        return {k: deserialize_datetime(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deserialize_datetime(v) for v in value]
    return value
