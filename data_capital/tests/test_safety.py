"""safety.py 단위 테스트 — timestamp 검증 + 상태 영속화."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from data_capital.safety import (
    StateManager,
    deserialize_datetime,
    validate_timestamp,
)


class TestValidateTimestamp:
    def test_fresh_timestamp_passes(self):
        now = datetime(2026, 4, 24, 10, 0, 0)
        data_time = now - timedelta(seconds=30)
        ok, _ = validate_timestamp(data_time, now=now, max_age_seconds=600)
        assert ok

    def test_stale_timestamp_rejected(self):
        now = datetime(2026, 4, 24, 10, 0, 0)
        data_time = now - timedelta(seconds=700)
        ok, reason = validate_timestamp(data_time, now=now, max_age_seconds=600)
        assert not ok
        assert "stale" in reason

    def test_future_timestamp_rejected(self):
        now = datetime(2026, 4, 24, 10, 0, 0)
        data_time = now + timedelta(seconds=120)
        ok, reason = validate_timestamp(data_time, now=now)
        assert not ok
        assert "미래" in reason

    def test_small_clock_skew_tolerated(self):
        # +30초는 미래지만 허용 범위(60초) 내
        now = datetime(2026, 4, 24, 10, 0, 0)
        data_time = now + timedelta(seconds=30)
        ok, _ = validate_timestamp(data_time, now=now)
        assert ok


class TestStateManager:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        sm = StateManager(tmp_path / "state.json")
        assert not sm.exists()

        state = {
            "current_capital": 3_200_000.0,
            "peak_capital": 3_200_000.0,
            "daily_pnl_pct": 0.002,
            "open_positions": {
                "gap": {"ticker": "069500", "shares": 10, "entry_price": 30000.0},
            },
        }
        sm.save(state)
        assert sm.exists()

        loaded = sm.load()
        assert loaded == state

    def test_datetime_survives_roundtrip(self, tmp_path: Path):
        sm = StateManager(tmp_path / "state.json")
        entry = datetime(2026, 4, 24, 10, 15, 30)
        sm.save({"entry_time": entry})

        loaded = sm.load()
        # _serialize → __type__ dict. 복원은 호출자 몫.
        restored = deserialize_datetime(loaded["entry_time"])
        assert restored == entry

    def test_load_missing_returns_none(self, tmp_path: Path):
        sm = StateManager(tmp_path / "nonexistent.json")
        assert sm.load() is None

    def test_load_corrupt_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text("this is not json{{{", encoding="utf-8")
        sm = StateManager(path)
        assert sm.load() is None

    def test_schema_mismatch_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"version": 999, "state": {"x": 1}}),
            encoding="utf-8",
        )
        sm = StateManager(path)
        assert sm.load() is None

    def test_atomic_write_no_partial_on_crash(self, tmp_path: Path, monkeypatch):
        """tempfile 경로에서 os.replace 전에 예외 나도 원본이 손상되지 않음."""
        sm = StateManager(tmp_path / "state.json")
        sm.save({"x": 1})
        original = sm.load()

        def boom(*a, **kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr("os.replace", boom)

        with pytest.raises(RuntimeError):
            sm.save({"x": 2})

        # 원본 보존되어야 함
        assert sm.load() == original
        # tempfile이 정리되었는지
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_clear_removes_file(self, tmp_path: Path):
        sm = StateManager(tmp_path / "state.json")
        sm.save({"x": 1})
        sm.clear()
        assert not sm.exists()
        # clear()는 멱등
        sm.clear()
