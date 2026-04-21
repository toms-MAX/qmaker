"""Firebase / 알림 연동."""

from __future__ import annotations

from datetime import datetime
from typing import Optional


class FirebaseSync:
    """
    Python 매매 엔진 ↔ Firestore 실시간 동기화.
    오프라인 모드에서는 조용히 무시한다.
    """

    def __init__(self, service_account_path: str = "serviceAccount.json"):
        self.db = None
        self._init(service_account_path)

    def _init(self, path: str):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred = credentials.Certificate(path)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
        except Exception:
            pass   # 오프라인 모드

    @property
    def online(self) -> bool:
        return self.db is not None

    def _write(self, collection: str, doc_id: Optional[str], data: dict):
        if not self.online:
            return
        try:
            from firebase_admin import firestore
            data["synced_at"] = firestore.SERVER_TIMESTAMP
            ref = self.db.collection(collection)
            if doc_id:
                ref.document(doc_id).set(data)
            else:
                ref.add(data)
        except Exception:
            pass

    def save_trade(self, trade: dict):
        self._write("trades", None, trade)

    def save_signal(self, signal: dict):
        self._write("signals", None, signal)

    def update_portfolio(self, portfolio: dict):
        self._write("portfolio", "live", portfolio)

    def save_consensus(self, consensus: dict):
        self._write("consensus", None, consensus)

    def alert(self, message: str, severity: str = "INFO"):
        self._write("alerts", None, {
            "message":   message,
            "severity":  severity,
            "timestamp": datetime.now().isoformat(),
        })
