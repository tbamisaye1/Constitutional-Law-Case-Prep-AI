"""push_changes must lock the stored row before checking the push's base.

Without the lock, an MCP write that commits between the base check and the
upsert is overwritten by a push that was checked against the older version.
"""

from datetime import datetime, timezone

from app.db import repository
from app.db.repository import _load_library_record


class RecordingCursor:
    def __init__(self):
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(" ".join(str(sql).split()))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


def test_loader_locks_only_when_asked():
    cursor = RecordingCursor()
    _load_library_record(cursor, "ws", "arguments", "main")
    _load_library_record(cursor, "ws", "arguments", "main", lock=True)
    assert "FOR UPDATE" not in cursor.sql[0]
    assert cursor.sql[1].endswith("FOR UPDATE")


def test_push_loads_library_rows_with_lock(monkeypatch):
    calls = []

    def fake_load(cursor, workspace_id, kind, record_id, *, lock=False):
        calls.append(lock)
        return None

    monkeypatch.setattr(repository, "_load_library_record", fake_load)
    row = {"kind": "arguments", "id": "main", "data": {"id": "main"}, "updatedAt": 1}
    try:
        repository.push_changes(
            RecordingCursor(), "ws", {"library_records": [row]}, datetime.now(timezone.utc)
        )
    except Exception:
        # Later steps may need a real database; only the first load matters.
        pass
    assert calls and calls[0] is True
