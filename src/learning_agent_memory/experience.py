"""Experience storage with SQLite backend."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Experience:
    """A single agent experience record."""

    action: str
    context: dict[str, Any]
    outcome: dict[str, Any]
    success: bool
    agent_id: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Experience:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    action TEXT NOT NULL,
    context TEXT NOT NULL,
    outcome TEXT NOT NULL,
    success INTEGER NOT NULL,
    category TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    timestamp REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agent ON experiences(agent_id);
CREATE INDEX IF NOT EXISTS idx_category ON experiences(category);
CREATE INDEX IF NOT EXISTS idx_success ON experiences(success);
CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp);
CREATE INDEX IF NOT EXISTS idx_action ON experiences(action);
"""


class ExperienceStore:
    """SQLite-backed experience storage."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def record(self, exp: Experience) -> str:
        self._conn.execute(
            "INSERT INTO experiences (id, agent_id, action, context, outcome, success, category, tags, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exp.id,
                exp.agent_id,
                exp.action,
                json.dumps(exp.context),
                json.dumps(exp.outcome),
                int(exp.success),
                exp.category,
                json.dumps(exp.tags),
                exp.timestamp,
                json.dumps(exp.metadata),
            ),
        )
        self._conn.commit()
        return exp.id

    def get(self, exp_id: str) -> Experience | None:
        row = self._conn.execute("SELECT * FROM experiences WHERE id = ?", (exp_id,)).fetchone()
        return self._row_to_exp(row) if row else None

    def query(
        self,
        agent_id: str | None = None,
        category: str | None = None,
        success: bool | None = None,
        action: str | None = None,
        limit: int = 100,
        since: float | None = None,
    ) -> list[Experience]:
        clauses, params = [], []
        if agent_id is not None:
            clauses.append("agent_id = ?"); params.append(agent_id)
        if category is not None:
            clauses.append("category = ?"); params.append(category)
        if success is not None:
            clauses.append("success = ?"); params.append(int(success))
        if action is not None:
            clauses.append("action = ?"); params.append(action)
        if since is not None:
            clauses.append("timestamp >= ?"); params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM experiences{where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def count(self, agent_id: str | None = None) -> int:
        if agent_id:
            return self._conn.execute("SELECT COUNT(*) FROM experiences WHERE agent_id = ?", (agent_id,)).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def delete(self, exp_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM experiences WHERE id = ?", (exp_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_exp(row: sqlite3.Row) -> Experience:
        return Experience(
            id=row["id"],
            agent_id=row["agent_id"],
            action=row["action"],
            context=json.loads(row["context"]),
            outcome=json.loads(row["outcome"]),
            success=bool(row["success"]),
            category=row["category"],
            tags=json.loads(row["tags"]),
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )
