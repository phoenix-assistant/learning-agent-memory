"""Bayesian belief updating from accumulated evidence."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Belief:
    """A belief with Bayesian probability tracking."""

    key: str
    category: str
    description: str
    prior: float  # initial probability
    posterior: float  # current probability after updates
    evidence_count: int = 0
    positive_evidence: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        """Confidence grows with evidence count (asymptotic to 1)."""
        return 1 - 1 / (1 + self.evidence_count * 0.3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "description": self.description,
            "prior": self.prior,
            "posterior": self.posterior,
            "evidence_count": self.evidence_count,
            "positive_evidence": self.positive_evidence,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
        }


_BELIEF_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT DEFAULT '',
    prior REAL NOT NULL,
    posterior REAL NOT NULL,
    evidence_count INTEGER DEFAULT 0,
    positive_evidence INTEGER DEFAULT 0,
    last_updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_belief_cat ON beliefs(category);
"""


class BeliefUpdater:
    """Maintains and updates beliefs using Bayesian updating."""

    def __init__(self, db_path: str | Path = ":memory:", learning_rate: float = 0.1):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_BELIEF_SCHEMA)
        self.learning_rate = learning_rate

    def close(self) -> None:
        self._conn.close()

    def add_belief(self, key: str, category: str, prior: float = 0.5, description: str = "") -> Belief:
        belief = Belief(key=key, category=category, description=description, prior=prior, posterior=prior)
        self._conn.execute(
            "INSERT OR REPLACE INTO beliefs (key, category, description, prior, posterior, evidence_count, positive_evidence, last_updated) "
            "VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
            (key, category, description, prior, prior, time.time()),
        )
        self._conn.commit()
        return belief

    def update(self, key: str, positive: bool, strength: float = 1.0) -> Belief | None:
        """Update a belief with new evidence using simplified Bayesian update."""
        row = self._conn.execute("SELECT * FROM beliefs WHERE key = ?", (key,)).fetchone()
        if not row:
            return None

        posterior = row["posterior"]
        # Bayesian update: likelihood ratio approach
        # For positive evidence, shift posterior up; for negative, shift down
        lr = self.learning_rate * strength
        if positive:
            # Increase posterior
            new_posterior = posterior + lr * (1 - posterior)
        else:
            # Decrease posterior
            new_posterior = posterior - lr * posterior

        new_posterior = float(np.clip(new_posterior, 0.001, 0.999))
        now = time.time()
        self._conn.execute(
            "UPDATE beliefs SET posterior = ?, evidence_count = evidence_count + 1, "
            "positive_evidence = positive_evidence + ?, last_updated = ? WHERE key = ?",
            (new_posterior, int(positive), now, key),
        )
        self._conn.commit()
        return self.get(key)

    def get(self, key: str) -> Belief | None:
        row = self._conn.execute("SELECT * FROM beliefs WHERE key = ?", (key,)).fetchone()
        return self._row_to_belief(row) if row else None

    def query(self, category: str | None = None, min_confidence: float = 0.0) -> list[Belief]:
        if category:
            rows = self._conn.execute("SELECT * FROM beliefs WHERE category = ?", (category,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM beliefs").fetchall()
        beliefs = [self._row_to_belief(r) for r in rows]
        return [b for b in beliefs if b.confidence >= min_confidence]

    def strongest(self, category: str | None = None, n: int = 10) -> list[Belief]:
        beliefs = self.query(category=category)
        return sorted(beliefs, key=lambda b: abs(b.posterior - 0.5) * b.confidence, reverse=True)[:n]

    @staticmethod
    def _row_to_belief(row: sqlite3.Row) -> Belief:
        return Belief(
            key=row["key"],
            category=row["category"],
            description=row["description"],
            prior=row["prior"],
            posterior=row["posterior"],
            evidence_count=row["evidence_count"],
            positive_evidence=row["positive_evidence"],
            last_updated=row["last_updated"],
        )
