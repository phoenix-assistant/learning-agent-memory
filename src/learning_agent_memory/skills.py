"""Skill compiler — compress repeated successful sequences into reusable procedures."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .experience import Experience


@dataclass
class CompiledSkill:
    """A compiled skill derived from repeated successful experience sequences."""

    name: str
    description: str
    steps: list[dict[str, Any]]
    category: str = ""
    success_rate: float = 0.0
    usage_count: int = 0
    source_count: int = 0  # how many experience sequences contributed
    preconditions: dict[str, Any] = field(default_factory=dict)
    postconditions: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "category": self.category,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "source_count": self.source_count,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "created_at": self.created_at,
        }


_SKILL_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    steps TEXT NOT NULL,
    category TEXT DEFAULT '',
    success_rate REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    source_count INTEGER DEFAULT 0,
    preconditions TEXT DEFAULT '{}',
    postconditions TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
"""


class SkillCompiler:
    """Compiles repeated successful action sequences into reusable skills."""

    def __init__(self, db_path: str | Path = ":memory:", min_occurrences: int = 2):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SKILL_SCHEMA)
        self.min_occurrences = min_occurrences

    def close(self) -> None:
        self._conn.close()

    def compile_from_experiences(
        self, name: str, experiences: list[Experience], description: str = ""
    ) -> CompiledSkill:
        """Compile a named skill from a set of successful experiences."""
        successful = [e for e in experiences if e.success]
        if not successful:
            raise ValueError("No successful experiences to compile from")

        # Extract action sequence
        sorted_exps = sorted(successful, key=lambda e: e.timestamp)
        steps = []
        for e in sorted_exps:
            steps.append({
                "action": e.action,
                "context_keys": list(e.context.keys()),
                "expected_outcome_keys": list(e.outcome.keys()),
            })

        # Common context across all experiences = preconditions
        common_context: dict[str, Any] = {}
        if sorted_exps:
            first_ctx = sorted_exps[0].context
            for key, val in first_ctx.items():
                if all(e.context.get(key) == val for e in sorted_exps):
                    common_context[key] = val

        success_rate = len(successful) / len(experiences) if experiences else 0.0
        skill = CompiledSkill(
            name=name,
            description=description or f"Compiled skill from {len(successful)} successful experiences",
            steps=steps,
            category=successful[0].category if successful else "",
            success_rate=success_rate,
            source_count=len(successful),
            preconditions=common_context,
        )

        self._conn.execute(
            "INSERT OR REPLACE INTO skills (name, description, steps, category, success_rate, usage_count, source_count, preconditions, postconditions, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (skill.name, skill.description, json.dumps(skill.steps), skill.category,
             skill.success_rate, skill.usage_count, skill.source_count,
             json.dumps(skill.preconditions), json.dumps(skill.postconditions), skill.created_at),
        )
        self._conn.commit()
        return skill

    def auto_compile(self, experiences: list[Experience], window: int = 3) -> list[CompiledSkill]:
        """Auto-detect repeated successful action sequences and compile them."""
        successful = sorted([e for e in experiences if e.success], key=lambda e: e.timestamp)
        if len(successful) < window:
            return []

        # Find repeated action n-grams
        ngrams: dict[tuple[str, ...], list[list[Experience]]] = defaultdict(list)
        for i in range(len(successful) - window + 1):
            chunk = successful[i:i + window]
            key = tuple(e.action for e in chunk)
            ngrams[key].append(chunk)

        skills = []
        for actions, groups in ngrams.items():
            if len(groups) < self.min_occurrences:
                continue
            name = "auto:" + "->".join(actions)
            flat_exps = [e for group in groups for e in group]
            skill = self.compile_from_experiences(name, flat_exps, f"Auto-compiled: {' → '.join(actions)}")
            skills.append(skill)
        return skills

    def get(self, name: str) -> CompiledSkill | None:
        row = self._conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self, category: str | None = None) -> list[CompiledSkill]:
        if category:
            rows = self._conn.execute("SELECT * FROM skills WHERE category = ?", (category,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM skills").fetchall()
        return [self._row_to_skill(r) for r in rows]

    def record_usage(self, name: str) -> None:
        self._conn.execute("UPDATE skills SET usage_count = usage_count + 1 WHERE name = ?", (name,))
        self._conn.commit()

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> CompiledSkill:
        return CompiledSkill(
            name=row["name"],
            description=row["description"],
            steps=json.loads(row["steps"]),
            category=row["category"],
            success_rate=row["success_rate"],
            usage_count=row["usage_count"],
            source_count=row["source_count"],
            preconditions=json.loads(row["preconditions"]),
            postconditions=json.loads(row["postconditions"]),
            created_at=row["created_at"],
        )
