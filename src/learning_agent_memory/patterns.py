"""Pattern extraction from accumulated experiences."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .experience import Experience


@dataclass
class Pattern:
    """A discovered pattern across experiences."""

    name: str
    category: str
    description: str
    frequency: int
    success_rate: float
    confidence: float
    context_signature: dict[str, Any] = field(default_factory=dict)
    sample_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "context_signature": self.context_signature,
            "sample_ids": self.sample_ids,
        }


class PatternExtractor:
    """Extract recurring patterns from experiences."""

    def __init__(self, min_frequency: int = 3, min_confidence: float = 0.6):
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence

    def extract(self, experiences: list[Experience], category: str | None = None) -> list[Pattern]:
        if category:
            experiences = [e for e in experiences if e.category == category]
        if not experiences:
            return []

        patterns: list[Pattern] = []
        patterns.extend(self._action_patterns(experiences))
        patterns.extend(self._context_patterns(experiences))
        patterns.extend(self._sequence_patterns(experiences))
        return [p for p in patterns if p.frequency >= self.min_frequency and p.confidence >= self.min_confidence]

    def _action_patterns(self, exps: list[Experience]) -> list[Pattern]:
        """Group by action and compute success rates."""
        by_action: dict[str, list[Experience]] = defaultdict(list)
        for e in exps:
            by_action[e.action].append(e)

        patterns = []
        for action, group in by_action.items():
            successes = sum(1 for e in group if e.success)
            rate = successes / len(group) if group else 0
            confidence = self._wilson_score(successes, len(group))
            patterns.append(Pattern(
                name=f"action:{action}",
                category=group[0].category or "general",
                description=f"Action '{action}' observed {len(group)} times with {rate:.0%} success rate",
                frequency=len(group),
                success_rate=rate,
                confidence=confidence,
                context_signature={"action": action},
                sample_ids=[e.id for e in group[:5]],
            ))
        return patterns

    def _context_patterns(self, exps: list[Experience]) -> list[Pattern]:
        """Find common context keys correlated with success/failure."""
        key_success: dict[str, list[bool]] = defaultdict(list)
        for e in exps:
            for key in e.context:
                key_success[f"{key}={e.context[key]}"].append(e.success)

        patterns = []
        for sig, outcomes in key_success.items():
            if len(outcomes) < self.min_frequency:
                continue
            successes = sum(outcomes)
            rate = successes / len(outcomes)
            confidence = self._wilson_score(successes, len(outcomes))
            if confidence < self.min_confidence:
                continue
            key, val = sig.split("=", 1)
            patterns.append(Pattern(
                name=f"context:{sig}",
                category="context",
                description=f"Context '{key}={val}' correlates with {rate:.0%} success ({len(outcomes)} obs)",
                frequency=len(outcomes),
                success_rate=rate,
                confidence=confidence,
                context_signature={key: val},
            ))
        return patterns

    def _sequence_patterns(self, exps: list[Experience]) -> list[Pattern]:
        """Detect action bigrams (sequential pairs)."""
        sorted_exps = sorted(exps, key=lambda e: e.timestamp)
        bigrams: dict[str, list[bool]] = defaultdict(list)
        for i in range(len(sorted_exps) - 1):
            a, b = sorted_exps[i], sorted_exps[i + 1]
            if a.agent_id == b.agent_id:
                key = f"{a.action}->{b.action}"
                bigrams[key].append(b.success)

        patterns = []
        for seq, outcomes in bigrams.items():
            if len(outcomes) < self.min_frequency:
                continue
            successes = sum(outcomes)
            rate = successes / len(outcomes)
            confidence = self._wilson_score(successes, len(outcomes))
            patterns.append(Pattern(
                name=f"sequence:{seq}",
                category="sequence",
                description=f"Sequence '{seq}' leads to {rate:.0%} success ({len(outcomes)} obs)",
                frequency=len(outcomes),
                success_rate=rate,
                confidence=confidence,
                context_signature={"sequence": seq},
            ))
        return patterns

    @staticmethod
    def _wilson_score(successes: int, total: int, z: float = 1.96) -> float:
        """Wilson score interval lower bound for confidence."""
        if total == 0:
            return 0.0
        p = successes / total
        denominator = 1 + z * z / total
        centre = p + z * z / (2 * total)
        spread = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        return float((centre - spread) / denominator)
