"""LearningMemory — unified high-level API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .experience import Experience, ExperienceStore
from .patterns import PatternExtractor, Pattern
from .beliefs import BeliefUpdater, Belief
from .skills import SkillCompiler, CompiledSkill
from .transfer import TransferBridge


class LearningMemory:
    """High-level API for the learning agent memory system."""

    def __init__(self, agent_id: str, db_path: str | Path = ":memory:"):
        self.agent_id = agent_id
        self._db_path = str(db_path)
        self.store = ExperienceStore(db_path)
        self.patterns_extractor = PatternExtractor()
        self.beliefs = BeliefUpdater(db_path)
        self.skill_compiler = SkillCompiler(db_path)
        self.transfer = TransferBridge()

    def close(self) -> None:
        self.store.close()
        self.beliefs.close()
        self.skill_compiler.close()

    def record_experience(
        self,
        action: str,
        context: dict[str, Any],
        outcome: dict[str, Any],
        success: bool = True,
        category: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a new experience and auto-update related beliefs."""
        exp = Experience(
            action=action,
            context=context,
            outcome=outcome,
            success=success,
            agent_id=self.agent_id,
            category=category,
            tags=tags or [],
            metadata=metadata or {},
        )
        exp_id = self.store.record(exp)

        # Auto-update belief for this action
        belief_key = f"action:{action}:success"
        if not self.beliefs.get(belief_key):
            self.beliefs.add_belief(belief_key, category=category or "actions", prior=0.5,
                                     description=f"Belief about success of action '{action}'")
        self.beliefs.update(belief_key, positive=success)

        return exp_id

    def learned_patterns(self, category: str | None = None, min_frequency: int = 3) -> list[Pattern]:
        """Get learned patterns, optionally filtered by category."""
        self.patterns_extractor.min_frequency = min_frequency
        exps = self.store.query(agent_id=self.agent_id, limit=1000)
        return self.patterns_extractor.extract(exps, category=category)

    def compile_skill(self, name: str, category: str | None = None, description: str = "") -> CompiledSkill:
        """Compile a skill from successful experiences matching the category/action pattern."""
        exps = self.store.query(agent_id=self.agent_id, category=category, limit=500)
        if not exps:
            raise ValueError(f"No experiences found to compile skill '{name}'")
        return self.skill_compiler.compile_from_experiences(name, exps, description)

    def auto_compile_skills(self, window: int = 3) -> list[CompiledSkill]:
        """Auto-detect and compile skills from repeated successful sequences."""
        exps = self.store.query(agent_id=self.agent_id, limit=1000)
        return self.skill_compiler.auto_compile(exps, window=window)

    def get_belief(self, key: str) -> Belief | None:
        return self.beliefs.get(key)

    def add_belief(self, key: str, category: str, prior: float = 0.5, description: str = "") -> Belief:
        return self.beliefs.add_belief(key, category, prior, description)

    def update_belief(self, key: str, positive: bool, strength: float = 1.0) -> Belief | None:
        return self.beliefs.update(key, positive, strength)

    def export_knowledge(self) -> dict[str, Any]:
        """Export all learned knowledge for transfer learning."""
        exps = self.store.query(agent_id=self.agent_id, limit=10000)
        patterns = self.learned_patterns(min_frequency=2)
        skills = self.skill_compiler.list_skills()
        beliefs_list = self.beliefs.query()
        return self.transfer.full_export(exps, patterns, skills, beliefs_list)

    def stats(self) -> dict[str, Any]:
        exps = self.store.query(agent_id=self.agent_id, limit=10000)
        return {
            "agent_id": self.agent_id,
            "total_experiences": len(exps),
            "successful": sum(1 for e in exps if e.success),
            "failed": sum(1 for e in exps if not e.success),
            "beliefs": len(self.beliefs.query()),
            "skills": len(self.skill_compiler.list_skills()),
        }
