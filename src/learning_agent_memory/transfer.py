"""Transfer learning bridge — export learned knowledge for LLM fine-tuning."""

from __future__ import annotations

import json
from typing import Any

from .experience import Experience
from .patterns import Pattern
from .skills import CompiledSkill
from .beliefs import Belief


class TransferBridge:
    """Export learned knowledge in formats compatible with LLM fine-tuning."""

    @staticmethod
    def experiences_to_jsonl(experiences: list[Experience]) -> str:
        """Export experiences as JSONL for fine-tuning."""
        lines = []
        for exp in experiences:
            record = {
                "messages": [
                    {"role": "system", "content": f"Agent performing '{exp.action}' in category '{exp.category}'."},
                    {"role": "user", "content": json.dumps(exp.context)},
                    {"role": "assistant", "content": json.dumps(exp.outcome)},
                ],
                "metadata": {
                    "success": exp.success,
                    "action": exp.action,
                    "category": exp.category,
                    "timestamp": exp.timestamp,
                },
            }
            lines.append(json.dumps(record))
        return "\n".join(lines)

    @staticmethod
    def patterns_to_system_prompt(patterns: list[Pattern]) -> str:
        """Convert learned patterns into a system prompt supplement."""
        if not patterns:
            return ""
        sections = ["# Learned Patterns\n"]
        for p in sorted(patterns, key=lambda x: x.confidence, reverse=True):
            sections.append(
                f"- **{p.name}** (confidence: {p.confidence:.2f}, success: {p.success_rate:.0%}, n={p.frequency}): {p.description}"
            )
        return "\n".join(sections)

    @staticmethod
    def skills_to_procedures(skills: list[CompiledSkill]) -> str:
        """Convert compiled skills into procedure documentation."""
        if not skills:
            return ""
        sections = ["# Compiled Procedures\n"]
        for skill in skills:
            sections.append(f"## {skill.name}\n")
            sections.append(f"{skill.description}\n")
            if skill.preconditions:
                sections.append(f"**Preconditions:** {json.dumps(skill.preconditions)}\n")
            sections.append("**Steps:**")
            for i, step in enumerate(skill.steps, 1):
                sections.append(f"  {i}. `{step['action']}` (context: {step.get('context_keys', [])})")
            sections.append(f"\n*Success rate: {skill.success_rate:.0%} | Sources: {skill.source_count}*\n")
        return "\n".join(sections)

    @staticmethod
    def beliefs_to_knowledge(beliefs: list[Belief]) -> str:
        """Convert beliefs into a knowledge summary."""
        if not beliefs:
            return ""
        sections = ["# Agent Beliefs\n"]
        for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
            direction = "likely true" if b.posterior > 0.5 else "likely false"
            sections.append(
                f"- **{b.key}** ({direction}, p={b.posterior:.3f}, confidence={b.confidence:.2f}): {b.description}"
            )
        return "\n".join(sections)

    @staticmethod
    def full_export(
        experiences: list[Experience],
        patterns: list[Pattern],
        skills: list[CompiledSkill],
        beliefs: list[Belief],
    ) -> dict[str, Any]:
        """Full export of all learned knowledge."""
        bridge = TransferBridge()
        return {
            "training_data": bridge.experiences_to_jsonl(experiences),
            "system_prompt_supplement": bridge.patterns_to_system_prompt(patterns),
            "procedures": bridge.skills_to_procedures(skills),
            "knowledge": bridge.beliefs_to_knowledge(beliefs),
            "stats": {
                "total_experiences": len(experiences),
                "total_patterns": len(patterns),
                "total_skills": len(skills),
                "total_beliefs": len(beliefs),
                "success_rate": sum(1 for e in experiences if e.success) / len(experiences) if experiences else 0,
            },
        }
