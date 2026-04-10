"""MCP server for learning-agent-memory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .memory import LearningMemory

mcp = FastMCP("learning-agent-memory")

_instances: dict[str, LearningMemory] = {}


def _get_memory(agent_id: str) -> LearningMemory:
    if agent_id not in _instances:
        db_dir = Path(os.environ.get("LAM_DB_DIR", "."))
        db_dir.mkdir(parents=True, exist_ok=True)
        _instances[agent_id] = LearningMemory(agent_id, db_path=db_dir / f"{agent_id}.db")
    return _instances[agent_id]


@mcp.tool()
def record_experience(
    agent_id: str,
    action: str,
    context: str,
    outcome: str,
    success: bool = True,
    category: str = "",
    tags: str = "[]",
) -> str:
    """Record an agent experience with action, context, outcome, and success flag."""
    mem = _get_memory(agent_id)
    exp_id = mem.record_experience(
        action=action,
        context=json.loads(context),
        outcome=json.loads(outcome),
        success=success,
        category=category,
        tags=json.loads(tags),
    )
    return json.dumps({"id": exp_id, "status": "recorded"})


@mcp.tool()
def get_patterns(agent_id: str, category: str = "", min_frequency: int = 3) -> str:
    """Get learned patterns from accumulated experiences."""
    mem = _get_memory(agent_id)
    patterns = mem.learned_patterns(category=category or None, min_frequency=min_frequency)
    return json.dumps([p.to_dict() for p in patterns])


@mcp.tool()
def compile_skill(agent_id: str, name: str, category: str = "", description: str = "") -> str:
    """Compile a skill from successful experiences."""
    mem = _get_memory(agent_id)
    try:
        skill = mem.compile_skill(name, category=category or None, description=description)
        return json.dumps(skill.to_dict())
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def auto_compile_skills(agent_id: str, window: int = 3) -> str:
    """Auto-detect and compile skills from repeated successful sequences."""
    mem = _get_memory(agent_id)
    skills = mem.auto_compile_skills(window=window)
    return json.dumps([s.to_dict() for s in skills])


@mcp.tool()
def get_belief(agent_id: str, key: str) -> str:
    """Get a specific belief."""
    mem = _get_memory(agent_id)
    belief = mem.get_belief(key)
    return json.dumps(belief.to_dict() if belief else {"error": "not found"})


@mcp.tool()
def update_belief(agent_id: str, key: str, positive: bool, strength: float = 1.0) -> str:
    """Update a belief with new evidence."""
    mem = _get_memory(agent_id)
    belief = mem.update_belief(key, positive=positive, strength=strength)
    return json.dumps(belief.to_dict() if belief else {"error": "not found"})


@mcp.tool()
def add_belief(agent_id: str, key: str, category: str, prior: float = 0.5, description: str = "") -> str:
    """Add a new belief."""
    mem = _get_memory(agent_id)
    belief = mem.add_belief(key, category=category, prior=prior, description=description)
    return json.dumps(belief.to_dict())


@mcp.tool()
def export_knowledge(agent_id: str) -> str:
    """Export all learned knowledge for transfer learning."""
    mem = _get_memory(agent_id)
    return json.dumps(mem.export_knowledge())


@mcp.tool()
def agent_stats(agent_id: str) -> str:
    """Get statistics for an agent's learning memory."""
    mem = _get_memory(agent_id)
    return json.dumps(mem.stats())


def main():
    mcp.run()


if __name__ == "__main__":
    main()
