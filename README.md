# learning-agent-memory

Persistent memory layer that learns and improves agent performance through experience consolidation.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LearningMemory API                  │
│  record_experience() | learned_patterns() | compile_skill()  │
├─────────────┬──────────────┬────────────┬───────────┤
│  Experience │   Pattern    │   Belief   │   Skill   │
│    Store    │  Extractor   │  Updater   │ Compiler  │
│  (SQLite)   │  (numpy)     │ (Bayesian) │ (n-gram)  │
├─────────────┴──────────────┴────────────┴───────────┤
│              Transfer Learning Bridge                │
│   JSONL export | System prompts | Procedures        │
├─────────────────────────────────────────────────────┤
│                   MCP Server                         │
│          9 tools for external integration            │
└─────────────────────────────────────────────────────┘
                         │
                    SQLite DB
```

## Quick Start

```bash
pip install learning-agent-memory
```

```python
from learning_agent_memory import LearningMemory

mem = LearningMemory(agent_id="my-agent", db_path="agent.db")

# Record experiences
mem.record_experience(
    action="api_call",
    context={"endpoint": "/users", "method": "GET"},
    outcome={"status": 200, "latency_ms": 45},
    success=True,
    category="api-calls",
)

# Discover patterns
patterns = mem.learned_patterns(category="api-calls")
for p in patterns:
    print(f"{p.name}: {p.success_rate:.0%} success ({p.frequency} observations)")

# Compile skills from repeated successes
skill = mem.compile_skill("handle-rate-limits", category="api-calls")
print(f"Skill '{skill.name}': {len(skill.steps)} steps, {skill.success_rate:.0%} success")

# Export for fine-tuning
export = mem.export_knowledge()
print(export["system_prompt_supplement"])
```

## Core Components

### Experience Store
Structured SQLite storage for agent interactions with full queryability — filter by agent, category, action, success, and time range.

### Pattern Extractor
Identifies recurring patterns using:
- **Action patterns** — success rates per action type
- **Context patterns** — context values correlated with success/failure
- **Sequence patterns** — action bigrams that predict outcomes

Uses Wilson score intervals for statistical confidence.

### Belief Updater
Bayesian-style belief tracking that updates probability estimates as evidence accumulates. Beliefs have:
- Prior and posterior probabilities
- Evidence counts
- Asymptotic confidence scores

### Skill Compiler
Compresses repeated successful action sequences into reusable procedures:
- Manual compilation from filtered experiences
- Auto-compilation detecting repeated n-gram sequences
- Usage tracking for compiled skills

### Transfer Learning Bridge
Exports learned knowledge in multiple formats:
- **JSONL** — OpenAI fine-tuning format
- **System prompts** — pattern summaries for prompt injection
- **Procedures** — skill documentation in markdown
- **Knowledge base** — belief summaries

## MCP Server

Run as an MCP server for tool-based integration:

```bash
learning-agent-memory-server
```

Available tools: `record_experience`, `get_patterns`, `compile_skill`, `auto_compile_skills`, `get_belief`, `update_belief`, `add_belief`, `export_knowledge`, `agent_stats`

### Docker

```bash
docker build -t learning-agent-memory .
docker run -v ./data:/data learning-agent-memory
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
