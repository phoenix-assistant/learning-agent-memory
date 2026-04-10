"""Tests for learning-agent-memory."""

import json
import time
import pytest

from learning_agent_memory import (
    LearningMemory, Experience, ExperienceStore,
    PatternExtractor, Pattern, BeliefUpdater, Belief,
    SkillCompiler, CompiledSkill, TransferBridge,
)


# --- Experience Store ---

def test_experience_record_and_get():
    store = ExperienceStore()
    exp = Experience(action="api_call", context={"url": "/test"}, outcome={"status": 200}, success=True, agent_id="a1")
    eid = store.record(exp)
    got = store.get(eid)
    assert got is not None
    assert got.action == "api_call"
    assert got.success is True
    store.close()


def test_experience_query_by_category():
    store = ExperienceStore()
    for i in range(5):
        store.record(Experience(action="a", context={}, outcome={}, success=True, agent_id="a1", category="cat1"))
    for i in range(3):
        store.record(Experience(action="b", context={}, outcome={}, success=False, agent_id="a1", category="cat2"))
    assert len(store.query(category="cat1")) == 5
    assert len(store.query(category="cat2")) == 3
    assert len(store.query(success=True)) == 5
    store.close()


def test_experience_count():
    store = ExperienceStore()
    store.record(Experience(action="x", context={}, outcome={}, success=True, agent_id="a1"))
    store.record(Experience(action="y", context={}, outcome={}, success=False, agent_id="a2"))
    assert store.count() == 2
    assert store.count(agent_id="a1") == 1
    store.close()


def test_experience_delete():
    store = ExperienceStore()
    exp = Experience(action="del", context={}, outcome={}, success=True, agent_id="a1")
    eid = store.record(exp)
    assert store.delete(eid) is True
    assert store.get(eid) is None
    store.close()


def test_experience_to_from_dict():
    exp = Experience(action="test", context={"k": "v"}, outcome={"r": 1}, success=True, agent_id="a1")
    d = exp.to_dict()
    exp2 = Experience.from_dict(d)
    assert exp2.action == exp.action
    assert exp2.context == exp.context


# --- Pattern Extractor ---

def _make_exps(action, n, success=True, category="test", context=None):
    return [
        Experience(action=action, context=context or {}, outcome={}, success=success,
                   agent_id="a1", category=category, timestamp=time.time() + i)
        for i in range(n)
    ]


def test_pattern_extraction_action():
    exps = _make_exps("fetch", 5, True) + _make_exps("fetch", 2, False)
    extractor = PatternExtractor(min_frequency=3, min_confidence=0.3)
    patterns = extractor.extract(exps)
    assert any(p.name == "action:fetch" for p in patterns)
    fetch = [p for p in patterns if p.name == "action:fetch"][0]
    assert fetch.frequency == 7
    assert 0.5 < fetch.success_rate < 1.0


def test_pattern_extraction_context():
    exps = [Experience(action="a", context={"env": "prod"}, outcome={}, success=True, agent_id="a1", timestamp=time.time()+i) for i in range(5)]
    exps += [Experience(action="a", context={"env": "prod"}, outcome={}, success=False, agent_id="a1", timestamp=time.time()+10+i) for i in range(2)]
    extractor = PatternExtractor(min_frequency=3, min_confidence=0.3)
    patterns = extractor.extract(exps)
    ctx_patterns = [p for p in patterns if p.name.startswith("context:")]
    assert len(ctx_patterns) > 0


def test_pattern_extraction_empty():
    extractor = PatternExtractor()
    assert extractor.extract([]) == []


# --- Belief Updater ---

def test_belief_add_and_get():
    bu = BeliefUpdater()
    b = bu.add_belief("test_key", "test_cat", prior=0.5, description="test belief")
    assert b.posterior == 0.5
    got = bu.get("test_key")
    assert got is not None
    assert got.key == "test_key"
    bu.close()


def test_belief_update_positive():
    bu = BeliefUpdater()
    bu.add_belief("k1", "c1", prior=0.5)
    for _ in range(5):
        bu.update("k1", positive=True)
    b = bu.get("k1")
    assert b.posterior > 0.5
    assert b.evidence_count == 5
    bu.close()


def test_belief_update_negative():
    bu = BeliefUpdater()
    bu.add_belief("k2", "c1", prior=0.5)
    for _ in range(5):
        bu.update("k2", positive=False)
    b = bu.get("k2")
    assert b.posterior < 0.5
    bu.close()


def test_belief_confidence_grows():
    bu = BeliefUpdater()
    bu.add_belief("k3", "c1")
    b0 = bu.get("k3")
    assert b0.confidence == 0.0
    bu.update("k3", positive=True)
    b1 = bu.get("k3")
    assert b1.confidence > 0
    for _ in range(10):
        bu.update("k3", positive=True)
    b2 = bu.get("k3")
    assert b2.confidence > b1.confidence
    bu.close()


# --- Skill Compiler ---

def test_skill_compile():
    sc = SkillCompiler()
    exps = _make_exps("step1", 3, True) + _make_exps("step2", 3, True)
    skill = sc.compile_from_experiences("test_skill", exps, "A test skill")
    assert skill.name == "test_skill"
    assert skill.source_count == 6
    assert skill.success_rate == 1.0
    got = sc.get("test_skill")
    assert got is not None
    sc.close()


def test_skill_compile_no_success():
    sc = SkillCompiler()
    exps = _make_exps("fail", 3, False)
    with pytest.raises(ValueError):
        sc.compile_from_experiences("fail_skill", exps)
    sc.close()


def test_skill_usage_tracking():
    sc = SkillCompiler()
    exps = _make_exps("a", 3, True)
    sc.compile_from_experiences("s1", exps)
    sc.record_usage("s1")
    sc.record_usage("s1")
    s = sc.get("s1")
    assert s.usage_count == 2
    sc.close()


# --- Transfer Bridge ---

def test_transfer_jsonl():
    exps = _make_exps("act", 3, True)
    jsonl = TransferBridge.experiences_to_jsonl(exps)
    lines = jsonl.strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert "messages" in data


def test_transfer_full_export():
    exps = _make_exps("act", 3, True)
    result = TransferBridge.full_export(exps, [], [], [])
    assert result["stats"]["total_experiences"] == 3
    assert result["stats"]["success_rate"] == 1.0


# --- LearningMemory (integration) ---

def test_learning_memory_integration():
    mem = LearningMemory(agent_id="test-agent")
    for i in range(5):
        mem.record_experience("api_call", {"endpoint": "/data"}, {"status": 200}, success=True, category="api")
    for i in range(2):
        mem.record_experience("api_call", {"endpoint": "/data"}, {"status": 500}, success=False, category="api")

    stats = mem.stats()
    assert stats["total_experiences"] == 7
    assert stats["successful"] == 5

    # Belief should exist and lean positive
    belief = mem.get_belief("action:api_call:success")
    assert belief is not None
    assert belief.posterior > 0.5

    # Export
    export = mem.export_knowledge()
    assert export["stats"]["total_experiences"] == 7
    mem.close()
