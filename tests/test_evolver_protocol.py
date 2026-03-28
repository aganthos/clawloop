"""Tests for internal Evolver interface and lifecycle types."""

from clawloop.core.evolver import (
    Evolver,
    EvolverContext,
    EvolverResult,
    HarnessSnapshot,
    Provenance,
)


class StubEvolver:
    """Minimal evolver that does nothing — proves interface is satisfiable."""

    def evolve(self, episodes, harness_state, context):
        return EvolverResult()

    def name(self):
        return "stub"


def test_stub_satisfies_interface():
    e = StubEvolver()
    result = e.evolve(
        episodes=[],
        harness_state=HarnessSnapshot(
            system_prompts={},
            playbook_entries=[],
            pareto_fronts={},
            playbook_generation=0,
            playbook_version=0,
        ),
        context=EvolverContext(
            reward_history=[],
            is_stagnating=False,
            iteration=0,
        ),
    )
    assert isinstance(result, EvolverResult)
    assert result.insights == []
    assert result.candidates == {}
    assert result.paradigm_shift is False
    assert result.run_id == ""
    assert e.name() == "stub"


def test_harness_snapshot_serializable():
    snap = HarnessSnapshot(
        system_prompts={"default": "You are helpful."},
        playbook_entries=[{"id": "e1", "content": "Be concise", "helpful": 3, "harmful": 0}],
        pareto_fronts={"default": [{"id": "pc-1", "text": "You are helpful.", "per_task_scores": {"t1": 0.8}, "generation": 0, "parent_id": None}]},
        playbook_generation=5,
        playbook_version=12,
    )
    d = snap.to_dict()
    assert d["playbook_generation"] == 5
    assert len(d["playbook_entries"]) == 1


def test_evolver_result_with_all_fields():
    from clawloop.layers.harness import Insight

    result = EvolverResult(
        insights=[Insight(action="add", content="test insight", tags=["test"])],
        candidates={"default": []},
        paradigm_shift=True,
        deprecation_targets=["entry_1", "entry_2"],
        run_id="ev-abc123",
        provenance=Provenance(backend="test", version="0.1", tokens_used=100),
    )
    assert result.paradigm_shift is True
    assert len(result.deprecation_targets) == 2
    assert result.run_id == "ev-abc123"
    assert result.provenance.backend == "test"


def test_evolver_context_defaults():
    ctx = EvolverContext()
    assert ctx.is_stagnating is False
    assert ctx.iteration == 0
    assert ctx.max_tokens is None


def test_fb_info_schema():
    """FBResult.info should follow standardized schema for lifecycle."""
    from clawloop.core.evolver import make_fb_info

    info = make_fb_info(
        status="ok",
        run_id="ev-001",
        candidates_tested=28,
        best_score=0.85,
        backend="local",
    )
    assert info["info_version"] == 1
    assert info["status"] == "ok"
    assert info["run_id"] == "ev-001"
