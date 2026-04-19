"""Tests for clawloop.learning_layers (harness, router, weights)."""

import json

from clawloop.learning_layers.harness import (
    Harness,
    Insight,
    ParetoFront,
    Playbook,
    PlaybookEntry,
    PromptCandidate,
    ToolConfig,
)
from clawloop.learning_layers.router import QueryFeatures, Router, Tier
from clawloop.learning_layers.weights import Weights

# -- ToolConfig --


class TestToolConfig:
    def test_to_dict(self) -> None:
        tc = ToolConfig(
            name="search",
            schema={"type": "function", "parameters": {}},
            owner="env",
            mutable=False,
            sandbox_required=True,
        )
        d = tc.to_dict()
        assert d["name"] == "search"
        assert d["owner"] == "env"
        assert d["sandbox_required"] is True


# -- PlaybookEntry --


class TestPlaybookEntry:
    def test_score(self) -> None:
        e = PlaybookEntry(id="s-1", content="strategy", helpful=5, harmful=2)
        assert e.score() == 3.0

    def test_new_id(self) -> None:
        a = PlaybookEntry.new_id("str")
        b = PlaybookEntry.new_id("str")
        assert a != b
        assert a.startswith("str-")


# -- Playbook --


class TestPlaybook:
    def test_add_and_lookup(self) -> None:
        pb = Playbook()
        entry = PlaybookEntry(id="s-1", content="test strategy")
        pb.add(entry)
        assert pb.lookup("s-1") is entry
        assert pb.lookup("nonexistent") is None

    def test_remove(self) -> None:
        pb = Playbook()
        pb.add(PlaybookEntry(id="s-1", content="a"))
        pb.add(PlaybookEntry(id="s-2", content="b"))
        assert pb.remove("s-1") is True
        assert pb.remove("s-1") is False
        assert len(pb.entries) == 1

    def test_prune(self) -> None:
        pb = Playbook()
        pb.add(PlaybookEntry(id="s-1", content="good", helpful=5, harmful=0))
        pb.add(PlaybookEntry(id="s-2", content="bad", helpful=0, harmful=3))
        pb.add(PlaybookEntry(id="s-3", content="ok", helpful=2, harmful=2))
        pruned = pb.prune(min_score=1.0)
        assert pruned == 2  # s-2 and s-3 removed
        assert len(pb.entries) == 1
        assert pb.entries[0].id == "s-1"

    def test_render(self) -> None:
        pb = Playbook()
        pb.add(PlaybookEntry(id="s-1", content="test", helpful=3, harmful=1))
        text = pb.render()
        assert "PLAYBOOK" in text
        assert "[s-1]" in text
        assert "test" in text

    def test_render_empty(self) -> None:
        assert Playbook().render() == ""


# -- ParetoFront --


class TestParetoFront:
    def test_add_and_best(self) -> None:
        front = ParetoFront()
        c1 = PromptCandidate(id="c1", text="prompt A", per_task_scores={"t1": 0.8})
        c2 = PromptCandidate(id="c2", text="prompt B", per_task_scores={"t1": 0.9})
        front.add(c1)
        front.add(c2)
        best = front.best()
        assert best is not None
        assert best.mean_score() >= 0.8

    def test_prune_dominated(self) -> None:
        front = ParetoFront()
        # c1 dominates c2 on all shared tasks
        c1 = PromptCandidate(id="c1", text="A", per_task_scores={"t1": 0.9, "t2": 0.8})
        c2 = PromptCandidate(id="c2", text="B", per_task_scores={"t1": 0.5, "t2": 0.4})
        front.add(c1)
        front.add(c2)
        # c2 should be pruned
        assert len(front.candidates) == 1
        assert front.candidates[0].id == "c1"

    def test_non_dominated_preserved(self) -> None:
        front = ParetoFront()
        # c1 better on t1, c2 better on t2 -> both non-dominated
        c1 = PromptCandidate(id="c1", text="A", per_task_scores={"t1": 0.9, "t2": 0.4})
        c2 = PromptCandidate(id="c2", text="B", per_task_scores={"t1": 0.5, "t2": 0.9})
        front.add(c1)
        front.add(c2)
        assert len(front.candidates) == 2


# -- Harness --


class TestHarness:
    def test_default_empty(self) -> None:
        h = Harness()
        assert h.system_prompts == {}
        assert h.tool_configs == []
        assert h.validators == {}
        assert len(h.playbook.entries) == 0

    def test_system_prompt_with_playbook(self) -> None:
        h = Harness(system_prompts={"test": "You are helpful."})
        h.playbook.add(PlaybookEntry(id="s-1", content="Always be concise", helpful=3))
        prompt = h.system_prompt("test")
        assert "You are helpful." in prompt
        assert "Always be concise" in prompt

    def test_apply_insights_add(self) -> None:
        h = Harness()
        insights = [
            Insight(content="New strategy", tags=["strategy"]),
            Insight(content="Another one"),
        ]
        applied = h.apply_insights(insights)
        assert applied == 2
        assert len(h.playbook.entries) == 2

    def test_apply_insights_update(self) -> None:
        h = Harness()
        h.playbook.add(PlaybookEntry(id="s-1", content="old", helpful=1))
        insights = [Insight(content="updated", action="update", target_entry_id="s-1")]
        h.apply_insights(insights)
        assert h.playbook.lookup("s-1").content == "updated"
        assert h.playbook.lookup("s-1").helpful == 2

    def test_apply_insights_remove(self) -> None:
        h = Harness()
        h.playbook.add(PlaybookEntry(id="s-1", content="gone"))
        insights = [Insight(content="", action="remove", target_entry_id="s-1")]
        h.apply_insights(insights)
        assert h.playbook.lookup("s-1") is None

    def test_update_pareto_promotes_best(self) -> None:
        h = Harness()
        c = PromptCandidate(id="c1", text="optimized prompt", per_task_scores={"t1": 0.9})
        h.update_pareto("bench1", c)
        assert h.system_prompts["bench1"] == "optimized prompt"

    def test_to_dict(self) -> None:
        h = Harness(
            system_prompts={"test": "prompt"},
            tool_configs=[ToolConfig(name="t", schema={}, owner="harness", mutable=True)],
        )
        d = h.to_dict()
        assert "system_prompts" in d
        assert "playbook" in d
        assert "pareto_fronts" in d
        assert len(d["tool_configs"]) == 1

    def test_to_dict_deterministic(self) -> None:
        h = Harness(system_prompts={"b": "2", "a": "1"})
        s1 = json.dumps(h.to_dict(), sort_keys=True)
        s2 = json.dumps(h.to_dict(), sort_keys=True)
        assert s1 == s2


# -- Router --


class TestRouter:
    def test_default_tiers(self) -> None:
        r = Router()
        assert Tier.LIGHT in r.tier_models
        assert Tier.REASONING in r.tier_models

    def test_classify_returns_tier(self) -> None:
        r = Router()
        features = QueryFeatures(token_count=10)
        tier = r.classify(features)
        assert tier in Tier.ALL

    def test_trivial_query_routes_light(self) -> None:
        """A zero-feature query must classify as LIGHT, not HEAVY."""
        r = Router()
        tier = r.classify(QueryFeatures())
        assert tier == Tier.LIGHT

    def test_route_with_models(self) -> None:
        r = Router(
            tier_models={
                Tier.LIGHT: "haiku",
                Tier.MEDIUM: "sonnet",
                Tier.HEAVY: "opus",
                Tier.REASONING: "opus",
            }
        )
        features = QueryFeatures(token_count=10)
        model = r.route(features)
        assert model in ("haiku", "sonnet", "opus")

    def test_route_fallback(self) -> None:
        r = Router(
            tier_models={t: "" for t in Tier.ALL},
            fallback_chains=["fallback-model"],
        )
        model = r.route(QueryFeatures())
        assert model == "fallback-model"

    def test_record_and_update(self) -> None:
        r = Router()
        for _ in range(5):
            r.record_outcome(QueryFeatures(token_count=10), "haiku", cost=1.0, reward=0.9)
            r.record_outcome(
                QueryFeatures(token_count=500, reasoning_markers=3),
                "opus",
                cost=10.0,
                reward=0.95,
            )
        deltas = r.update_weights()
        assert len(deltas) > 0
        # Weights should sum to ~1.0
        assert abs(sum(r.score_weights.values()) - 1.0) < 0.01

    def test_to_dict(self) -> None:
        r = Router(fallback_chains=["gpt-4", "gpt-3.5-turbo"])
        d = r.to_dict()
        assert "tier_models" in d
        assert "score_weights" in d
        assert len(d["fallback_chains"]) == 2


# -- Weights --


class TestWeights:
    def test_default_empty(self) -> None:
        w = Weights()
        assert w.model_ref == ""
        assert w.adapter_refs == []
        assert w.active_adapter is None

    def test_record_training_step(self) -> None:
        w = Weights(model_ref="meta-llama/Llama-3-8B")
        w.record_training_step("lora-v1", {"loss": 0.5, "reward_mean": 0.7})
        assert w.active_adapter == "lora-v1"
        assert len(w.training_history) == 1

    def test_grpo_config_defaults(self) -> None:
        w = Weights()
        assert w.grpo_config.n_samples_per_prompt == 4
        assert w.grpo_config.clip_ratio == 0.2

    def test_to_dict(self) -> None:
        w = Weights(
            model_ref="meta-llama/Llama-3-8B",
            adapter_refs=["lora-v1"],
        )
        d = w.to_dict()
        assert d["model_ref"] == "meta-llama/Llama-3-8B"
        assert "lora-v1" in d["adapter_refs"]
        assert "grpo_config" in d
        assert d["grpo_config"]["n_samples_per_prompt"] == 4
