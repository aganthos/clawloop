"""Tests for clawloop.core.state."""

from clawloop.core.state import StateID, _canonical_json
from clawloop.learning_layers.harness import Harness, ToolConfig
from clawloop.learning_layers.router import Router
from clawloop.learning_layers.weights import Weights


class TestCanonicalJson:
    def test_sorted_keys(self) -> None:
        assert _canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_no_whitespace(self) -> None:
        result = _canonical_json({"key": "value"})
        assert " " not in result

    def test_deterministic(self) -> None:
        obj = {"z": [3, 2, 1], "a": {"nested": True}}
        assert _canonical_json(obj) == _canonical_json(obj)


class TestStateID:
    def _make_layers(self) -> tuple[Harness, Router, Weights]:
        h = Harness(
            system_prompts={"test": "You are a test agent."},
            tool_configs=[
                ToolConfig(
                    name="search",
                    schema={"type": "function"},
                    owner="harness",
                    mutable=True,
                )
            ],
        )
        r = Router(
            tier_models={"light": "gpt-3.5-turbo", "heavy": "gpt-4"},
            token_budgets={"default": 4096},
            fallback_chains=["gpt-4", "gpt-3.5-turbo"],
        )
        w = Weights(
            model_ref="meta-llama/Llama-3-8B",
            adapter_refs=[],
        )
        return h, r, w

    def test_from_layers(self) -> None:
        h, r, w = self._make_layers()
        sid = StateID.from_layers(h, r, w)
        assert len(sid.combined_hash) == 64  # SHA-256 hex
        assert len(sid.harness_hash) == 64
        assert len(sid.router_hash) == 64
        assert len(sid.weights_hash) == 64

    def test_deterministic(self) -> None:
        h, r, w = self._make_layers()
        sid1 = StateID.from_layers(h, r, w)
        sid2 = StateID.from_layers(h, r, w)
        assert sid1.combined_hash == sid2.combined_hash
        assert sid1.harness_hash == sid2.harness_hash

    def test_different_layers_different_hash(self) -> None:
        h, r, w = self._make_layers()
        sid1 = StateID.from_layers(h, r, w)

        h2 = Harness(system_prompts={"test": "Different prompt."})
        sid2 = StateID.from_layers(h2, r, w)

        assert sid1.combined_hash != sid2.combined_hash
        assert sid1.harness_hash != sid2.harness_hash
        # Router and weights unchanged
        assert sid1.router_hash == sid2.router_hash
        assert sid1.weights_hash == sid2.weights_hash

    def test_from_dicts(self) -> None:
        h, r, w = self._make_layers()
        sid_layers = StateID.from_layers(h, r, w)
        sid_dicts = StateID.from_dicts(h.to_dict(), r.to_dict(), w.to_dict())
        assert sid_layers.combined_hash == sid_dicts.combined_hash

    def test_frozen(self) -> None:
        h, r, w = self._make_layers()
        sid = StateID.from_layers(h, r, w)
        try:
            sid.combined_hash = "tampered"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # expected — frozen dataclass
