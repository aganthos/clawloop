"""Tests for clawloop.core.paradigm — ParadigmBreakthrough stagnation escape."""

from __future__ import annotations

import json

from clawloop.core.paradigm import ParadigmBreakthrough, ParadigmConfig
from clawloop.learning_layers.harness import Insight, Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient


class TestParadigmConfigDefaults:
    def test_defaults(self) -> None:
        cfg = ParadigmConfig()
        assert cfg.max_paradigms == 3
        assert cfg.temperature == 0.9
        assert cfg.max_tokens == 1500


class TestGenerateReturnsInsights:
    def test_generate_returns_insights(self) -> None:
        """Valid JSON response produces tagged insights with action='add'."""
        response_payload = json.dumps([
            {"content": "Try a tree-of-thought approach instead of chain-of-thought"},
            {"content": "Use adversarial self-play to discover edge cases"},
        ])
        client = MockLLMClient(responses=[response_payload])
        pb = ParadigmBreakthrough(client=client, config=ParadigmConfig())

        playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Always double-check arithmetic"),
        ])
        reward_history = [0.4, 0.42, 0.41]
        tried_paradigms: list[str] = []

        insights = pb.generate(playbook, reward_history, tried_paradigms)

        assert len(insights) == 2
        for ins in insights:
            assert isinstance(ins, Insight)
            assert ins.action == "add"
            assert "paradigm" in ins.tags

        assert "tree-of-thought" in insights[0].content
        assert "adversarial self-play" in insights[1].content


class TestGenerateIncludesTriedParadigms:
    def test_generate_includes_tried_paradigms(self) -> None:
        """Previously tried paradigms appear in the prompt sent to the LLM."""
        response_payload = json.dumps([
            {"content": "A brand-new direction"},
        ])
        client = MockLLMClient(responses=[response_payload])
        pb = ParadigmBreakthrough(client=client, config=ParadigmConfig())

        playbook = Playbook()
        reward_history = [0.3]
        tried_paradigms = ["tree-of-thought", "adversarial self-play"]

        pb.generate(playbook, reward_history, tried_paradigms)

        # Inspect the messages sent to the mock client
        assert len(client.call_log) == 1
        messages, _kwargs = client.call_log[0]
        # Concatenate all message contents to search for tried paradigms
        all_text = " ".join(m["content"] for m in messages)
        assert "tree-of-thought" in all_text
        assert "adversarial self-play" in all_text


class TestBadJsonReturnsEmpty:
    def test_bad_json_returns_empty(self) -> None:
        """Graceful degradation: malformed JSON returns an empty list."""
        client = MockLLMClient(responses=["this is not valid json {{{"])
        pb = ParadigmBreakthrough(client=client, config=ParadigmConfig())

        playbook = Playbook()
        reward_history = [0.2, 0.2]
        tried_paradigms: list[str] = []

        insights = pb.generate(playbook, reward_history, tried_paradigms)

        assert insights == []
