"""Tests for PlaybookCurator.check_prompt_playbook_coherence()."""

from clawloop.core.curator import CuratorConfig, PlaybookCurator
from clawloop.core.embeddings import MockEmbedding
from clawloop.learning_layers.harness import Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient


class TestCoherence:
    """PlaybookCurator.check_prompt_playbook_coherence() conflict detection."""

    def test_no_conflicts_empty_playbook(self) -> None:
        """Empty playbook returns no conflicts."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["[]"])
        curator = PlaybookCurator(embeddings, llm)

        result = curator.check_prompt_playbook_coherence(
            "You are a helpful assistant.", Playbook()
        )

        assert result == []

    def test_conflicts_detected(self) -> None:
        """LLM returning a JSON array of conflicts is parsed correctly."""
        conflict_desc = "Prompt says be concise but playbook says be verbose"
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=[f'["{conflict_desc}"]'])
        curator = PlaybookCurator(embeddings, llm)

        playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Always provide verbose explanations"),
        ])

        result = curator.check_prompt_playbook_coherence(
            "You are a concise assistant. Keep answers short.", playbook,
        )

        assert len(result) == 1
        assert result[0] == conflict_desc

    def test_coherence_llm_failure_returns_empty(self) -> None:
        """When the LLM raises, coherence check returns empty list."""

        class _FailingLLM:
            def complete(self, messages, **kwargs):
                raise RuntimeError("LLM unavailable")

        embeddings = MockEmbedding()
        curator = PlaybookCurator(embeddings, _FailingLLM())  # type: ignore[arg-type]

        playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="some entry"),
        ])

        result = curator.check_prompt_playbook_coherence(
            "You are a helpful assistant.", playbook,
        )

        assert result == []
