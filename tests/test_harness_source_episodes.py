"""Tests for source_episode_ids propagation in PlaybookEntry."""

from clawloop.learning_layers.harness import Harness, Insight, Playbook, PlaybookEntry


class TestPlaybookEntrySourceEpisodeIds:
    def test_playbook_entry_has_source_episode_ids(self) -> None:
        entry = PlaybookEntry(id="e1", content="tip")
        assert entry.source_episode_ids == []

    def test_playbook_entry_with_source_ids(self) -> None:
        entry = PlaybookEntry(id="e1", content="tip", source_episode_ids=["ep-1", "ep-2"])
        assert entry.source_episode_ids == ["ep-1", "ep-2"]

    def test_to_dict_includes_source_episode_ids(self) -> None:
        entry = PlaybookEntry(id="e1", content="tip", source_episode_ids=["ep-42"])
        d = entry.to_dict()
        assert "source_episode_ids" in d
        assert d["source_episode_ids"] == ["ep-42"]

    def test_apply_insights_propagates_source_episode_ids(self) -> None:
        h = Harness()
        insight = Insight(content="use retries", source_episode_ids=["ep-10", "ep-11"])
        h.apply_insights([insight])
        assert len(h.playbook.entries) == 1
        entry = h.playbook.entries[0]
        assert entry.source_episode_ids == ["ep-10", "ep-11"]

    def test_load_state_restores_source_episode_ids(self) -> None:
        state = {
            "system_prompts": {},
            "playbook": {
                "entries": [
                    {
                        "id": "e1",
                        "content": "tip",
                        "helpful": 0,
                        "harmful": 0,
                        "tags": [],
                        "source_episode_ids": ["ep-99"],
                    }
                ]
            },
        }
        h = Harness()
        h.load_state(state)
        assert len(h.playbook.entries) == 1
        assert h.playbook.entries[0].source_episode_ids == ["ep-99"]
