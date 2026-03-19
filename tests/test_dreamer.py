"""Tests for EpisodeDreamer — cross-episode meta-pattern analysis."""

import json

from lfx.core.background import BackgroundState, EpisodeDreamer
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.layers.harness import Playbook, PlaybookEntry
from lfx.llm import MockLLMClient


def _make_episode(task_id="t1", reward=0.5):
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id=task_id,
        bench="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_playbook():
    return Playbook(entries=[
        PlaybookEntry(id="e1", content="Always greet the user"),
    ])


def _make_state(episodes=None, episodes_since=25, time_since=700.0, idle=True):
    return BackgroundState(
        episodes_since_last_run=episodes_since,
        time_since_last_run=time_since,
        is_user_idle=idle,
        playbook=_make_playbook(),
        recent_episodes=episodes or [_make_episode() for _ in range(25)],
    )


class TestEpisodeDreamer:
    def test_dreamer_produces_tagged_insights(self) -> None:
        mock_response = json.dumps([
            {
                "action": "add",
                "content": "Pattern: failures cluster around X",
                "tags": ["meta-pattern"],
            },
        ])
        llm = MockLLMClient(responses=[mock_response])
        dreamer = EpisodeDreamer(llm=llm, episode_threshold=5)
        state = _make_state()

        insights = dreamer.run(state)

        assert len(insights) == 1
        assert insights[0].content == "Pattern: failures cluster around X"
        assert "meta-pattern" in insights[0].tags
        assert insights[0].action == "add"

    def test_dreamer_returns_empty_without_llm(self) -> None:
        dreamer = EpisodeDreamer(llm=None)
        state = _make_state()

        insights = dreamer.run(state)

        assert insights == []

    def test_dreamer_should_run_conditions(self) -> None:
        llm = MockLLMClient(responses=["[]"])
        dreamer = EpisodeDreamer(
            llm=llm,
            episode_threshold=20,
            min_interval=600.0,
        )

        # All conditions met
        state = _make_state(episodes_since=25, time_since=700.0, idle=True)
        assert dreamer.should_run(state) is True

        # Not enough episodes
        state = _make_state(episodes_since=10, time_since=700.0, idle=True)
        assert dreamer.should_run(state) is False

        # Not enough time elapsed
        state = _make_state(episodes_since=25, time_since=300.0, idle=True)
        assert dreamer.should_run(state) is False

        # User not idle
        state = _make_state(episodes_since=25, time_since=700.0, idle=False)
        assert dreamer.should_run(state) is False

        # No LLM configured
        no_llm = EpisodeDreamer(llm=None)
        state = _make_state(episodes_since=100, time_since=9999.0, idle=True)
        assert no_llm.should_run(state) is False

    def test_dreamer_handles_invalid_llm_response(self) -> None:
        llm = MockLLMClient(responses=["this is not json at all {{{"])
        dreamer = EpisodeDreamer(llm=llm, episode_threshold=5)
        state = _make_state()

        insights = dreamer.run(state)

        assert insights == []
