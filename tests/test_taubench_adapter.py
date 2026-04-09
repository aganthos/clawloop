"""Unit tests for TauBenchAdapter — tau2 library is mocked throughout."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from clawloop.core.episode import Message
from clawloop.environments.taubench import TauBenchAdapter, _compute_step_boundaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tau2_message(role: str, content: str):
    m = MagicMock()
    m.role = MagicMock()
    m.role.value = role
    m.content = content
    return m


def _make_sim_run(
    reward: float = 1.0,
    messages=None,
    termination_reason: str = "SUCCESS",
    duration: float = 1.5,
):
    sim_run = MagicMock()
    sim_run.reward_info = MagicMock()
    sim_run.reward_info.reward = reward
    sim_run.reward_info.db_check = None
    sim_run.reward_info.env_assertions = []
    sim_run.reward_info.action_checks = []
    sim_run.messages = messages or []
    tr = MagicMock()
    tr.value = termination_reason
    sim_run.termination_reason = tr
    sim_run.duration = duration
    sim_run.agent_cost = 0.01
    sim_run.user_cost = 0.005
    return sim_run


def _make_task(task_id: str):
    t = MagicMock()
    t.id = task_id
    return t


def _make_agent_state(harness_prompt: str = "You are helpful."):
    state = MagicMock()
    state.harness.system_prompt.return_value = harness_prompt
    state.state_id.return_value.combined_hash = "abc123"
    return state


# ---------------------------------------------------------------------------
# _compute_step_boundaries
# ---------------------------------------------------------------------------

class TestComputeStepBoundaries:
    def test_empty_returns_empty(self):
        assert _compute_step_boundaries([]) == []

    def test_single_user_message(self):
        msgs = [Message(role="user", content="hi")]
        assert _compute_step_boundaries(msgs) == [0]

    def test_user_assistant_user(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="bye"),
        ]
        assert _compute_step_boundaries(msgs) == [0, 2]

    def test_consecutive_user_messages_count_as_one_boundary(self):
        msgs = [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
            Message(role="assistant", content="c"),
            Message(role="user", content="d"),
        ]
        assert _compute_step_boundaries(msgs) == [0, 3]

    def test_only_assistant_messages_returns_zero(self):
        msgs = [Message(role="assistant", content="a")]
        assert _compute_step_boundaries(msgs) == [0]


# ---------------------------------------------------------------------------
# TauBenchAdapter._map_to_episode
# ---------------------------------------------------------------------------

class TestMapToEpisode:
    def _adapter(self, domain: str = "retail") -> TauBenchAdapter:
        a = TauBenchAdapter()
        a._domain = domain
        a._iteration_count = 0
        return a

    def test_successful_episode_has_outcome_signal(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(reward=1.0)
        ep = adapter._map_to_episode(sim_run, "retail_0", "abc123")
        assert ep.task_id == "taubench:retail_0"
        assert ep.bench == "taubench"
        # total_reward=1.0 maps to outcome signal value=1.0
        assert ep.summary.signals["outcome"].value == pytest.approx(1.0)

    def test_failed_episode_outcome_zero(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(reward=0.0)
        ep = adapter._map_to_episode(sim_run, "retail_1", "abc123")
        assert ep.summary.signals["outcome"].value == pytest.approx(-1.0)

    def test_messages_are_mapped(self):
        adapter = self._adapter()
        tau_messages = [
            _make_tau2_message("user", "Hello"),
            _make_tau2_message("assistant", "Hi there"),
        ]
        sim_run = _make_sim_run(messages=tau_messages)
        ep = adapter._map_to_episode(sim_run, "retail_0", "abc123")
        assert len(ep.messages) == 2
        assert ep.messages[0].role == "user"
        assert ep.messages[0].content == "Hello"
        assert ep.messages[1].role == "assistant"

    def test_score_breakdown_contains_reward(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(reward=1.0)
        ep = adapter._map_to_episode(sim_run, "retail_0", "state42")
        assert ep.summary.score_breakdown["reward"] == pytest.approx(1.0)

    def test_state_id_propagated(self):
        adapter = self._adapter()
        sim_run = _make_sim_run()
        ep = adapter._map_to_episode(sim_run, "retail_0", "mystate")
        assert ep.state_id == "mystate"

    def test_max_steps_metadata_set(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(termination_reason="MAX_STEPS_REACHED")
        ep = adapter._map_to_episode(sim_run, "retail_0", "")
        assert ep.metadata.get("truncated") is True

    def test_reward_info_none_gives_zero_reward(self):
        adapter = self._adapter()
        sim_run = _make_sim_run()
        sim_run.reward_info = None
        ep = adapter._map_to_episode(sim_run, "retail_0", "")
        assert ep.summary.signals["outcome"].value == pytest.approx(-1.0)

    def test_db_check_in_score_breakdown(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(reward=1.0)
        db_check = MagicMock()
        db_check.model_dump.return_value = {"passed": True}
        sim_run.reward_info.db_check = db_check
        ep = adapter._map_to_episode(sim_run, "retail_0", "")
        assert ep.summary.score_breakdown["db_check"] == {"passed": True}

    def test_domain_in_metadata(self):
        adapter = self._adapter("airline")
        sim_run = _make_sim_run()
        ep = adapter._map_to_episode(sim_run, "airline_0", "")
        assert ep.metadata["domain"] == "airline"

    def test_max_errors_reached_sets_filtered(self):
        adapter = self._adapter()
        sim_run = _make_sim_run(termination_reason="MAX_ERRORS_REACHED")
        ep = adapter._map_to_episode(sim_run, "retail_0", "")
        assert ep.summary.filtered is True


# ---------------------------------------------------------------------------
# TauBenchAdapter._make_failed_episode
# ---------------------------------------------------------------------------

class TestMakeFailedEpisode:
    def test_failed_episode_has_negative_outcome(self):
        adapter = TauBenchAdapter()
        adapter._domain = "retail"
        ep = adapter._make_failed_episode("retail_0", "state1", "task_not_found")
        assert ep.summary.signals["outcome"].value == pytest.approx(-1.0)
        assert ep.metadata["error"] == "task_not_found"
        assert ep.task_id == "taubench:retail_0"


# ---------------------------------------------------------------------------
# TauBenchAdapter.list_tasks
# ---------------------------------------------------------------------------

class TestListTasks:
    @patch("clawloop.environments.taubench.get_tasks")
    def test_returns_task_ids(self, mock_get_tasks):
        mock_get_tasks.return_value = [_make_task("retail_0"), _make_task("retail_1")]
        adapter = TauBenchAdapter()
        adapter._domain = "retail"
        result = adapter.list_tasks("test")
        assert result == ["retail_0", "retail_1"]
        mock_get_tasks.assert_called_once_with(task_set_name="retail", task_split_name="test")


# ---------------------------------------------------------------------------
# TauBenchAdapter.run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    @patch("clawloop.environments.taubench.run_single_task")
    @patch("clawloop.environments.taubench.get_tasks")
    @patch("clawloop.environments.taubench._register_clawloop_agent")
    @patch("clawloop.environments.taubench.TextRunConfig")
    def test_run_batch_returns_one_episode_per_task(
        self, mock_config, mock_register, mock_get_tasks, mock_run_single
    ):
        mock_get_tasks.return_value = [_make_task("retail_0"), _make_task("retail_1")]
        mock_run_single.return_value = _make_sim_run(reward=1.0)

        adapter = TauBenchAdapter()
        adapter._domain = "retail"
        adapter._llm_agent = "openai/gpt-4o-mini"
        adapter._llm_user = "openai/gpt-4o-mini"
        adapter._max_steps = 30
        adapter._max_concurrency = 2
        adapter._task_split = "test"
        adapter._iteration_count = 0

        agent_state = _make_agent_state()
        episodes = adapter.run_batch(agent_state, ["retail_0", "retail_1"])

        assert len(episodes) == 2
        assert all(ep.bench == "taubench" for ep in episodes)
        mock_register.assert_called_once()

    @patch("clawloop.environments.taubench.run_single_task")
    @patch("clawloop.environments.taubench.get_tasks")
    @patch("clawloop.environments.taubench._register_clawloop_agent")
    @patch("clawloop.environments.taubench.TextRunConfig")
    def test_missing_task_produces_failed_episode(
        self, mock_config, mock_register, mock_get_tasks, mock_run_single
    ):
        mock_get_tasks.return_value = [_make_task("retail_0")]  # retail_99 not here

        adapter = TauBenchAdapter()
        adapter._domain = "retail"
        adapter._llm_agent = "openai/gpt-4o-mini"
        adapter._llm_user = "openai/gpt-4o-mini"
        adapter._max_steps = 30
        adapter._max_concurrency = 2
        adapter._task_split = "test"
        adapter._iteration_count = 0

        agent_state = _make_agent_state()
        episodes = adapter.run_batch(agent_state, ["retail_99"])

        assert len(episodes) == 1
        assert episodes[0].metadata["error"] == "task_not_found"



# ---------------------------------------------------------------------------
# TauBenchAdapter.setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_reads_config(self):
        adapter = TauBenchAdapter()
        adapter.setup({
            "domain": "airline",
            "llm_agent": "openai/gpt-4o",
            "llm_user": "openai/gpt-4o",
            "max_steps": 50,
            "max_concurrency": 4,
            "task_split": "dev",
            "num_tasks": 5,
        })
        assert adapter._domain == "airline"
        assert adapter._llm_agent == "openai/gpt-4o"
        assert adapter._max_steps == 50
        assert adapter._max_concurrency == 4
        assert adapter._task_split == "dev"
        assert adapter._num_tasks == 5
        assert adapter._iteration_count == 0


# ---------------------------------------------------------------------------
# Harness prompt passthrough
# ---------------------------------------------------------------------------

class TestHarnessPromptPassthrough:
    @patch("clawloop.environments.taubench.TextRunConfig")
    @patch("clawloop.environments.taubench._register_clawloop_agent")
    @patch("clawloop.environments.taubench.get_tasks")
    @patch("clawloop.environments.taubench.run_single_task")
    def test_harness_prompt_passed_to_register(
        self, mock_run, mock_get_tasks, mock_register, mock_config
    ):
        """Confirm the current harness prompt is forwarded to _register_clawloop_agent."""
        from clawloop.core.loop import AgentState
        from clawloop.learning_layers.harness import Harness

        mock_get_tasks.return_value = [_make_task("retail_0")]
        mock_run.return_value = _make_sim_run(reward=1.0)

        harness = Harness(system_prompts={"taubench": "Follow the policy carefully."})
        agent_state = AgentState(harness=harness)

        adapter = TauBenchAdapter()
        adapter.setup({"domain": "retail", "task_split": "test"})
        adapter.run_batch(agent_state, ["retail_0"])

        mock_register.assert_called_once_with("Follow the policy carefully.")
