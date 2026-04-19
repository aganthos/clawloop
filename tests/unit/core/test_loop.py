"""Unit tests for clawloop.core.loop."""


def test_agent_state_has_sampling_client_field_default_none():
    from clawloop.core.loop import AgentState
    state = AgentState()
    assert hasattr(state, "sampling_client")
    assert state.sampling_client is None


def test_agent_state_has_renderer_and_tokenizer_fields_default_none():
    from clawloop.core.loop import AgentState
    state = AgentState()
    assert hasattr(state, "renderer") and state.renderer is None
    assert hasattr(state, "tokenizer") and state.tokenizer is None


def test_learning_loop_refreshes_sampling_client_and_calls_save_state_per_iter():
    """Each iteration: refresh agent_state.sampling_client from backend + save_state at end."""
    from unittest.mock import MagicMock

    from clawloop.core.episode import Episode, EpisodeSummary
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.core.reward import RewardSignal
    from clawloop.core.types import FBResult, Future, OptimResult, SaveResult

    # --- Build a fake backend that has current_sampling_client + save_state
    sampling_clients = [MagicMock(name=f"sc_{i}") for i in range(3)]
    saved: list[str] = []

    def _current_sampling_client():
        # Returns sc_0 before any save, sc_1 after first save, sc_2 after second.
        return sampling_clients[len(saved)]

    def _save_state(name):
        saved.append(name)
        fut = MagicMock()
        fut.result.return_value = SaveResult(name=name, status="ok")
        return fut

    backend = MagicMock()
    backend.current_sampling_client.side_effect = _current_sampling_client
    backend.save_state.side_effect = _save_state
    backend.renderer = MagicMock(name="renderer")
    backend.tokenizer = MagicMock(name="tokenizer")

    # --- Fake Weights wrapper so _backend lookup works
    weights = MagicMock()
    weights._backend = backend
    weights.forward_backward = MagicMock(
        return_value=Future.immediate(FBResult(status="ok", metrics={"n_datums": 1})),
    )
    weights.optim_step = MagicMock(
        return_value=Future.immediate(OptimResult(status="ok", updates_applied=1, metrics={})),
    )
    weights.to_dict = MagicMock(return_value={})
    weights.load_state = MagicMock(return_value=Future.immediate(object()))
    weights.clear_pending_state = MagicMock()

    # --- Fake adapter recording which sampling_client was in effect at rollout time
    captured_clients: list = []

    def _fake_run_episode(task_id, agent_state):
        captured_clients.append(agent_state.sampling_client)
        summary = EpisodeSummary(
            signals={
                "outcome": RewardSignal(name="outcome", value=0.0, confidence=1.0),
            },
        )
        return Episode(
            id="fake-id",
            state_id="",
            task_id=task_id,
            bench="test",
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=summary,
        )

    adapter = MagicMock()
    adapter.run_episode.side_effect = _fake_run_episode
    # Force the loop to use run_episode (not run_batch / run_episodes_batch).
    if hasattr(adapter, "run_batch"):
        del adapter.run_batch
    if hasattr(adapter, "run_episodes_batch"):
        del adapter.run_episodes_batch

    # Seed state and run 2 iters.
    agent_state = AgentState(weights=weights)
    learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=["t1"],
        n_episodes=1,
        n_iterations=2,
        active_layers=["weights"],
    )

    # Iter 0 rollouts saw sc_0; iter 1 rollouts saw sc_1 (after iter-0 save).
    assert len(captured_clients) == 2
    assert captured_clients[0] is sampling_clients[0]
    assert captured_clients[1] is sampling_clients[1]
    # save_state called with iter_0 and iter_1.
    assert saved == ["iter_0", "iter_1"]


def test_learning_loop_tolerates_backend_without_hooks():
    """Existing SkyRL-style backend lacking current_sampling_client must still run."""
    from unittest.mock import MagicMock

    from clawloop.core.episode import Episode, EpisodeSummary
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.core.reward import RewardSignal
    from clawloop.core.types import FBResult, Future, OptimResult

    # Build a minimal backend WITHOUT current_sampling_client / save_state
    # (pure spec so attribute access returns AttributeError for those).
    backend = MagicMock(spec=["forward_backward", "optim_step"])
    weights = MagicMock()
    weights._backend = backend
    weights.forward_backward = MagicMock(
        return_value=Future.immediate(FBResult(status="ok", metrics={"n_datums": 1})),
    )
    weights.optim_step = MagicMock(
        return_value=Future.immediate(OptimResult(status="ok", updates_applied=1, metrics={})),
    )
    weights.to_dict = MagicMock(return_value={})
    weights.load_state = MagicMock(return_value=Future.immediate(object()))
    weights.clear_pending_state = MagicMock()

    def _fake_run_episode(task_id, agent_state):
        summary = EpisodeSummary(
            signals={
                "outcome": RewardSignal(name="outcome", value=0.0, confidence=1.0),
            },
        )
        return Episode(
            id="fake-id",
            state_id="",
            task_id=task_id,
            bench="test",
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=summary,
        )

    adapter = MagicMock()
    adapter.run_episode.side_effect = _fake_run_episode
    if hasattr(adapter, "run_batch"):
        del adapter.run_batch
    if hasattr(adapter, "run_episodes_batch"):
        del adapter.run_episodes_batch

    agent_state = AgentState(weights=weights)
    # Should not raise.
    learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=["t1"],
        n_episodes=1,
        n_iterations=1,
        active_layers=["weights"],
    )
