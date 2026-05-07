"""Minimal example: MlflowSink logging ClawLoop episode metrics to MLflow.

Run with::

    uv sync --extra mlflow
    python examples/observability/mlflow_sink_demo.py

Or use the sink as a learning-loop callback::

    from clawloop.integrations.mlflow import MlflowSink

    sink = MlflowSink(experiment_name="clawloop-demo")
    learning_loop(..., after_iteration=sink.after_iteration)
    sink.finish()
"""

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta, Timing, TokenUsage
from clawloop.core.reward import RewardSignal
from clawloop.integrations.mlflow import MlflowSink


def _make_demo_episode(iteration: int) -> Episode:
    reward_value = min(0.5 + iteration * 0.1, 1.0)
    mapped = reward_value * 2.0 - 1.0

    return Episode(
        id=f"ep-{iteration:04d}",
        state_id=f"state-{iteration}",
        task_id="demo-task",
        bench="demo",
        messages=[
            Message(role="user", content=f"Iteration {iteration} input"),
            Message(role="assistant", content=f"Iteration {iteration} response"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward_value, done=True, timing_ms=150.0)],
        summary=EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=mapped, confidence=1.0)},
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            timing=Timing(total_ms=150.0, per_step_ms=[150.0]),
        ),
        model="gpt-4o-mini",
    )


def main() -> None:
    sink = MlflowSink(experiment_name="clawloop-demo", run_name="mlflow-sink-demo")

    for i in range(10):
        episodes = [_make_demo_episode(i) for _ in range(4)]
        sink.log_episodes(episodes, iteration=i)
        print(f"Logged iteration {i}")

    sink.finish()
    print("Done - run `mlflow ui` to inspect the run.")


if __name__ == "__main__":
    main()
