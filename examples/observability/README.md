# Observability Sinks

ClawLoop observability sinks are optional adapters that consume episode and iteration data without changing the learning loop. Use `WandbSink` with `uv sync --extra wandb` to log reward curves and episode tables to Weights & Biases, or `MlflowSink` with `uv sync --extra mlflow` to log reward metrics, state hashes, playbook artifacts, and episode summaries to MLflow. Sinks are designed to fail soft so a backend outage does not break a training run.
