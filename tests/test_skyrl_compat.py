"""CI compat gate: validates SkyRL submodule types are importable
and the Episode → GeneratorOutput translation path works."""

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.exporters.skyrl import SkyRLExporter


def _skyrl_available() -> bool:
    try:
        import skyrl.tinker.types
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(not _skyrl_available(), reason="SkyRL submodule not available")
class TestSkyRLCompat:
    def test_tinker_types_importable(self):
        from skyrl.tinker.types import ForwardBackwardInput, OptimStepInput
        assert ForwardBackwardInput is not None
        assert OptimStepInput is not None

    def test_backend_importable(self):
        from skyrl.backends.backend import AbstractBackend
        assert AbstractBackend is not None

    def test_lora_config_importable(self):
        from skyrl.tinker.types import LoraConfig
        assert LoraConfig is not None

    def test_exporter_produces_valid_output(self):
        """Episode → GeneratorOutput via SkyRLExporter — the full translation path."""
        from tests.test_skyrl_export import FakeTokenizer

        ep = Episode(
            id="compat-test", state_id="abc", task_id="t1", bench="test",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=100.0)],
            summary=EpisodeSummary(total_reward=1.0),
        )
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export([ep])

        # Validate GeneratorOutput structure
        assert "prompt_token_ids" in output
        assert "response_ids" in output
        assert "loss_masks" in output
        assert "rewards" in output
        assert "trajectory_ids" in output
        assert len(output["prompt_token_ids"]) > 0
        assert len(output["response_ids"]) == len(output["prompt_token_ids"])
        assert len(output["loss_masks"]) == len(output["prompt_token_ids"])
        assert len(output["rewards"]) == len(output["prompt_token_ids"])


class TestExporterAlwaysWorks:
    """These tests don't need SkyRL — they validate the exporter itself."""

    def test_exporter_basic(self):
        from tests.test_skyrl_export import FakeTokenizer

        ep = Episode(
            id="basic-test", state_id="abc", task_id="t1", bench="test",
            messages=[
                Message(role="system", content="System."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.5, done=True, timing_ms=100.0)],
            summary=EpisodeSummary(total_reward=0.5),
        )
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export([ep])
        assert len(output["prompt_token_ids"]) == 1
        assert output["rewards"][0] == 0.5  # Terminal step carries reward
