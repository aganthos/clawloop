"""CI compat gate: validates SkyRL submodule types are importable
and the Episode -> GeneratorOutput translation path works."""

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.exporters.skyrl import SkyRLExporter


def _skyrl_available() -> bool:
    try:
        import skyrl.tinker.types  # noqa: F401

        return True
    except ImportError:
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

    def test_full_translation_path(self):
        """Episode -> GeneratorOutput via SkyRLExporter."""
        from tests.test_skyrl_export import FakeTokenizer

        ep = Episode(
            id="test-ep",
            state_id="abc",
            task_id="t1",
            bench="test",
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

        assert "prompt_token_ids" in output
        assert "response_ids" in output
        assert "loss_masks" in output
        assert "rewards" in output
        assert "trajectory_ids" in output
        assert len(output["prompt_token_ids"]) > 0


class TestSkyRLExporterBasic:
    """These tests run without SkyRL — just verify exporter output shape."""

    def test_export_produces_required_keys(self):
        from tests.test_skyrl_export import FakeTokenizer

        ep = Episode(
            id="test",
            state_id="abc",
            task_id="t1",
            bench="test",
            messages=[
                Message(role="system", content="Hi"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=100.0)],
            summary=EpisodeSummary(total_reward=1.0),
        )
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export([ep])

        for key in (
            "prompt_token_ids",
            "response_ids",
            "rewards",
            "loss_masks",
            "trajectory_ids",
            "is_last_step",
        ):
            assert key in output, f"Missing key: {key}"
