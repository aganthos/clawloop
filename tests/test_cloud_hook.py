"""Tests for cloud_url, cloud_api_key, and trace_level parameters on wrap()."""

from unittest.mock import MagicMock

import pytest

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.wrapper import WrappedClient, wrap


def _make_collector() -> EpisodeCollector:
    return EpisodeCollector(pipeline=MagicMock(spec=RewardPipeline))


def test_wrap_accepts_cloud_url_and_api_key():
    client = MagicMock()
    collector = _make_collector()
    wrapped = wrap(
        client,
        collector,
        cloud_url="https://api.clawloop.dev",
        cloud_api_key="cl-key-test",
    )
    assert isinstance(wrapped, WrappedClient)
    assert wrapped._cloud_url == "https://api.clawloop.dev"
    assert wrapped._cloud_api_key == "cl-key-test"


def test_wrap_without_cloud_url_defaults_to_none():
    client = MagicMock()
    collector = _make_collector()
    wrapped = wrap(client, collector)
    assert wrapped._cloud_url is None
    assert wrapped._cloud_api_key is None


@pytest.mark.parametrize("level", ["minimal", "standard", "full"])
def test_wrap_accepts_all_valid_trace_levels(level):
    wrapped = wrap(MagicMock(), _make_collector(), trace_level=level)
    assert wrapped._trace_level == level


def test_wrap_trace_level_defaults_to_minimal():
    client = MagicMock()
    collector = _make_collector()
    wrapped = wrap(client, collector)
    assert wrapped._trace_level == "minimal"


def test_wrap_rejects_invalid_trace_level():
    client = MagicMock()
    collector = _make_collector()
    with pytest.raises(ValueError, match="trace_level must be one of"):
        wrap(client, collector, trace_level="verbose")


def test_wrap_rejects_cloud_url_without_api_key():
    with pytest.raises(ValueError, match="cloud_api_key is required"):
        wrap(MagicMock(), _make_collector(), cloud_url="https://api.clawloop.dev")


@pytest.mark.parametrize("api_key", ["", "   "])
def test_wrap_rejects_cloud_url_with_invalid_api_key(api_key):
    with pytest.raises(ValueError, match="cloud_api_key must be non-empty"):
        wrap(
            MagicMock(),
            _make_collector(),
            cloud_url="https://api.clawloop.dev",
            cloud_api_key=api_key,
        )


@pytest.mark.parametrize("url", ["", "   "])
def test_wrap_rejects_invalid_cloud_url(url):
    with pytest.raises(ValueError, match="cloud_url must be non-empty"):
        wrap(MagicMock(), _make_collector(), cloud_url=url)


def test_wrap_accepts_api_key_without_cloud_url():
    wrapped = wrap(MagicMock(), _make_collector(), cloud_api_key="cl-key-test")
    assert wrapped._cloud_url is None
    assert wrapped._cloud_api_key == "cl-key-test"
