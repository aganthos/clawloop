"""ClawLoop — Learning from Experience unified learning API."""

__version__ = "0.1.0"

from clawloop.agent import ClawLoopAgent
from clawloop.archive import (
    AgentVariant,
    ArchiveStore,
    EpisodeRecord,
    IterationRecord,
    JsonlArchiveStore,
    NullArchiveStore,
    RunRecord,
)
from clawloop.callbacks.litellm_cb import ClawLoopCallback
from clawloop.collector import EpisodeCollector
from clawloop.completion import CompletionResult
from clawloop.core.env import EvalResult, Sample, StaticTaskEnvironment
from clawloop.core.episode import TokenLogProb
from clawloop.core.reward import RewardPipeline, RewardSignal
from clawloop.harness_backends.local import LocalEvolver
from clawloop.learner import AsyncLearner
from clawloop.llm import LiteLLMClient, MockLLMClient
from clawloop.wrapper import wrap

__all__ = [
    "AgentVariant",
    "ArchiveStore",
    "AsyncLearner",
    "ClawLoopAgent",
    "ClawLoopCallback",
    "CompletionResult",
    "EpisodeCollector",
    "EpisodeRecord",
    "EvalResult",
    "IterationRecord",
    "JsonlArchiveStore",
    "LiteLLMClient",
    "LocalEvolver",
    "MockLLMClient",
    "NullArchiveStore",
    "RewardPipeline",
    "RewardSignal",
    "RunRecord",
    "Sample",
    "StaticTaskEnvironment",
    "TokenLogProb",
    "wrap",
]
