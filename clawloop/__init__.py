"""ClawLoop — Learning from Experience unified learning API."""

__version__ = "0.0.1"

from clawloop.agent import ClawLoopAgent
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
    "AsyncLearner",
    "CompletionResult",
    "EpisodeCollector",
    "EvalResult",
    "ClawLoopAgent",
    "ClawLoopCallback",
    "LiteLLMClient",
    "LocalEvolver",
    "MockLLMClient",
    "RewardPipeline",
    "RewardSignal",
    "Sample",
    "StaticTaskEnvironment",
    "TokenLogProb",
    "wrap",
]
