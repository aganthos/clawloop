"""LfXAgent — high-level convenience wrapper for the LfX learning loop.

Bundles the learning loop into a simple, user-facing API with three levels
of usage:

- **Level 1**: ``agent.learn(env, iterations)`` — full plug-and-play
- **Level 2**: ``agent.ingest(episodes)`` — bring your own traces
- **Level 3**: Use :func:`~lfx.core.loop.learning_loop` directly — full control
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from lfx.core.env import EvalResult, Sample, TaskEnvironment
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.paradigm import ParadigmBreakthrough
from lfx.core.reflector import Reflector
from lfx.core.types import Datum
from lfx.layers.harness import Harness, Playbook, PlaybookEntry

log = logging.getLogger(__name__)


@dataclass
class LfXAgent:
    """High-level wrapper that bundles the LfX learning loop into a simple API.

    Parameters
    ----------
    task_client:
        LLMClient used for task execution (generating agent responses).
    reflector_client:
        LLMClient used for reflection (analysing traces and producing insights).
    bench:
        Benchmark/domain name used as the key for system prompts.
    base_system_prompt:
        Initial system prompt before any learning has occurred.
    """

    task_client: Any
    reflector_client: Any
    bench: str = "default"
    base_system_prompt: str = ""

    # Internal state (init=False)
    _harness: Harness = field(init=False)
    _intensity: AdaptiveIntensity = field(init=False)
    _tried_paradigms: list[str] = field(init=False)

    def __post_init__(self) -> None:
        reflector = Reflector(client=self.reflector_client)
        self._harness = Harness(
            system_prompts={self.bench: self.base_system_prompt},
            reflector=reflector,
        )
        self._intensity = AdaptiveIntensity()
        self._tried_paradigms = []

    # ------------------------------------------------------------------
    # Level 1: Full plug-and-play learning
    # ------------------------------------------------------------------

    def learn(
        self,
        env: TaskEnvironment,
        iterations: int = 5,
        episodes_per_iter: int = 5,
    ) -> dict:
        """Run the full learning loop over a task environment.

        For each iteration:
        1. Sample tasks randomly and run them via the task LLM.
        2. Compute average reward and record in intensity tracker.
        3. If intensity says reflect: run forward_backward + optim_step.
        4. If stagnating: fire paradigm breakthrough.

        Returns a dict with ``"rewards"`` (list of per-iteration avg rewards),
        ``"playbook"`` (serialised playbook dict), and ``"n_entries"``
        (number of playbook entries).
        """
        tasks = env.get_tasks()
        rewards: list[float] = []

        for i in range(iterations):
            log.info("LfXAgent iteration %d/%d", i + 1, iterations)

            # 1. Sample tasks and run episodes
            n_sample = min(episodes_per_iter, len(tasks))
            sampled = random.sample(tasks, n_sample)
            episodes: list[Episode] = []
            for sample in sampled:
                ep = self._run_one(sample, env)
                episodes.append(ep)

            # 2. Compute average reward
            avg_reward = (
                sum(ep.summary.total_reward for ep in episodes) / len(episodes)
                if episodes
                else 0.0
            )
            self._intensity.record_reward(avg_reward)
            rewards.append(avg_reward)
            log.info("  avg_reward=%.4f", avg_reward)

            # 3. Reflect if intensity says so
            if self._intensity.should_reflect(i):
                log.info("  reflecting...")
                datum = Datum(episodes=episodes)
                self._harness.forward_backward(datum)
                self._harness.optim_step()

            # 4. Paradigm breakthrough on stagnation
            if self._intensity.is_stagnating():
                log.info("  stagnation — triggering paradigm breakthrough")
                pb = ParadigmBreakthrough(client=self.reflector_client)
                insights = pb.generate(
                    playbook=self._harness.playbook,
                    reward_history=self._intensity._rewards,
                    tried_paradigms=self._tried_paradigms,
                )
                if insights:
                    for ins in insights:
                        self._tried_paradigms.append(ins.content)
                    self._harness.apply_insights(insights)

        return {
            "rewards": rewards,
            "playbook": self._harness.playbook.to_dict(),
            "n_entries": len(self._harness.playbook.entries),
        }

    # ------------------------------------------------------------------
    # Level 2: Bring your own traces
    # ------------------------------------------------------------------

    def ingest(self, episodes: list[Episode]) -> None:
        """Ingest externally-collected episodes into the learning system.

        Runs forward_backward (which triggers the reflector) and then
        optim_step to apply insights to the playbook.
        """
        datum = Datum(episodes=episodes)
        self._harness.forward_backward(datum)
        self._harness.optim_step()

    # ------------------------------------------------------------------
    # Prompt and playbook access
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return the current system prompt (base + playbook)."""
        return self._harness.system_prompt(self.bench)

    def save_playbook(self, path: str) -> None:
        """Save the current playbook to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._harness.playbook.to_dict(), f, indent=2)

    def load_playbook(self, path: str) -> None:
        """Load a playbook from a JSON file and set it on the harness."""
        with open(path) as f:
            data = json.load(f)
        entries = [
            PlaybookEntry(
                id=e["id"],
                content=e["content"],
                helpful=e.get("helpful", 0),
                harmful=e.get("harmful", 0),
                tags=e.get("tags", []),
            )
            for e in data.get("entries", [])
        ]
        self._harness.playbook = Playbook(entries=entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self, sample: Sample, env: TaskEnvironment) -> Episode:
        """Execute a single task sample and return an Episode."""
        system_prompt = self._harness.system_prompt(self.bench)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample.question},
        ]

        # Call the task LLM
        response_text = self.task_client.complete(messages)

        # Evaluate
        eval_result = env.evaluate(sample, response_text)

        # Build episode
        ep_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=sample.question),
            Message(role="assistant", content=response_text),
        ]

        step = StepMeta(
            t=0,
            reward=eval_result.score,
            done=True,
            timing_ms=0.0,
        )

        episode = Episode(
            id=Episode.new_id(),
            state_id="agent",
            task_id=sample.question,
            bench=self.bench,
            messages=ep_messages,
            step_boundaries=[0],
            steps=[step],
            summary=EpisodeSummary(total_reward=eval_result.score),
        )

        return episode
