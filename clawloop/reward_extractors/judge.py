"""LLM-as-judge reward extractor.

Sends the episode's conversation to an LLM judge that scores the agent's
response as {-1, 0, +1} based on instruction alignment and task completion.

Same approach as MetaClaw's PRM (arXiv 2603.17187) — not a trained reward
model, just a carefully prompted LLM call with majority voting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from clawloop.core.reward import RewardSignal

if TYPE_CHECKING:
    from clawloop.core.episode import Episode

log = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are a quality reviewer for AI agent responses.
Evaluate whether the assistant's response adequately addresses the user's instruction.

Score:
  +1 = Response clearly follows and substantially completes the instruction.
  -1 = Response is off-task, incorrect, or fails to address core requirements.
   0 = Completion quality is ambiguous or evidence is insufficient.

Respond with ONLY a single integer: -1, 0, or 1."""

JUDGE_USER_TEMPLATE = """\
## Instruction
{instruction}

## Assistant Response
{response}

Score:"""


class JudgeExtractor:
    """LLM-as-judge reward extractor (MetaClaw PRM-style).

    Calls an LLM to score the agent's response. Uses majority voting
    across ``n_votes`` independent calls for robustness.

    The extractor reports ``name="judge"`` so the RewardPipeline
    automatically skips it when higher-priority signals (execution,
    user feedback) already provide high-confidence scores.
    """

    name: str = "judge"

    def __init__(
        self,
        client: Any,
        n_votes: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 8,
    ) -> None:
        self.client = client  # LLMClient protocol (has .complete())
        self.n_votes = n_votes
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, episode: Episode) -> RewardSignal | None:
        """Score the episode via LLM judge with majority voting."""
        instruction, response = self._extract_pair(episode)
        if not instruction or not response:
            return None

        votes: list[int] = []
        for _ in range(self.n_votes):
            score = self._single_vote(instruction, response)
            if score is not None:
                votes.append(score)

        if not votes:
            log.warning("Judge produced no valid votes for episode %s", episode.id)
            return None

        # Majority vote
        from collections import Counter

        counts = Counter(votes)
        majority_score, majority_count = counts.most_common(1)[0]

        confidence = majority_count / len(votes)
        return RewardSignal(
            name="judge",
            value=float(majority_score),
            confidence=confidence * 0.8,  # Cap at 0.8 — judge is fallible
        )

    def _single_vote(self, instruction: str, response: str) -> int | None:
        """Make one judge call, return -1/0/+1 or None on failure."""
        user_msg = JUDGE_USER_TEMPLATE.format(
            instruction=instruction[:2000],
            response=response[:4000],
        )
        try:
            result = self.client.complete(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = result.text.strip()
            # Parse: look for -1, 0, or 1
            for token in text.split():
                token = token.strip(".,;:!?")
                if token in ("-1", "0", "1"):
                    return int(token)
            log.debug("Judge returned unparseable response: %r", text[:100])
            return None
        except Exception:
            log.warning("Judge LLM call failed", exc_info=True)
            return None

    @staticmethod
    def _extract_pair(episode: Episode) -> tuple[str, str]:
        """Extract the last user instruction and assistant response."""
        instruction = ""
        response = ""
        for msg in episode.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.role == "user":
                instruction = content
            elif msg.role == "assistant":
                response = content
        return instruction, response
