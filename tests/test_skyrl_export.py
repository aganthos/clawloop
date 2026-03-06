"""Tests for lfx.exporters.skyrl — Episode -> GeneratorOutput serialization."""

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta, TokenUsage
from lfx.exporters.skyrl import SkyRLExporter, TrajectoryID


class FakeTokenizer:
    """Minimal tokenizer for testing (word-level split)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple: each word gets a unique-ish token ID
        words = text.split()
        return [hash(w) % 10000 for w in words] if words else []

    def apply_chat_template(
        self,
        conversation: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        # Concatenate all content and tokenize
        parts = []
        for msg in conversation:
            parts.append(f"<{msg['role']}>")
            parts.append(msg.get("content", "") or "")
        text = " ".join(parts)
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text


def _make_episode(n_steps: int = 2, reward: float = 1.0) -> Episode:
    messages: list[Message] = []
    step_boundaries: list[int] = []
    steps: list[StepMeta] = []

    messages.append(Message(role="system", content="You are helpful."))

    for t in range(n_steps):
        step_boundaries.append(len(messages))
        messages.append(Message(role="user", content=f"Do step {t}"))
        messages.append(Message(role="assistant", content=f"Done with step {t}"))
        is_terminal = t == n_steps - 1
        steps.append(
            StepMeta(
                t=t,
                reward=reward if is_terminal else 0.0,
                done=is_terminal,
                timing_ms=50.0,
            )
        )

    return Episode(
        id="ep-001",
        state_id="state-abc",
        task_id="task-xyz",
        bench="test",
        messages=messages,
        step_boundaries=step_boundaries,
        steps=steps,
        summary=EpisodeSummary(
            total_reward=reward,
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
        ),
    )


class TestSkyRLExporter:
    def setup_method(self) -> None:
        self.exporter = SkyRLExporter(tokenizer=FakeTokenizer())

    def test_export_one_episode(self) -> None:
        ep = _make_episode(n_steps=2, reward=0.8)
        result = self.exporter.export([ep])

        assert "prompt_token_ids" in result
        assert "response_ids" in result
        assert "rewards" in result
        assert "loss_masks" in result
        assert "trajectory_ids" in result
        assert "is_last_step" in result

        # 2 steps -> 2 transitions
        assert len(result["prompt_token_ids"]) == 2
        assert len(result["response_ids"]) == 2
        assert len(result["rewards"]) == 2
        assert len(result["loss_masks"]) == 2

    def test_sparse_rewards(self) -> None:
        ep = _make_episode(n_steps=3, reward=0.9)
        result = self.exporter.export([ep])

        # Only the last step (terminal) should have reward
        assert result["rewards"][0] == 0.0
        assert result["rewards"][1] == 0.0
        assert result["rewards"][2] == 0.9

    def test_is_last_step_flags(self) -> None:
        ep = _make_episode(n_steps=3)
        result = self.exporter.export([ep])

        assert result["is_last_step"] == [False, False, True]

    def test_loss_masks_all_ones_for_assistant_only_response(self) -> None:
        ep = _make_episode(n_steps=1, reward=1.0)
        result = self.exporter.export([ep])

        # Response should only contain assistant tokens (user is in prompt)
        mask = result["loss_masks"][0]
        assert len(mask) > 0
        assert all(m == 1 for m in mask)

    def test_prompt_includes_user_message(self) -> None:
        """The prompt must include the user turn so the model is conditioned
        on the same context it sees at inference time."""
        ep = _make_episode(n_steps=2, reward=1.0)
        result = self.exporter.export([ep])

        # Step 0: prompt = system + user ("Do step 0")
        # Step 1: prompt = system + user + assistant + user ("Do step 1")
        # Both prompts must be non-empty (they include at least the user msg)
        for p in result["prompt_token_ids"]:
            assert len(p) > 0

        # Response should only contain assistant tokens -> all-1 masks
        for mask in result["loss_masks"]:
            assert all(m == 1 for m in mask)

    def test_trajectory_ids(self) -> None:
        ep = _make_episode(n_steps=2)
        result = self.exporter.export([ep])

        for tid in result["trajectory_ids"]:
            assert isinstance(tid, TrajectoryID)
            assert tid.instance_id == "task-xyz"

    def test_trajectory_ids_distinct_per_episode(self) -> None:
        ep1 = _make_episode(n_steps=1, reward=0.5)
        ep2 = _make_episode(n_steps=1, reward=0.7)
        result = self.exporter.export([ep1, ep2])

        # Two episodes of the same task must have different repetition_ids
        assert result["trajectory_ids"][0].repetition_id == 0
        assert result["trajectory_ids"][1].repetition_id == 1

    def test_multiple_episodes(self) -> None:
        ep1 = _make_episode(n_steps=2, reward=0.5)
        ep2 = _make_episode(n_steps=3, reward=0.7)
        result = self.exporter.export([ep1, ep2])

        # 2 + 3 = 5 transitions total
        assert len(result["prompt_token_ids"]) == 5
        assert len(result["rewards"]) == 5

    def test_tool_call_tokens_in_response(self) -> None:
        """Assistant tool-call messages must produce non-empty response tokens."""
        from lfx.core.episode import ToolCall

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Search for X"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="search",
                        arguments='{"q": "X"}',
                    )
                ],
            ),
            Message(role="tool", content="Found X", name="search", tool_call_id="tc-1"),
            Message(role="assistant", content="Here is X."),
        ]
        ep = Episode(
            id="ep-tc",
            state_id="s",
            task_id="task-tc",
            bench="test",
            messages=messages,
            step_boundaries=[1],  # user msg at index 1
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=10.0)],
            summary=EpisodeSummary(total_reward=1.0),
        )
        result = self.exporter.export([ep])

        # Response tokens must be non-empty (tool call serialized)
        assert len(result["response_ids"][0]) > 0
        # Loss mask: assistant tokens = 1, tool result = 0
        mask = result["loss_masks"][0]
        assert 1 in mask
        assert 0 in mask

    def test_prompt_grows_with_steps(self) -> None:
        ep = _make_episode(n_steps=3)
        result = self.exporter.export([ep])

        # Each subsequent step should have a longer prompt (prefix-sharing)
        prompt_lengths = [len(p) for p in result["prompt_token_ids"]]
        for i in range(1, len(prompt_lengths)):
            assert prompt_lengths[i] > prompt_lengths[i - 1]

    def test_export_one_delegates(self) -> None:
        ep = _make_episode()
        r1 = self.exporter.export_one(ep)
        r2 = self.exporter.export([ep])
        assert r1["rewards"] == r2["rewards"]


class TestTrajectoryID:
    def test_to_string(self) -> None:
        tid = TrajectoryID(instance_id="task-1", repetition_id=2)
        assert tid.to_string() == "task-1_2"
