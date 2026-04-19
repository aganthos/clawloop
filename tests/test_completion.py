"""Tests for CompletionResult — rich LLM response type."""

from clawloop.completion import CompletionResult
from clawloop.core.episode import TokenLogProb, TokenUsage, ToolCall


class TestCompletionResultBasic:
    def test_text_only(self) -> None:
        r = CompletionResult(text="Hello world")
        assert r.text == "Hello world"
        assert r.model is None
        assert r.tool_calls is None
        assert r.usage is None
        assert r.logprobs is None
        assert r.latency_ms is None

    def test_full_construction(self) -> None:
        r = CompletionResult(
            text="result",
            model="gpt-4o",
            tool_calls=[ToolCall(id="tc-1", name="search", arguments='{"q":"x"}')],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            logprobs=[TokenLogProb(token="result", logprob=-0.2)],
            latency_ms=123.4,
        )
        assert r.model == "gpt-4o"
        assert len(r.tool_calls) == 1
        assert r.usage.total_tokens == 15
        assert r.logprobs[0].logprob == -0.2
        assert r.latency_ms == 123.4


class TestCompletionResultStringCompat:
    """CompletionResult must behave like a string in common contexts."""

    def test_str_returns_text(self) -> None:
        r = CompletionResult(text="hello")
        assert str(r) == "hello"

    def test_eq_with_string(self) -> None:
        r = CompletionResult(text="hello")
        assert r == "hello"
        assert not (r == "world")

    def test_eq_with_completion_result(self) -> None:
        r1 = CompletionResult(text="hello")
        r2 = CompletionResult(text="hello", model="gpt-4")
        assert r1 == r2  # equality is on text only

    def test_eq_with_other_type(self) -> None:
        r = CompletionResult(text="hello")
        assert r != 42

    def test_hash_matches_text(self) -> None:
        r = CompletionResult(text="hello")
        assert hash(r) == hash("hello")

    def test_bool_truthy(self) -> None:
        assert bool(CompletionResult(text="hello"))
        assert not bool(CompletionResult(text=""))

    def test_contains(self) -> None:
        r = CompletionResult(text="hello world")
        assert "world" in r

    def test_len(self) -> None:
        r = CompletionResult(text="hello")
        assert len(r) == 5

    def test_repr_shows_text(self) -> None:
        r = CompletionResult(text="hi")
        assert "hi" in repr(r)

    def test_format_string(self) -> None:
        r = CompletionResult(text="answer")
        assert f"The {r}" == "The answer"

    def test_add_string(self) -> None:
        r = CompletionResult(text="hello")
        assert r + " world" == "hello world"

    def test_radd_string(self) -> None:
        r = CompletionResult(text="world")
        assert "hello " + r == "hello world"
