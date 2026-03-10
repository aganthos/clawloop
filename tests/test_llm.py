"""Tests for lfx.llm — LLM client abstraction."""

from lfx.llm import LiteLLMClient, MockLLMClient


class TestMockLLMClient:
    def test_returns_canned_response(self) -> None:
        client = MockLLMClient(responses=["hello world"])
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    def test_cycles_responses(self) -> None:
        client = MockLLMClient(responses=["a", "b"])
        assert client.complete([]) == "a"
        assert client.complete([]) == "b"
        assert client.complete([]) == "a"  # wraps around

    def test_records_calls(self) -> None:
        client = MockLLMClient(responses=["ok"])
        msgs = [{"role": "user", "content": "test"}]
        client.complete(msgs, temperature=0.5)
        assert len(client.call_log) == 1
        recorded_msgs, recorded_kwargs = client.call_log[0]
        assert recorded_msgs == msgs
        assert recorded_kwargs["temperature"] == 0.5

    def test_default_response(self) -> None:
        client = MockLLMClient()
        assert client.complete([]) == "mock response"


class TestLiteLLMClient:
    def test_init_stores_config(self) -> None:
        client = LiteLLMClient(model="gpt-4o")
        assert client.model == "gpt-4o"
        assert client.api_key is None
        assert client.default_kwargs == {}

    def test_init_with_api_key(self) -> None:
        client = LiteLLMClient(model="gpt-4o", api_key="sk-test")
        assert client.api_key == "sk-test"

    def test_init_with_kwargs(self) -> None:
        client = LiteLLMClient(
            model="gpt-4o", temperature=0.7, max_tokens=100
        )
        assert client.default_kwargs == {"temperature": 0.7, "max_tokens": 100}
