"""Tests for Reflector JSON trace format, sanitization, and robust parsing."""

import json

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reflector import Reflector, _sanitize_obj, _sanitize_str
from clawloop.learning_layers.harness import Playbook, PlaybookEntry


def _make_episode(
    ep_id: str = "ep-1",
    task_id: str = "t-1",
    bench: str = "test",
    content: str = "Hello",
) -> Episode:
    return Episode(
        id=ep_id,
        state_id="s0",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="user", content=content),
            Message(role="assistant", content="World"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=0.5, done=True, timing_ms=10.0)],
        summary=EpisodeSummary(total_reward=0.5),
    )


class _FakeLLM:
    """Fake LLM that returns a canned response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict] | None = None

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.last_messages = messages
        return self._response


class TestReflectorJsonTraceFormat:
    def test_prompt_contains_json_block(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        ep = _make_episode()
        r.reflect([ep], Playbook())

        user_prompt = llm.last_messages[1]["content"]
        assert "```json" in user_prompt
        assert "```" in user_prompt

        # Extract JSON from fenced block
        import re

        match = re.search(r"```json\s*(.*?)\s*```", user_prompt, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "ep-1"
        assert data[0]["task_id"] == "t-1"
        assert data[0]["bench"] == "test"

    def test_trace_messages_only_role_and_content(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        ep = _make_episode()
        r.reflect([ep], Playbook())

        user_prompt = llm.last_messages[1]["content"]
        import re

        match = re.search(r"```json\s*(.*?)\s*```", user_prompt, re.DOTALL)
        data = json.loads(match.group(1))
        for msg in data[0]["messages"]:
            assert set(msg.keys()) == {"role", "content"}


class TestSanitization:
    def test_sanitize_str_strips_null_bytes(self) -> None:
        assert _sanitize_str("hello\x00world") == "helloworld"

    def test_sanitize_str_no_change(self) -> None:
        assert _sanitize_str("clean text") == "clean text"

    def test_sanitize_obj_recursive(self) -> None:
        obj = {"key\x00": ["val\x00ue", {"nested\x00": "data\x00"}]}
        result = _sanitize_obj(obj)
        assert result == {"key": ["value", {"nested": "data"}]}

    def test_null_bytes_stripped_in_episode_traces(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        ep = _make_episode(content="Hello\x00World")
        r.reflect([ep], Playbook())

        user_prompt = llm.last_messages[1]["content"]
        assert "\x00" not in user_prompt

    def test_null_bytes_stripped_in_playbook(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        pb = Playbook(entries=[PlaybookEntry(id="e1", content="tip\x00injected")])
        r.reflect([_make_episode()], pb)

        user_prompt = llm.last_messages[1]["content"]
        assert "\x00" not in user_prompt

    def test_null_bytes_stripped_in_sibling_context(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        r.reflect(
            [_make_episode()],
            Playbook(),
            sibling_context="context\x00injected",
        )

        user_prompt = llm.last_messages[1]["content"]
        assert "\x00" not in user_prompt


class TestParseResponseRobustness:
    def test_non_dict_items_skipped(self) -> None:
        response = json.dumps(
            [
                {"action": "add", "content": "good insight", "tags": []},
                "not a dict",
                42,
                None,
                {"action": "add", "content": "another good one", "tags": []},
            ]
        )
        llm = _FakeLLM(response)
        r = Reflector(client=llm)
        insights = r.reflect([_make_episode()], Playbook())
        assert len(insights) == 2

    def test_empty_response(self) -> None:
        llm = _FakeLLM("[]")
        r = Reflector(client=llm)
        insights = r.reflect([_make_episode()], Playbook())
        assert insights == []

    def test_invalid_json(self) -> None:
        llm = _FakeLLM("not json at all")
        r = Reflector(client=llm)
        insights = r.reflect([_make_episode()], Playbook())
        assert insights == []

    def test_json_with_fencing(self) -> None:
        inner = json.dumps([{"action": "add", "content": "tip", "tags": ["perf"]}])
        response = f"```json\n{inner}\n```"
        llm = _FakeLLM(response)
        r = Reflector(client=llm)
        insights = r.reflect([_make_episode()], Playbook())
        assert len(insights) == 1
        assert insights[0].content == "tip"
