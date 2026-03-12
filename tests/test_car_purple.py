# tests/test_car_purple.py
"""Tests for CAR-bench A2A purple agent."""

import json
from unittest.mock import MagicMock, patch

from lfx.adapters._car_purple import CarPurpleAgent
from lfx.layers.harness import Harness


def _make_harness(prompt: str = "") -> Harness:
    h = Harness()
    if prompt:
        h.system_prompts["car"] = prompt
    return h


class TestParseFirstMessage:
    """_parse_first_message extracts system prompt and user text."""

    def test_standard_format(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        system, user = agent._parse_first_message(
            "System: You are a car assistant.\n\nUser: Book a service."
        )
        assert system == "You are a car assistant."
        assert user == "Book a service."

    def test_no_system_prefix(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        system, user = agent._parse_first_message("Just a user message")
        assert system == ""
        assert user == "Just a user message"

    def test_multiline_system(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        raw = "System: Line one.\nLine two.\nLine three.\n\nUser: Hello"
        system, user = agent._parse_first_message(raw)
        assert "Line one" in system
        assert "Line three" in system
        assert user == "Hello"


class TestToolSchemaConversion:
    """_convert_tools_to_openai converts CAR tool format to OpenAI."""

    def test_basic_conversion(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        car_tools = [
            {"name": "get_location", "description": "Get current location",
             "parameters": {"type": "object", "properties": {}}}
        ]
        result = agent._convert_tools_to_openai(car_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_location"
        assert result[0]["function"]["description"] == "Get current location"

    def test_missing_description(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        car_tools = [{"name": "fn", "parameters": {}}]
        result = agent._convert_tools_to_openai(car_tools)
        assert result[0]["function"]["description"] == ""


class TestFormatA2AResponse:
    """_format_a2a_response builds correct A2A message parts."""

    def test_text_only_response(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        msg = MagicMock()
        msg.content = "Hello there"
        msg.tool_calls = None

        result = agent._format_a2a_response(msg)
        parts = result["message"]["parts"]
        assert parts[0]["kind"] == "text"
        assert parts[0]["text"] == "Hello there"
        assert len(parts) == 1  # no data part

    def test_tool_call_response(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "get_location"
        tc.function.arguments = '{"city": "Zurich"}'

        msg = MagicMock()
        msg.content = "Let me check"
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        parts = result["message"]["parts"]
        assert len(parts) == 2
        assert parts[1]["kind"] == "data"
        assert parts[1]["data"]["tool_calls"][0]["tool_name"] == "get_location"
        assert parts[1]["data"]["tool_calls"][0]["arguments"] == {"city": "Zurich"}

    def test_malformed_arguments_fallback(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "fn"
        tc.function.arguments = "not valid json {"

        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        tool_call = result["message"]["parts"][1]["data"]["tool_calls"][0]
        assert tool_call["arguments"] == {"raw": "not valid json {"}

    def test_none_arguments(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "fn"
        tc.function.arguments = None

        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        tool_call = result["message"]["parts"][1]["data"]["tool_calls"][0]
        assert tool_call["arguments"] == {}


class TestNormalizeAssistantMsg:
    """_normalize_assistant_msg produces a stable dict for conversation history."""

    def test_text_only(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        msg = MagicMock()
        msg.content = "Hello"
        msg.tool_calls = None

        result = agent._normalize_assistant_msg(msg)
        assert result == {"role": "assistant", "content": "Hello"}
        assert "tool_calls" not in result

    def test_with_tool_calls(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "fn"
        tc.function.arguments = '{"a": 1}'

        msg = MagicMock()
        msg.content = "Calling fn"
        msg.tool_calls = [tc]

        result = agent._normalize_assistant_msg(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"


class TestHarnessInjection:
    """Harness system prompt is prepended to CAR's system prompt."""

    def test_harness_prepended(self):
        harness = _make_harness("## PLAYBOOK\nAlways be polite.")
        agent = CarPurpleAgent(model="test", harness=harness)

        # Simulate first message handling
        system, user = agent._parse_first_message(
            "System: You are a car assistant.\n\nUser: Hi"
        )
        harness_prompt = agent.harness.system_prompt("car")
        combined = f"{harness_prompt}\n\n{system}"

        assert "PLAYBOOK" in combined
        assert "car assistant" in combined

    def test_no_harness_no_prefix(self):
        agent = CarPurpleAgent(model="test", harness=_make_harness())
        system, _ = agent._parse_first_message(
            "System: Original prompt.\n\nUser: Hi"
        )
        harness_prompt = agent.harness.system_prompt("car")
        assert harness_prompt == ""


class TestReconcileToolCallId:
    """_reconcile_tool_call_id rewrites assistant tool_call ids to match green's."""

    def test_rewrites_matching_tool_name(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_call_1", "type": "function",
                     "function": {"name": "get_location", "arguments": "{}"}},
                ],
            },
        ]
        CarPurpleAgent._reconcile_tool_call_id(messages, "get_location", "green_id_99")
        assert messages[2]["tool_calls"][0]["id"] == "green_id_99"

    def test_no_match_leaves_unchanged(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_call_1", "type": "function",
                     "function": {"name": "other_tool", "arguments": "{}"}},
                ],
            },
        ]
        CarPurpleAgent._reconcile_tool_call_id(messages, "get_location", "green_id_99")
        assert messages[0]["tool_calls"][0]["id"] == "llm_call_1"

    def test_multiple_tool_calls_rewrites_correct_one(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_a", "type": "function",
                     "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "llm_b", "type": "function",
                     "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            },
        ]
        CarPurpleAgent._reconcile_tool_call_id(messages, "tool_b", "green_b")
        assert messages[0]["tool_calls"][0]["id"] == "llm_a"  # unchanged
        assert messages[0]["tool_calls"][1]["id"] == "green_b"  # rewritten

    def test_empty_messages_no_crash(self):
        CarPurpleAgent._reconcile_tool_call_id([], "fn", "id")
