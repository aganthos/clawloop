# tests/test_entropic_purple.py
"""Tests for Entropic CRMArenaPro A2A purple agent."""

import json
from unittest.mock import MagicMock, patch

from clawloop.adapters._entropic_purple import EntropicPurpleAgent
from clawloop.layers.harness import Harness


def _make_harness(prompt: str = "") -> Harness:
    h = Harness()
    if prompt:
        h.system_prompts["entropic"] = prompt
    return h


class TestToolSchemaConversion:
    """_convert_tools_to_openai converts raw tool format to OpenAI."""

    def test_basic_conversion(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tools = [
            {"name": "query_leads", "description": "Query lead records",
             "parameters": {"type": "object", "properties": {}}}
        ]
        result = agent._convert_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "query_leads"

    def test_already_openai_format(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tools = [{"type": "function", "function": {"name": "fn", "description": "", "parameters": {}}}]
        result = agent._convert_tools_to_openai(tools)
        assert result == tools

    def test_missing_description(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tools = [{"name": "fn", "parameters": {}}]
        result = agent._convert_tools_to_openai(tools)
        assert result[0]["function"]["description"] == ""


class TestFormatA2AResponse:
    """_format_a2a_response builds correct A2A message parts."""

    def test_text_only_response(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        msg = MagicMock()
        msg.content = "42 leads found"
        msg.tool_calls = None

        result = agent._format_a2a_response(msg)
        parts = result["parts"]
        assert parts[0]["kind"] == "text"
        assert parts[0]["text"] == "42 leads found"
        assert len(parts) == 1

    def test_tool_call_response(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "query_leads"
        tc.function.arguments = '{"status": "active"}'

        msg = MagicMock()
        msg.content = "Let me check"
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        parts = result["parts"]
        assert len(parts) == 2
        assert parts[1]["kind"] == "data"
        assert parts[1]["data"]["tool_calls"][0]["tool_name"] == "query_leads"

    def test_malformed_arguments_fallback(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "fn"
        tc.function.arguments = "not valid json {"

        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        tool_call = result["parts"][1]["data"]["tool_calls"][0]
        assert tool_call["arguments"] == {"raw": "not valid json {"}

    def test_none_arguments(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.function.name = "fn"
        tc.function.arguments = None

        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]

        result = agent._format_a2a_response(msg)
        tool_call = result["parts"][1]["data"]["tool_calls"][0]
        assert tool_call["arguments"] == {}


class TestNormalizeAssistantMsg:
    """_normalize_assistant_msg produces stable dict."""

    def test_text_only(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        msg = MagicMock()
        msg.content = "Hello"
        msg.tool_calls = None

        result = agent._normalize_assistant_msg(msg)
        assert result == {"role": "assistant", "content": "Hello"}
        assert "tool_calls" not in result

    def test_with_tool_calls(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "fn"
        tc.function.arguments = '{"a": 1}'

        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]

        result = agent._normalize_assistant_msg(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"


class TestApiBasePassthrough:
    """api_base and api_key are stored for litellm calls."""

    def test_api_base_stored(self):
        agent = EntropicPurpleAgent(
            model="test", harness=_make_harness(),
            api_base="http://localhost:9999", api_key="sk-test",
        )
        assert agent.api_base == "http://localhost:9999"
        assert agent.api_key == "sk-test"

    def test_defaults_to_none(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        assert agent.api_base is None
        assert agent.api_key is None


class TestHarnessInjection:
    """Harness system prompt is injected as system message."""

    def test_harness_used_as_system(self):
        harness = _make_harness("## PLAYBOOK\nBe accurate with CRM data.")
        agent = EntropicPurpleAgent(model="test", harness=harness)

        prompt = agent.harness.system_prompt("entropic")
        assert "PLAYBOOK" in prompt
        assert "CRM data" in prompt

    def test_no_harness_uses_default(self):
        agent = EntropicPurpleAgent(model="test", harness=_make_harness())
        prompt = agent.harness.system_prompt("entropic")
        assert prompt == ""


class TestReconcileToolCallId:
    """_reconcile_tool_call_id rewrites assistant tool_call ids."""

    def test_rewrites_matching_tool_name(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_1", "type": "function",
                     "function": {"name": "query_leads", "arguments": "{}"}},
                ],
            },
        ]
        EntropicPurpleAgent._reconcile_tool_call_id(messages, "query_leads", "green_99")
        assert messages[2]["tool_calls"][0]["id"] == "green_99"

    def test_no_match_leaves_unchanged(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_1", "type": "function",
                     "function": {"name": "other_tool", "arguments": "{}"}},
                ],
            },
        ]
        EntropicPurpleAgent._reconcile_tool_call_id(messages, "query_leads", "green_99")
        assert messages[0]["tool_calls"][0]["id"] == "llm_1"

    def test_empty_messages_no_crash(self):
        EntropicPurpleAgent._reconcile_tool_call_id([], "fn", "id")

    def test_duplicate_tool_names(self):
        """Two calls to same tool get different green IDs."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "llm_1", "type": "function",
                     "function": {"name": "query", "arguments": '{"q":"a"}'}},
                    {"id": "llm_2", "type": "function",
                     "function": {"name": "query", "arguments": '{"q":"b"}'}},
                ],
            },
        ]
        EntropicPurpleAgent._reconcile_tool_call_id(messages, "query", "green_1")
        assert messages[0]["tool_calls"][0]["id"] == "green_1"
        assert messages[0]["tool_calls"][1]["id"] == "llm_2"

        messages.append({"role": "tool", "tool_call_id": "green_1", "content": "r1"})
        EntropicPurpleAgent._reconcile_tool_call_id(messages, "query", "green_2")
        assert messages[0]["tool_calls"][1]["id"] == "green_2"
