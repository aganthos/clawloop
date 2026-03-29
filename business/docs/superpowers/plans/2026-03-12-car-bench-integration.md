# CAR-bench Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up CAR-bench as the first real benchmark for the lfx harness improvement loop, with a custom A2A purple agent that injects harness state into LLM calls.

**Architecture:** CARAdapter runs `agentbeats-run` per iteration with a generated scenario.toml. Our thin A2A purple agent server injects the harness system prompt + playbook into LLM calls via litellm. Results parsed from CAR's results.json and mapped to lfx Episodes with per-metric RewardSignals.

**Tech Stack:** litellm, uvicorn/starlette (A2A server), httpx (health checks), subprocess (green agent), CAR-bench git submodule

**Spec:** `docs/specs/2026-03-12-car-bench-integration.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `lfx/adapters/_car_purple.py` | Create | A2A purple agent server: message handling, harness injection, LLM calls |
| `lfx/adapters/car.py` | Rewrite | CARAdapter: scenario generation, agentbeats-run orchestration, results parsing |
| `lfx/adapters/_car_rewards.py` | Create | `map_car_scores()` reward mapping + constants |
| `lfx/adapters/base.py` | Modify | Add optional `run_batch()` to EnvAdapter |
| `lfx/core/loop.py` | Modify | Use `run_batch()` when available |
| `lfx/cli.py` | Modify | Add CAR-specific CLI args to `lfx run` |
| `tests/test_car_rewards.py` | Create | Unit tests for reward mapping |
| `tests/test_car_purple.py` | Create | Unit tests for A2A message parsing + formatting |
| `tests/test_car_adapter.py` | Create | Integration tests for CARAdapter with mock agentbeats-run |
| `benchmarks/car-bench/` | Create | Git submodule |

---

## Chunk 1: Submodule + Reward Mapping

### Task 1: Add CAR-bench git submodule

**Files:**
- Create: `benchmarks/car-bench/` (submodule)
- Modify: `.gitmodules`
- Modify: `.gitignore`

- [ ] **Step 1: Create benchmarks directory and add submodule**

```bash
cd /Users/robertmueller/Desktop/aganthos
mkdir -p benchmarks
git submodule add https://github.com/CAR-bench/car-bench-agentbeats.git benchmarks/car-bench
```

- [ ] **Step 2: Add gitignore entries for benchmark artifacts**

Append to `.gitignore`:
```
benchmarks/*/output/
benchmarks/*/.venv/
benchmarks/*/.env
```

- [ ] **Step 3: Commit**

```bash
git add .gitmodules benchmarks/car-bench .gitignore
git commit -m "feat: add car-bench as submodule under benchmarks/"
```

---

### Task 2: Reward mapping — tests first

**Files:**
- Create: `lfx/adapters/_car_rewards.py`
- Create: `tests/test_car_rewards.py`

- [ ] **Step 1: Write failing tests for `map_car_scores()`**

```python
# tests/test_car_rewards.py
"""Tests for CAR-bench reward mapping."""

from lfx.adapters._car_rewards import map_car_scores, DEFAULT_CAR_WEIGHTS


class TestMapCarScores:
    """map_car_scores converts CAR metrics to lfx RewardSignals."""

    def test_perfect_scores(self):
        """All metrics 1.0 → outcome signal near +1.0."""
        reward_info = {k: 1.0 for k in DEFAULT_CAR_WEIGHTS}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert signals["outcome"].value == 1.0
        assert signals["outcome"].confidence == 1.0
        for name in DEFAULT_CAR_WEIGHTS:
            assert signals[name].value == 1.0
            assert signals[name].confidence == 1.0

    def test_zero_scores(self):
        """All metrics 0.0 → outcome signal -1.0."""
        reward_info = {k: 0.0 for k in DEFAULT_CAR_WEIGHTS}
        signals, breakdown = map_car_scores(reward_info, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        for name in DEFAULT_CAR_WEIGHTS:
            assert signals[name].value == -1.0

    def test_mixed_scores(self):
        """Mixed metrics produce weighted composite."""
        reward_info = {
            "r_actions_final": 1.0,
            "r_actions_intermediate": 0.0,
            "r_tool_subset": 1.0,
            "r_tool_execution_errors": 1.0,
            "r_policy_errors": 0.0,
            "r_user_end_conversation": 1.0,
        }
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        # task_reward drives outcome
        assert signals["outcome"].value == 1.0
        assert signals["r_actions_final"].value == 1.0
        assert signals["r_actions_intermediate"].value == -1.0

    def test_missing_metrics(self):
        """Missing metrics default to 0 with warning, not crash."""
        signals, breakdown = map_car_scores({}, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        assert "r_actions_final" not in signals  # not created if missing

    def test_unknown_metrics_stored_in_breakdown(self):
        """Extra CAR metrics are stored but not mapped to signals."""
        reward_info = {"r_actions_final": 1.0, "r_new_metric": 0.5}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert "r_new_metric" not in signals
        assert breakdown["r_new_metric"] == 0.5

    def test_out_of_range_clamped(self):
        """Values outside [0,1] are clamped."""
        reward_info = {"r_actions_final": 1.5, "r_policy_errors": -0.3}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert signals["r_actions_final"].value == 1.0  # clamped 1.5→1.0→mapped to +1
        assert signals["r_policy_errors"].value == -1.0  # clamped -0.3→0.0→mapped to -1

    def test_non_numeric_metric(self):
        """Non-numeric values are skipped with warning."""
        reward_info = {"r_actions_final": "bad", "r_policy_errors": 1.0}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert "r_actions_final" not in signals
        assert signals["r_policy_errors"].value == 1.0

    def test_custom_weights(self):
        """Custom weights override defaults."""
        custom = {"r_actions_final": 1.0}
        reward_info = {"r_actions_final": 1.0, "r_policy_errors": 0.0}
        signals, breakdown = map_car_scores(
            reward_info, task_reward=1.0, weights=custom
        )

        # Only r_actions_final mapped (custom weights has only that)
        assert "r_actions_final" in signals
        assert "r_policy_errors" not in signals

    def test_non_binary_confidence(self):
        """Non-binary values (not 0.0 or 1.0) get confidence 0.8."""
        reward_info = {"r_actions_final": 0.5}
        signals, _ = map_car_scores(reward_info, task_reward=0.5)

        assert signals["r_actions_final"].confidence == 0.8

    def test_breakdown_contains_all_known(self):
        """Breakdown includes all validated metrics."""
        reward_info = {k: 1.0 for k in DEFAULT_CAR_WEIGHTS}
        _, breakdown = map_car_scores(reward_info, task_reward=1.0)

        for name in DEFAULT_CAR_WEIGHTS:
            assert name in breakdown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_rewards.py -v`
Expected: ImportError — `_car_rewards` module doesn't exist yet.

- [ ] **Step 3: Implement `_car_rewards.py`**

```python
# lfx/adapters/_car_rewards.py
"""CAR-bench reward mapping — converts CAR metrics to lfx RewardSignals."""

from __future__ import annotations

import logging
from typing import Any

from lfx.core.reward import RewardSignal

log = logging.getLogger(__name__)

DEFAULT_CAR_WEIGHTS: dict[str, float] = {
    "r_actions_final": 0.30,
    "r_actions_intermediate": 0.20,
    "r_tool_subset": 0.15,
    "r_tool_execution_errors": 0.15,
    "r_policy_errors": 0.10,
    "r_user_end_conversation": 0.10,
}


def map_car_scores(
    reward_info: dict[str, Any],
    task_reward: float,
    weights: dict[str, float] = DEFAULT_CAR_WEIGHTS,
) -> tuple[dict[str, RewardSignal], dict[str, Any]]:
    """Map CAR-bench metrics to lfx RewardSignals.

    Parameters
    ----------
    reward_info:
        Per-metric dict from CAR's detailed_results (e.g. r_actions_final: 0/1).
    task_reward:
        Binary task reward (0.0 or 1.0) from CAR's top-level scoring.
    weights:
        Metric weights for composite score. Defaults to DEFAULT_CAR_WEIGHTS.

    Returns
    -------
    tuple of (signals dict, breakdown dict)
        signals: named RewardSignals for the learning loop
        breakdown: raw validated values for score_breakdown
    """
    signals: dict[str, RewardSignal] = {}
    breakdown: dict[str, Any] = {}

    # Primary outcome from task_reward (binary, ground truth)
    clamped_reward = max(0.0, min(1.0, float(task_reward)))
    signals["outcome"] = RewardSignal(
        name="outcome",
        value=clamped_reward * 2.0 - 1.0,
        confidence=1.0,
    )

    # Per-metric signals
    for name in weights:
        raw = reward_info.get(name)
        if raw is None:
            log.debug("Missing CAR metric %s", name)
            continue
        if not isinstance(raw, (int, float)):
            log.warning("Non-numeric CAR metric %s=%r, skipping", name, raw)
            continue
        val = max(0.0, min(1.0, float(raw)))
        conf = 1.0 if val in (0.0, 1.0) else 0.8
        signals[name] = RewardSignal(name=name, value=val * 2.0 - 1.0, confidence=conf)
        breakdown[name] = val

    # Unknown metrics: store but don't map
    for k, v in reward_info.items():
        if k not in weights:
            breakdown[k] = v

    return signals, breakdown
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_rewards.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lfx/adapters/_car_rewards.py tests/test_car_rewards.py
git commit -m "feat: add CAR-bench reward mapping with tests"
```

---

## Chunk 2: A2A Purple Agent

### Task 3: Purple agent message parsing — tests first

**Files:**
- Create: `lfx/adapters/_car_purple.py`
- Create: `tests/test_car_purple.py`

- [ ] **Step 1: Write failing tests for message parsing and A2A formatting**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_purple.py -v`
Expected: ImportError — `_car_purple` doesn't exist yet.

- [ ] **Step 3: Implement `_car_purple.py`**

```python
# lfx/adapters/_car_purple.py
"""A2A purple agent server for CAR-bench with lfx harness injection."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any
from uuid import uuid4

import litellm
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lfx.layers.harness import Harness

log = logging.getLogger(__name__)


class CarPurpleAgent:
    """A2A-compliant purple agent that injects lfx harness state into LLM calls."""

    def __init__(self, model: str, harness: Harness, bench: str = "car"):
        self.model = model
        self.harness = harness
        self.bench = bench
        self._sessions: dict[str, list[dict]] = {}
        self._tool_cache: dict[str, list[dict]] = {}
        self._captured: dict[str, list[dict]] = {}

    def update_harness(self, harness: Harness) -> None:
        self.harness = harness

    def clear_all_sessions(self) -> None:
        self._sessions.clear()
        self._tool_cache.clear()
        self._captured.clear()

    # -- Message parsing --

    @staticmethod
    def _parse_first_message(raw_text: str) -> tuple[str, str]:
        """Parse 'System: ...\\n\\nUser: ...' format from green agent."""
        if "System:" in raw_text and "\n\nUser:" in raw_text:
            parts = raw_text.split("\n\nUser:", 1)
            system = parts[0].replace("System:", "", 1).strip()
            user = parts[1].strip()
            return system, user
        return "", raw_text

    @staticmethod
    def _convert_tools_to_openai(car_tools: list[dict]) -> list[dict]:
        """Convert CAR tool schemas to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
            for t in car_tools
        ]

    @staticmethod
    def _normalize_assistant_msg(litellm_msg: Any) -> dict:
        """Normalize litellm response to stable internal format."""
        normalized: dict[str, Any] = {
            "role": "assistant",
            "content": litellm_msg.content or "",
        }
        if litellm_msg.tool_calls:
            normalized["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in litellm_msg.tool_calls
            ]
        return normalized

    def _format_a2a_response(self, assistant_msg: Any) -> dict:
        """Format LLM response as A2A result body."""
        parts: list[dict] = [{"kind": "text", "text": assistant_msg.content or ""}]

        if assistant_msg.tool_calls:
            tool_calls = []
            for tc in assistant_msg.tool_calls:
                args = tc.function.arguments
                if args is None:
                    args = {}
                elif isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        log.warning("Malformed tool args for %s", tc.function.name)
                        args = {"raw": args}
                tool_calls.append(
                    {"tool_name": tc.function.name, "arguments": args}
                )
            parts.append({"kind": "data", "data": {"tool_calls": tool_calls}})

        return {
            "message": {
                "messageId": uuid4().hex,
                "role": "agent",
                "parts": parts,
            }
        }

    # -- Core message handling --

    def handle_message_sync(self, jsonrpc_request: dict) -> dict:
        """Handle one message/send request (sync — litellm.completion is sync)."""
        params = jsonrpc_request["params"]
        msg = params["message"]
        context_id = params.get("contextId", "default")

        text_parts = [p["text"] for p in msg["parts"] if p.get("kind") == "text"]
        data_parts = [p["data"] for p in msg["parts"] if p.get("kind") == "data"]

        # Initialize session
        if context_id not in self._sessions:
            self._sessions[context_id] = []
            self._captured[context_id] = []

        messages = self._sessions[context_id]

        if not messages:
            # First message: extract system prompt + tools
            raw_text = text_parts[0] if text_parts else ""
            system_prompt, user_text = self._parse_first_message(raw_text)

            # HARNESS INJECTION
            harness_prompt = self.harness.system_prompt(self.bench)
            if harness_prompt:
                system_prompt = f"{harness_prompt}\n\n{system_prompt}"

            messages.append({"role": "system", "content": system_prompt})
            if user_text:
                messages.append({"role": "user", "content": user_text})

            # Cache tools
            for d in data_parts:
                if "tools" in d:
                    self._tool_cache[context_id] = self._convert_tools_to_openai(
                        d["tools"]
                    )
        else:
            # Subsequent: tool results and/or user text
            for d in data_parts:
                if "tool_results" in d:
                    for tr in d["tool_results"]:
                        # Reconcile tool_call_ids: rewrite last assistant msg's
                        # tool_call ids to match green's generated ids
                        green_id = tr["tool_call_id"]
                        tool_name = tr.get("tool_name", "")
                        self._reconcile_tool_call_id(messages, tool_name, green_id)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": green_id,
                            "content": tr["content"],
                        })
            for text in text_parts:
                if text.strip():
                    messages.append({"role": "user", "content": text})

        # Call LLM
        tools = self._tool_cache.get(context_id)
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if tools:
            completion_kwargs["tools"] = tools

        response = litellm.completion(**completion_kwargs)
        assistant_msg = response.choices[0].message

        # Normalize and store
        normalized = self._normalize_assistant_msg(assistant_msg)
        messages.append(normalized)
        self._captured[context_id].append(normalized)

        return self._format_a2a_response(assistant_msg)

    @staticmethod
    def _reconcile_tool_call_id(
        messages: list[dict], tool_name: str, green_id: str
    ) -> None:
        """Rewrite last assistant message's tool_call id to match green's id.

        Green generates its own tool_call_ids. The LLM needs matching ids between
        assistant tool_calls and tool-role messages. We rewrite the assistant msg's
        id to match what green sent back.
        """
        # Walk backwards to find the last assistant message with tool_calls
        for msg in reversed(messages):
            if msg.get("role") != "assistant" or "tool_calls" not in msg:
                continue
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == tool_name and tc["id"] != green_id:
                    tc["id"] = green_id
                    return
            return  # found assistant msg but no matching tool name


def create_app(agent: CarPurpleAgent, port: int = 0) -> Starlette:
    """Create the A2A Starlette app."""

    async def agent_card(request: Request) -> JSONResponse:
        return JSONResponse({
            "name": "lfx-purple-agent",
            "description": "LfX harness-optimized agent under test",
            "url": f"http://127.0.0.1:{port}/",
            "version": "0.1.0",
            "capabilities": {"streaming": False, "pushNotifications": False},
        })

    async def handle_jsonrpc(request: Request) -> JSONResponse:
        body = await request.json()
        if body.get("jsonrpc") != "2.0" or "id" not in body:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32600, "message": "Invalid Request"}}
            )

        method = body.get("method")
        if method != "message/send":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": body["id"],
                 "error": {"code": -32601, "message": f"Method not found: {method}"}}
            )

        # Run sync litellm call in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, agent.handle_message_sync, body
        )

        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})

    return Starlette(
        routes=[
            Route("/.well-known/agent.json", agent_card, methods=["GET"]),
            Route("/", handle_jsonrpc, methods=["POST"]),
        ]
    )


def start_purple_server(
    agent: CarPurpleAgent, host: str = "127.0.0.1", port: int = 0
) -> tuple[threading.Thread, int]:
    """Start the purple agent server in a background thread. Returns (thread, actual_port)."""
    app = create_app(agent, port)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Get actual bound port
    actual_port = port
    if port == 0:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        actual_port = sock.getsockname()[1]
        sock.close()
        config.port = actual_port
        app = create_app(agent, actual_port)
        config = uvicorn.Config(app, host=host, port=actual_port, log_level="warning")
        server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return thread, actual_port
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_purple.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lfx/adapters/_car_purple.py tests/test_car_purple.py
git commit -m "feat: add A2A purple agent server for CAR-bench"
```

---

## Chunk 3: CARAdapter + Loop Integration

### Task 4: Add `run_batch()` to EnvAdapter base

**Files:**
- Modify: `lfx/adapters/base.py`

- [ ] **Step 1: Add optional `run_batch` method to EnvAdapter**

Add after `list_tasks`:
```python
def run_batch(
    self, agent_state: "AgentState", task_ids: list[Any]
) -> list[Episode]:
    """Run a batch of tasks. Default falls back to sequential run_episode."""
    return [self.run_episode(task_id, agent_state) for task_id in task_ids]
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add lfx/adapters/base.py
git commit -m "feat: add optional run_batch to EnvAdapter"
```

---

### Task 5: Update learning loop for batch support

**Files:**
- Modify: `lfx/core/loop.py`

- [ ] **Step 1: Modify episode collection in learning_loop to use run_batch when available**

Replace the episode collection block (lines ~100-113) with:
```python
        # 1. Collect episodes
        if not tasks or n_episodes <= 0:
            episodes: list[Episode] = []
        else:
            if n_episodes <= len(tasks):
                selected_tasks = random.sample(tasks, n_episodes)
            else:
                selected_tasks = random.choices(tasks, k=n_episodes)

            if hasattr(adapter, "run_batch") and callable(
                getattr(adapter, "run_batch", None)
            ):
                episodes = adapter.run_batch(agent_state, selected_tasks)
            else:
                episodes = []
                for task in selected_tasks:
                    ep = adapter.run_episode(task, agent_state)
                    episodes.append(ep)
```

- [ ] **Step 2: Run existing loop tests**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_loop_icl.py tests/test_integration_icl.py -v`
Expected: All pass (existing adapters don't have run_batch, so fallback path used).

- [ ] **Step 3: Commit**

```bash
git add lfx/core/loop.py
git commit -m "feat: support run_batch in learning loop"
```

---

### Task 6: Implement CARAdapter

**Files:**
- Rewrite: `lfx/adapters/car.py`
- Create: `tests/test_car_adapter.py`

- [ ] **Step 1: Write integration test with mock agentbeats-run**

```python
# tests/test_car_adapter.py
"""Integration tests for CARAdapter with mock agentbeats-run."""

import json
import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from lfx.adapters.car import CARAdapter
from lfx.core.loop import AgentState


@pytest.fixture
def mock_car_bench(tmp_path):
    """Create a fake car-bench directory with mock agentbeats-run."""
    bench_dir = tmp_path / "car-bench"
    bench_dir.mkdir()

    # Create a mock agentbeats-run script that writes canned results
    results = {
        "score": 2.0,
        "max_score": 3,
        "pass_rate": 66.7,
        "detailed_results_by_split": {
            "base": [
                {
                    "task_id": "base_0",
                    "reward": 1.0,
                    "trial": 0,
                    "reward_info": {
                        "r_actions_final": 1.0,
                        "r_actions_intermediate": 1.0,
                        "r_tool_subset": 1.0,
                        "r_tool_execution_errors": 1.0,
                        "r_policy_errors": 1.0,
                        "r_user_end_conversation": 1.0,
                    },
                    "trajectory": [
                        {"role": "user", "content": "Book a service"},
                        {"role": "assistant", "content": "Done!"},
                    ],
                    "total_agent_cost": 0.01,
                    "total_llm_latency_ms": 500.0,
                },
                {
                    "task_id": "base_1",
                    "reward": 0.0,
                    "trial": 0,
                    "reward_info": {
                        "r_actions_final": 0.0,
                        "r_actions_intermediate": 0.0,
                        "r_tool_subset": 1.0,
                        "r_tool_execution_errors": 1.0,
                        "r_policy_errors": 0.0,
                        "r_user_end_conversation": 0.0,
                    },
                    "trajectory": [
                        {"role": "user", "content": "Cancel booking"},
                        {"role": "assistant", "content": "I can't do that."},
                    ],
                    "total_agent_cost": 0.02,
                    "total_llm_latency_ms": 800.0,
                },
            ]
        },
    }

    # Mock script writes results to --output path
    script = bench_dir / "mock_agentbeats"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import sys, json
        # Find --output arg
        output_path = None
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
        if output_path:
            with open(output_path, "w") as f:
                json.dump({json.dumps(results)}, f)
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    return bench_dir, script


class TestCARAdapterResultsParsing:
    """CARAdapter parses results.json into Episodes."""

    def test_maps_results_to_episodes(self, mock_car_bench, tmp_path):
        bench_dir, mock_script = mock_car_bench
        output_dir = tmp_path / "output"

        adapter = CARAdapter()
        adapter._car_bench_path = bench_dir
        adapter._output_dir = output_dir
        adapter._task_type = "base"
        adapter._task_split = "test"
        adapter._model = "test-model"
        adapter._iteration_count = 0
        adapter._agentbeats_cmd = str(mock_script)

        # Skip purple server for this test — just test results parsing
        agent_state = AgentState()
        adapter._purple = None  # no purple needed for parse-only test

        # Write canned results directly
        iter_dir = output_dir / "iter_0"
        iter_dir.mkdir(parents=True)
        results_path = iter_dir / "results.json"
        results_path.write_text(json.dumps({
            "detailed_results_by_split": {
                "base": [
                    {
                        "task_id": "base_0",
                        "reward": 1.0,
                        "reward_info": {"r_actions_final": 1.0},
                        "trajectory": [{"role": "user", "content": "Hi"}],
                        "total_agent_cost": 0.01,
                        "total_llm_latency_ms": 500.0,
                    }
                ]
            }
        }))

        episodes = adapter._parse_results(results_path, ["base_0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "car:base_0"
        assert episodes[0].bench == "car"
        assert episodes[0].summary.signals["outcome"].value == 1.0

    def test_missing_task_creates_failed_episode(self, tmp_path):
        adapter = CARAdapter()
        adapter._model = "test"
        adapter._output_dir = tmp_path
        adapter._iteration_count = 0

        iter_dir = tmp_path / "iter_0"
        iter_dir.mkdir(parents=True)
        results_path = iter_dir / "results.json"
        results_path.write_text(json.dumps({
            "detailed_results_by_split": {"base": []}
        }))

        episodes = adapter._parse_results(results_path, ["base_0", "base_1"])
        # Should have 2 failed episodes for missing tasks
        assert len(episodes) == 2
        assert all(ep.summary.signals["outcome"].value == -1.0 for ep in episodes)


class TestScenarioGeneration:
    """_generate_scenario produces valid TOML."""

    def test_generates_valid_scenario(self):
        adapter = CARAdapter()
        adapter._purple_port = 9999
        adapter._task_split = "test"

        scenario = adapter._generate_scenario(["base_0", "base_2"])
        assert "task_split" in scenario
        assert '"test"' in scenario
        assert "base_0" in scenario
        assert "9999" in scenario
        # Unused types zeroed out
        assert "tasks_hallucination_num_tasks = 0" in scenario
        assert "tasks_disambiguation_num_tasks = 0" in scenario

    def test_mixed_task_types(self):
        adapter = CARAdapter()
        adapter._purple_port = 9999
        adapter._task_split = "test"

        scenario = adapter._generate_scenario(["base_0", "hallucination_1"])
        assert "base_0" in scenario
        assert "hallucination_1" in scenario
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_adapter.py -v`
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement CARAdapter**

```python
# lfx/adapters/car.py
"""CAR-bench adapter — orchestrates agentbeats-run with lfx harness injection.

Uses a custom A2A purple agent server that injects harness system prompt +
playbook into LLM calls. Results parsed from CAR's results.json.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lfx.adapters._car_rewards import DEFAULT_CAR_WEIGHTS, map_car_scores
from lfx.adapters.base import EnvAdapter
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta

if TYPE_CHECKING:
    from lfx.adapters._car_purple import CarPurpleAgent
    from lfx.core.loop import AgentState

log = logging.getLogger(__name__)

# All task types in CAR-bench
_ALL_TASK_TYPES = ("base", "hallucination", "disambiguation")

REWARD_METRICS = list(DEFAULT_CAR_WEIGHTS.keys())


class CARAdapter(EnvAdapter):
    """Adapter for CAR-bench. Runs agentbeats-run per learning iteration."""

    CAR_BENCH_TESTED_COMMIT = "TBD"

    def setup(self, config: dict[str, Any]) -> None:
        self._model = config.get("model", "anthropic/claude-haiku-4-5-20251001")
        self._car_bench_path = Path(
            config.get("car_bench_path", "benchmarks/car-bench")
        )
        self._output_dir = Path(
            config.get("output", f"./runs/car/{int(time.time())}")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task_type = config.get("task_type", "base")
        self._task_split = config.get("task_split", "test")
        self._agentbeats_cmd = config.get("agentbeats_cmd", "agentbeats-run")
        self._iteration_count = 0
        self._purple: CarPurpleAgent | None = None
        self._purple_port: int = 0
        self._config = config

    def _start_purple(self) -> None:
        """Start the purple agent server (lazy — called on first run_batch)."""
        if self._purple is not None:
            return
        from lfx.adapters._car_purple import CarPurpleAgent, start_purple_server
        from lfx.layers.harness import Harness

        self._purple = CarPurpleAgent(
            model=self._model, harness=Harness(), bench="car"
        )
        _, self._purple_port = start_purple_server(self._purple)
        log.info("Purple agent started on port %d", self._purple_port)

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        """Run a single task. Delegates to run_batch with one task."""
        episodes = self.run_batch(agent_state, [task])
        return episodes[0] if episodes else self._make_failed_episode(str(task), "empty")

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks via agentbeats-run."""
        self._start_purple()
        assert self._purple is not None

        # Update harness + clear sessions
        self._purple.update_harness(agent_state.harness)
        self._purple.clear_all_sessions()

        # Generate scenario
        str_ids = [str(tid) for tid in task_ids]
        scenario = self._generate_scenario(str_ids)
        iter_dir = self._output_dir / f"iter_{self._iteration_count}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = iter_dir / "scenario.toml"
        scenario_path.write_text(scenario)
        results_path = iter_dir / "results.json"

        # Run agentbeats-run
        try:
            result = subprocess.run(
                [self._agentbeats_cmd, str(scenario_path), "--show-logs",
                 "--output", str(results_path)],
                cwd=str(self._car_bench_path),
                capture_output=True, text=True, timeout=600,
            )
            (iter_dir / "green_agent.log").write_text(
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            if result.returncode != 0:
                log.warning(
                    "agentbeats-run exited %d", result.returncode
                )
        except subprocess.TimeoutExpired:
            log.error("agentbeats-run timed out")
            self._iteration_count += 1
            return [self._make_failed_episode(tid, "timeout") for tid in str_ids]

        # Parse results
        episodes = self._parse_results(results_path, str_ids)

        # Save harness state
        harness_path = iter_dir / "harness_state.json"
        harness_path.write_text(json.dumps(agent_state.harness.to_dict(), indent=2))

        self._iteration_count += 1
        return episodes

    def _parse_results(
        self, results_path: Path, expected_task_ids: list[str]
    ) -> list[Episode]:
        """Parse results.json into Episodes."""
        try:
            raw = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error("Failed to parse results: %s", e)
            return [
                self._make_failed_episode(tid, "parse_error")
                for tid in expected_task_ids
            ]

        episodes = []
        detailed = raw.get("detailed_results_by_split", {})
        for task_type_results in detailed.values():
            for task_result in task_type_results:
                episodes.append(self._map_to_episode(task_result))

        # Check for missing tasks
        found_ids = {ep.task_id for ep in episodes}
        for tid in expected_task_ids:
            if f"car:{tid}" not in found_ids:
                episodes.append(self._make_failed_episode(tid, "missing_result"))

        return episodes

    def _generate_scenario(self, task_ids: list[str]) -> str:
        """Generate scenario.toml for this batch."""
        by_type: dict[str, list[str]] = {}
        for tid in task_ids:
            # "base_0" → "base", "hallucination_3" → "hallucination"
            parts = tid.rsplit("_", 1)
            task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else "base"
            by_type.setdefault(task_type, []).append(tid)

        lines = []
        for tt in _ALL_TASK_TYPES:
            if tt in by_type:
                lines.append(
                    f'tasks_{tt}_task_id_filter = {json.dumps(by_type[tt])}'
                )
            else:
                lines.append(f"tasks_{tt}_num_tasks = 0")

        filter_block = "\n".join(lines)

        return f"""\
[green_agent]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/green_car_bench_agent/server.py --host 127.0.0.1 --port 8081"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:{self._purple_port}"

[config]
task_split = "{self._task_split}"
{filter_block}
num_trials = 1
max_steps = 50
"""

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map a CAR detailed result to an lfx Episode."""
        car_task_id = task_result["task_id"]
        task_id = f"car:{car_task_id}"

        signals, breakdown = map_car_scores(
            task_result.get("reward_info", {}),
            task_reward=task_result.get("reward", 0.0),
        )

        # Convert trajectory to lfx Messages
        messages = []
        for msg in task_result.get("trajectory", []):
            messages.append(
                Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )
            )

        summary = EpisodeSummary(
            signals=signals,
            score_breakdown=breakdown,
        )

        return Episode(
            id=uuid4().hex,
            state_id="",
            task_id=task_id,
            bench="car",
            model=self._model,
            messages=messages,
            step_boundaries=[0] if messages else [],
            steps=[StepMeta(t=0, reward=task_result.get("reward", 0.0),
                            done=True, timing_ms=task_result.get("total_llm_latency_ms", 0.0))],
            summary=summary,
            created_at=time.time(),
            metadata={
                "car_raw_reward": task_result.get("reward"),
                "car_agent_cost": task_result.get("total_agent_cost"),
                "car_llm_latency_ms": task_result.get("total_llm_latency_ms"),
            },
        )

    def _make_failed_episode(self, task_id: str, reason: str) -> Episode:
        """Create a failed episode placeholder."""
        from lfx.core.reward import RewardSignal

        signals = {
            "outcome": RewardSignal(name="outcome", value=-1.0, confidence=0.5)
        }
        return Episode(
            id=uuid4().hex,
            state_id="",
            task_id=f"car:{task_id}",
            bench="car",
            model=self._model,
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=EpisodeSummary(signals=signals),
            metadata={"error": reason},
        )

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "car", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        # TODO: parse from CAR-bench task definitions (HuggingFace auto-download)
        # For now, return numbered task IDs
        raise NotImplementedError(
            "list_tasks requires CAR-bench data. Use run_batch with explicit task_ids."
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_car_adapter.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -v --timeout=30`
Expected: All existing + new tests pass.

- [ ] **Step 6: Commit**

```bash
git add lfx/adapters/car.py tests/test_car_adapter.py
git commit -m "feat: implement CARAdapter with scenario generation and results parsing"
```

---

## Chunk 4: CLI Extension + Final Wiring

### Task 7: Extend CLI for CAR-bench

**Files:**
- Modify: `lfx/cli.py`

- [ ] **Step 1: Add CAR-specific args to `lfx run` command**

In `_build_parser()`, add to the `run_p` subparser:
```python
    run_p.add_argument("--model", type=str, default=None, help="LLM model (litellm format)")
    run_p.add_argument("--task-type", type=str, default="base",
                       help="Task type: base, hallucination, disambiguation")
    run_p.add_argument("--task-split", type=str, default="test",
                       help="Data split: train, test")
    run_p.add_argument("--output", type=str, default=None, help="Output directory")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")
```

- [ ] **Step 2: Update `cmd_run` to pass config to adapter**

```python
def cmd_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    # CLI args override config file
    if args.model:
        config["model"] = args.model
    if args.output:
        config["output"] = args.output
    if hasattr(args, "task_type"):
        config["task_type"] = args.task_type
    if hasattr(args, "task_split"):
        config["task_split"] = args.task_split
    if hasattr(args, "seed") and args.seed is not None:
        config["seed"] = args.seed

    adapter = _get_adapter(args.bench)
    adapter.setup(config)

    if args.seed is not None:
        import random
        random.seed(args.seed)

    agent_state = AgentState()
    tasks = adapter.list_tasks()

    _, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=args.episodes,
        n_iterations=args.iterations,
    )
    print(f"Final state: {state_id.combined_hash}")
```

- [ ] **Step 3: Commit**

```bash
git add lfx/cli.py
git commit -m "feat: add CAR-bench CLI args to lfx run"
```

---

### Task 8: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new dependencies**

Add to `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
car = [
    "starlette>=0.27",
    "uvicorn>=0.20",
    "httpx>=0.24",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-timeout>=2.0",
]
```

- [ ] **Step 2: Install and verify**

```bash
cd /Users/robertmueller/Desktop/aganthos && uv sync --extra car --extra dev
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add car-bench dependencies"
```

---

### Task 9: Final integration smoke test

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -v --timeout=30
```
Expected: All tests pass including new test_car_rewards, test_car_purple, test_car_adapter.

- [ ] **Step 2: Verify imports work**

```bash
cd /Users/robertmueller/Desktop/aganthos && python -c "
from lfx.adapters.car import CARAdapter
from lfx.adapters._car_purple import CarPurpleAgent, create_app
from lfx.adapters._car_rewards import map_car_scores, DEFAULT_CAR_WEIGHTS
print('All imports OK')
"
```

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A
git commit -m "chore: final wiring for CAR-bench integration"
```
