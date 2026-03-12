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
        loop = asyncio.get_running_loop()
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
    import socket
    import time

    # Bind socket first to avoid race condition with port 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]
    sock.close()

    app = create_app(agent, actual_port)
    config = uvicorn.Config(app, host=host, port=actual_port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll for readiness
    import httpx
    for _ in range(50):
        try:
            r = httpx.get(f"http://{host}:{actual_port}/.well-known/agent.json", timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        log.warning("Purple server did not become ready within 5s")

    return thread, actual_port
