# clawloop/adapters/_entropic_purple.py
"""A2A purple agent server for Entropic CRMArenaPro with clawloop harness injection.

The entropic green agent sends CRM task prompts as plain text via A2A
``message/send``.  This purple agent:
  1. Injects the clawloop harness system prompt.
  2. Forwards the task to the configured LLM.
  3. Returns the answer as an A2A text response.

Multi-turn is supported: the green agent may send follow-up messages when the
agent requests clarification or when tool results are returned.
"""

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

from clawloop.layers.harness import Harness

log = logging.getLogger(__name__)


class EntropicPurpleAgent:
    """A2A-compliant purple agent for CRM tasks with clawloop harness injection."""

    def __init__(
        self,
        model: str,
        harness: Harness,
        bench: str = "entropic",
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.harness = harness
        self.bench = bench
        self.api_base = api_base
        self.api_key = api_key
        self._sessions: dict[str, list[dict]] = {}
        self._tool_cache: dict[str, list[dict]] = {}

    def update_harness(self, harness: Harness) -> None:
        self.harness = harness

    def clear_all_sessions(self) -> None:
        self._sessions.clear()
        self._tool_cache.clear()

    # -- Tool schema helpers --

    @staticmethod
    def _convert_tools_to_openai(raw_tools: list[dict]) -> list[dict]:
        """Normalize tool schemas to OpenAI function-calling format."""
        result = []
        for t in raw_tools:
            if t.get("type") == "function" and "function" in t:
                result.append(t)
            else:
                result.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                })
        return result

    @staticmethod
    def _normalize_assistant_msg(litellm_msg: Any) -> dict:
        """Normalize litellm response to stable internal dict."""
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

        # Return Message directly (not wrapped) — a2a-sdk expects result=Message
        return {
            "kind": "message",
            "messageId": uuid4().hex,
            "role": "agent",
            "parts": parts,
        }

    # -- CRM task formatting --

    @staticmethod
    def _format_crm_task(raw_text: str) -> str:
        """Parse CRM task JSON and format as a readable prompt.

        The green agent sends ``json.dumps(task_context)`` as an A2A TextPart.
        We extract the structured fields and present them clearly to the LLM.
        If the text isn't valid JSON, return it unchanged.
        """
        import json as _json
        try:
            ctx = _json.loads(raw_text)
        except (ValueError, TypeError):
            return raw_text

        if not isinstance(ctx, dict) or "prompt" not in ctx:
            return raw_text

        parts: list[str] = []

        persona = ctx.get("persona", "")
        if persona:
            parts.append(f"Persona: {persona}")

        parts.append(f"\nTask: {ctx['prompt']}")

        required = ctx.get("required_context", "")
        if required and required.strip():
            parts.append(f"\nContext:\n{required}")

        entropy = ctx.get("entropy")
        if entropy:
            parts.append(
                f"\nNote: Column names may have been modified (drift_level={entropy.get('drift_level','?')}). "
                "Adapt to any schema changes in the context."
            )

        parts.append(
            "\nProvide a direct, concise answer. "
            "If the task asks for IDs or specific values, return only those values."
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_task_tags(raw_text: str) -> set[str] | None:
        """Extract task category from CRM task JSON for selective playbook retrieval."""
        import json as _json
        try:
            ctx = _json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(ctx, dict):
            return None
        tags: set[str] = set()
        cat = ctx.get("task_category")
        if cat:
            tags.add(cat)
        return tags or None

    # -- Core message handling --

    def handle_message_sync(self, jsonrpc_request: dict) -> dict:
        """Handle one message/send request (sync)."""
        params = jsonrpc_request["params"]
        msg = params["message"]
        context_id = params.get("contextId", "default")

        text_parts = [p["text"] for p in msg["parts"] if p.get("kind") == "text"]
        data_parts = [p["data"] for p in msg["parts"] if p.get("kind") == "data"]

        # Initialize session
        if context_id not in self._sessions:
            self._sessions[context_id] = []

        messages = self._sessions[context_id]

        if not messages:
            # First message — inject harness as system prompt.
            # Extract task tags for selective playbook retrieval (ACE/DC-RS).
            raw_text = "\n".join(text_parts)
            task_tags = self._extract_task_tags(raw_text)
            harness_prompt = self.harness.system_prompt(self.bench, task_tags=task_tags)
            system_content = harness_prompt or "You are a helpful CRM assistant."
            messages.append({"role": "system", "content": system_content})

            user_text = raw_text
            user_text = self._format_crm_task(user_text)
            if user_text.strip():
                messages.append({"role": "user", "content": user_text})

            # Cache tools if provided
            for d in data_parts:
                if "tools" in d:
                    self._tool_cache[context_id] = self._convert_tools_to_openai(
                        d["tools"]
                    )
        else:
            # Subsequent turns — tool results and/or user text
            for d in data_parts:
                if "tool_results" in d:
                    for tr in d["tool_results"]:
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
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        response = litellm.completion(**completion_kwargs)
        assistant_msg = response.choices[0].message

        normalized = self._normalize_assistant_msg(assistant_msg)
        messages.append(normalized)

        return self._format_a2a_response(assistant_msg)

    @staticmethod
    def _reconcile_tool_call_id(
        messages: list[dict], tool_name: str, green_id: str
    ) -> None:
        """Rewrite last assistant tool_call id to match green's id."""
        used_green_ids = {
            m["tool_call_id"]
            for m in messages
            if m.get("role") == "tool" and "tool_call_id" in m
        }
        for msg in reversed(messages):
            if msg.get("role") != "assistant" or "tool_calls" not in msg:
                continue
            for tc in msg["tool_calls"]:
                if (
                    tc["function"]["name"] == tool_name
                    and tc["id"] not in used_green_ids
                ):
                    tc["id"] = green_id
                    return
            return


def create_app(agent: EntropicPurpleAgent, port: int = 0) -> Starlette:
    """Create the A2A Starlette app for the entropic purple agent."""

    async def agent_card(request: Request) -> JSONResponse:
        return JSONResponse({
            "name": "clawloop-entropic-purple-agent",
            "description": "ClawLoop harness-optimized CRM agent under test",
            "url": f"http://127.0.0.1:{port}/",
            "version": "0.1.0",
            "protocol_version": "0.3.0",
            "preferred_transport": "JSONRPC",
            "default_input_modes": ["text/plain"],
            "default_output_modes": ["text/plain"],
            "capabilities": {"streaming": False, "push_notifications": False},
            "skills": [
                {
                    "id": "crm_assistant",
                    "name": "CRM Assistant",
                    "description": "Agent under test for Entropic CRMArenaPro evaluation",
                    "tags": ["benchmark", "entropic", "crmarena"],
                }
            ],
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

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, agent.handle_message_sync, body
        )

        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})

    return Starlette(
        routes=[
            Route("/.well-known/agent.json", agent_card, methods=["GET"]),
            Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
            Route("/", handle_jsonrpc, methods=["POST"]),
        ]
    )


def start_purple_server(
    agent: EntropicPurpleAgent, host: str = "127.0.0.1", port: int = 0
) -> tuple[threading.Thread, int]:
    """Start the purple agent server in a background thread. Returns (thread, actual_port)."""
    import socket
    import time

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]

    app = create_app(agent, actual_port)
    config = uvicorn.Config(app, host=host, port=actual_port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()

    import httpx
    for _ in range(50):
        try:
            r = httpx.get(f"http://{host}:{actual_port}/.well-known/agent-card.json", timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        log.warning("Entropic purple server did not become ready within 5s")

    return thread, actual_port
