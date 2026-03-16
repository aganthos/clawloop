"""lfx-server — HTTP layer for n8n integration."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from lfx.collector import EpisodeCollector
from lfx.core.loop import AgentState
from lfx.core.reflector import Reflector, ReflectorConfig
from lfx.core.reward import RewardPipeline
from lfx.layers.harness import Harness
from lfx.learner import AsyncLearner

log = logging.getLogger(__name__)

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class LfxServer:
    """State container for the lfx-server."""

    def __init__(
        self,
        seed_prompt: str,
        bench: str = "n8n",
        batch_size: int = 5,
        reflector: Reflector | None = None,
    ) -> None:
        self.bench = bench
        self.seed_prompt = seed_prompt
        self._batch_size = batch_size
        self._reflector = reflector

        self.harness = Harness(
            system_prompts={bench: seed_prompt},
            reflector=reflector,
        )
        self.agent_state = AgentState(harness=self.harness)
        self.learner = AsyncLearner(
            agent_state=self.agent_state,
            active_layers=["harness"],
            on_learn_complete=self._on_learn_complete,
        )
        self.collector = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            batch_size=batch_size,
            on_batch=self._on_batch,
        )

        self._state_lock = threading.RLock()
        self._learning_status = "idle"
        self._last_error: str | None = None
        self._prompt_updated_at: str = datetime.now(timezone.utc).isoformat()
        self._reward_trend: list[float] = []
        self._reward_episode_ids: list[str] = []  # parallel to _reward_trend
        self._recent_insights: list[dict[str, Any]] = []

        # SSE: list of (queue, event_loop) tuples
        self._event_subscribers: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._subscribers_lock = threading.Lock()

    def _on_batch(self, episodes: list) -> None:
        enqueued = self.learner.on_batch(episodes)
        if enqueued:
            self.set_learning_status("learning")
            self.broadcast_event("learning_started", {
                "playbook_version": self.harness.playbook_version,
                "batch_size": len(episodes),
            })

    def _on_learn_complete(
        self, episodes: list, *, success: bool, error: str | None,
    ) -> None:
        # Only transition to "idle" if no more batches are queued
        queue_empty = self.learner.metrics["queue_size"] == 0

        with self._state_lock:
            if success:
                if queue_empty:
                    self._learning_status = "idle"
                self._last_error = None
                self._prompt_updated_at = datetime.now(timezone.utc).isoformat()
                self._recent_insights.clear()
                for entry in self.harness.playbook.entries:
                    if entry.source_episode_ids:
                        self._recent_insights.append({
                            "content": entry.content,
                            "source_episodes": entry.source_episode_ids,
                        })
            else:
                if queue_empty:
                    self._learning_status = "idle"
                self._last_error = error

        if success:
            self.broadcast_event("learning_completed", {
                "playbook_version": self.harness.playbook_version,
                "new_entries": len(self.harness.playbook.entries),
            })

    def set_learning_status(self, status: str) -> None:
        with self._state_lock:
            self._learning_status = status

    @property
    def learning_status(self) -> str:
        with self._state_lock:
            return self._learning_status

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            entries = [
                {
                    "id": e.id,
                    "content": e.content,
                    "tags": e.tags,
                    "helpful": e.helpful,
                    "harmful": e.harmful,
                    "source_episode_ids": e.source_episode_ids,
                }
                for e in self.harness.playbook.entries
            ]
            return {
                "system_prompt": self.harness.system_prompt(self.bench),
                "playbook_entries": entries,
                "playbook_version": self.harness.playbook_version,
                "prompt_updated_at": self._prompt_updated_at,
                "learning_status": self._learning_status,
                "last_error": self._last_error,
                "metrics": {
                    "episodes_collected": self.collector.metrics["episodes_collected"],
                    "episodes_filtered": self.collector.metrics["episodes_filtered"],
                    "feedback_received": self.collector.metrics["feedback_received"],
                },
            }

    def broadcast_event(self, event_type: str, data: dict) -> None:
        event = {"event": event_type, "data": data}
        with self._subscribers_lock:
            dead = []
            for q, loop in self._event_subscribers:
                try:
                    # Check fullness before scheduling — QueueFull would be
                    # raised inside the event loop where we can't catch it.
                    if q.full():
                        dead.append((q, loop))
                        continue
                    loop.call_soon_threadsafe(q.put_nowait, event)
                except RuntimeError:
                    # Event loop closed
                    dead.append((q, loop))
            for item in dead:
                self._event_subscribers.remove(item)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._subscribers_lock:
            self._event_subscribers.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._subscribers_lock:
            self._event_subscribers = [
                (qq, ll) for qq, ll in self._event_subscribers if qq is not q
            ]

    def start(self) -> None:
        self.learner.start()

    def stop(self) -> None:
        self.learner.stop()

    def reset(self) -> None:
        self.learner.stop()
        self.harness = Harness(
            system_prompts={self.bench: self.seed_prompt},
            reflector=self._reflector,
        )
        self.agent_state.harness = self.harness
        self.collector = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            batch_size=self._batch_size,
            on_batch=self._on_batch,
        )
        self.learner = AsyncLearner(
            agent_state=self.agent_state,
            active_layers=["harness"],
            on_learn_complete=self._on_learn_complete,
        )
        with self._state_lock:
            self._learning_status = "idle"
            self._last_error = None
            self._reward_trend.clear()
            self._reward_episode_ids.clear()
            self._recent_insights.clear()
            self._prompt_updated_at = datetime.now(timezone.utc).isoformat()
        self.learner.start()


def _validate_ingest(body: dict) -> str | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return "messages must be a non-empty list of {role, content} objects"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"messages[{i}] must be a dict"
        if "role" not in msg or "content" not in msg:
            return f"messages[{i}] must have 'role' and 'content'"
        if msg["role"] not in VALID_ROLES:
            return f"messages[{i}].role must be one of {sorted(VALID_ROLES)}"
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return "metadata must be a dict"
    return None


async def ingest(request: Request) -> JSONResponse:
    server: LfxServer = request.app.state.server
    body = await request.json()
    error = _validate_ingest(body)
    if error:
        return JSONResponse({"error": "validation_error", "detail": error}, status_code=422)

    metadata = body.get("metadata", {})
    ep = server.collector.ingest_external(
        messages=body["messages"],
        task_id=metadata.get("conversation_id", ""),
        session_id=metadata.get("conversation_id", ""),
        model=metadata.get("model"),
        usage=metadata.get("usage"),
        bench=server.bench,
    )

    with server._state_lock:
        server._reward_trend.append(ep.summary.normalized_reward())
        server._reward_episode_ids.append(ep.id)

    # Extract user query and assistant response for dashboard display
    messages = body["messages"]
    user_query = ""
    assistant_response = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_response = msg.get("content", "")

    server.broadcast_event("episode_ingested", {
        "episode_id": ep.id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query[:200],
        "assistant_response": assistant_response[:300],
        "reward_signals": {
            k: {"value": s.value, "confidence": s.confidence}
            for k, s in ep.summary.signals.items()
        },
    })

    return JSONResponse({
        "episode_id": ep.id,
        "playbook_version": server.harness.playbook_version,
        "learning_status": server.learning_status,
    })


async def feedback(request: Request) -> JSONResponse:
    server: LfxServer = request.app.state.server
    body = await request.json()
    episode_id = body.get("episode_id", "")
    score = body.get("score", 0.0)
    if not isinstance(score, (int, float)):
        return JSONResponse({"error": "validation_error", "detail": "score must be a number"}, status_code=422)

    found = server.collector.submit_feedback(episode_id, float(score))
    if not found:
        return JSONResponse({"error": "not_found", "detail": f"episode {episode_id} not found"}, status_code=404)

    # Update reward trend to reflect feedback
    with server._state_lock:
        try:
            idx = server._reward_episode_ids.index(episode_id)
            # Re-read the episode's reward now that feedback is attached
            ep = server.collector._episode_index.get(episode_id)
            if ep:
                server._reward_trend[idx] = ep.summary.normalized_reward()
        except (ValueError, IndexError):
            pass

    server.collector.flush_buffer()
    server.broadcast_event("feedback_received", {"episode_id": episode_id, "score": score})
    return JSONResponse({"ok": True})


async def state(request: Request) -> JSONResponse:
    server: LfxServer = request.app.state.server
    return JSONResponse(server.get_state_snapshot())


async def reset_handler(request: Request) -> JSONResponse:
    server: LfxServer = request.app.state.server
    server.reset()
    return JSONResponse({"ok": True, "playbook_version": 0, "learning_status": "idle"})


async def metrics(request: Request) -> JSONResponse:
    server: LfxServer = request.app.state.server
    cm = server.collector.metrics
    with server._state_lock:
        return JSONResponse({
            "episodes_collected": cm["episodes_collected"],
            "episodes_filtered": cm["episodes_filtered"],
            "feedback_received": cm["feedback_received"],
            "playbook_version": server.harness.playbook_version,
            "learning_status": server._learning_status,
            "last_error": server._last_error,
            "reward_trend": list(server._reward_trend),
            "recent_insights": list(server._recent_insights),
        })


async def events(request: Request) -> StreamingResponse:
    server: LfxServer = request.app.state.server
    loop = asyncio.get_running_loop()
    q = server.subscribe(loop)

    async def event_stream():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            server.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    seed_prompt_path: str | None = None,
    seed_prompt: str | None = None,
    bench: str = "n8n",
    batch_size: int = 5,
    reflector: Reflector | None = None,
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    api_key: str | None = None,
) -> Starlette:
    import os

    if seed_prompt is None:
        if seed_prompt_path:
            seed_prompt = Path(seed_prompt_path).read_text().strip()
        else:
            seed_prompt = "You are a helpful assistant."

    # Auto-create Reflector: explicit api_base/api_key, or env vars
    if reflector is None:
        has_creds = api_base or (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if has_creds:
            try:
                from lfx.llm import LiteLLMClient
                client = LiteLLMClient(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                )
                reflector = Reflector(client=client, config=ReflectorConfig())
                log.info("Auto-created Reflector with %s (api_base=%s)", model, api_base or "default")
            except Exception:
                log.warning("Could not create Reflector — learning will not generate insights", exc_info=True)

    server = LfxServer(
        seed_prompt=seed_prompt, bench=bench,
        batch_size=batch_size, reflector=reflector,
    )

    routes = [
        Route("/ingest", ingest, methods=["POST"]),
        Route("/feedback", feedback, methods=["POST"]),
        Route("/state", state, methods=["GET"]),
        Route("/reset", reset_handler, methods=["POST"]),
        Route("/metrics", metrics, methods=["GET"]),
        Route("/events", events, methods=["GET"]),
    ]

    app = Starlette(routes=routes)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        from starlette.staticfiles import StaticFiles
        app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True))

    app.state.server = server

    @app.on_event("startup")
    async def startup():
        server.start()

    @app.on_event("shutdown")
    async def shutdown():
        server.stop()

    return app


def main() -> None:
    import argparse
    import os
    parser = argparse.ArgumentParser(description="lfx-server for n8n integration")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--seed-prompt", default="config/seed_prompt.txt")
    parser.add_argument("--bench", default="n8n")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default=None, help="LLM model for Reflector (litellm format)")
    parser.add_argument("--api-base", default=None, help="LLM API base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # CLI args override env vars
    api_base = args.api_base or os.environ.get("LFX_API_BASE") or None
    api_key = args.api_key or os.environ.get("LFX_API_KEY") or None
    model = args.model or os.environ.get("LFX_MODEL") or "gpt-4o-mini"

    app = create_app(
        seed_prompt_path=args.seed_prompt, bench=args.bench,
        batch_size=args.batch_size, model=model,
        api_base=api_base, api_key=api_key,
    )
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
