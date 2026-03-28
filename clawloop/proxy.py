"""ProxyApp — OpenAI-compatible reverse proxy with trace capture.

Compatibility:
    - Serves **Chat Completions** only: `POST /v1/chat/completions` (and streaming).
      This proxy does not implement `/v1/completions`, `/v1/embeddings`,
      `/v1/responses`, etc.

Modes:
    - bench_mode=True (default): intended for local benchmark/training runs.
      Requires `X-ClawLoop-Run-Id` so traces can be correlated into sessions.
    - bench_mode=False ("live mode"): intended for a deployed proxy.
      Requires `proxy_key` and enforces `Authorization: Bearer <proxy_key>`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from clawloop.proxy_config import ProxyConfig
from clawloop.proxy_session import SessionTracker
from clawloop.proxy_skills import inject_skills, strip_skills
from clawloop.proxy_sse import parse_json_response, parse_sse_bytes

log = logging.getLogger(__name__)


class ProxyApp:
    """OpenAI-compatible reverse proxy with trace capture and post-processing.

    Constructor accepts:
        config: ProxyConfig
        collector: optional EpisodeCollector
        harness: optional Harness (provides playbook.render())
        mount_prefix: prefix for standalone mode (default "/v1"); set to ""
            when the app is mounted under an external Mount("/v1", ...).
    """

    def __init__(
        self,
        config: ProxyConfig,
        collector: Any | None = None,
        harness: Any | None = None,
        mount_prefix: str = "/v1",
    ) -> None:
        self.config = config
        self.collector = collector
        self.harness = harness
        self.session_tracker = SessionTracker()
        self.drops_total: int = 0

        # Populated during lifespan
        self._http_client: httpx.AsyncClient | None = None
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []

        # The route is always "/chat/completions" (no /v1 prefix).
        # In standalone mode (default), mount_prefix="/v1" wraps it in a
        # Mount so the full path /v1/chat/completions works.
        # When mounted externally (server.py), mount_prefix="" skips the
        # wrapper so the external Mount("/v1", ...) can supply the prefix.
        from starlette.routing import Mount

        chat_route = Route(
            "/chat/completions",
            self._handle_chat_completions,
            methods=["POST"],
        )

        if mount_prefix:
            routes = [Mount(mount_prefix, routes=[chat_route])]
        else:
            routes = [chat_route]

        self.asgi_app = Starlette(
            routes=routes,
            lifespan=self._lifespan,
        )

    # ------------------------------------------------------------------
    # Lifespan — startup / shutdown can be called externally when the
    # proxy is mounted as a sub-app (server.py) or internally via the
    # Starlette lifespan context manager (standalone mode).
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise HTTP client and background workers."""
        if self._http_client is not None:
            return  # already started (guard against double-start)
        self._check_single_worker()

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.upstream_connect_timeout_s,
                read=self.config.upstream_read_timeout_s,
                write=30.0,
                pool=30.0,
            ),
            follow_redirects=False,
            trust_env=False,
        )

        self._queue = asyncio.Queue(maxsize=64)
        self._workers = [
            asyncio.create_task(self._post_process_worker(i))
            for i in range(self.config.max_post_process_tasks)
        ]

    async def shutdown(self) -> None:
        """Drain workers and close HTTP client."""
        if self._http_client is None:
            return  # not started or already shut down

        # Send poison pills
        assert self._queue is not None
        for _ in self._workers:
            await self._queue.put(None)

        # Wait with 10s grace period
        done, pending = await asyncio.wait(
            self._workers, timeout=10.0,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=2.0)

        await self._http_client.aclose()
        self._http_client = None

    @asynccontextmanager
    async def _lifespan(self, app: Starlette):
        await self.startup()
        yield
        await self.shutdown()

    # ------------------------------------------------------------------
    # Single-worker guard
    # ------------------------------------------------------------------

    @staticmethod
    def _check_single_worker() -> None:
        concurrency = os.environ.get("WEB_CONCURRENCY", "1")
        if int(concurrency) > 1:
            raise RuntimeError(
                f"ProxyApp requires WEB_CONCURRENCY=1, got {concurrency}. "
                "Session state is in-process only."
            )

    # ------------------------------------------------------------------
    # Route handler: POST /chat/completions (served at /v1 via mount prefix)
    # ------------------------------------------------------------------

    async def _handle_chat_completions(self, request: Request) -> Response:
        cfg = self.config

        # 1. Auth check
        if not cfg.bench_mode and cfg.proxy_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {cfg.proxy_key}":
                return JSONResponse(
                    {"error": "unauthorized", "detail": "Invalid or missing API key"},
                    status_code=401,
                )

        # 2. Parse body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "bad_request", "detail": "Invalid JSON body"},
                status_code=400,
            )

        # 3. Session correlation
        run_id = request.headers.get("x-clawloop-run-id")
        session_id_header = request.headers.get("x-clawloop-session-id")

        if cfg.bench_mode and run_id is None:
            return JSONResponse(
                {"error": "bad_request", "detail": "X-ClawLoop-Run-Id header required in bench mode"},
                status_code=400,
            )

        session_id, _attributed = self.session_tracker.resolve_session(
            run_id, session_id_header,
        )

        # 4. Turn ordering
        turn_idx = self.session_tracker.next_turn(session_id)
        t_start = time.monotonic_ns()

        # 5. No-train header
        no_train = request.headers.get("x-clawloop-no-train") == "1"

        # 6. Skill injection
        messages = body.get("messages", [])
        if self.harness is not None:
            try:
                skills_text = self.harness.playbook.render()
                messages = inject_skills(messages, skills_text)
                body["messages"] = messages
            except Exception:
                log.warning("skill injection failed", exc_info=True)

        # 7. stream_options
        is_streaming = body.get("stream", False)
        if is_streaming and cfg.upstream_supports_stream_usage:
            so = body.get("stream_options", {})
            so["include_usage"] = True
            body["stream_options"] = so

        # 8. Forward upstream — build safe headers
        forward_headers: dict[str, str] = {}
        for hname in cfg.FORWARD_HEADERS:
            val = request.headers.get(hname)
            if val is not None:
                forward_headers[hname] = val
        forward_headers["authorization"] = (
            f"Bearer {cfg.upstream_api_key.get_secret_value()}"
        )

        upstream_url = f"{cfg.upstream_url}/chat/completions"

        assert self._http_client is not None

        # 9. Tee response
        truncated = False
        tee_buffer = bytearray()
        max_tee = cfg.max_tee_bytes

        if is_streaming:
            # Use send(stream=True) so bytes flow through without buffering
            # the full response in memory.  The finally block ensures the
            # upstream connection is closed even if the client disconnects.
            req = self._http_client.build_request(
                "POST", upstream_url,
                content=json.dumps(body).encode(),
                headers=forward_headers,
            )
            try:
                upstream_resp = await self._http_client.send(req, stream=True)
            except httpx.HTTPError as exc:
                log.error("upstream request failed: %s", exc)
                return JSONResponse(
                    {"error": "upstream_error", "detail": str(exc)},
                    status_code=502,
                )

            async def _stream_and_tee():
                nonlocal truncated
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                        if not truncated:
                            if len(tee_buffer) + len(chunk) <= max_tee:
                                tee_buffer.extend(chunk)
                            else:
                                truncated = True

                    # Enqueue after stream completes
                    await self._enqueue_post_process(
                        body=body,
                        tee_bytes=bytes(tee_buffer),
                        truncated=truncated,
                        no_train=no_train,
                        session_id=session_id,
                        turn_idx=turn_idx,
                        t_start=t_start,
                        is_streaming=True,
                    )
                finally:
                    await upstream_resp.aclose()

            resp_headers = {}
            ct = upstream_resp.headers.get("content-type")
            if ct:
                resp_headers["content-type"] = ct

            return StreamingResponse(
                _stream_and_tee(),
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )

        else:
            # Non-streaming: eagerly read full response
            try:
                upstream_resp = await self._http_client.post(
                    upstream_url,
                    content=json.dumps(body).encode(),
                    headers=forward_headers,
                )
            except httpx.HTTPError as exc:
                log.error("upstream request failed: %s", exc)
                return JSONResponse(
                    {"error": "upstream_error", "detail": str(exc)},
                    status_code=502,
                )

            content = upstream_resp.content
            if len(content) <= max_tee:
                tee_buffer.extend(content)
            else:
                tee_buffer.extend(content[:max_tee])
                truncated = True

            # 10. Enqueue post-processing
            await self._enqueue_post_process(
                body=body,
                tee_bytes=bytes(tee_buffer),
                truncated=truncated,
                no_train=no_train,
                session_id=session_id,
                turn_idx=turn_idx,
                t_start=t_start,
                is_streaming=False,
            )

            # 11. Return response
            resp_headers = {}
            ct = upstream_resp.headers.get("content-type")
            if ct:
                resp_headers["content-type"] = ct

            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )

    # ------------------------------------------------------------------
    # Post-processing enqueue
    # ------------------------------------------------------------------

    async def _enqueue_post_process(
        self,
        *,
        body: dict,
        tee_bytes: bytes,
        truncated: bool,
        no_train: bool,
        session_id: str,
        turn_idx: int,
        t_start: int,
        is_streaming: bool,
    ) -> None:
        work_item = {
            "body": body,
            "tee_bytes": tee_bytes,
            "truncated": truncated,
            "no_train": no_train,
            "session_id": session_id,
            "turn_idx": turn_idx,
            "t_start": t_start,
            "is_streaming": is_streaming,
        }
        assert self._queue is not None
        try:
            self._queue.put_nowait(work_item)
        except asyncio.QueueFull:
            self.drops_total += 1
            log.warning(
                "post-process queue full, dropping work item "
                "(session=%s turn=%d drops_total=%d)",
                session_id,
                turn_idx,
                self.drops_total,
            )

    # ------------------------------------------------------------------
    # Post-processing workers
    # ------------------------------------------------------------------

    async def _post_process_worker(self, worker_id: int) -> None:
        assert self._queue is not None
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return

            if item is None:
                # Poison pill — shut down
                return

            try:
                await self._process_item(item)
            except Exception:
                log.error(
                    "post-process worker %d failed", worker_id, exc_info=True,
                )

    async def _process_item(self, item: dict) -> None:
        if item["truncated"] or item["no_train"]:
            return

        tee_bytes: bytes = item["tee_bytes"]
        body: dict = item["body"]
        session_id: str = item["session_id"]

        # Apply redaction hook before any parsing / persistence
        if self.config.redaction_hook is not None:
            try:
                body = self.config.redaction_hook(body)
                item["body"] = body
            except Exception:
                log.error(
                    "redaction_hook failed for session=%s, dropping item",
                    session_id,
                    exc_info=True,
                )
                return

        # Parse response
        if item["is_streaming"]:
            message, usage, _complete = parse_sse_bytes(tee_bytes)
        else:
            message, usage, _model = parse_json_response(tee_bytes)

        if message is None:
            log.debug("could not parse response for session=%s", session_id)
            return

        # Strip skills from the request messages before storing
        request_messages = strip_skills(body.get("messages", []))

        # Get model from body
        model = body.get("model")

        # Normalize usage to dict[str, int]
        usage_dict: dict[str, int] | None = None
        if isinstance(usage, dict):
            usage_dict = {
                k: int(v) for k, v in usage.items() if isinstance(v, (int, float))
            }

        # Call collector if available
        if self.collector is not None:
            # Build the full message list: request messages + assistant response
            all_messages = list(request_messages)
            all_messages.append(message)

            try:
                self.collector.ingest_external(
                    messages=all_messages,
                    task_id=session_id,
                    session_id=session_id,
                    model=model,
                    usage=usage_dict,
                    bench=self.config.bench,
                )
            except Exception:
                log.error(
                    "collector.ingest_external failed for session=%s",
                    session_id,
                    exc_info=True,
                )
