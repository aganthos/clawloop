"""ProxyConfig — configuration for the OpenClaw reverse-proxy adapter.

The proxy only serves the OpenAI **Chat Completions** endpoint
(`POST /v1/chat/completions`). It does not implement `/v1/completions`,
`/v1/embeddings`, `/v1/responses`, etc.

bench_mode:
    - True (default): local benchmark/training mode. Requires `X-ClawLoop-Run-Id`.
    - False: live/deployed mode. Requires `proxy_key` and enforces Authorization.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, SecretStr, model_validator


class ProxyConfig(BaseModel):
    """Configuration for proxying upstream LLM API calls."""

    model_config = {"arbitrary_types_allowed": True}

    FORWARD_HEADERS: ClassVar[frozenset[str]] = frozenset({"content-type", "accept", "user-agent"})

    upstream_url: str
    upstream_api_key: SecretStr
    bench: str = "openclaw"
    bench_mode: bool = True
    proxy_key: str = ""
    max_tee_bytes: int = 524288  # 512 KB
    live_idle_timeout_s: int = 300
    upstream_connect_timeout_s: float = 10.0
    upstream_read_timeout_s: float = 120.0
    upstream_supports_stream_usage: bool = True
    max_post_process_tasks: int = 8
    redaction_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> ProxyConfig:
        # Validate upstream_url scheme
        parsed = urlparse(self.upstream_url)
        if parsed.scheme == "http":
            hostname = parsed.hostname or ""
            if hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(
                    "upstream_url must use https for remote hosts " f"(got http://{hostname})"
                )
        elif parsed.scheme != "https":
            raise ValueError(f"upstream_url must use https (got {parsed.scheme}://)")

        # Live mode requires proxy_key
        if not self.bench_mode and not self.proxy_key:
            raise ValueError("proxy_key is required when bench_mode=False (live mode)")

        return self
