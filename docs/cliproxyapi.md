# CLIProxyAPI Setup for lfx

CLIProxyAPI wraps Claude OAuth tokens as an OpenAI-compatible API.
We use it to route Haiku and Sonnet calls without managing raw Anthropic API keys.

## Endpoint

```
URL:  http://127.0.0.1:8317/v1
Key:  kuhhandel-bench-key
```

Models use `openai/` prefix for litellm routing:
- `openai/claude-haiku-4-5-20251001` (agent under test)
- `openai/claude-sonnet-4-20250514` (reflector)

## Installation

```bash
brew install cliproxyapi
```

Config lives at `~/.cli-proxy-api/config.yaml`:
```yaml
host: "127.0.0.1"
port: 8317
auth-dir: "~/.cli-proxy-api"
api-keys:
  - "kuhhandel-bench-key"
```

## tool_prefix_disabled (critical)

CLIProxyAPI adds a `proxy_` prefix to tool call names for Claude OAuth tokens
(anti-fingerprinting measure). This breaks CAR-bench evaluation because the
green agent doesn't recognize `proxy_open_close_sunshade`.

**Fix:** Add `"metadata":{"tool_prefix_disabled":true}` to the Claude auth JSON:

```json
// ~/.cli-proxy-api/claude-*.json
{
  "access_token": "sk-ant-oat01-...",
  "type": "claude",
  "metadata": {
    "tool_prefix_disabled": true
  }
}
```

This was introduced in PR #1625 (2026-02-18). Requires CLIProxyAPI >= 6.8.x.

**History:**
- v6.7.x: `proxy_` prefix added for all Claude OAuth tokens, no way to disable
- v6.8.x: `tool_prefix_disabled` metadata option added (PR #1625)
- v6.8.50+: `claudeToolPrefix` set to empty string in cloak mode (PR #1750)
- Issue #1694: confirmed the prefix leaks through OpenAI-compat translation path
- Issue #1530: `proxy_` on `tool_choice.name` but not `tools[].name` caused 400 errors

## Running the service

```bash
# Via brew services (uses default config path)
brew services start cliproxyapi

# Manual with custom config
cliproxyapi -config ~/.cli-proxy-api/config.yaml
```

The service auto-reloads when auth files change (file watcher).

## Gotcha: GOOGLE_API_KEY

litellm prefers `GOOGLE_API_KEY` over `GEMINI_API_KEY`. If the system has a
free-tier Google API key in the environment, it will be used instead of the
paid Gemini key from `.env`. The CARAdapter strips `GOOGLE_API_KEY` from the
subprocess environment to prevent this.
