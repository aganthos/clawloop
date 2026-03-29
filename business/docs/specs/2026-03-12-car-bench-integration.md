# CAR-bench Harness Improvement Loop — Integration Spec

## Goal

Wire up CAR-bench as the first real benchmark for the lfx harness improvement
loop. Run iterative prompt/playbook optimization on CAR's 254 tasks and measure
improvement over iterations.

## Non-goals (v1)

- Entropic CRM, tau2, or other benchmarks (next)
- Router/Weights layers (harness-only)
- Parallel episodes
- Checkpoint restore
- Deploy gating

---

## Phase 0 Findings (from CAR-bench source study)

All answers verified against source at
`https://github.com/CAR-bench/car-bench-agentbeats`.

### 1. Task Orchestration

`agentbeats-run` is the entry point (`agentbeats.run_scenario:main`). It:
1. Starts purple + green agent subprocesses (from `cmd` fields in scenario.toml)
2. Waits for health checks (polls agent card endpoints)
3. Launches `client_cli.py` which sends an `EvalRequest` to the green agent

**Task selection** is fully configurable in `scenario.toml`:
```toml
[config]
task_split = "test"                    # "train" or "test"
tasks_base_num_tasks = 10              # first N tasks, -1 for all
tasks_hallucination_num_tasks = 0      # skip this type
tasks_disambiguation_num_tasks = 0     # skip this type
# tasks_base_task_id_filter = ["base_0", "base_2"]  # overrides num_tasks
num_trials = 1
max_steps = 50
```

**Key**: We can select specific task IDs per type via `task_id_filter`, or take
the first N via `num_tasks`. Task types are skipped if their count/filter is
absent.

### 2. Score Delivery

Scores land in `output/results.json` (path configurable via `--output`).

**Format** (from `calculate_evaluation_results`):
```json
{
  "score": 5.0,
  "max_score": 10,
  "pass_rate": 50.0,
  "task_rewards_by_split": {"base": {"base_0": 1.0, "base_1": 0.0, ...}},
  "detailed_results_by_split": {
    "base": [
      {
        "task_id": "base_0",
        "reward": 1.0,
        "trial": 0,
        "reward_info": { ... },
        "trajectory": [ ... ],
        "user_cost": 0.001,
        "total_agent_cost": 0.05,
        "total_llm_latency_ms": 2300.0
      }
    ]
  }
}
```

- `reward` is binary (0.0 or 1.0) per task per trial
- `reward_info` contains the 6 sub-metrics
- `trajectory` contains the full message sequence (messages minus system prompt)
- `pass_rate` is 0-100, `Pass^k`/`Pass@k` are 0.0-1.0

### 3. Purple Agent Interface

File: `src/purple_car_bench_agent/car_bench_agent.py`

- Uses `litellm.completion()` (sync, not async, not streaming)
- `CARBenchAgentExecutor(AgentExecutor)` — implements a2a-sdk's executor interface
- Model configurable via CLI `--agent-llm` or env `AGENT_LLM`
- System prompt comes FROM the green agent (not hardcoded in purple)

### 4. System Prompt Flow

The system prompt originates in the car-bench environment (policies + tools +
task context). Green agent relays it to purple in the first message:
```
"System: {system_prompt}\n\nUser: {initial_observation}"
```
Purple parses this apart and sets `messages[0] = {"role": "system", ...}`.

**Implication**: To inject our harness playbook, we prepend it to the system
prompt that arrives from green, before sending to the LLM.

### 5. Injection Strategy: Write Our Own Purple Agent (Strategy C)

The DEVELOPMENT_GUIDE.md explicitly states:
> "You are not limited to the baseline approach. You can use any LLM provider,
> any framework. The only requirement is conforming to the A2A message protocol."

**Decision**: Write our own thin A2A purple agent server. This gives us:
- Full control over system prompt injection (prepend harness playbook)
- Full message capture for Episode construction
- No dependency on CAR's purple agent code
- No monkey-patching or source modification
- Clean separation: CAR submodule stays pristine

### 6. A2A Method Contract

Green calls purple via exactly two HTTP endpoints:
1. `GET /.well-known/agent.json` — agent card (health check + discovery)
2. `POST /` — JSON-RPC 2.0 `message/send`

No other methods. No streaming green→purple. Sequential, one request at a time
(`max_concurrency=1` hardcoded in green).

### 7. Message Protocol

**First message** (green → purple):
- TextPart: `"System: {policies+tools+context}\n\nUser: {initial_observation}"`
- DataPart: `{"tools": [{"name": str, "description": str, "parameters": JSONSchema}, ...]}`

**Subsequent messages** (green → purple, after tool execution):
- DataPart: `{"tool_results": [{"tool_name": str, "tool_call_id": str, "content": str}, ...]}`
- Optional TextPart: user follow-up text

**Purple responses**:
- TextPart: agent's text response
- DataPart (if tool calls): `{"tool_calls": [{"tool_name": str, "arguments": dict}, ...]}`

**Task boundary**: Each task = fresh `RemoteA2AAgent` instance with new
`context_id`. No explicit "done" signal to purple — green just stops sending
messages when car-bench `run()` decides the task is complete.

### 8. Concurrency

Sequential. `max_concurrency=1`. No multiprocessing. Purple handles one
request at a time.

---

## Terminology

CAR-bench has two separate concepts that must not be conflated:

- **task_split**: `"train"` or `"test"` — which data partition to use
- **task_type**: `"base"`, `"hallucination"`, or `"disambiguation"` — task category

Our CLI `--task-type` maps to task_type (default: base).
Our CLI `--task-split` maps to task_split (default: test).

---

## Architecture

```
aganthos/
├── benchmarks/
│   └── car-bench/              # git submodule @ pinned commit
├── lfx/
│   ├── adapters/
│   │   ├── base.py             # EnvAdapter ABC (exists, add run_batch)
│   │   ├── car.py              # CARAdapter (NEW)
│   │   └── _car_purple.py      # Our A2A purple agent server (NEW)
│   ├── core/
│   │   └── loop.py             # Minor: support run_batch if available
│   └── cli.py                  # Extended: `lfx run car ...`
```

### _car_purple.py — Our Purple Agent (~200 lines)

Thin A2A server that:
1. Serves agent card at `/.well-known/agent.json`
2. Handles `message/send` JSON-RPC at `POST /`
3. Parses green's messages (TextPart system prompt + DataPart tools/tool_results)
4. Calls LLM via litellm with harness-augmented system prompt
5. Returns correctly formatted A2A responses
6. Captures full message trajectory for Episode construction

#### A2A Response Format (exact)

Purple must return JSON-RPC response with A2A message envelope:
```json
{
  "jsonrpc": "2.0",
  "id": "<echo request id>",
  "result": {
    "message": {
      "messageId": "<uuid4>",
      "role": "agent",
      "parts": [
        {"kind": "text", "text": "<agent text response>"},
        {"kind": "data", "data": {
          "tool_calls": [
            {"tool_name": "fn_name", "arguments": {"key": "val"}}
          ]
        }}
      ]
    }
  }
}
```
- TextPart always present (agent's text response, or empty string if only tool calls)
- DataPart with `tool_calls` only present if LLM returned tool calls
- `messageId` is a fresh UUID4 per response

#### Tool call ID flow

In CAR's A2A protocol, **purple does NOT send tool_call_ids** in its response.
The reference purple agent sends only `{"tool_name": ..., "arguments": ...}`.
Green's `RemoteA2AAgent._parse_response()` receives these, generates its own
internal tool_call_ids, executes the tools, and sends results back with those IDs.

Flow:
1. Purple → Green: `{"tool_calls": [{"tool_name": "X", "arguments": {...}}]}`
   (no ID field — green generates its own)
2. Green executes tools, generates `tool_call_id` per call
3. Green → Purple: `{"tool_results": [{"tool_call_id": "green_gen_id", ...}]}`
4. Purple uses `tool_call_id` from tool_results in the `role: tool` messages
   sent to the LLM (so the LLM sees correct correlation)

**Our code**: We store the incoming `tool_call_id` from green's tool_results
and use it in the conversation history. We do NOT generate or send IDs ourselves.

#### JSON-RPC envelope responsibility

`_format_a2a_response()` returns the `result` body only. The JSON-RPC envelope
(`{"jsonrpc": "2.0", "id": <request_id>, "result": ...}`) is added by the
FastAPI route handler which has access to the request ID. This separation keeps
the formatting function clean.

#### Message normalization

All messages stored in session history use a normalized schema:
```python
def _normalize_assistant_msg(self, litellm_msg) -> dict:
    """Normalize litellm response to stable internal format."""
    normalized = {"role": "assistant", "content": litellm_msg.content or ""}
    if litellm_msg.tool_calls:
        normalized["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments}}
            for tc in litellm_msg.tool_calls
        ]
    return normalized
```
This prevents provider-specific schema drift.

#### Tool schema caching

Tool schemas are cached per context_id in `_tool_cache: dict[str, list[dict]]`.
Parsed from the first message's DataPart, converted to OpenAI function format:
```python
def _convert_tools_to_openai(self, car_tools: list[dict]) -> list[dict]:
    """Convert CAR tool schemas to OpenAI function-calling format."""
    return [
        {"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }}
        for t in car_tools
    ]
```

```python
class CarPurpleAgent:
    """A2A-compliant purple agent with lfx harness injection."""

    def __init__(self, model: str, harness: Harness, bench: str = "car"):
        self.model = model
        self.harness = harness
        self.bench = bench
        self._sessions: dict[str, list[dict]] = {}   # context_id → messages
        self._tool_cache: dict[str, list[dict]] = {}  # context_id → openai tools
        self._captured: dict[str, list[dict]] = {}    # context_id → captured msgs
        self._context_order: list[str] = []           # tracks context_id creation order

    def update_harness(self, harness: Harness) -> None:
        """Called between iterations to update harness state."""
        self.harness = harness

    async def handle_message(self, jsonrpc_request: dict) -> dict:
        """Handle one message/send request from green agent."""
        params = jsonrpc_request["params"]
        msg = params["message"]
        context_id = params.get("contextId", "default")

        text_parts = [p["text"] for p in msg["parts"] if p.get("kind") == "text"]
        data_parts = [p["data"] for p in msg["parts"] if p.get("kind") == "data"]

        # Initialize session
        if context_id not in self._sessions:
            self._sessions[context_id] = []
            self._captured[context_id] = []
            self._context_order.append(context_id)

        messages = self._sessions[context_id]

        # First message: extract system prompt + tools
        if not messages:
            raw_text = text_parts[0] if text_parts else ""
            system_prompt, user_text = self._parse_first_message(raw_text)

            # HARNESS INJECTION: prepend playbook to system prompt
            harness_prompt = self.harness.system_prompt(self.bench)
            if harness_prompt:
                system_prompt = f"{harness_prompt}\n\n{system_prompt}"

            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})

            # Cache tool schemas
            for d in data_parts:
                if "tools" in d:
                    self._tool_cache[context_id] = self._convert_tools_to_openai(d["tools"])
        else:
            # Subsequent: tool results AND/OR user text (handle both)
            for d in data_parts:
                if "tool_results" in d:
                    for tr in d["tool_results"]:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_call_id"],
                            "content": tr["content"],
                        })
            for text in text_parts:
                if text.strip():
                    messages.append({"role": "user", "content": text})

        # Call LLM
        tools = self._tool_cache.get(context_id)
        completion_kwargs = {"model": self.model, "messages": messages, "temperature": 0.0}
        if tools:
            completion_kwargs["tools"] = tools

        response = litellm.completion(**completion_kwargs)
        assistant_msg = response.choices[0].message

        # Normalize and append to conversation
        normalized = self._normalize_assistant_msg(assistant_msg)
        messages.append(normalized)

        # Capture for Episode construction
        self._captured[context_id].append(normalized)

        # Format A2A response
        return self._format_a2a_response(assistant_msg)

    def _format_a2a_response(self, assistant_msg) -> dict:
        """Format LLM response as A2A result body (JSON-RPC envelope added by server)."""
        parts = [{"kind": "text", "text": assistant_msg.content or ""}]

        if assistant_msg.tool_calls:
            tool_calls = []
            for tc in assistant_msg.tool_calls:
                # Robust argument parsing: handle str, dict, or malformed JSON
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        log.warning("Malformed tool args for %s, passing raw", tc.function.name)
                        args = {"raw": args}
                tool_calls.append({"tool_name": tc.function.name, "arguments": args})
            parts.append({"kind": "data", "data": {"tool_calls": tool_calls}})

        return {
            "message": {
                "messageId": uuid4().hex,
                "role": "agent",
                "parts": parts,
            }
        }

    def _parse_first_message(self, raw_text: str) -> tuple[str, str]:
        """Parse 'System: ...\n\nUser: ...' format from green agent."""
        if "System:" in raw_text and "\n\nUser:" in raw_text:
            parts = raw_text.split("\n\nUser:", 1)
            system = parts[0].replace("System:", "", 1).strip()
            user = parts[1].strip()
            return system, user
        return "", raw_text  # no system prompt, all user text

    def get_completed_contexts(self) -> list[str]:
        """Return context_ids in creation order (for task_id mapping)."""
        return list(self._context_order)

    def collect_episode_data(self, context_id: str) -> dict:
        """Retrieve captured messages for Episode construction, then cleanup."""
        data = {
            "messages": self._sessions.pop(context_id, []),
            "captured": self._captured.pop(context_id, []),
        }
        self._tool_cache.pop(context_id, None)
        return data

    def clear_all_sessions(self) -> None:
        """Clear all session state between iterations."""
        self._sessions.clear()
        self._tool_cache.clear()
        self._captured.clear()
        self._context_order.clear()
```

### car.py — CARAdapter

```python
class CARAdapter(EnvAdapter):
    """Adapter for CAR-bench. Runs agentbeats-run per iteration."""

    CAR_BENCH_TESTED_COMMIT = "TBD"  # set after first successful run

    def setup(self, config):
        self._model = config.get("model", "anthropic/claude-haiku-4-5-20251001")
        self._car_bench_path = Path(config["car_bench_path"])  # benchmarks/car-bench
        self._output_dir = Path(config.get("output", f"./runs/car/{int(time.time())}"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._task_type = config.get("task_type", "base")     # base/hallucination/disambiguation
        self._task_split = config.get("task_split", "test")    # train/test
        self._seed = config.get("seed")
        self._verify_submodule()

        # Start our purple agent server (persistent across iterations)
        self._harness = Harness()
        self._purple = CarPurpleAgent(model=self._model, harness=self._harness)
        self._purple_port = self._start_purple_server()

    def list_tasks(self, task_type: str = "base") -> list[str]:
        """Load task IDs from CAR-bench task definitions."""
        # Parse from car-bench data (HuggingFace auto-download)
        ...

    def run_batch(self, agent_state: AgentState, task_ids: list[str]) -> list[Episode]:
        """Run one learning iteration: selected tasks via agentbeats-run."""
        # 1. Update purple agent's harness state + clear previous sessions
        self._purple.update_harness(agent_state.harness)
        self._purple.clear_all_sessions()

        # 2. Generate scenario.toml for this batch
        scenario = self._generate_scenario(task_ids)
        iter_dir = self._output_dir / f"iter_{self._iteration_count}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = iter_dir / "scenario.toml"
        scenario_path.write_text(scenario)

        # 3. Run agentbeats-run (starts green, connects to our purple)
        results_path = iter_dir / "results.json"
        try:
            result = subprocess.run(
                ["agentbeats-run", str(scenario_path), "--show-logs",
                 "--output", str(results_path)],
                cwd=str(self._car_bench_path),
                capture_output=True, text=True, timeout=600,
            )
            (iter_dir / "green_agent.log").write_text(
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        except subprocess.TimeoutExpired:
            log.error("agentbeats-run timed out")
            return [self._make_failed_episode(tid, "timeout") for tid in task_ids]

        # 4. Handle non-zero exit: still try to parse partial results
        if result.returncode != 0:
            log.warning("agentbeats-run exited %d, attempting partial result parse",
                        result.returncode)

        # 5. Parse results.json (robust: handles missing file, partial writes)
        try:
            raw_results = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error("Failed to parse results: %s", e)
            return [self._make_failed_episode(tid, "parse_error") for tid in task_ids]

        # Save raw results verbatim
        (iter_dir / "raw_results.json").write_text(json.dumps(raw_results, indent=2))

        # 5. Map to Episodes (results.json is the source of truth for task_id,
        #    reward, trajectory — no context_id mapping needed)
        episodes = []
        detailed = raw_results.get("detailed_results_by_split", {})
        for task_type_results in detailed.values():
            for task_result in task_type_results:
                episode = self._map_to_episode(task_result)
                episodes.append(episode)

        # Validate: expected vs actual episode count
        if len(episodes) != len(task_ids):
            log.warning("Expected %d episodes, got %d", len(task_ids), len(episodes))
            # Create failed episodes for missing tasks
            found_ids = {ep.task_id for ep in episodes}
            for tid in task_ids:
                if f"car:{tid}" not in found_ids:
                    episodes.append(self._make_failed_episode(tid, "missing_result"))

        self._iteration_count += 1
        return episodes

    def _generate_scenario(self, task_ids: list[str]) -> str:
        """Generate scenario.toml pointing green at our purple server."""
        # Group task_ids by type (base_0 → base, hallucination_3 → hallucination)
        by_type: dict[str, list[str]] = {}
        for tid in task_ids:
            task_type = tid.rsplit("_", 1)[0]  # "base_0" → "base"
            by_type.setdefault(task_type, []).append(tid)

        # Build task filter lines
        filter_lines = []
        for task_type, ids in by_type.items():
            filter_lines.append(
                f'tasks_{task_type}_task_id_filter = {json.dumps(ids)}'
            )

        return f'''[green_agent]
endpoint = "http://127.0.0.1:8081"
cmd = "python src/green_car_bench_agent/server.py --host 127.0.0.1 --port 8081"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:{self._purple_port}"

[config]
task_split = "{self._task_split}"
{chr(10).join(filter_lines)}
num_trials = 1
max_steps = 50
'''

    def _map_to_episode(self, task_result: dict) -> Episode:
        """Map CAR detailed result to lfx Episode."""
        car_task_id = task_result["task_id"]
        task_id = f"car:{car_task_id}"

        # Map rewards
        signals, breakdown = map_car_scores(
            task_result.get("reward_info", {}),
            task_reward=task_result["reward"],
        )

        # Build Episode from CAR trajectory + captured messages
        # CAR trajectory has the message sequence; our purple captured the full
        # conversation including injected harness prompt
        messages = self._build_episode_messages(task_result.get("trajectory", []))

        summary = EpisodeSummary(signals=signals, score_breakdown=breakdown)
        return Episode(
            id=uuid4().hex,
            state_id=self._current_state_id or "",
            task_id=task_id,
            bench="car",
            model=self._model,
            messages=messages,
            summary=summary,
            metadata={
                "car_raw_reward": task_result["reward"],
                "car_agent_cost": task_result.get("total_agent_cost"),
                "car_llm_latency_ms": task_result.get("total_llm_latency_ms"),
            },
        )
```

### Reward Mapping

```python
DEFAULT_CAR_WEIGHTS = {
    "r_actions_final": 0.30,
    "r_actions_intermediate": 0.20,
    "r_tool_subset": 0.15,
    "r_tool_execution_errors": 0.15,
    "r_policy_errors": 0.10,
    "r_user_end_conversation": 0.10,
}

def map_car_scores(reward_info: dict, task_reward: float,
                   weights: dict = DEFAULT_CAR_WEIGHTS
                   ) -> tuple[dict[str, RewardSignal], dict]:
    """Map CAR metrics to lfx RewardSignals."""
    signals = {}
    breakdown = {}

    # Use task_reward as primary outcome (binary, from car-bench scoring)
    signals["outcome"] = RewardSignal(
        name="outcome", value=task_reward * 2.0 - 1.0, confidence=1.0
    )

    # Per-metric signals from reward_info (if available)
    for name, weight in weights.items():
        val = reward_info.get(name)
        if val is not None:
            val = max(0.0, min(1.0, float(val)))
            signals[name] = RewardSignal(
                name=name, value=val * 2.0 - 1.0,
                confidence=1.0 if val in (0.0, 1.0) else 0.8
            )
            breakdown[name] = val

    # Unknown metrics stored but not mapped
    for k, v in reward_info.items():
        if k not in weights:
            breakdown[k] = v

    return signals, breakdown
```

### Learning Loop Integration

Minor change to `loop.py` — if adapter has `run_batch()`, use it:

```python
# In learning_loop():
if hasattr(adapter, 'run_batch') and callable(adapter.run_batch):
    task_ids = [t for t in random.sample(tasks, min(n_episodes, len(tasks)))]
    episodes = adapter.run_batch(agent_state, task_ids)
else:
    # existing per-task logic
    ...
```

### CLI Extension

```bash
lfx run car [--model MODEL] [--iterations N] [--episodes N] \
            [--task-type TYPE] [--task-split SPLIT] [--output DIR] [--seed SEED]
```

- `--model`: litellm string, default `anthropic/claude-haiku-4-5-20251001`
- `--iterations`: default 5
- `--episodes`: per iteration, default 10
- `--task-type`: `base`|`hallucination`|`disambiguation`, default `base`
- `--task-split`: `train`|`test`, default `test`
- `--output`: default `./runs/car/<timestamp>/`
- `--seed`: for reproducible task sampling (lfx side; CAR has its own RNG)

---

## Error Handling

### Episode-level (never abort run)

| Failure | Action | Metadata |
|---------|--------|----------|
| agentbeats-run timeout (600s) | Fail all episodes in batch | error=timeout |
| agentbeats-run non-zero exit | Parse partial results if available, fail rest | error=process_error |
| Score parse failure | Log raw, fail episode | error=parse_error |
| Missing task in results | Fail that episode | error=missing_result |

### Run-level

| Failure | Action |
|---------|--------|
| Purple server won't start | Abort with error |
| Submodule not found/wrong commit | Abort with instructions |
| All episodes in iteration failed | Skip fb/optim, continue |

### Process Lifecycle

- Purple server: started in setup(), persistent across iterations, stopped in teardown()
- Green agent: started/stopped by agentbeats-run per batch (subprocess managed by agentbeats)
- Context manager: `async with CARAdapter(config) as adapter:`
- atexit fallback: kill purple server

---

## Persistence

- `{output}/harness_state_{iter}.json` — harness via to_dict()
- `{output}/rewards.jsonl` — per iteration metrics
- `{output}/raw_results/iter_{N}.json` — verbatim CAR results.json per iteration
- `{output}/run_config.json` — CLI args, model, seed, commit hash
- No A2A traffic logged by default. `--verbose` for DEBUG level.

---

## Dependencies to Add

- `a2a-sdk[http-server]>=0.3.5` (for A2A server types, or implement raw)
- `httpx` (health checks)
- `uvicorn` (purple agent server)

---

## Testing

- **Unit**: `map_car_scores()` edge cases, scenario.toml generation, message parsing
- **Integration**: mock agentbeats-run (write canned results.json), full loop iteration
- **E2E**: real CAR-bench, 1 iteration, 3 tasks (EXPERIMENT CHECKLIST required)

---

## Implementation Steps

1. Add car-bench submodule under benchmarks/
2. Implement `_car_purple.py` — thin A2A server with harness injection
3. Implement `car.py` — CARAdapter with scenario generation + results parsing
4. Implement `map_car_scores()` reward mapping
5. Extend `loop.py` with `run_batch()` support
6. Extend `cli.py` with `lfx run car` command
7. Unit + integration tests
8. Manual E2E validation with real CAR-bench
