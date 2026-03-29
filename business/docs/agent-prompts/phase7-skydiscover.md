# Agent Prompt: Phase 7 — SkyDiscover AdaEvolve Integration

Implement Phase 7: SkyDiscover AdaEvolve as an enterprise evolution backend for ClawLoop.

IMPORTANT: Work in an ISOLATED GIT WORKTREE to avoid conflicts with parallel work on main.

## Setup worktree FIRST

```bash
cd /Users/robertmueller/Desktop/aganthos
git worktree add ../aganthos-skydiscover feat/skydiscover-backend -b feat/skydiscover-backend
```

Then work ONLY in /Users/robertmueller/Desktop/aganthos-skydiscover for all changes.

## Context

ClawLoop has:
- clawloop/core/evolver.py — Evolver interface (evolve(episodes, harness_state, context) -> EvolverResult)
- clawloop/evolvers/local.py — LocalEvolver (community default)
- The Harness uses evolver= parameter internally, Layer Protocol (forward_backward/optim_step) externally

SkyDiscover (github.com/skydiscover-ai/skydiscover, Apache 2.0, Berkeley) provides:
- AdaEvolve: multi-island adaptive search with paradigm breakthrough, UCB island selection
- EvoX: co-evolves the search strategy itself (future Phase 8)
- Clean API: run_discovery(evaluator, search="adaevolve", iterations=N)
- pip install skydiscover (or from git)

We want to wrap SkyDiscover as an enterprise Evolver backend. This code goes in enterprise/ (PRIVATE, never synced to public repo).

## What to build

### 1. SkyDiscover evaluator adapter (enterprise/evolution/backends/skydiscover_evaluator.py)

SkyDiscover needs an evaluator function: evaluate(program_path) -> {"combined_score": float}

For ClawLoop, a "program" is an agent configuration (system prompt + playbook). The evaluator:
- Takes a candidate config file path
- Loads the config (system prompt + playbook JSON)
- Creates a temporary Harness with the candidate config
- Runs ClawLoop's reward pipeline on a batch of tasks via an adapter
- Returns the aggregate reward as combined_score

```python
class ClawLoopEvaluator:
    def __init__(self, adapter, tasks, n_episodes=5):
        self.adapter = adapter  # EnvAdapter (e.g., CARAdapter, EntropicAdapter)
        self.tasks = tasks
        self.n_episodes = n_episodes

    def __call__(self, program_path: str) -> dict:
        # Load candidate config from program_path (JSON with system_prompt + playbook)
        # Create a temporary Harness with the candidate
        # Run adapter.run_batch() or similar to collect episodes
        # Compute mean reward
        # Return {"combined_score": mean_reward, "n_episodes": len(episodes)}
```

### 2. Serialization helpers (enterprise/evolution/backends/skydiscover_utils.py)

harness_to_program(snapshot: HarnessSnapshot, output_path: str) -> str:
  Serialize HarnessSnapshot to a JSON file that SkyDiscover can evolve.
  Format:
  ```json
  {
    "system_prompt": "...",
    "playbook": [{"content": "...", "tags": [...], "helpful": N, "harmful": N}, ...],
    "model": "..."
  }
  ```
  Returns the file path.

program_to_evolver_result(program_path: str, original: HarnessSnapshot) -> EvolverResult:
  Parse the evolved program, diff against original, produce:
  - Insights for playbook changes (add/update/remove)
  - PromptCandidates if system_prompt changed
  - Provenance with backend="skydiscover_adaevolve"

### 3. SkyDiscover Evolver backend (enterprise/evolution/backends/skydiscover_adaevolve.py)

```python
class SkyDiscoverAdaEvolve:
    """Wraps SkyDiscover's AdaEvolve as a ClawLoop Evolver.

    On evolve():
    1. Serialize current HarnessSnapshot to a seed program file
    2. Create a ClawLoopEvaluator with the provided adapter + tasks
    3. Call skydiscover.run_discovery(evaluator, initial_program=seed, search="adaevolve", ...)
    4. Parse the best result back into an EvolverResult
    """

    def __init__(
        self,
        adapter,       # EnvAdapter for evaluation
        tasks: list[str],
        iterations: int = 20,
        model: str = "claude-sonnet-4-6",
        num_islands: int = 2,
        population_size: int = 20,
    ):
        ...

    def evolve(self, episodes, harness_state, context) -> EvolverResult:
        # 1. Write harness_state to temp dir as seed program
        seed_path = harness_to_program(harness_state, self._work_dir)

        # 2. Create evaluator
        evaluator = ClawLoopEvaluator(self._adapter, self._tasks)

        # 3. Run SkyDiscover (this is the expensive part)
        result = run_discovery(
            evaluator=evaluator,
            initial_program=seed_path,
            search="adaevolve",
            model=self._model,
            iterations=self._iterations,
        )

        # 4. Parse result back to EvolverResult
        return program_to_evolver_result(result.best_program, harness_state)

    def name(self) -> str:
        return "skydiscover_adaevolve"
```

### 4. Tests

enterprise/tests/test_skydiscover_evaluator.py — mock adapter, verify score returned
enterprise/tests/test_skydiscover_utils.py — roundtrip serialization (harness → program → evolver_result)
enterprise/tests/test_skydiscover_adaevolve.py — mock run_discovery, verify EvolverResult shape

Use mocks for SkyDiscover calls (don't run real evolution in tests). Test the serialization and result parsing thoroughly.

### 5. Dependencies

Install SkyDiscover in the worktree venv:
```bash
pip install git+https://github.com/skydiscover-ai/skydiscover.git
```

Do NOT add to main pyproject.toml. This is enterprise-only.

## Commit sequence

1. `feat: serialization helpers for SkyDiscover harness programs`
2. `feat: ClawLoop evaluator adapter for SkyDiscover`
3. `feat: SkyDiscover AdaEvolve as enterprise Evolver backend`

## Rules
- Commit format: `fix:`, `feat:`, or `chore:` + one line. NO Co-Authored-By. NO multi-line bodies.
- Work ONLY in /Users/robertmueller/Desktop/aganthos-skydiscover (the worktree)
- All code goes in enterprise/ (PRIVATE) — NOT in clawloop/
- Do NOT modify any files in clawloop/ — only import from it
- Run pytest enterprise/tests/ -v after each chunk
- When done, push: git push origin feat/skydiscover-backend
- Do NOT merge — just push the branch
