"""Playbook Learning Demo — watch the harness learn from failure episodes.

Run with real LLM (requires API key):
    python examples/playbook_demo.py

Run in dry-run mode (no API calls, finishes in seconds):
    python examples/playbook_demo.py --dry-run

This script walks through the ClawLoop learning loop step by step,
showing exactly what happens at each stage:

  1. Agent fails at math problems
  2. Reflector (LLM) analyses the failures and extracts strategies
  3. Strategies become playbook entries
  4. Playbook entries get injected into the system prompt
  5. Over iterations, entries accumulate helpful/harmful scores
  6. Bad entries decay and get pruned, good ones persist

The playbook is the agent's evolving memory — it learns general
strategies from specific failures.
"""

import argparse
import json
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Local dev: allow running from the repo root without pip install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.types import Datum
from clawloop.harness_backends.local import LocalEvolver
from clawloop.learning_layers.harness import Harness, PlaybookEntry
from clawloop.llm import LiteLLMClient, MockLLMClient

# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playbook Learning Demo — watch the harness learn from failure episodes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock LLM clients (no API calls, fast)",
    )
    return parser.parse_args()


# ── Mock reflector responses for --dry-run ──────────────────────────────


def _build_mock_reflector_responses() -> list[str]:
    """Canned reflector outputs that drive the demo without real LLM calls.

    The reflector is called three times during the demo:
      1. After initial failures  — produces math-strategy insights
      2. After success episodes  — may produce refinement insights
      3. After another failure   — may produce additional insights
    """

    def _insight_json(*insights: dict) -> str:
        return json.dumps(list(insights))

    return [
        # Call 1: analyse initial math failures -> two strategy insights
        _insight_json(
            {
                "action": "add",
                "content": "Show intermediate calculation steps for arithmetic",
                "target_entry_id": None,
                "tags": ["math", "strategy"],
                "source_episode_ids": [],
            },
            {
                "action": "add",
                "content": "Double-check multiplication by estimating the expected magnitude first",
                "target_entry_id": None,
                "tags": ["math", "verification"],
                "source_episode_ids": [],
            },
        ),
        # Call 2: after success episodes -> no new insights needed
        "[]",
        # Call 3: after another failure -> one new insight
        _insight_json(
            {
                "action": "add",
                "content": "For division and root problems, verify by multiplying the result back",
                "target_entry_id": None,
                "tags": ["math", "verification"],
                "source_episode_ids": [],
            },
        ),
        # Extra responses in case of additional reflect calls
        "[]",
        "[]",
        "[]",
    ]


# ── Helpers ──────────────────────────────────────────────────────────────


def banner(text: str) -> None:
    width = 70
    print()
    print("═" * width)
    print(f"  {text}")
    print("═" * width)


def show_playbook(harness: Harness, bench: str = "math") -> None:
    """Pretty-print the current playbook state."""
    entries = harness.playbook.entries
    if not entries:
        print("  (empty)")
        return

    for i, e in enumerate(entries, 1):
        tags = ", ".join(e.tags) if e.tags else "none"
        score = f"+{e.helpful}/-{e.harmful}"
        eff = f"eff={e.effective_score():.2f}"
        print(f"  [{i}] {e.id}  score={score}  {eff}  tags=[{tags}]")
        # Show structured fields if present
        if e.name:
            print(f"      Name: {e.name}")
        if e.description:
            print(f"      When: {e.description}")
        if e.anti_patterns:
            print(f"      Anti-pattern: {e.anti_patterns}")
        # Content (wrapped)
        wrapped = textwrap.fill(
            e.content, width=64, initial_indent="      ", subsequent_indent="      "
        )
        print(wrapped)
        print()


def show_prompt(harness: Harness, bench: str = "math") -> None:
    """Show the full system prompt the agent would see."""
    prompt = harness.system_prompt(bench)
    print(prompt)


def make_episode(
    task_id: str,
    question: str,
    answer: str,
    reward: float,
    bench: str = "math",
) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="demo",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="system", content="You are a math solver."),
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


# ── Main demo ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("=== DRY-RUN MODE (mock LLM clients, no API calls) ===\n")
        llm = MockLLMClient(responses=_build_mock_reflector_responses())
    else:
        # Connect to an LLM — configure via environment variables:
        #   CLAWLOOP_MODEL    — litellm model string (e.g. "gemini/gemini-2.0-flash-lite")
        #   CLAWLOOP_API_BASE — optional API base URL
        #   CLAWLOOP_API_KEY  — optional API key (or use provider-specific env vars)
        llm = LiteLLMClient(
            model=os.environ.get("CLAWLOOP_MODEL", "gemini/gemini-2.0-flash-lite"),
            api_base=os.environ.get("CLAWLOOP_API_BASE") or None,
            api_key=os.environ.get("CLAWLOOP_API_KEY") or None,
        )

    # ┌─────────────────────────────────────────────────────────┐
    # │  PART 1: The basics — Reflector produces insights       │
    # └─────────────────────────────────────────────────────────┘

    banner("PART 1: Learning from failures")
    print("""
    The agent tried 3 math problems and got them all wrong.
    The Reflector (an LLM) will analyse these failures and
    extract reusable strategies — NOT task-specific answers.
    """)

    # Set up harness with evolver
    reflector = Reflector(client=llm, config=ReflectorConfig(reflection_batch_size=5))
    evolver = LocalEvolver(reflector=reflector)
    harness = Harness(
        system_prompts={"math": "You are a math problem solver."},
        evolver=evolver,
    )

    print("  Initial system prompt:")
    print(f"  > {harness.system_prompt('math')}")
    print()
    print("  Initial playbook:")
    show_playbook(harness)

    # Simulate 3 failure episodes
    failures = [
        make_episode("q1", "What is 17 + 28?", "The answer is 43.", 0.1),
        make_episode("q2", "What is 15 × 13?", "The answer is 165.", 0.1),
        make_episode("q3", "What is 144 ÷ 12?", "The answer is 14.", 0.0),
    ]

    print("  Agent's failed episodes:")
    for ep in failures:
        q = ep.messages[1].content
        a = ep.messages[2].content
        r = ep.summary.total_reward
        print(f"    Q: {q}")
        print(f"    A: {a}  (reward={r:.1f})")
    print()

    # Step 1: forward_backward — accumulates signals without mutating state
    print("  Running forward_backward (Reflector analyses traces)...")
    from clawloop.core.evolver import EvolverContext

    harness.set_evolver_context(EvolverContext())
    datum = Datum(episodes=failures)
    fb_result = harness.forward_backward(datum).result()

    print(f"  Result: {fb_result.metrics.get('insights_generated', 0)} insights generated")
    print()
    print("  Pending insights (not yet applied):")
    for i, insight in enumerate(harness._pending.insights, 1):
        tags = ", ".join(insight.tags) if insight.tags else "none"
        print(f"    {i}. [{insight.action}] [{tags}]")
        wrapped = textwrap.fill(
            insight.content, width=60, initial_indent="       ", subsequent_indent="       "
        )
        print(wrapped)
    print()

    # Note: playbook is STILL empty — insights are pending
    print("  Playbook after forward_backward (still empty — two-phase protocol):")
    show_playbook(harness)

    # Step 2: optim_step — atomically applies all pending signals
    print("  Running optim_step (applies insights to playbook)...")
    optim_result = harness.optim_step().result()
    print(f"  Result: {optim_result.updates_applied} updates applied")
    print()

    print("  Playbook after optim_step:")
    show_playbook(harness)

    print("  System prompt NOW (base + playbook injected):")
    print("  ─" * 35)
    show_prompt(harness)
    print("  ─" * 35)

    # ┌─────────────────────────────────────────────────────────┐
    # │  PART 2: Helpful/harmful scoring                        │
    # └─────────────────────────────────────────────────────────┘

    banner("PART 2: Entries earn helpful/harmful scores")
    print("""
    When the agent succeeds, entries that were active get +helpful.
    When the agent fails, active entries get +harmful.
    Over time, bad strategies decay and good ones persist.
    """)

    # Simulate some success episodes (the strategies helped!)
    successes = [
        make_episode("q4", "What is 23 + 19?", "Step by step: 23+19=42. The answer is 42.", 0.9),
        make_episode("q5", "What is 8 × 7?", "8×7 = 56. The answer is 56.", 0.85),
    ]

    print("  Success episodes (strategies helped):")
    for ep in successes:
        print(f"    Q: {ep.messages[1].content}  →  reward={ep.summary.total_reward:.1f}")

    harness.set_evolver_context(EvolverContext())
    harness.forward_backward(Datum(episodes=successes))
    harness.optim_step()

    print()
    print("  Playbook after successes (helpful counts increased):")
    show_playbook(harness)

    # Now simulate a failure (the strategy didn't help this time)
    more_failures = [
        make_episode("q6", "What is √289?", "The answer is 15.", 0.0),
    ]

    print("  Failure episode (strategy didn't help):")
    for ep in more_failures:
        print(f"    Q: {ep.messages[1].content}  →  reward={ep.summary.total_reward:.1f}")

    harness.set_evolver_context(EvolverContext())
    harness.forward_backward(Datum(episodes=more_failures))
    harness.optim_step()

    print()
    print("  Playbook after failure (harmful counts increased):")
    show_playbook(harness)

    # ┌─────────────────────────────────────────────────────────┐
    # │  PART 3: Structured (skill) entries                     │
    # └─────────────────────────────────────────────────────────┘

    banner("PART 3: Structured skill entries")
    print("""
    Entries can also be structured skills with name, description,
    and anti-patterns. These render differently in the prompt —
    as formatted skill blocks instead of flat text.

    The PlaybookCurator can promote flat entries to structured skills,
    but here we add one manually to show the difference.
    """)

    # Add a structured entry manually
    structured_entry = PlaybookEntry(
        id=PlaybookEntry.new_id("skill"),
        content="Break the problem into smaller sub-problems. Solve each independently, then combine.",
        name="Divide and Conquer",
        description="Complex multi-step math problems",
        anti_patterns="Trying to solve everything in one mental step",
        category="strategy",
        tags=["math", "strategy"],
        helpful=5,
        harmful=0,
    )
    harness.playbook.add(structured_entry)

    print("  Playbook with structured entry added:")
    show_playbook(harness)

    print("  How it renders in the system prompt:")
    print("  ─" * 35)
    print(harness.playbook.render())
    print("  ─" * 35)

    # ┌─────────────────────────────────────────────────────────┐
    # │  PART 4: Selective retrieval by tags                    │
    # └─────────────────────────────────────────────────────────┘

    banner("PART 4: Tag-based selective retrieval")
    print("""
    When the agent faces a specific task category, only matching
    playbook entries are included in the prompt. This keeps the
    context focused and avoids irrelevant strategies.
    """)

    # Add a non-math entry
    coding_entry = PlaybookEntry(
        id=PlaybookEntry.new_id(),
        content="Always validate input types before processing to avoid runtime errors.",
        tags=["coding", "defensive-programming"],
        helpful=3,
    )
    harness.playbook.add(coding_entry)

    print("  All entries:")
    show_playbook(harness)

    print("  Prompt with tags={'math'}  (only math entries):")
    print("  ─" * 35)
    print(harness.system_prompt("math", task_tags={"math"}))
    print("  ─" * 35)
    print()
    print("  Prompt with tags={'coding'}  (only coding entries):")
    print("  ─" * 35)
    print(harness.system_prompt("math", task_tags={"coding"}))
    print("  ─" * 35)

    # ┌─────────────────────────────────────────────────────────┐
    # │  PART 5: JSON export                                    │
    # └─────────────────────────────────────────────────────────┘

    banner("PART 5: Serialization")
    print("""
    The entire playbook can be serialized to JSON for persistence,
    analysis, or transfer to another agent.
    """)
    print(json.dumps(harness.playbook.to_dict(), indent=2, default=str))

    banner("DONE")
    print(f"""
    The playbook is the agent's evolving memory:

    • Reflector extracts strategies from episode traces (LLM call)
    • Entries accumulate helpful/harmful scores from outcomes
    • Bad entries decay over time (decay_rate) and get pruned
    • Tags enable selective retrieval per task category
    • Structured entries render as skill blocks in prompts
    • The Evolver orchestrates all of this inside forward_backward

    Current state: {len(harness.playbook.entries)} entries,
    version={harness.playbook_version}, generation={harness.playbook_generation}
    """)


if __name__ == "__main__":
    main()
