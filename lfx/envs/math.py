"""MATH/AIME-style competition math environment.

Provides a set of built-in competition math problems with deterministic
exact-match scoring.  No external dependencies required.
"""

from __future__ import annotations

import re

from lfx.core.env import EvalResult, Sample


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    r"""Extract the final answer from an LLM response.

    Strategy (in priority order):
    1. Last ``\boxed{...}`` match (handles nested braces).
    2. "answer is X" pattern (captures the token after "is").
    3. Last number on the last non-empty line.
    4. Full response text, stripped.
    """
    # 1. \boxed{...}  — grab the last one, handling nested braces
    boxed_matches: list[str] = []
    for m in re.finditer(r"\\boxed\{", response):
        start = m.end()
        depth = 1
        pos = start
        while pos < len(response) and depth > 0:
            if response[pos] == "{":
                depth += 1
            elif response[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            boxed_matches.append(response[start : pos - 1])
    if boxed_matches:
        return boxed_matches[-1]

    # 2. "answer is ..."
    m = re.search(r"(?i)\banswer\s+is\s+([^\s.,;]+)", response)
    if m:
        return m.group(1)

    # 3. Last number on last non-empty line
    lines = [l for l in response.splitlines() if l.strip()]
    if lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", lines[-1])
        if nums:
            return nums[-1]

    # 4. Fallback
    return response.strip()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_answer(answer: str) -> str:
    r"""Normalize an answer string for comparison.

    * strip whitespace
    * lowercase
    * remove ``$`` signs
    * remove ``\text{...}`` wrappers (keeps inner text)
    * collapse remaining whitespace
    """
    s = answer.strip().lower()
    s = s.replace("$", "")
    # Remove \text{...} but keep the content inside
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s


# ---------------------------------------------------------------------------
# Built-in problems
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict] = [
    # --- Arithmetic (easy) ---
    {
        "question": "What is 17 + 28?",
        "answer": "45",
        "difficulty": "easy",
        "source": "arithmetic",
    },
    {
        "question": "What is 144 / 12?",
        "answer": "12",
        "difficulty": "easy",
        "source": "arithmetic",
    },
    {
        "question": "What is 15 * 13?",
        "answer": "195",
        "difficulty": "easy",
        "source": "arithmetic",
    },
    # --- Algebra (easy/medium) ---
    {
        "question": "Solve for x: 3x + 7 = 22.",
        "answer": "5",
        "difficulty": "easy",
        "source": "algebra",
    },
    {
        "question": "Solve for x: x^2 - 5x + 6 = 0. Give the larger root.",
        "answer": "3",
        "difficulty": "easy",
        "source": "algebra",
    },
    {
        "question": "If f(x) = 2x + 3, what is f(7)?",
        "answer": "17",
        "difficulty": "easy",
        "source": "algebra",
    },
    {
        "question": "What is the sum of the roots of x^2 - 7x + 12 = 0?",
        "answer": "7",
        "difficulty": "easy",
        "source": "algebra",
    },
    {
        "question": "Simplify: (x^3 * x^4). Express as x^n; what is n?",
        "answer": "7",
        "difficulty": "easy",
        "source": "algebra",
    },
    # --- Number theory (easy/medium) ---
    {
        "question": "What is the greatest common divisor of 36 and 48?",
        "answer": "12",
        "difficulty": "easy",
        "source": "number_theory",
    },
    {
        "question": "What is the least common multiple of 6 and 8?",
        "answer": "24",
        "difficulty": "easy",
        "source": "number_theory",
    },
    {
        "question": "How many positive divisors does 60 have?",
        "answer": "12",
        "difficulty": "medium",
        "source": "number_theory",
    },
    {
        "question": "What is the remainder when 2^10 is divided by 7?",
        "answer": "2",
        "difficulty": "medium",
        "source": "number_theory",
    },
    # --- Combinatorics (easy/medium) ---
    {
        "question": "How many ways can you choose 3 items from a set of 5? (i.e., C(5,3))",
        "answer": "10",
        "difficulty": "easy",
        "source": "combinatorics",
    },
    {
        "question": "How many ways can 4 distinct books be arranged on a shelf?",
        "answer": "24",
        "difficulty": "easy",
        "source": "combinatorics",
    },
    {
        "question": "What is 7! / 5! ?",
        "answer": "42",
        "difficulty": "easy",
        "source": "combinatorics",
    },
    {
        "question": "How many subsets does a set with 4 elements have?",
        "answer": "16",
        "difficulty": "easy",
        "source": "combinatorics",
    },
    # --- Geometry / misc (medium) ---
    {
        "question": "A right triangle has legs of length 5 and 12. What is the length of the hypotenuse?",
        "answer": "13",
        "difficulty": "easy",
        "source": "geometry",
    },
    {
        "question": "What is the area of a triangle with base 10 and height 7?",
        "answer": "35",
        "difficulty": "easy",
        "source": "geometry",
    },
    {
        "question": "What is the sum of the interior angles (in degrees) of a hexagon?",
        "answer": "720",
        "difficulty": "medium",
        "source": "geometry",
    },
    {
        "question": "What is the value of the sum 1 + 2 + 3 + ... + 20?",
        "answer": "210",
        "difficulty": "easy",
        "source": "arithmetic",
    },
]


# ---------------------------------------------------------------------------
# MathEnvironment
# ---------------------------------------------------------------------------

class MathEnvironment:
    """MATH/AIME-style environment with built-in problems and exact-match scoring."""

    def __init__(self, problems: list[dict] | None = None) -> None:
        self._problems = problems if problems is not None else _BUILTIN_PROBLEMS

    # -- TaskEnvironment interface ------------------------------------------

    def get_tasks(self) -> list[Sample]:
        """Return a :class:`Sample` for each problem."""
        return [
            Sample(
                question=p["question"],
                ground_truth=p["answer"],
                metadata={
                    k: v for k, v in p.items() if k not in ("question", "answer")
                },
            )
            for p in self._problems
        ]

    def evaluate(self, sample: Sample, response: str) -> EvalResult:
        """Score *response* by exact match (after normalization).

        Returns score 1.0 for correct, 0.0 for incorrect.
        """
        extracted = extract_answer(response)
        expected = sample.ground_truth or ""

        norm_extracted = _normalize_answer(extracted)
        norm_expected = _normalize_answer(expected)

        correct = norm_extracted == norm_expected

        if correct:
            feedback = f"Correct! Expected '{expected}', got '{extracted}'."
        else:
            feedback = f"Incorrect. Expected '{expected}', got '{extracted}'."

        return EvalResult(
            score=1.0 if correct else 0.0,
            feedback=feedback,
            metrics={"exact_match": 1.0 if correct else 0.0},
        )
