"""Tests for the MATH/AIME environment."""

from __future__ import annotations

import pytest

from clawloop.core.env import EvalResult, Sample
from clawloop.envs.math import MathEnvironment, extract_answer


# ---------------------------------------------------------------------------
# TestExtractAnswer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """Tests for the extract_answer helper."""

    def test_boxed_answer(self) -> None:
        response = "The answer is obvious.\n$$\\boxed{42}$$"
        assert extract_answer(response) == "42"

    def test_boxed_fraction(self) -> None:
        response = "So we get \\boxed{\\frac{3}{4}}."
        assert extract_answer(response) == "\\frac{3}{4}"

    def test_plain_number_last_line(self) -> None:
        response = "Let me think step by step.\nFirst we add.\nThen we get 17"
        assert extract_answer(response) == "17"

    def test_answer_is_prefix(self) -> None:
        response = "After working through the problem, the answer is 256."
        assert extract_answer(response) == "256"

    def test_no_answer_returns_full(self) -> None:
        response = "I have no idea"
        assert extract_answer(response) == "I have no idea"

    def test_negative_number(self) -> None:
        response = "Calculating:\n-7"
        assert extract_answer(response) == "-7"


# ---------------------------------------------------------------------------
# TestMathEnvironment
# ---------------------------------------------------------------------------


class TestMathEnvironment:
    """Tests for the MathEnvironment class."""

    def test_get_tasks_returns_samples(self) -> None:
        env = MathEnvironment()
        tasks = env.get_tasks()
        assert len(tasks) >= 20
        for t in tasks:
            assert isinstance(t, Sample)
            assert t.question
            assert t.ground_truth is not None

    def test_evaluate_correct(self) -> None:
        env = MathEnvironment()
        sample = Sample(question="What is 2+2?", ground_truth="4")
        result = env.evaluate(sample, "The answer is \\boxed{4}.")
        assert isinstance(result, EvalResult)
        assert result.score == 1.0

    def test_evaluate_incorrect(self) -> None:
        env = MathEnvironment()
        sample = Sample(question="What is 2+2?", ground_truth="4")
        result = env.evaluate(sample, "I think it's 5.")
        assert result.score == 0.0

    def test_evaluate_feedback_is_informative(self) -> None:
        env = MathEnvironment()
        sample = Sample(question="What is 2+2?", ground_truth="4")
        result = env.evaluate(sample, "Hmm, maybe 5?")
        assert "4" in result.feedback
        assert "5" in result.feedback
