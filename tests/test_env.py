"""Tests for clawloop.core.env — Sample, EvalResult, TaskEnvironment, StaticTaskEnvironment."""

from __future__ import annotations

from clawloop.core.env import EvalResult, Sample, StaticTaskEnvironment, TaskEnvironment


class TestSample:
    def test_creation(self) -> None:
        s = Sample(question="What is 2+2?")
        assert s.question == "What is 2+2?"
        assert s.context == ""
        assert s.ground_truth is None
        assert s.metadata == {}

    def test_with_metadata(self) -> None:
        s = Sample(
            question="Translate hello",
            context="English to French",
            ground_truth="bonjour",
            metadata={"difficulty": "easy", "category": "translation"},
        )
        assert s.question == "Translate hello"
        assert s.context == "English to French"
        assert s.ground_truth == "bonjour"
        assert s.metadata == {"difficulty": "easy", "category": "translation"}


class TestEvalResult:
    def test_creation(self) -> None:
        r = EvalResult(score=0.85)
        assert r.score == 0.85

    def test_defaults(self) -> None:
        r = EvalResult(score=1.0)
        assert r.feedback == ""
        assert r.metrics == {}

    def test_with_metrics(self) -> None:
        r = EvalResult(
            score=0.75,
            feedback="Partially correct",
            metrics={"precision": 0.8, "recall": 0.7},
        )
        assert r.score == 0.75
        assert r.feedback == "Partially correct"
        assert r.metrics == {"precision": 0.8, "recall": 0.7}


class TestStaticTaskEnvironment:
    def test_get_tasks(self) -> None:
        samples = [
            Sample(question="Q1", ground_truth="A1"),
            Sample(question="Q2", ground_truth="A2"),
        ]
        env = StaticTaskEnvironment(
            tasks=samples,
            evaluate_fn=lambda s, r: EvalResult(score=1.0),
        )
        assert env.get_tasks() == samples

    def test_evaluate_calls_fn(self) -> None:
        sample = Sample(question="What is 2+2?", ground_truth="4")
        calls: list[tuple[Sample, str]] = []

        def score_fn(s: Sample, response: str) -> EvalResult:
            calls.append((s, response))
            return EvalResult(
                score=1.0 if response == s.ground_truth else 0.0,
                feedback="checked",
            )

        env = StaticTaskEnvironment(tasks=[sample], evaluate_fn=score_fn)
        result = env.evaluate(sample, "4")

        assert len(calls) == 1
        assert calls[0] == (sample, "4")
        assert result.score == 1.0
        assert result.feedback == "checked"
