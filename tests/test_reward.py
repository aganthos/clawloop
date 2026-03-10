"""Tests for lfx.core.reward and EpisodeSummary signal integration."""

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.reward import RewardExtractor, RewardPipeline, RewardSignal


# ── RewardSignal tests ──────────────────────────────────────────────────


class TestRewardSignal:
    def test_creation(self) -> None:
        sig = RewardSignal(name="outcome", value=0.5, confidence=0.9)
        assert sig.name == "outcome"
        assert sig.value == 0.5
        assert sig.confidence == 0.9

    def test_frozen(self) -> None:
        sig = RewardSignal(name="outcome", value=0.5, confidence=0.9)
        with pytest.raises(AttributeError):
            sig.value = 0.8  # type: ignore[misc]

    def test_value_clamped_high(self) -> None:
        sig = RewardSignal(name="x", value=5.0, confidence=0.5)
        assert sig.value == 1.0

    def test_value_clamped_low(self) -> None:
        sig = RewardSignal(name="x", value=-3.0, confidence=0.5)
        assert sig.value == -1.0

    def test_confidence_clamped_high(self) -> None:
        sig = RewardSignal(name="x", value=0.0, confidence=2.0)
        assert sig.confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        sig = RewardSignal(name="x", value=0.0, confidence=-0.5)
        assert sig.confidence == 0.0

    def test_boundary_values(self) -> None:
        sig = RewardSignal(name="x", value=1.0, confidence=1.0)
        assert sig.value == 1.0
        assert sig.confidence == 1.0
        sig2 = RewardSignal(name="x", value=-1.0, confidence=0.0)
        assert sig2.value == -1.0
        assert sig2.confidence == 0.0


# ── EpisodeSummary tests ────────────────────────────────────────────────


class TestEpisodeSummaryEffectiveReward:
    def test_empty_signals_returns_zero(self) -> None:
        s = EpisodeSummary()
        assert s.effective_reward() == 0.0

    def test_outcome_signal(self) -> None:
        s = EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=0.6, confidence=1.0)},
        )
        assert s.effective_reward() == pytest.approx(0.6)

    def test_user_overrides_outcome(self) -> None:
        s = EpisodeSummary(
            signals={
                "outcome": RewardSignal(name="outcome", value=0.6, confidence=1.0),
                "user": RewardSignal(name="user", value=-0.5, confidence=1.0),
            },
        )
        assert s.effective_reward() == pytest.approx(-0.5)

    def test_execution_high_confidence(self) -> None:
        s = EpisodeSummary(
            signals={
                "execution": RewardSignal(name="execution", value=0.8, confidence=0.9),
            },
        )
        assert s.effective_reward() == pytest.approx(0.8)

    def test_execution_low_confidence_falls_through(self) -> None:
        s = EpisodeSummary(
            signals={
                "execution": RewardSignal(name="execution", value=0.8, confidence=0.5),
            },
        )
        assert s.effective_reward() == 0.0

    def test_execution_low_confidence_falls_to_judge(self) -> None:
        s = EpisodeSummary(
            signals={
                "execution": RewardSignal(name="execution", value=0.8, confidence=0.5),
                "judge": RewardSignal(name="judge", value=0.3, confidence=0.8),
            },
        )
        assert s.effective_reward() == pytest.approx(0.3)

    def test_priority_order_full(self) -> None:
        """user > outcome > execution > judge."""
        signals = {
            "judge": RewardSignal(name="judge", value=0.1, confidence=1.0),
            "execution": RewardSignal(name="execution", value=0.2, confidence=0.9),
            "outcome": RewardSignal(name="outcome", value=0.3, confidence=1.0),
            "user": RewardSignal(name="user", value=0.4, confidence=1.0),
        }
        s = EpisodeSummary(signals=signals)
        assert s.effective_reward() == pytest.approx(0.4)  # user wins

    def test_execution_boundary_confidence_0_7(self) -> None:
        """Confidence exactly 0.7 should qualify."""
        s = EpisodeSummary(
            signals={
                "execution": RewardSignal(name="execution", value=0.5, confidence=0.7),
            },
        )
        assert s.effective_reward() == pytest.approx(0.5)


class TestEpisodeSummaryNormalizedReward:
    def test_maps_minus_one_to_zero(self) -> None:
        s = EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=-1.0, confidence=1.0)},
        )
        assert s.normalized_reward() == pytest.approx(0.0)

    def test_maps_plus_one_to_one(self) -> None:
        s = EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=1.0, confidence=1.0)},
        )
        assert s.normalized_reward() == pytest.approx(1.0)

    def test_maps_zero_to_half(self) -> None:
        s = EpisodeSummary()
        assert s.normalized_reward() == pytest.approx(0.5)

    def test_maps_arbitrary(self) -> None:
        s = EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=0.6, confidence=1.0)},
        )
        assert s.normalized_reward() == pytest.approx(0.8)


class TestEpisodeSummaryTotalRewardCompat:
    def test_setter_and_getter_roundtrip(self) -> None:
        """Setting total_reward=0.8 should read back as 0.8."""
        s = EpisodeSummary(total_reward=0.8)
        assert s.total_reward == pytest.approx(0.8)

    def test_setter_stores_outcome_signal(self) -> None:
        s = EpisodeSummary(total_reward=0.8)
        assert "outcome" in s.signals
        # 0.8 in [0,1] → 0.6 in [-1,1]
        assert s.signals["outcome"].value == pytest.approx(0.6)
        assert s.signals["outcome"].confidence == 1.0

    def test_setter_zero(self) -> None:
        s = EpisodeSummary(total_reward=0.0)
        assert s.total_reward == pytest.approx(0.0)
        assert s.signals["outcome"].value == pytest.approx(-1.0)

    def test_setter_one(self) -> None:
        s = EpisodeSummary(total_reward=1.0)
        assert s.total_reward == pytest.approx(1.0)
        assert s.signals["outcome"].value == pytest.approx(1.0)

    def test_setter_half(self) -> None:
        s = EpisodeSummary(total_reward=0.5)
        assert s.total_reward == pytest.approx(0.5)
        assert s.signals["outcome"].value == pytest.approx(0.0)

    def test_keyword_only(self) -> None:
        """EpisodeSummary(total_reward=X) must work as keyword arg."""
        s = EpisodeSummary(total_reward=0.75)
        assert s.total_reward == pytest.approx(0.75)

    def test_with_score_breakdown(self) -> None:
        """Backward compat: pass total_reward + score_breakdown."""
        s = EpisodeSummary(
            total_reward=0.8,
            score_breakdown={"functional": 0.8, "efficiency": 0.6},
        )
        assert s.total_reward == pytest.approx(0.8)
        assert s.score_breakdown == {"functional": 0.8, "efficiency": 0.6}


class TestEpisodeSummaryNeedsJudge:
    def test_empty_signals_needs_judge(self) -> None:
        s = EpisodeSummary()
        assert s.needs_judge() is True

    def test_outcome_present(self) -> None:
        s = EpisodeSummary(total_reward=0.5)
        assert s.needs_judge() is False

    def test_user_present(self) -> None:
        s = EpisodeSummary(
            signals={"user": RewardSignal(name="user", value=0.5, confidence=1.0)},
        )
        assert s.needs_judge() is False

    def test_execution_high_confidence(self) -> None:
        s = EpisodeSummary(
            signals={"execution": RewardSignal(name="execution", value=0.5, confidence=0.9)},
        )
        assert s.needs_judge() is False

    def test_execution_low_confidence_needs_judge(self) -> None:
        s = EpisodeSummary(
            signals={"execution": RewardSignal(name="execution", value=0.5, confidence=0.5)},
        )
        assert s.needs_judge() is True

    def test_judge_only_still_needs_judge_false(self) -> None:
        """Having a judge signal doesn't affect needs_judge (it checks
        whether a judge is *needed*, not whether one exists)."""
        s = EpisodeSummary(
            signals={"judge": RewardSignal(name="judge", value=0.5, confidence=0.8)},
        )
        # No outcome/user/high-conf execution → still needs judge
        assert s.needs_judge() is True


class TestEpisodeSummaryFiltered:
    def test_default_false(self) -> None:
        s = EpisodeSummary()
        assert s.filtered is False

    def test_set_true(self) -> None:
        s = EpisodeSummary(filtered=True)
        assert s.filtered is True


# ── RewardExtractor protocol tests ────────────────────────────────────


class _FixedExtractor:
    def __init__(self, signal: RewardSignal) -> None:
        self.name = signal.name
        self._signal = signal

    def extract(self, episode: Episode) -> RewardSignal | None:
        return self._signal


class _NoneExtractor:
    def __init__(self, name: str = "none") -> None:
        self.name = name

    def extract(self, episode: Episode) -> RewardSignal | None:
        return None


def _pipeline_episode(*, with_outcome: bool = False) -> Episode:
    summary = EpisodeSummary(total_reward=0.5) if with_outcome else EpisodeSummary()
    return Episode(
        id=Episode.new_id(),
        state_id="s0",
        task_id="t0",
        bench="test",
        messages=[
            Message(role="user", content="solve 2+2"),
            Message(role="assistant", content="4"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=10.0)],
        summary=summary,
    )


class TestRewardExtractorProtocol:
    def test_fixed_extractor_satisfies_protocol(self) -> None:
        sig = RewardSignal(name="outcome", value=1.0, confidence=1.0)
        ext = _FixedExtractor(sig)
        assert isinstance(ext, RewardExtractor)

    def test_none_extractor_satisfies_protocol(self) -> None:
        ext = _NoneExtractor()
        assert isinstance(ext, RewardExtractor)


class TestRewardPipeline:
    def test_enriches_signals(self) -> None:
        sig = RewardSignal(name="outcome", value=1.0, confidence=1.0)
        pipeline = RewardPipeline([_FixedExtractor(sig)])
        ep = _pipeline_episode()
        pipeline.enrich(ep)
        assert "outcome" in ep.summary.signals
        assert ep.summary.signals["outcome"].value == 1.0

    def test_none_extractor_skipped(self) -> None:
        pipeline = RewardPipeline([_NoneExtractor(name="broken")])
        ep = _pipeline_episode()
        pipeline.enrich(ep)
        assert len(ep.summary.signals) == 0

    def test_judge_skipped_when_outcome_present(self) -> None:
        outcome = RewardSignal(name="outcome", value=1.0, confidence=1.0)
        judge = RewardSignal(name="judge", value=0.5, confidence=0.8)
        pipeline = RewardPipeline([_FixedExtractor(outcome), _FixedExtractor(judge)])
        ep = _pipeline_episode()
        pipeline.enrich(ep)
        assert "outcome" in ep.summary.signals
        assert "judge" not in ep.summary.signals

    def test_judge_fires_when_no_outcome(self) -> None:
        judge = RewardSignal(name="judge", value=0.7, confidence=0.8)
        pipeline = RewardPipeline([_FixedExtractor(judge)])
        ep = _pipeline_episode()
        pipeline.enrich(ep)
        assert "judge" in ep.summary.signals
        assert ep.summary.signals["judge"].value == pytest.approx(0.7)

    def test_multiple_signals_effective_reward(self) -> None:
        low = RewardSignal(name="execution", value=0.3, confidence=0.4)
        high = RewardSignal(name="outcome", value=0.9, confidence=1.0)
        pipeline = RewardPipeline([_FixedExtractor(low), _FixedExtractor(high)])
        ep = _pipeline_episode()
        pipeline.enrich(ep)
        assert len(ep.summary.signals) == 2
        assert ep.summary.effective_reward() == pytest.approx(0.9)
