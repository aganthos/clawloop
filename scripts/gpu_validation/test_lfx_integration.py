"""LfX Weight Layer Integration Test.

Tests the full pipeline: Episode -> SkyRLExporter -> PreparedModelPassBatch
-> SkyRLTrainBackend.forward_backward -> optim_step on a real GPU.

This validates that the LfX integration correctly translates episodes into
SkyRL's training format and that actual GPU training occurs.
"""
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("lfx_integration_test")


def make_synthetic_episodes(n: int = 4):
    """Create synthetic math episodes for testing."""
    from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
    from lfx.core.reward import RewardSignal

    problems = [
        ("What is 2 + 3?", "The answer is 5. #### 5", 1.0),
        ("What is 7 * 8?", "7 * 8 = 56. #### 56", 1.0),
        ("What is 100 / 4?", "100 / 4 = 25. #### 25", 1.0),
        ("What is 15 - 9?", "15 - 9 = 7. #### 7", 0.0),  # wrong answer
    ]

    episodes = []
    for i in range(n):
        q, a, reward = problems[i % len(problems)]
        summary = EpisodeSummary(total_reward=reward)
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=reward * 2 - 1, confidence=1.0,
        )
        ep = Episode(
            id=Episode.new_id(),
            state_id="test-state",
            task_id=f"math-{i % len(problems)}",
            bench="math",
            messages=[
                Message(role="system", content="You are a math tutor. Solve step by step."),
                Message(role="user", content=q),
                Message(role="assistant", content=a),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
            summary=summary,
        )
        episodes.append(ep)
    return episodes


def test_export_pipeline():
    """Test Episode -> SkyRLExporter -> GeneratorOutput -> PreparedModelPassBatch."""
    from transformers import AutoTokenizer

    from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
    from lfx.exporters.skyrl import SkyRLExporter
    from skyrl.tinker.types import PreparedModelPassBatch

    log.info("=== Test 1: Export Pipeline (no GPU) ===")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    exporter = SkyRLExporter(tokenizer=tokenizer)

    episodes = make_synthetic_episodes(4)
    gen_output = exporter.export(episodes)

    log.info("Exported %d transitions from %d episodes", len(gen_output["prompt_token_ids"]), len(episodes))
    assert len(gen_output["prompt_token_ids"]) > 0
    assert len(gen_output["response_ids"]) > 0
    assert len(gen_output["rewards"]) > 0

    # Build a backend with mocks just to test _to_prepared_batch
    backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
    backend._config = SkyRLWeightsConfig(
        base_model=model_name,
        backend_type="skyrl_train",
        training_config={"loss_fn": "cross_entropy"},
    )
    backend._model_id = f"lfx-{model_name.replace('/', '-')}"
    backend._exporter = exporter

    batch = backend._to_prepared_batch(gen_output)
    assert isinstance(batch, PreparedModelPassBatch)

    n = len(batch.all_model_inputs)
    log.info("PreparedModelPassBatch: %d sequences", n)
    assert n > 0
    assert len(batch.all_targets) == n
    assert len(batch.all_advantages) == n
    assert len(batch.all_model_ids) == n

    # Verify ModelInput structure
    for mi in batch.all_model_inputs:
        assert len(mi.chunks) == 1
        assert hasattr(mi.chunks[0], "tokens")
        assert len(mi.chunks[0].tokens) > 0

    log.info("PASSED: Export pipeline produces valid PreparedModelPassBatch")
    return True


def test_gpu_training():
    """Test full GPU training: SkyRLWeightsBackend with SkyRLTrainBackend."""
    from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
    from lfx.core.types import Datum

    log.info("=== Test 2: GPU Training via LfX Weight Layer ===")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    config = SkyRLWeightsConfig(
        base_model=model_name,
        backend_type="skyrl_train",
        backend_config={
            "strategy": "fsdp2",
            "trainer.placement.colocate_all": True,
            "trainer.placement.policy_num_gpus_per_node": 1,
            "trainer.placement.ref_num_gpus_per_node": 1,
            "generator.inference_engine.num_engines": 1,
            "generator.inference_engine.tensor_parallel_size": 1,
            "trainer.train_batch_size": 4,
            "trainer.policy_mini_batch_size": 4,
            "trainer.micro_forward_batch_size_per_gpu": 2,
            "trainer.micro_train_batch_size_per_gpu": 2,
            "trainer.max_prompt_length": 256,
            "generator.sampling_params.max_generate_length": 256,
            "generator.inference_engine.gpu_memory_utilization": 0.4,
            "trainer.use_sample_packing": False,
        },
        lora_config={
            "rank": 8,
            "alpha": 16.0,
            "seed": 42,
        },
        training_config={
            "loss_fn": "cross_entropy",
            "adam_params": {
                "learning_rate": 1e-5,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
    )

    log.info("Initializing SkyRLWeightsBackend (this downloads model + starts Ray)...")
    t0 = time.time()
    backend = SkyRLWeightsBackend(config)
    log.info("Backend initialized in %.1fs", time.time() - t0)

    # Create episodes and run forward_backward
    episodes = make_synthetic_episodes(4)
    datum = Datum(episodes=episodes)

    log.info("Running forward_backward...")
    t0 = time.time()
    fb_result = backend.forward_backward(datum).result()
    log.info("forward_backward completed in %.1fs: status=%s metrics=%s",
             time.time() - t0, fb_result.status, fb_result.metrics)

    if fb_result.status == "error":
        log.error("forward_backward failed: %s", fb_result.metrics)
        return False

    # Run optim_step
    log.info("Running optim_step...")
    t0 = time.time()
    optim_result = backend.optim_step().result()
    log.info("optim_step completed in %.1fs: status=%s updates=%d",
             time.time() - t0, optim_result.status, optim_result.updates_applied)

    if optim_result.status == "error":
        log.error("optim_step failed: %s", optim_result.metrics)
        return False

    # Save checkpoint
    log.info("Saving checkpoint...")
    save_result = backend.save_state("test-ckpt-1").result()
    log.info("save_state: %s", save_result.status)

    log.info("PASSED: Full GPU training pipeline works")
    return True


def main():
    log.info("=" * 60)
    log.info("LfX Weight Layer Integration Test")
    log.info("=" * 60)

    results = {}

    # Test 1: Export pipeline (no GPU needed)
    try:
        results["export"] = test_export_pipeline()
    except Exception:
        log.exception("Export pipeline test FAILED")
        results["export"] = False

    # Test 2: GPU training
    try:
        results["gpu_training"] = test_gpu_training()
    except Exception:
        log.exception("GPU training test FAILED")
        results["gpu_training"] = False

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        log.info("  %s: %s", name, status)
    log.info("=" * 60)

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
