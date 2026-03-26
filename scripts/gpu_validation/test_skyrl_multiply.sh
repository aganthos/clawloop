#!/usr/bin/env bash
# SkyRL multiply env — multi-turn RL with custom environment.
# Adapted for single A10 GPU with 0.5B model and tiny batches.
set -euxo pipefail

cd ~/aganthos
source .venv/bin/activate

DATA_DIR="$HOME/data/multiply"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Multiply data not found. Run setup.sh first."
    exit 1
fi

# Run from the SkyRL submodule directory so the custom entry point resolves
cd clawloop/skyrl

python -m examples.train.multiply.main_multiply \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=16 \
    trainer.policy_mini_batch_size=8 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.eval_batch_size=16 \
    trainer.eval_before_train=false \
    trainer.eval_interval=999 \
    trainer.ckpt_interval=999 \
    trainer.max_prompt_length=256 \
    generator.sampling_params.max_generate_length=256 \
    trainer.policy.optimizer_config.lr=3.0e-5 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    environment.env_class=multiply \
    generator.n_samples_per_prompt=2 \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    trainer.use_sample_packing=false \
    trainer.logger="console" \
    trainer.project_name="clawloop_validation" \
    trainer.run_name="multiply_0.5b_test" \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/multiply_test" \
    "$@"

echo ""
echo "=== SkyRL multiply env training completed successfully ==="
