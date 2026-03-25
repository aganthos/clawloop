#!/usr/bin/env bash
# Weight tuning: GSM8K GRPO full finetune (no LoRA) on Qwen2.5-0.5B.
#
# Full finetune is supported by SkyRLTrainBackend when lora.rank=0.
# Only feasible on small models — 0.5B fits on a single A10 (23GB).
# For larger models, use gsm8k_lora.sh instead.
#
# Setup:
#   pip install -e lfx/skyrl[fsdp,dev]
#   python lfx/skyrl/examples/train/gsm8k/gsm8k_dataset.py --output_dir ~/data/gsm8k
#
# Run:
#   bash examples/weight_tuning/gsm8k_full_finetune.sh
set -euxo pipefail

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
EPOCHS="${EPOCHS:-1}"
LOGGER="${LOGGER:-console}"

cd "$(dirname "$0")/../../lfx/skyrl"

python -m skyrl.train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="$MODEL" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    generator.inference_engine.num_engines=$NUM_GPUS \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=$EPOCHS \
    trainer.eval_before_train=false \
    trainer.eval_interval=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=16 \
    trainer.policy_mini_batch_size=4 \
    trainer.micro_forward_batch_size_per_gpu=2 \
    trainer.micro_train_batch_size_per_gpu=2 \
    trainer.max_prompt_length=256 \
    generator.sampling_params.max_generate_length=512 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    environment.env_class=gsm8k \
    generator.n_samples_per_prompt=2 \
    generator.inference_engine.gpu_memory_utilization=0.4 \
    trainer.use_sample_packing=false \
    trainer.logger="$LOGGER" \
    trainer.project_name="lfx_examples" \
    trainer.run_name="gsm8k_full_ft_${MODEL##*/}" \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/gsm8k_full_ft" \
    "$@"
