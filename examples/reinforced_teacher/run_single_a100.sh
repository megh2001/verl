#!/bin/bash
# ReinforcedTeacher GRPO Training - Optimized for Single A100
#
# This config is optimized for:
# - 1x A100 40GB: Use 7B teacher model (recommended)
# - 1x A100 80GB: Use 7B teacher model (comfortable) or try 13B

set -x

# ============================================
# CONFIGURATION - EDIT THESE
# ============================================

# Using 3B model - perfect for single A100!
TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Alternative 3B options:
# TEACHER_MODEL="Qwen/Qwen2-3B-Instruct"

# If you want to try larger models later:
# TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"  # 7B
# TEACHER_MODEL="Qwen/Qwen2.5-14B-Instruct"  # 14B (A100 80GB)

# Data paths - uses current working directory
MATH_TRAIN_PATH=./data/math_small/train.parquet
MATH_TEST_PATH=./data/math_small/test.parquet

train_files="['$MATH_TRAIN_PATH']"
test_files="['$MATH_TEST_PATH']"

# Output directory
OUTPUT_DIR="./outputs/reinforced_teacher_single_a100"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "ReinforcedTeacher GRPO - Single A100"
echo "=========================================="
echo "Teacher Model: $TEACHER_MODEL"
echo "Train Data: $MATH_TRAIN_PATH"
echo "Test Data: $MATH_TEST_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ============================================
# TRAINING CONFIGURATION
# Optimized for single A100
# ============================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    `# Data Configuration` \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    `# Model Configuration` \
    actor_rollout_ref.model.path=$TEACHER_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# Actor (Training) Configuration` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# Rollout (Generation) Configuration` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    \
    `# Reference Model Configuration` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# GRPO Algorithm Configuration` \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    `# Trainer Configuration` \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='reinforced_teacher' \
    trainer.experiment_name='single_a100_3b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=20 \
    trainer.default_hdfs_dir=$OUTPUT_DIR \
    \
    `# Custom Reward Function` \
    +custom_reward_function.path=$(pwd)/examples/reinforced_teacher/reward_function.py \
    +custom_reward_function.name=compute_score $@

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="

# ============================================
# MEMORY OPTIMIZATION TIPS
# ============================================
# If you run out of memory, try these in order:
#
# 1. Reduce batch size:
#    data.train_batch_size=64
#    actor_rollout_ref.actor.ppo_mini_batch_size=32
#
# 2. Reduce rollout samples:
#    actor_rollout_ref.rollout.n=3
#
# 3. Enable optimizer offloading:
#    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
#
# 4. Reduce vLLM memory:
#    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
#
# 5. Use smaller model:
#    TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"
