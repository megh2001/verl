#!/bin/bash
# ReinforcedTeacher GRPO Training Script
#
# This trains a teacher model to generate helpful hints for a student model
# using GRPO (Group Relative Policy Optimization).
#
# Training loop:
# 1. Teacher generates hints (multiple samples via GRPO rollout)
# 2. For each hint, student attempts to solve the problem
# 3. Teacher receives reward based on student's success
# 4. Teacher policy updated with GRPO

set -x

# Model paths - CHANGE THESE TO YOUR MODELS
TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"          # 7B teacher model
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"        # 2.5B student model

# Data paths - uses current working directory
MATH_TRAIN_PATH=./data/math_small/train.parquet
MATH_TEST_PATH=./data/math_small/test.parquet

# Training configuration
train_files="['$MATH_TRAIN_PATH']"
test_files="['$MATH_TEST_PATH']"

# Output directory
OUTPUT_DIR="./outputs/reinforced_teacher_grpo"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "ReinforcedTeacher GRPO Training"
echo "=========================================="
echo "Teacher Model: $TEACHER_MODEL"
echo "Student Model: $STUDENT_MODEL"
echo "Train Data: $MATH_TRAIN_PATH"
echo "Test Data: $MATH_TEST_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# NOTE: This is a simplified single-model training setup
# For true ReinforcedTeacher with separate teacher/student coordination,
# you would need to implement a custom reward manager that:
# 1. Takes teacher's generated hints
# 2. Runs student model to get solutions
# 3. Evaluates student correctness
# 4. Returns rewards to teacher based on student performance
#
# For this POC, we train the teacher model with direct rewards
# and show how to integrate the custom reward function.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$TEACHER_MODEL \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='reinforced_teacher' \
    trainer.experiment_name='teacher_grpo_7b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=20 \
    trainer.default_hdfs_dir=$OUTPUT_DIR \
    ++custom_reward_function.path=./reward_function.py \
    ++custom_reward_function.name=compute_score $@

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
