#!/bin/bash
# Quickstart script for ReinforcedTeacher GRPO experiments
# This script runs all experiments in sequence:
# 1. Data preparation
# 2. Baseline evaluation (student only)
# 3. Frozen hints evaluation
# 4. ReinforcedTeacher GRPO training

set -e  # Exit on error

echo "=========================================="
echo "ReinforcedTeacher GRPO - Quickstart"
echo "=========================================="
echo ""

# Configuration - uses current working directory
NUM_TRAIN=500
NUM_TEST=100
DATA_DIR=./data/math_small

echo "Configuration:"
echo "  Training examples: $NUM_TRAIN"
echo "  Test examples: $NUM_TEST"
echo "  Data directory: $DATA_DIR"
echo ""

# Step 1: Prepare dataset
echo "=========================================="
echo "Step 1: Preparing dataset..."
echo "=========================================="
python examples/reinforced_teacher/dataset_prep_math.py \
    --local_save_dir=$DATA_DIR \
    --num_train=$NUM_TRAIN \
    --num_test=$NUM_TEST

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Dataset preparation failed!"
    exit 1
fi

echo "✓ Dataset prepared successfully"
echo ""

# Step 2: Baseline evaluation
echo "=========================================="
echo "Step 2: Running baseline evaluation..."
echo "=========================================="
echo "This evaluates the student model without any hints."
echo ""

bash examples/reinforced_teacher/eval_baseline.sh

echo "✓ Baseline evaluation completed"
echo ""

# Step 3: Frozen hints evaluation
echo "=========================================="
echo "Step 3: Running frozen hints evaluation..."
echo "=========================================="
echo "This evaluates the student with hints from untrained teacher."
echo ""

bash examples/reinforced_teacher/eval_frozen_hints.sh

echo "✓ Frozen hints evaluation completed"
echo ""

# Step 4: Main training
echo "=========================================="
echo "Step 4: Starting ReinforcedTeacher GRPO training..."
echo "=========================================="
echo "This will train the teacher model to generate better hints."
echo "Training will run for 20 epochs. You can monitor progress in WandB."
echo ""

bash examples/reinforced_teacher/run_reinforced_teacher_grpo.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  1. Baseline (student only): Check WandB project 'reinforced_teacher/baseline_student_only'"
echo "  2. Frozen hints: Check WandB project 'reinforced_teacher/eval_frozen_hints'"
echo "  3. GRPO training: Check WandB project 'reinforced_teacher/teacher_grpo_7b'"
echo ""
echo "Model checkpoints saved to: ./outputs/reinforced_teacher_grpo/"
echo ""
echo "Next steps:"
echo "  - Compare accuracy across all three experiments"
echo "  - Inspect teacher's learned hints in the WandB logs"
echo "  - Try training for more epochs or with different hyperparameters"
echo "=========================================="
