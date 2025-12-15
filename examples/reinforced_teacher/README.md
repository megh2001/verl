# ReinforcedTeacher GRPO Training

A proof-of-concept implementation of ReinforcedTeacher using GRPO (Group Relative Policy Optimization) for training a teacher model to generate helpful hints for a student model on math problems.

## Overview

### Architecture

- **Teacher Model**: 3B model (Qwen2.5-3B-Instruct) that generates hints
- **Student Model**: Same 3B model (Qwen2.5-3B-Instruct) for evaluation
- **Dataset**: Small subset of GSM8K math problems (500 train, 100 test)
- **Hardware**: Optimized for single A100 GPU (40GB or 80GB)

### Training Loop

1. **Teacher generates hints**: Multiple hints per question (GRPO rollout, n=4)
2. **Student attempts solution**: Uses hints to solve the math problem
3. **Reward computation**:
   - Teacher: `+1` if student gets correct answer with hint, `0` otherwise
   - Student: `+1` if answer is correct, `0` otherwise
4. **Policy update**: Teacher updated via GRPO to generate better hints

## Setup

### 1. Prepare Dataset

```bash
# Create small math dataset (500 train, 100 test)
python examples/reinforced_teacher/dataset_prep_math.py \
    --local_save_dir=$HOME/data/math_small \
    --num_train=500 \
    --num_test=100
```

This will download GSM8K and create:
- `~/data/math_small/train.parquet` (500 examples)
- `~/data/math_small/test.parquet` (100 examples)

### 2. Models Pre-Configured

All scripts are pre-configured to use Qwen 3B model:

```bash
# All scripts use:
TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"    # 3B teacher
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"    # 3B student (same model)
```

**Perfect for single A100 GPU!** No configuration changes needed.

## Experiments

### Baseline 1: Student Only (No Hints)

Evaluate the student model's baseline performance without any teacher hints:

```bash
bash examples/reinforced_teacher/eval_baseline.sh
```

This runs 1 epoch of evaluation and logs to WandB project `reinforced_teacher` with experiment name `baseline_student_only`.

**Expected outputs**:
- Baseline accuracy on test set
- Shows student performance without any guidance

### Baseline 2: Student with Frozen Teacher Hints

Generate hints from the teacher model (before training) and evaluate student performance:

```bash
bash examples/reinforced_teacher/eval_frozen_hints.sh
```

This script:
1. Generates hints from untrained teacher model
2. Augments test set with these frozen hints
3. Evaluates student with the hints

**Expected outputs**:
- Accuracy with frozen (untrained) teacher hints
- Comparison to baseline shows if hints help at all

### Main Training: ReinforcedTeacher GRPO

Train the teacher model to generate better hints using GRPO:

```bash
bash examples/reinforced_teacher/run_single_a100.sh
```

**Training configuration** (optimized for single A100):
- Algorithm: GRPO (Group Relative Policy Optimization)
- Model: Qwen2.5-3B-Instruct
- Rollout samples: n=4 hints per question
- Batch size: 128
- Learning rate: 1e-6
- Total epochs: 20
- Test frequency: Every 2 epochs
- GPUs: 1 A100 GPU
- **Training time**: ~2-3 hours for 20 epochs

**Outputs**:
- Checkpoints saved to `./outputs/reinforced_teacher_single_a100/`
- WandB logging: project `reinforced_teacher`, experiment `single_a100_3b`
- Metrics: teacher reward, student accuracy, KL divergence

## File Structure

```
examples/reinforced_teacher/
├── README.md                           # This file
├── dataset_prep_math.py                # Prepare small math dataset
├── reward_function.py                  # Custom reward functions
│   ├── compute_score()                 # Main reward for ReinforcedTeacher
│   ├── compute_score_baseline()        # Baseline evaluation
│   └── compute_score_frozen_hints()    # Frozen hints evaluation
├── generate_teacher_hints.py           # Generate frozen hints from teacher
├── train_reinforced_teacher.py         # Reference implementation (not used directly)
├── eval_baseline.sh                    # Run baseline evaluation (Qwen 3B)
├── eval_frozen_hints.sh                # Run frozen hints evaluation (Qwen 3B)
├── run_single_a100.sh                  # Main training script (Qwen 3B, single GPU)
├── run_reinforced_teacher_grpo.sh      # Alternative training script (multi-GPU)
├── quickstart.sh                       # Run all experiments automatically
├── SETUP_GUIDE.md                      # Quick setup and troubleshooting
├── RUN_EVALUATIONS_FIRST.md            # Step-by-step evaluation guide
└── HPC_SETUP.md                        # Guide for running on HPC clusters
```

## Reward Function Details

### `compute_score()` - Main ReinforcedTeacher Reward

Located in `reward_function.py`:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    is_teacher = extra_info.get("is_teacher", False)

    if is_teacher:
        # Teacher reward based on student success
        student_correct = extra_info.get("student_correct", False)
        return 1.0 if student_correct else 0.0
    else:
        # Student reward based on own correctness
        predicted = extract_solution(solution_str)
        return 1.0 if predicted == ground_truth else 0.0
```

**Key points**:
- Teacher gets reward `+1` only if student succeeds with the hint
- Student gets standard correctness reward
- This encourages teacher to generate helpful hints

### Answer Extraction

The reward function uses regex to extract numerical answers:
- Primary: `#### <number>` format (GSM8K style)
- Fallback: Last number in response (flexible matching)

## Hyperparameter Guide

### For Single A100 (default - Qwen 3B)

**Using `run_single_a100.sh`:**

```bash
# Model
TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"

# GRPO configuration
actor_rollout_ref.rollout.n=4              # 4 hints per question
data.train_batch_size=128                  # Total batch size
actor_rollout_ref.actor.ppo_mini_batch_size=64

# Memory optimization (3B model fits easily)
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # No TP needed
actor_rollout_ref.rollout.gpu_memory_utilization=0.45   # vLLM memory
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# Training
trainer.n_gpus_per_node=1
trainer.total_epochs=20
```

**Memory usage on A100**: ~24GB (plenty of headroom on 40GB A100)

### For A100 80GB (can use larger model)

```bash
# Use 7B model
TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"

# Increase batch sizes
data.train_batch_size=256
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.rollout.n=5
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

### For Multi-GPU Setup

```bash
# 4 GPUs with 7B model
TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"
data.train_batch_size=512
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.rollout.tensor_model_parallel_size=4
trainer.n_gpus_per_node=4
```

## Expected Results

### Baseline Performance (Qwen 3B)
- **Student only**: ~20-30% accuracy (3B model on GSM8K subset)
- **Student + frozen hints**: ~25-35% accuracy (slight improvement from untrained hints)

### After GRPO Training
- **Student + trained hints**: Target 40-50% accuracy
- Teacher learns to generate hints that improve student success rate
- Monitor `teacher_reward_mean` metric - should increase over epochs

**Training Time**: ~2-3 hours for 20 epochs on single A100 with 3B model

## Monitoring Training

### WandB Metrics

Key metrics to watch:
- `test/reward_mean`: Average reward on test set
- `train/teacher_reward_mean`: Teacher's reward (student success rate)
- `train/kl_loss`: KL divergence from reference policy
- `train/actor_loss`: GRPO policy gradient loss
- `test/accuracy`: Direct accuracy metric

### Console Output

The trainer prints decoded responses every few batches:
```
[Epoch 5] Decoded responses:
Question: ...
Hint: Start by identifying the key variables...
Student solution: ...
Reward: 1.0
```

## Troubleshooting

### OOM Errors

1. Reduce batch size: `data.train_batch_size=128`
2. Increase model parallelism: `tensor_model_parallel_size=4`
3. Enable offloading: `actor.fsdp_config.param_offload=True`
4. Reduce rollout samples: `rollout.n=3`

### Slow Training

1. Increase micro batch size: `ppo_micro_batch_size_per_gpu=16`
2. Reduce logging: `trainer.test_freq=5`
3. Use more GPUs or increase TP

### Low Rewards

1. Check dataset quality: Ensure ground truth answers are correct
2. Verify reward function: Test `extract_solution()` on sample outputs
3. Adjust learning rate: Try `lr=5e-7` or `lr=2e-6`
4. Increase rollout samples: `rollout.n=8` for more diverse hints

## Implementation Notes

### Current Limitations

This is a simplified POC implementation. For production ReinforcedTeacher:

1. **Separate teacher/student coordination**: Currently uses single-model training. True ReinforcedTeacher needs:
   - Custom reward manager that runs student model
   - Batched student inference for each teacher hint
   - Reward routing based on student outcomes

2. **Student model training**: Current setup trains only the teacher. You can:
   - Train student separately with best hints
   - Implement joint training (more complex)
   - Use fixed student for faster iteration

3. **Multi-turn interaction**: Could extend to:
   - Multiple hint rounds
   - Student asking for clarification
   - Dynamic hint generation based on student progress

### Extending the Implementation

To implement full ReinforcedTeacher with separate models:

1. Create custom reward manager (inherit from `AbstractRewardManager`)
2. Load student model in reward manager initialization
3. In `__call__()`:
   - Extract teacher's hint from response
   - Run student model with hint-augmented prompt
   - Evaluate student's solution
   - Return reward to teacher based on student correctness
4. Register custom reward manager and use in training

See `verl/workers/reward_manager/` for examples.

## Citation

If you use this implementation, please cite VERL:

```bibtex
@article{sheng2024verl,
  title={VERL: A Unified Framework for Verifiable Reinforcement Learning},
  author={Sheng, Ying and others},
  journal={arXiv preprint arXiv:2410.xxxxx},
  year={2024}
}
```

## References

- GRPO Paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- VERL Documentation: [docs/algo/grpo.md](../../docs/algo/grpo.md)
- GSM8K Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
