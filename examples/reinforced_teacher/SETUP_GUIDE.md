# ReinforcedTeacher GRPO - Setup Guide

## Quick Start (TL;DR)

```bash
# 1. Run everything at once
bash examples/reinforced_teacher/quickstart.sh

# OR run step by step:

# 2. Prepare dataset (500 train, 100 test)
python examples/reinforced_teacher/dataset_prep_math.py \
    --local_save_dir=$HOME/data/math_small \
    --num_train=500 \
    --num_test=100

# 3. Baseline: Student only (no hints)
bash examples/reinforced_teacher/eval_baseline.sh

# 4. Baseline: Student with frozen teacher hints
bash examples/reinforced_teacher/eval_frozen_hints.sh

# 5. Train teacher with GRPO (single A100, Qwen 3B)
bash examples/reinforced_teacher/run_single_a100.sh
```

## What You Get

### Files Created

```
examples/reinforced_teacher/
├── README.md                           # Full documentation
├── SETUP_GUIDE.md                      # This file
├── quickstart.sh                       # Run all experiments
│
├── dataset_prep_math.py                # Prepare GSM8K subset
├── reward_function.py                  # Custom reward functions
├── generate_teacher_hints.py           # Generate frozen hints
├── train_reinforced_teacher.py         # Reference implementation
│
├── eval_baseline.sh                    # Baseline: student only (Qwen 3B)
├── eval_frozen_hints.sh                # Baseline: student + frozen hints (Qwen 3B)
├── run_single_a100.sh                  # Main GRPO training (Qwen 3B, single A100)
└── run_reinforced_teacher_grpo.sh      # Alternative training (multi-GPU)
```

### Experiments

**Experiment 1: Baseline (Student Only)**
- Script: `eval_baseline.sh`
- Model: Qwen2.5-3B-Instruct
- Purpose: Measure student performance without hints
- Expected: ~20-30% accuracy on math problems
- WandB: `reinforced_teacher/baseline_student_only`

**Experiment 2: Frozen Hints (Student + Untrained Teacher)**
- Script: `eval_frozen_hints.sh`
- Model: Qwen2.5-3B-Instruct
- Purpose: Check if any hints help (even from untrained teacher)
- Expected: ~25-35% accuracy
- WandB: `reinforced_teacher/eval_frozen_hints`

**Experiment 3: GRPO Training (Student + Trained Teacher)**
- Script: `run_single_a100.sh`
- Model: Qwen2.5-3B-Instruct
- Purpose: Train teacher to generate helpful hints
- Expected: ~40-50% accuracy after training
- WandB: `reinforced_teacher/single_a100_3b`
- Duration: ~2-3 hours for 20 epochs on single A100

## System Requirements

### Hardware
- **Recommended**: 1x A100 GPU (40GB or 80GB)
- **Also works on**: V100 32GB, RTX 3090 24GB (with reduced batch size)
- **Multi-GPU**: Can use 4-8 GPUs with alternative scripts

### Software
- Python 3.8+
- PyTorch 2.0+
- VERL library (already installed in this repo)
- Transformers, datasets, pandas
- (Optional) WandB for logging

### Model (Pre-Configured)
- **Teacher & Student**: Qwen2.5-3B-Instruct (same model)
- Perfect for single A100
- No configuration changes needed!

## Configuration Guide

### Default Settings (4 GPUs)

```bash
# In run_reinforced_teacher_grpo.sh:
TEACHER_MODEL="Qwen/Qwen2-7B-Instruct"
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"

trainer.n_gpus_per_node=4
data.train_batch_size=256
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.rollout.n=5              # 5 hints per question
actor_rollout_ref.rollout.tensor_model_parallel_size=2
trainer.total_epochs=20
```

### For 8 GPUs

```bash
trainer.n_gpus_per_node=8
data.train_batch_size=512
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### For Testing (1 GPU)

```bash
trainer.n_gpus_per_node=1
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.n=3
trainer.total_epochs=5
```

## Understanding the Training Flow

### GRPO Algorithm

ReinforcedTeacher uses GRPO (Group Relative Policy Optimization):

1. **Rollout Phase**: Teacher generates `n=5` different hints per question
2. **Evaluation Phase**: (In full version) Student attempts to solve with each hint
3. **Reward Computation**:
   - Teacher gets `+1` if student succeeds, `0` if fails
   - Rewards grouped by question
4. **Advantage Calculation**: Each hint's advantage = `(reward - mean_reward_for_question) / std`
5. **Policy Update**: Teacher updated to increase probability of good hints

### Reward Function

Located in `reward_function.py`:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Teacher reward based on whether student succeeds with hint.
    Student reward based on own correctness.
    """
    is_teacher = extra_info.get("is_teacher", False)

    if is_teacher:
        # Teacher: Did student succeed with this hint?
        student_correct = extra_info.get("student_correct", False)
        return 1.0 if student_correct else 0.0
    else:
        # Student: Is answer correct?
        predicted = extract_solution(solution_str)
        return 1.0 if predicted == ground_truth else 0.0
```

### Dataset Format

Each example in the parquet files:

```python
{
    "data_source": "math_small",
    "prompt": [{"role": "user", "content": "Solve: ..."}],
    "ability": "math",
    "reward_model": {
        "style": "reinforced_teacher",
        "ground_truth": "42"  # Correct numerical answer
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "question": "...",
        "answer": "#### 42"
    }
}
```

## Monitoring Training

### WandB Dashboard

Key metrics to monitor:

| Metric | Description | Good Trend |
|--------|-------------|-----------|
| `test/reward_mean` | Average reward on test set | ↑ Increasing |
| `train/teacher_reward_mean` | Teacher's reward (student success) | ↑ Increasing |
| `train/kl_loss` | KL divergence from reference | → Stable, small |
| `train/actor_loss` | Policy gradient loss | ↓ Decreasing |
| `test/accuracy` | Direct accuracy metric | ↑ Increasing |

### Console Output

Training prints example outputs:

```
[Epoch 5, Batch 10]
Question: A store sells 3 apples for $2. How much for 12 apples?
Teacher Hint: Think about how many groups of 3 apples...
Student Solution: 12/3 = 4 groups, so 4 * $2 = $8 #### 8
Ground Truth: 8
Reward: 1.0 ✓
```

### Checkpoints

Saved to `./outputs/reinforced_teacher_grpo/`:
- `epoch_5/`: Checkpoint after epoch 5
- `epoch_10/`: Checkpoint after epoch 10
- etc.

## Troubleshooting

### "CUDA out of memory"

**Solution 1**: Reduce batch size
```bash
data.train_batch_size=128
actor_rollout_ref.actor.ppo_mini_batch_size=64
```

**Solution 2**: Increase model parallelism
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

**Solution 3**: Enable offloading
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

**Solution 4**: Reduce rollout samples
```bash
actor_rollout_ref.rollout.n=3  # Instead of 5
```

### "Training is too slow"

**Solution 1**: Increase micro batch size
```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
```

**Solution 2**: Reduce evaluation frequency
```bash
trainer.test_freq=5  # Test every 5 epochs instead of 2
```

**Solution 3**: Use more GPUs
```bash
trainer.n_gpus_per_node=8
```

### "Rewards are always 0"

**Check 1**: Verify dataset
```python
import pandas as pd
df = pd.read_parquet("~/data/math_small/train.parquet")
print(df.iloc[0])  # Check format
```

**Check 2**: Test reward function
```python
from examples.reinforced_teacher.reward_function import extract_solution
test_response = "The answer is #### 42"
print(extract_solution(test_response))  # Should print "42"
```

**Check 3**: Inspect model outputs
- Check WandB logs for decoded responses
- Verify model is generating numerical answers

**Solution**: Adjust learning rate
```bash
actor_rollout_ref.actor.optim.lr=5e-7  # Lower
# or
actor_rollout_ref.actor.optim.lr=2e-6  # Higher
```

### "ModuleNotFoundError: No module named 'verl'"

Make sure you're in the VERL root directory:
```bash
cd /path/to/verl
python -m verl.trainer.main_ppo ...
```

## Advanced Usage

### Custom Dataset

1. Create your own dataset processor (see `dataset_prep_math.py`)
2. Format as parquet with required fields
3. Update paths in shell scripts

### Different Models

Edit shell scripts:
```bash
TEACHER_MODEL="your/teacher-model"
STUDENT_MODEL="your/student-model"
```

Supported models:
- Qwen2 family
- Llama 3
- Mistral
- Any HuggingFace causal LM

### Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim.lr` | 1e-6 | Learning rate |
| `rollout.n` | 5 | Hints per question |
| `kl_loss_coef` | 0.001 | KL penalty weight |
| `ppo_mini_batch_size` | 128 | Mini-batch size |
| `train_batch_size` | 256 | Total batch size |

### Extending to Multi-Turn

Modify `reward_function.py` to track conversation history:
```python
extra_info["conversation_history"] = [...]
# Generate follow-up hints based on student's attempt
```

## Next Steps

After running the experiments:

1. **Compare Results**:
   - Baseline vs Frozen Hints vs Trained Teacher
   - Plot accuracy curves
   - Analyze which types of hints work best

2. **Improve Training**:
   - Try longer training (50+ epochs)
   - Experiment with different reward functions
   - Add intermediate rewards (e.g., partial credit)

3. **Scale Up**:
   - Use larger teacher model (13B, 70B)
   - Train on full GSM8K or MATH dataset
   - Fine-tune student model jointly

4. **Implement Full ReinforcedTeacher**:
   - Create custom reward manager with student model
   - Real-time student evaluation during training
   - See implementation notes in README.md

## Support

For issues or questions:
- Check the main README.md for detailed documentation
- Review VERL documentation: `docs/algo/grpo.md`
- Open an issue on the VERL GitHub repository

## References

- **GRPO**: DeepSeekMath paper ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
- **VERL**: Verifiable Reinforcement Learning framework
- **GSM8K**: Grade School Math dataset ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k))
