# How to Run Evaluations First - Step by Step Guide

This guide shows you how to run baseline evaluations BEFORE training. This is important to establish baseline performance metrics.

## Why Run Evaluations First?

Before training the teacher with GRPO, you want to know:
1. **Baseline**: How well does the 3B model perform without any hints?
2. **Frozen Hints**: Do hints from an untrained model help at all?

Then after GRPO training, you can compare and see the improvement!

---

## Prerequisites

Make sure you have:
- ✅ VERL installed and working
- ✅ Access to HuggingFace models (Qwen2.5-3B-Instruct)
- ✅ Single A100 GPU
- ✅ Python 3.8+, PyTorch, etc.

---

## Step-by-Step Instructions

### **Step 0: Navigate to the Directory**

```bash
cd examples/reinforced_teacher
```

All commands below assume you're in this directory.

---

### **Step 1: Prepare the Dataset**

First, create the small math dataset (500 train, 100 test):

```bash
python dataset_prep_math.py \
    --local_save_dir=$HOME/data/math_small \
    --num_train=500 \
    --num_test=100
```

**What this does:**
- Downloads GSM8K dataset from HuggingFace
- Takes first 500 examples for training
- Takes first 100 examples for testing
- Saves to `~/data/math_small/train.parquet` and `~/data/math_small/test.parquet`

**Expected output:**
```
Downloading data...
Saved 500 training examples to /home/username/data/math_small/train.parquet
Saved 100 test examples to /home/username/data/math_small/test.parquet
```

**Verify it worked:**
```bash
ls -lh ~/data/math_small/
# Should show train.parquet and test.parquet
```

---

### **Step 2: Run Baseline Evaluation (No Hints)**

This evaluates the 3B model WITHOUT any hints - just raw performance.

```bash
bash eval_baseline.sh
```

**What this does:**
- Loads Qwen2.5-3B-Instruct model
- Runs evaluation on 100 test examples
- Model solves math problems without any hints
- Measures accuracy (% correct answers)
- Logs to WandB: `reinforced_teacher/baseline_student_only`

**Expected output:**
```
[Starting evaluation...]
Loading model: Qwen/Qwen2.5-3B-Instruct
Processing test batch 1/20...
Processing test batch 2/20...
...
Test accuracy: 0.25 (25/100 correct)
Baseline evaluation completed!
```

**Typical baseline accuracy:** 20-30% on GSM8K

**Time:** ~10-20 minutes depending on GPU

**Troubleshooting:**
- If OOM: The script is already optimized for single GPU
- If model not found: Make sure you have HuggingFace access
- If data not found: Go back to Step 1

---

### **Step 3: Run Frozen Hints Evaluation (Optional but Recommended)**

This evaluates the 3B model WITH hints from an untrained teacher model.

```bash
bash eval_frozen_hints.sh
```

**What this does:**
- Uses the SAME 3B model as "teacher" (untrained)
- Generates hints for each test question
- Saves hints to `~/data/math_small/test_with_hints.parquet`
- Evaluates student WITH these frozen hints
- Measures if hints help at all
- Logs to WandB: `reinforced_teacher/eval_frozen_hints`

**Expected output:**
```
[Generating hints from teacher...]
Loading teacher model: Qwen/Qwen2.5-3B-Instruct
Generating hints: 100%|████████████| 100/100 [02:15<00:00]
Saved 100 examples with hints to: ~/data/math_small/test_with_hints.parquet

[Evaluating student with frozen hints...]
Test accuracy: 0.28 (28/100 correct)
Frozen hints evaluation completed!
```

**Typical accuracy with frozen hints:** 25-35% (slight improvement)

**Time:** ~20-30 minutes (hint generation + evaluation)

**Note:** This step is optional but useful for comparison.

---

### **Step 4: Review Baseline Results**

Before moving to training, check your baseline metrics:

**Option A: Check WandB Dashboard**
1. Go to https://wandb.ai
2. Find project: `reinforced_teacher`
3. Compare experiments:
   - `baseline_student_only`: ~20-30% accuracy
   - `eval_frozen_hints`: ~25-35% accuracy

**Option B: Check Console Output**
Look at the final lines of each script's output:
```
Test accuracy: 0.25
Average reward: 0.25
```

**What to expect:**
- Baseline: 20-30% correct
- Frozen hints: Slightly better, 25-35% correct
- These are your "before training" numbers

---

### **Step 5: Now Train with GRPO!**

Once you have baselines, run the main training:

```bash
bash run_single_a100.sh
```

**What this does:**
- Trains the 3B teacher model with GRPO
- Teacher learns to generate helpful hints
- Runs for 20 epochs (~2-4 hours)
- Saves checkpoints every 5 epochs
- Tests every 2 epochs
- Logs to WandB: `reinforced_teacher/single_a100_3b`

**Expected output:**
```
[Epoch 1/20]
Training batch 1/100...
Average teacher reward: 0.23
...
[Epoch 5/20]
Testing...
Test reward: 0.32
...
[Epoch 20/20]
Test reward: 0.48
Training completed!
Checkpoints saved to: ./outputs/reinforced_teacher_single_a100/
```

**Target after training:** 40-50% accuracy (improvement over baseline!)

**Time:** 2-4 hours for 20 epochs

---

### **Step 6: Compare Results**

After training, compare all three:

| Experiment | Accuracy | Description |
|------------|----------|-------------|
| Baseline (no hints) | ~25% | Raw 3B model performance |
| Frozen hints | ~28% | With untrained hints |
| **After GRPO training** | **~45%** | **With trained hints** ✨ |

**Improvement:** 25% → 45% = 20 percentage point gain!

---

## Quick Reference Commands

```bash
# Full sequence (run in order):
cd examples/reinforced_teacher

# 1. Prepare data
python dataset_prep_math.py --num_train=500 --num_test=100

# 2. Baseline evaluation (REQUIRED)
bash eval_baseline.sh

# 3. Frozen hints evaluation (OPTIONAL)
bash eval_frozen_hints.sh

# 4. Train with GRPO
bash run_single_a100.sh

# 5. Check results in WandB
# Go to https://wandb.ai → project "reinforced_teacher"
```

---

## Files Created During Evaluation

After running evaluations, you'll have:

```
~/data/math_small/
├── train.parquet              # 500 training examples
├── test.parquet               # 100 test examples
└── test_with_hints.parquet    # 100 test examples + frozen hints

./outputs/
└── reinforced_teacher_single_a100/
    ├── epoch_5/               # Checkpoint after 5 epochs
    ├── epoch_10/              # Checkpoint after 10 epochs
    └── epoch_20/              # Final checkpoint
```

---

## Monitoring During Evaluation

### **Console Output**
The scripts print progress:
```
[Batch 5/20] Processing examples 25-30
[Batch 10/20] Processing examples 50-55
...
[EVALUATION] Question: A store sells 3 apples for $2...
[EVALUATION] Generated answer: #### 8
[EVALUATION] Ground truth: 8
[EVALUATION] Reward: 1.0 ✓
```

### **WandB Metrics**
Key metrics to check:
- `test/reward_mean`: Average reward (0-1 scale)
- `test/accuracy`: Direct accuracy percentage
- `test/reward_std`: Variance in rewards

---

## Troubleshooting

### **"ModuleNotFoundError: No module named 'verl'"**
```bash
# Make sure you're in the VERL root directory
cd /path/to/verl
python -m verl.trainer.main_ppo ...
```

### **"FileNotFoundError: train.parquet not found"**
```bash
# Run dataset preparation first:
python examples/reinforced_teacher/dataset_prep_math.py
```

### **"CUDA out of memory"**
The scripts are already optimized for single A100. If still OOM:

**In eval_baseline.sh**, reduce batch size:
```bash
data.train_batch_size=64        # Instead of 128
actor_rollout_ref.rollout.n=3   # Instead of 5
```

### **"Model not found on HuggingFace"**
```bash
# Login to HuggingFace first:
huggingface-cli login

# Or set token:
export HF_TOKEN="your_token_here"
```

### **Evaluation is too slow**
Normal times:
- Baseline eval: 10-20 min
- Frozen hints: 20-30 min (includes hint generation)
- Training: 2-4 hours

If much slower, check GPU utilization:
```bash
nvidia-smi
# Should show high GPU usage
```

---

## What to Look For in Results

### **Good Baseline Results:**
- Model generates numerical answers (not just text)
- Some answers are correct (20-30%)
- Console shows decoded responses with `#### <number>` format

### **Bad Baseline Results:**
- Reward always 0.0
- Model doesn't generate `#### <number>` format
- Check reward_function.py - might need to adjust extraction

### **Example Good Output:**
```
Question: Janet has 6 apples and gives 2 to her friend. How many does she have?
Generated: Let me solve this. 6 - 2 = 4 apples. #### 4
Ground truth: 4
Reward: 1.0 ✓
```

### **Example Bad Output:**
```
Question: Janet has 6 apples...
Generated: She would have some apples left.
Ground truth: 4
Reward: 0.0 ✗
```

If you see many bad outputs, the model might need:
- Better prompt engineering
- Different temperature settings
- More training data

---

## Next Steps After Evaluation

Once evaluations are complete:

1. **If baseline is good (>15% accuracy)**: Proceed to training!
   ```bash
   bash run_single_a100.sh
   ```

2. **If baseline is poor (<15% accuracy)**:
   - Check dataset format
   - Verify model is loaded correctly
   - Test reward function manually

3. **After training**: Compare all three experiments in WandB

4. **To improve further**:
   - Train for more epochs (50+)
   - Use full GSM8K dataset (7.5K examples)
   - Try larger models (7B, 13B)
   - Tune hyperparameters

---

## Summary

**Minimum required steps:**
1. ✅ Prepare dataset
2. ✅ Run baseline evaluation
3. ✅ Train with GRPO

**Recommended steps:**
1. ✅ Prepare dataset
2. ✅ Run baseline evaluation
3. ✅ Run frozen hints evaluation (for comparison)
4. ✅ Train with GRPO
5. ✅ Compare all results

**Total time:** ~3-5 hours (mostly training)

Good luck! 🚀
