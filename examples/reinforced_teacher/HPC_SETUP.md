# HPC Setup Guide for ReinforcedTeacher GRPO

This guide helps you set up and run the ReinforcedTeacher GRPO training on an HPC cluster (SLURM-based).

## Before You Pull to HPC

### 1. **Commit Your Changes Locally**

```bash
# On your local machine (Windows)
cd "C:\Users\megh2\OneDrive\Desktop\NYU_sem3\Eff_AI\Project\verl"

# Add the new files
git add examples/reinforced_teacher/

# Commit
git commit -m "Add ReinforcedTeacher GRPO training setup with Qwen 3B"

# Push to your remote (GitHub/GitLab)
git push origin main  # or your branch name
```

### 2. **What You'll Need on HPC**

- ✅ Python 3.8+ (usually via module)
- ✅ CUDA 11.8+ / 12.0+ (for A100)
- ✅ Access to at least 1 A100 GPU (40GB or 80GB)
- ✅ Internet access (to download models from HuggingFace)
- ✅ ~50GB disk space for models and datasets

---

## HPC Setup Steps

### **Step 1: Clone/Pull Repository**

```bash
# SSH into HPC
ssh your_username@hpc.cluster.edu

# Navigate to your workspace
cd /scratch/your_username/  # or wherever you have space

# Clone repo (first time)
git clone https://github.com/your_username/verl.git
cd verl

# OR pull changes (if already cloned)
cd /scratch/your_username/verl
git pull origin main
```

### **Step 2: Load Required Modules**

Most HPC systems use `module` to manage software. Create a setup script:

```bash
# Create setup script
nano setup_env.sh
```

Add this content (adjust module names for your HPC):

```bash
#!/bin/bash
# setup_env.sh - Load required modules for VERL

# Load Python (adjust version as available)
module load python/3.10
# OR
# module load anaconda3

# Load CUDA (for A100, use CUDA 11.8+ or 12.0+)
module load cuda/12.1
# OR
# module load cudnn/8.9-cuda12

# Load compiler if needed
module load gcc/11.2.0

# Show loaded modules
module list

echo "Environment ready!"
```

Make it executable:
```bash
chmod +x setup_env.sh
```

### **Step 3: Create Python Virtual Environment**

```bash
# Load modules first
source setup_env.sh

# Create virtual environment
python -m venv verl_env

# Activate it
source verl_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### **Step 4: Install Dependencies**

```bash
# Make sure virtual env is activated
source verl_env/bin/activate

# Install PyTorch (adjust CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install VERL dependencies
pip install -e .

# Install additional requirements
pip install transformers datasets accelerate
pip install vllm  # For fast inference
pip install wandb  # For logging
pip install pandas pyarrow  # For parquet files

# Install Flash Attention (optional but recommended for speed)
pip install flash-attn --no-build-isolation
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import verl; print('VERL installed successfully')"
```

### **Step 5: Configure HuggingFace**

```bash
# Login to HuggingFace (needed to download models)
pip install huggingface-hub
huggingface-cli login

# Enter your HF token when prompted
# Get token from: https://huggingface.co/settings/tokens

# OR set token as environment variable
export HF_TOKEN="your_token_here"
```

### **Step 6: Configure WandB (Optional but Recommended)**

```bash
# Login to WandB for experiment tracking
wandb login

# Enter your WandB API key when prompted
# Get key from: https://wandb.ai/authorize

# OR set as environment variable
export WANDB_API_KEY="your_api_key_here"

# If you want to run offline (no internet during training)
export WANDB_MODE=offline
```

---

## Creating SLURM Job Scripts

### **Job Script 1: Prepare Dataset**

Create `slurm_prepare_data.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=cpu  # or your CPU partition name

# Load environment
source setup_env.sh
source verl_env/bin/activate

# Create logs directory
mkdir -p logs

# Prepare dataset
cd examples/reinforced_teacher
python dataset_prep_math.py \
    --local_save_dir=$HOME/data/math_small \
    --num_train=500 \
    --num_test=100

echo "Dataset preparation complete!"
```

Submit:
```bash
sbatch slurm_prepare_data.sh
```

### **Job Script 2: Baseline Evaluation**

Create `slurm_eval_baseline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --output=logs/eval_baseline_%j.out
#SBATCH --error=logs/eval_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=a100:1  # Request 1 A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu  # Adjust to your GPU partition name

# Load environment
source setup_env.sh
source verl_env/bin/activate

# Set environment variables
export HF_TOKEN="your_token_here"  # If needed
export WANDB_API_KEY="your_key_here"  # If using WandB

# Create logs directory
mkdir -p logs

# Run baseline evaluation
cd examples/reinforced_teacher
bash eval_baseline.sh

echo "Baseline evaluation complete!"
```

Submit:
```bash
sbatch slurm_eval_baseline.sh
```

### **Job Script 3: GRPO Training (Main)**

Create `slurm_train_grpo.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=grpo_train
#SBATCH --output=logs/grpo_train_%j.out
#SBATCH --error=logs/grpo_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=a100:1  # Request 1 A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB  # More memory for training
#SBATCH --time=08:00:00  # 8 hours (adjust as needed)
#SBATCH --partition=gpu

# Load environment
source setup_env.sh
source verl_env/bin/activate

# Set environment variables
export HF_TOKEN="your_token_here"
export WANDB_API_KEY="your_key_here"
export CUDA_VISIBLE_DEVICES=0

# Create logs and output directories
mkdir -p logs
mkdir -p outputs

# Run GRPO training
cd examples/reinforced_teacher
bash run_single_a100.sh

echo "GRPO training complete!"
```

Submit:
```bash
sbatch slurm_train_grpo.sh
```

---

## HPC-Specific Configuration Changes

### **1. Update Paths for HPC**

Edit the shell scripts to use HPC paths:

In `eval_baseline.sh`, `run_single_a100.sh`, etc., change:

```bash
# FROM:
MATH_TRAIN_PATH=$HOME/data/math_small/train.parquet
MATH_TEST_PATH=$HOME/data/math_small/test.parquet

# TO: (use absolute path)
MATH_TRAIN_PATH=/scratch/your_username/data/math_small/train.parquet
MATH_TEST_PATH=/scratch/your_username/data/math_small/test.parquet

# OR keep $HOME if you have space there:
MATH_TRAIN_PATH=$HOME/data/math_small/train.parquet
MATH_TEST_PATH=$HOME/data/math_small/test.parquet
```

### **2. Adjust Output Directories**

In `run_single_a100.sh`:

```bash
# Use scratch space for outputs (faster, more space)
OUTPUT_DIR="/scratch/your_username/outputs/reinforced_teacher_single_a100"
mkdir -p $OUTPUT_DIR
```

### **3. Disable Interactive WandB**

Add to your scripts:

```bash
# Before training
export WANDB_MODE=offline  # If no internet during training
# OR
export WANDB_CONSOLE=off  # Disable interactive prompts
```

### **4. Cache Models Locally**

To avoid re-downloading models, set cache directory:

```bash
export HF_HOME="/scratch/your_username/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/your_username/.cache/huggingface/transformers"
mkdir -p $HF_HOME
```

---

## Common HPC Issues & Solutions

### **Issue 1: Out of Memory (OOM)**

**Solution:** Reduce batch size and rollout samples

In `run_single_a100.sh`:
```bash
data.train_batch_size=64  # Reduce from 128
actor_rollout_ref.rollout.n=3  # Reduce from 4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4  # Reduce from 8
```

### **Issue 2: Job Time Limit**

**Solution:** Save checkpoints more frequently

```bash
trainer.save_freq=2  # Save every 2 epochs instead of 5
trainer.total_epochs=10  # Reduce for testing
```

Or split training into multiple jobs that resume from checkpoints.

### **Issue 3: Module Not Found**

**Solution:** Ensure virtual environment is activated

```bash
# In SLURM script, always do:
source verl_env/bin/activate

# Verify:
which python  # Should point to verl_env/bin/python
```

### **Issue 4: CUDA Version Mismatch**

**Solution:** Match PyTorch CUDA version with loaded CUDA module

```bash
# Check loaded CUDA
module list | grep cuda

# Install matching PyTorch
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### **Issue 5: Network/Internet Access**

Some HPC systems restrict internet on compute nodes.

**Solution 1:** Pre-download models on login node

```bash
# On login node (has internet):
python -c "from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

**Solution 2:** Run WandB in offline mode

```bash
export WANDB_MODE=offline
# Then sync later: wandb sync logs/
```

---

## Monitoring Jobs

### **Check Job Status**

```bash
# View your jobs
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>
```

### **View Logs in Real-Time**

```bash
# Tail output log
tail -f logs/grpo_train_<job_id>.out

# Tail error log
tail -f logs/grpo_train_<job_id>.err
```

### **Check GPU Usage**

If you have access to the compute node:

```bash
# SSH to compute node
ssh compute-node-name

# Monitor GPU
watch -n 1 nvidia-smi
```

---

## Complete Setup Checklist

Before running on HPC:

- [ ] Commit and push code from local machine
- [ ] Clone/pull repo on HPC
- [ ] Load required modules (Python, CUDA)
- [ ] Create virtual environment
- [ ] Install PyTorch with correct CUDA version
- [ ] Install VERL and dependencies
- [ ] Login to HuggingFace CLI
- [ ] (Optional) Login to WandB
- [ ] Update paths in shell scripts for HPC
- [ ] Create SLURM job scripts
- [ ] Create `logs/` and `outputs/` directories
- [ ] Test with small job first

---

## Quick Start on HPC

```bash
# 1. Setup (one-time)
git clone <your_repo>
cd verl
source setup_env.sh
python -m venv verl_env
source verl_env/bin/activate
pip install -e .
pip install transformers datasets accelerate vllm wandb pandas pyarrow
huggingface-cli login
wandb login

# 2. Prepare data (5 min)
sbatch slurm_prepare_data.sh

# 3. Run baseline (1-2 hours)
sbatch slurm_eval_baseline.sh

# 4. Train with GRPO (4-8 hours)
sbatch slurm_train_grpo.sh

# 5. Monitor
squeue -u $USER
tail -f logs/grpo_train_*.out
```

---

## Tips for HPC

1. **Test first**: Run with 1 epoch on login node to catch errors early
   ```bash
   # Quick test (on login node, if allowed)
   cd examples/reinforced_teacher
   python dataset_prep_math.py --num_train=10 --num_test=5
   ```

2. **Use scratch space**: Store data/models on `/scratch`, not `$HOME`

3. **Set up cleanup**: Delete old checkpoints to save space
   ```bash
   # In SLURM script, after training:
   find outputs/ -name "epoch_*" -type d | head -n -3 | xargs rm -rf
   # Keeps only last 3 checkpoints
   ```

4. **Save environment**: Create `requirements.txt` for reproducibility
   ```bash
   pip freeze > requirements.txt
   ```

5. **Use job arrays**: For hyperparameter sweeps
   ```bash
   #SBATCH --array=0-4  # Run 5 jobs with different params
   ```

---

## Contact HPC Support

If you encounter issues specific to your HPC:

- Check your HPC's documentation
- Contact support: `support@hpc.cluster.edu`
- Ask about:
  - GPU partition names
  - Available CUDA versions
  - Module names for Python/CUDA
  - Scratch space location
  - Network access policies

---

## Summary

**Local (Windows):**
1. Commit and push changes

**HPC (Linux):**
1. Pull code
2. Setup environment (modules, venv, packages)
3. Prepare dataset
4. Run evaluations
5. Train with GRPO

All scripts are ready - just need to adjust paths and create SLURM wrappers!

Good luck! 🚀
