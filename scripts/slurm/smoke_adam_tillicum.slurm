#!/bin/bash
# DreamZero Adam — Tillicum SMOKE TEST. Run this BEFORE the paid full-node chain.
#
# Validates env + dataset + embodiment registration + the DeepSpeed/NCCL path on a
# few steps, writing to a throwaway OUTPUT_DIR so it never touches the real run.
#
# Two modes (pick by editing the SBATCH block below):
#   A) debug QOS, 1 GPU, <=1h  — cheapest; checks env/data/forward+backward.
#      Within the free 100 GPU-hr tier. NCCL multi-GPU path NOT exercised here.
#   B) normal QOS, 8 GPU, ~15m — exercises the real 8-way NCCL/DeepSpeed path.
#      Costs ~2 GPU-hr (still inside the free tier). Use this once before the chain.
#
# As shipped: mode A (debug, 1 GPU). Flip the commented directives for mode B.

#SBATCH --job-name=adam_smoke
#SBATCH --account=macsvlarobotics
#SBATCH --qos=debug                  # mode A. For mode B: --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1                     # mode A. For mode B: --gpus=8
#SBATCH --cpus-per-task=8            # mode A. For mode B: --cpus-per-task=64
#SBATCH --mem=180G                   # mode A (~200G/GPU). For mode B: --mem=1500G
#SBATCH --time=01:00:00              # debug QOS hard cap is 1h
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/%x_%j.out

set -euo pipefail

ALLOC=/gpfs/projects/macsvlarobotics
PROJECT=$ALLOC/dreamzero
SCRUBBED=/gpfs/scrubbed/$USER/dreamzero
export CONDA_ENV=$ALLOC/env

mkdir -p "$SCRUBBED/logs"
module load conda
conda activate "$CONDA_ENV"
module load gcc/13.4.0                        # host compiler only — do NOT load the cuda module:
export CUDA_HOME=/gpfs/software/cuda/12.9.1    # its lib64 ships cuDNN 9.14 which shadows torch's bundled
export PATH="$CUDA_HOME/bin:$PATH"             # cuDNN -> CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH at conv3d.
                                               # CUDA_HOME+nvcc on PATH satisfies deepspeed; torch keeps its cuDNN.
cd "$PROJECT"

# Throwaway output so the smoke run can't be mistaken for / resumed by the real chain.
export OUTPUT_DIR=$SCRUBBED/smoke_$SLURM_JOB_ID
export ADAM_DATA_ROOT=$SCRUBBED/data_merged
export PRETRAINED_MODEL_PATH=$PROJECT/checkpoints/DreamZero-AgiBot
export WAN_CKPT_DIR=$PROJECT/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=$PROJECT/checkpoints/umt5-xxl
export NUM_GPUS=$(nvidia-smi -L | wc -l)   # 1 in mode A, 8 in mode B
export MAX_STEPS=20                          # just enough to prove fwd/bwd + a save path
export SAVE_STEPS=10
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reclaim fragmented VRAM (bs=4 ~ H200 limit)

echo "SMOKE host=$(hostname) gpus=$NUM_GPUS out=$OUTPUT_DIR"
nvidia-smi -L
bash scripts/train/adam_stage_a.sh
echo "Smoke OK. Clean up: rm -rf $OUTPUT_DIR"
