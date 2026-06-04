#!/bin/bash
# DreamZero Adam Stage A — one 24h slice on UW Tillicum (full 8× H200 node).
#
# The training entry point auto-resumes: groot/vla/experiment/base.py:644 calls
# get_checkpoint_path(output_dir) and resumes from the latest checkpoint if one
# exists, or exits 0 ("Skip training") once max_steps is reached. So this job is
# safe to run as a chain of identical 24h slices against the SAME OUTPUT_DIR —
# see submit_chain.sh. Each slice resumes; the first slice that finds training
# complete exits in seconds.
#
# Full node: --gpus=8 => 64 CPUs (8/GPU) + up to 1600G RAM (200G/GPU) on Tillicum.
# DreamZero launches via `torchrun --nproc_per_node 8 --standalone` (single-node
# self-rendezvous on localhost), so we pin --nodes=1 and DON'T need srun / c10d
# rdzv plumbing — that's only required for the multi-node (16-GPU/2-node) case.
# Directives below match the verified UW recipe (josecols/ft-llms-tillicum) +
# tillicum-slurm docs; --gpus=8 is the canonical Tillicum GPU request.

#SBATCH --job-name=adam_stageA
#SBATCH --account=macsvlarobotics    # REQUIRED — else falls to wrong/no default account
#SBATCH --qos=normal                 # 24h cap; request 'long' QOS (help@uw.edu) to avoid chaining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one torchrun launcher; it spawns the 8 ranks
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G                  # near node max (1600G); lower if you want to share the node
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/%x_%j.out

set -euo pipefail

# ---- Tillicum paths for the macsvlarobotics allocation ----
# Project storage: 1TB, backed up daily, PURGED at end of project. Holds repo + env +
# weights + LoRA checkpoints. Move anything you want to keep off-cluster before project end.
ALLOC=/gpfs/projects/macsvlarobotics                # 1TB project allocation (backed up)
PROJECT=$ALLOC/dreamzero                            # git repo + weights + checkpoints
SCRUBBED=/gpfs/scrubbed/$USER/dreamzero             # dataset + logs (NOT backed up, 60d idle purge)
export CONDA_ENV=$ALLOC/env                          # conda --prefix env, SIBLING of the repo (not inside it)
# Billing: free 100 GPU-hr tier, then worktag PG225985 / TILLICUM_GPU_macsvlarobotics @ $0.90/GPU-hr.
# 8 GPUs => $7.20/hr => ~$173 per full 24h slice. Monitor with: hyakusage
# -----------------------------------------------------------

mkdir -p "$SCRUBBED/logs"

module load conda
conda activate "$CONDA_ENV"
module load gcc/13.4.0                        # host compiler only — do NOT load the cuda module:
export CUDA_HOME=/gpfs/software/cuda/12.9.1    # its lib64 ships cuDNN 9.14 which shadows torch's bundled cuDNN
export PATH="$CUDA_HOME/bin:$PATH"             # (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). CUDA_HOME+nvcc satisfies deepspeed.

cd "$PROJECT"

# All large artifacts must already live on /gpfs (pre-staged on the login node —
# compute nodes should do no downloads). Point the launcher at them explicitly so
# its HuggingFace auto-download branch is never hit.
export OUTPUT_DIR=$PROJECT/checkpoints/adam_stage_a_lora
export ADAM_DATA_ROOT=$SCRUBBED/wam_geer_us    # new WAM data (same as the full-FT run) for a parallel LoRA-vs-full comparison
export PRETRAINED_MODEL_PATH=$PROJECT/checkpoints/DreamZero-AgiBot
export WAN_CKPT_DIR=$PROJECT/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=$PROJECT/checkpoints/umt5-xxl
export NUM_GPUS=8
# per_device_bs=3 (NOT 4): bs=4 fits on 1 GPU but OOMs on the 8-GPU run — each card also
# needs NCCL/DeepSpeed comm buffers, and bs=4 leaves no room. bs=3 (global 24) is confirmed
# fitting on 8x H200 (~22.5 s/step). LR left at the recipe default 1e-5 (linear scaling for
# 24 vs 32 would be 7.5e-6; the modest drop doesn't require it).
export PER_DEVICE_BS=${PER_DEVICE_BS:-3}        # overridable; LoRA uses LESS GPU than full-FT (no full-grad buffers) so likely > 3
export MAX_STEPS=${MAX_STEPS:-10000}           # ~240k-sample budget / (bs3 * 8) == same total samples as full-FT 7500@bs4
export SAVE_STEPS=${SAVE_STEPS:-2000}          # < ~3840 reachable in 24h @ ~22.5s/step -> each slice checkpoints before the kill
export REQUIRE_UNIFORM_CHUNKS=1                # bs=3 on wam_geer_us hits the same edge-window collate bug as full-FT bs>1

# W&B live logging (compute nodes have internet; `wandb login` done on Tillicum).
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=$SCRUBBED
export WANDB_RUN_ID=adam_lora_stageA   # stable id -> chained 24h slices append to ONE run
export WANDB_RESUME=allow

# Avoid HF/torch trying to phone home from the compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# NCCL/distributed stability — both UW ft-llms train scripts set these; matters
# more here because DreamZero trains with DeepSpeed zero2 over NCCL on 8 GPUs.
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Recover fragmented VRAM — at per_device_bs=4 the 14B LoRA run sits right at the H200's
# 141GB; expandable_segments reclaims the ~9GB lost to fragmentation. (zero2 does NOT shard
# the frozen base for LoRA, so per-GPU memory ~ single-GPU; this matters on all 8 cards.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) gpus=$NUM_GPUS env=$CONDA_ENV"
nvidia-smi -L

bash scripts/train/adam_stage_a.sh
