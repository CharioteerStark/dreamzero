#!/bin/bash
#SBATCH --job-name=bsprobe
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/bsprobe_%j.out
set -uo pipefail
ALLOC=/gpfs/projects/macsvlarobotics
module load conda; conda activate $ALLOC/env
module load gcc/13.4.0
export CUDA_HOME=/gpfs/software/cuda/12.9.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ALLOC/dreamzero
export NUM_GPUS=1 MAX_STEPS=4 SAVE_STEPS=100000 PER_DEVICE_BS=${PROBE_BS:?set PROBE_BS}
export OUTPUT_DIR=/gpfs/scrubbed/$USER/dreamzero/bsprobe/bs${PROBE_BS}_$SLURM_JOB_ID
export ADAM_DATA_ROOT=/gpfs/scrubbed/$USER/dreamzero/data_merged
export PRETRAINED_MODEL_PATH=$ALLOC/dreamzero/checkpoints/DreamZero-AgiBot
export WAN_CKPT_DIR=$ALLOC/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=$ALLOC/dreamzero/checkpoints/umt5-xxl
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN
echo "===== PROBE per_device_bs=$PROBE_BS host=$(hostname) ====="
if bash scripts/train/adam_stage_a.sh; then echo "===== RESULT bs=$PROBE_BS FIT ====="; else echo "===== RESULT bs=$PROBE_BS OOM_OR_FAIL ====="; fi
