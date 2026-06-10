#!/bin/bash
# DreamZero-Flash Adam — full-parameter fine-tune, one 24h slice on UW Tillicum (8x H200).
#
# WHAT: continues training from the Stage-A full-FT result (adam_stage_a_full/checkpoint-7500)
# with the paper's "-Flash" DECOUPLED noise schedule (arXiv:2602.15922): video timesteps biased
# toward HIGH noise via t_video = 1 - eta, eta ~ Beta(7,1) (E[sigma] ~ 0.97 after the shift-5
# warp), while ACTION timesteps stay uniform. Teaches the model to predict clean actions from a
# noisy video context -> enables 1-step diffusion inference at deploy (paper: 1-step 52% -> 74%
# with Flash; ~2.3x faster chunks).
#
# HOW (verified against the code, 2026-06-10):
#  - The training entrypoint builds the model ENTIRELY from the launch hydra config
#    (experiment/base.py create_model -> instantiate(cfg.model)); PRETRAINED_MODEL_PATH loads
#    WEIGHTS ONLY via load_state_dict(strict=False) from the checkpoint's sharded safetensors.
#    The checkpoint's own config.json is NOT read in training, so the EXTRA_OVERRIDES below
#    take effect. checkpoint-7500 has config.json + index + shards -> loads like a base model.
#  - Flash flags are consumed in wan_flow_matching_action_tf.py:680 (video Beta sampling) and
#    :707 (action independent uniform). Keys exist in the action-head yaml -> plain overrides
#    (no ++). Smoke signal: slice log prints "[NOISE] Mode=DECOUPLED | Video: Beta(7.0,1) ...".
#  - Fresh run: new OUTPUT_DIR (step 0, fresh optimizer/LR schedule) + FRESH wandb id
#    (never reuse an id: the old corrupted-run lesson).
#
# Recipe otherwise identical to train_adam_full_tillicum.slurm (zero2_offload CPU-Adam, bs=4,
# LR 2e-5, REQUIRE_UNIFORM_CHUNKS=1, gcc11.5 cpu_adam prebuild, ~31.5 s/step). 7500 steps
# @ eff-batch 32. Chain slices: bash scripts/slurm/submit_chain.sh 4 train_adam_flash_tillicum.slurm
# Auto-resumes from OUTPUT_DIR (base.py:644) -> identical slices continue one run.

#SBATCH --job-name=adam_flash
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1800G                  # full-FT offloads ~198GB fp32 Adam state to CPU + job RAM
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/%x_%j.out

set -euo pipefail
ALLOC=/gpfs/projects/macsvlarobotics
PROJECT=$ALLOC/dreamzero
SCRUBBED=/gpfs/scrubbed/$USER/dreamzero
export CONDA_ENV=$ALLOC/env

mkdir -p "$SCRUBBED/logs"
module load conda
conda activate "$CONDA_ENV"
module load gcc/11.5.0                          # gcc 11.5 NOT 13.4: gcc13 -march=native emits AVX512-FP16
export CUDA_HOME=/gpfs/software/cuda/12.9.1      # (vmovw) that the assembler rejects when building cpu_adam.
export PATH="$CUDA_HOME/bin:$PATH"               # And NOT the cuda module (its cuDNN 9.14 shadows torch's).

cd "$PROJECT"

# Full-FT checkpoints are ~230 GB each -> write to scrubbed (~100 TB), NOT projects (1 TB).
export OUTPUT_DIR=$SCRUBBED/adam_flash_full
export ADAM_DATA_ROOT=$SCRUBBED/wam_geer_us        # same WAM GEAR dataset as Stage A
# Init = Stage-A full-FT result (weights only; fresh optimizer + step 0 in the new OUTPUT_DIR).
export PRETRAINED_MODEL_PATH=$SCRUBBED/adam_stage_a_full/checkpoint-7500
export WAN_CKPT_DIR=$PROJECT/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=$PROJECT/checkpoints/umt5-xxl
export NUM_GPUS=8
export PER_DEVICE_BS=4                 # validated optimum; REQUIRES REQUIRE_UNIFORM_CHUNKS below
export MAX_STEPS=7500                  # same budget as the Stage-A full run (eff-batch 32)
export SAVE_STEPS=1000                 # < ~2700 reachable in 24h @ 31.5s/it -> each slice checkpoints before the kill
export SAVE_TOTAL_LIMIT=8              # 8x230GB ~1.84TB on scrubbed
export LEARNING_RATE=2e-5              # keep the Stage-A LR (YAM-exact philosophy)
export DEEPSPEED_CFG=zero2_offload
export REQUIRE_UNIFORM_CHUNKS=1        # drop non-uniform edge windows so bs>1 collate works

# ============ FLASH: decoupled video/action noise (the only model-level change) ============
# Keys exist in groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml
# (defaults: false / 3.0 / 1.0) -> plain overrides, no '++'. Paper uses Beta(7,1).
export EXTRA_OVERRIDES="action_head_cfg.config.decouple_video_action_noise=true action_head_cfg.config.video_noise_beta_alpha=7.0 action_head_cfg.config.video_noise_beta_beta=1.0"
# ===========================================================================================

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=$SCRUBBED
export WANDB_RUN_ID=adam_flash_full_bs4    # FRESH id (never reuse: corrupted-run lesson). Stable across slices.
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export WANDB_RESUME=allow
export WANDB_LOG_MODEL=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1            # else DeepSpeedCPUAdam JIT fails (CUDAMismatchException)

echo "host=$(hostname) gpus=$NUM_GPUS env=$CONDA_ENV ds=$DEEPSPEED_CFG bs=$PER_DEVICE_BS"
echo "FLASH overrides: $EXTRA_OVERRIDES"
echo "init from: $PRETRAINED_MODEL_PATH"
nvidia-smi -L

# Pre-build DeepSpeed cpu_adam ONCE (single process) so the 8 ranks don't race the JIT build.
echo "pre-building cpu_adam (single process)..."
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load(); print('cpu_adam ready')"

bash scripts/train/adam_stage_a_full.sh
