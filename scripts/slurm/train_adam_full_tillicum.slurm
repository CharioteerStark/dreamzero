#!/bin/bash
# DreamZero Adam Stage A — FULL fine-tune, one 24h slice on UW Tillicum (full 8x H200 node).
# Counterpart to train_adam_tillicum.slurm (LoRA). Runs on its OWN node, so it can run
# concurrently with the LoRA job on a different node (Tillicum has 24 H200 nodes;
# normal-QOS concurrent cap = 48 GPUs = 6 full nodes).
#
# Recipe: full-parameter, DeepSpeed zero2_offload (CPU Adam), bs=4, LR 2e-5, 7500 steps.
# bs=4 REQUIRES REQUIRE_UNIFORM_CHUNKS=1 (set below). Root cause of the earlier bs=4 crash:
# wam_geer_us emits some edge-of-episode windows with < max_chunk_size chunks (state (3,64) vs
# (4,64), video 25 vs 33 frames). They are model-valid (so bs=1 trains) but won't np.stack with
# full windows -> bs>=2 collate crash at ~step 18 (dreamzero_cotrain.py:163, key='state'). The
# flag makes lerobot_sharded.py get_step_data drop those non-uniform edge windows, so every
# sample shares one shape and bs>1 batches cleanly. ~31.5 s/step -> ~2.7 days for 7500 steps.
# See scripts/train/adam_stage_a_full.sh for why zero2_offload (not zero3: zero3 shards the
# frozen VAE and breaks tiled encode; not plain zero2: OOMs VRAM).
# Auto-resumes from OUTPUT_DIR (base.py:644), so chain 24h slices with submit_chain-style deps.

#SBATCH --job-name=adam_full
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
export OUTPUT_DIR=$SCRUBBED/adam_stage_a_full
export ADAM_DATA_ROOT=$SCRUBBED/wam_geer_us        # new WAM GEAR dataset (RichtechRD/wa-adam-geer-us)
export PRETRAINED_MODEL_PATH=$PROJECT/checkpoints/DreamZero-AgiBot
export WAN_CKPT_DIR=$PROJECT/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=$PROJECT/checkpoints/umt5-xxl
export NUM_GPUS=8
export PER_DEVICE_BS=4                 # sweep optimum (bs1..5 fit, bs6 OOMs); REQUIRES REQUIRE_UNIFORM_CHUNKS below
export MAX_STEPS=7500                  # 7500 @ bs4 (eff batch 32) == same total samples as 30k @ bs1
export SAVE_STEPS=1000                 # < ~2700 reachable in 24h @ 31.5s/it -> each slice checkpoints before the kill
export SAVE_TOTAL_LIMIT=8              # keep all 8 ckpts (1k..7k + final 7500); 8x230GB ~1.84TB on scrubbed
export LEARNING_RATE=2e-5              # sqrt LR scaling for effective batch 8 -> 32
export DEEPSPEED_CFG=zero2_offload
export REQUIRE_UNIFORM_CHUNKS=1        # drop edge windows with < max_chunk_size chunks so bs>1 collate works (see header)

export WANDB_MODE=${WANDB_MODE:-online}     # live dashboard. NOTE: compute-node wandb.init() has intermittently
                                            # hit a 90s CommError (NETWORK, not auth) and killed a slice at step 0.
                                            # The afterany chain + step-1000 checkpoints limit the damage; if it
                                            # recurs, fall back to WANDB_MODE=offline + scripts/slurm/wandb_sync_loop.sh.
export WANDB_DIR=$SCRUBBED
export WANDB_RUN_ID=adam_full_bs4_v2       # FRESH id: the old adam_full_stageA_bs4 run got corrupted server-side
                                           # (rename + offline/online sync + 4 resumes) and wandb.init(resume) HANGS on
                                           # it (even an API read CommErrors). A clean id inits instantly. Stable across slices.
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}   # avoid a transient 90s init flake killing a slice (saw it on g018)
export WANDB_RESUME=allow
export WANDB_LOG_MODEL=false               # log metrics live, but DO NOT upload checkpoints to wandb
                                           # (checkpoints save locally to OUTPUT_DIR on scrubbed via save_strategy=steps)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1            # else DeepSpeedCPUAdam JIT fails (CUDAMismatchException)

echo "host=$(hostname) gpus=$NUM_GPUS env=$CONDA_ENV ds=$DEEPSPEED_CFG bs=$PER_DEVICE_BS"
nvidia-smi -L

# Pre-build DeepSpeed cpu_adam ONCE (single process, on this compute node) so the 8 torchrun
# ranks load the cached .so instead of racing to JIT-build it (race -> missing cpu_adam.so).
echo "pre-building cpu_adam (single process)..."
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load(); print('cpu_adam ready')"

bash scripts/train/adam_stage_a_full.sh
