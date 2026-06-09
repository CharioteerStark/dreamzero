#!/usr/bin/env bash
# Launch the WAM inference server for the Adam bimanual robot.
#
# The server speaks an openpi-style wire protocol (msgpack over websocket).
# Any client that implements this protocol can connect; this repo does not
# ship a real-robot client.
#
# Adam embodiment (14-DoF total): 2x 6-DoF arms + 2 grippers, 3 cameras
# (top + 2 wrists).
#
# Wire layout:
#   Client -> Server obs dict keys:
#     observation/head_left      : (H, W, 3) uint8 RGB  (expected 640x360)
#     observation/left_wrist     : (H, W, 3) uint8 RGB  (expected 640x360)
#     observation/right_wrist    : (H, W, 3) uint8 RGB  (expected 640x360)
#     observation/state          : (14,) float32 = [L-arm(6), L-grip(1), R-arm(6), R-grip(1)]
#     prompt                     : str
#   Server -> Client response:
#     {"actions": (24, 14) float32, "server_timing": {...}}
#   Handshake metadata advertises action_horizon=24, action_dim=14,
#   state_dim=14, expected_image_resolution=[640, 360].
#
# Usage:
#   bash scripts/inference/serve_wam.sh [MODEL_PATH] [PORT] [N_GPUS] [CUDA_DEVICES]
#
# Examples:
#   # Default: 2-GPU tensor parallel on GPUs 0,2 + fixed merged checkpoint (just run it)
#   bash scripts/inference/serve_wam.sh
#
#   # Explicit 2-GPU
#   bash scripts/inference/serve_wam.sh ./checkpoints/adam_stage_a_merged_19000 5000 2 0,2
#
#   # 1-GPU fallback (e.g. to use cfg_scale=1.0, which is incompatible with TP)
#   bash scripts/inference/serve_wam.sh ./checkpoints/adam_stage_a_merged_19000 5000 1 2

set -euo pipefail

# Defaults: 2-GPU tensor parallel (~1.7x) on GPUs 0,2 (skip GPU 1), and the FIXED merged
# checkpoint (LoRA merged into the correct DreamZero-AgiBot base). Override via positional args:
#   serve_wam.sh [MODEL_PATH] [PORT] [N_GPUS] [CUDA_DEVICES]
MODEL_PATH="${1:-./checkpoints/adam_stage_a_merged_19000}"
PORT="${2:-5000}"
N_GPUS="${3:-2}"
CUDA_DEVICES="${4:-0,2}"

# Default prompt drawn from data/meta/tasks.jsonl (overridable via env var).
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Pick up the yellow cube and place it on the pink circular pad.}"
# Override the default save dir (./world_model_videos under the repo root) if needed.
SAVE_VIDEO_DIR="${SAVE_VIDEO_DIR:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export ATTENTION_BACKEND="TE"
export HYDRA_FULL_ERROR=1
# Diffusion denoising steps to generate one action chunk (read by the action head at startup).
# Default 4 = DreamZero's deploy baseline (~83% task progress, ~2x faster than 16). Override:
#   WAM_NUM_INFERENCE_STEPS=8 bash scripts/inference/serve_wam.sh ...
export WAM_NUM_INFERENCE_STEPS="${WAM_NUM_INFERENCE_STEPS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Pick the Python environment the same way training does:
#   - The 'dreamzero' conda env ships its own CUDA toolkit (nvcc), which DeepSpeed
#     needs at import time (transformers.modeling_utils imports deepspeed which
#     touches CUDA_HOME). The repo's .venv has tyro but no CUDA toolkit.
CONDA_ENV_ROOT="${CONDA_ENV_ROOT:-/home/thematrix/miniconda3/envs/dreamzero}"
if [[ ! -x "${CONDA_ENV_ROOT}/bin/python" ]]; then
    echo "Error: conda env not found at ${CONDA_ENV_ROOT}." >&2
    echo "       Override by setting CONDA_ENV_ROOT, or create the env per repo README." >&2
    exit 1
fi
export PATH="${CONDA_ENV_ROOT}/bin:${PATH}"
export CUDA_HOME="${CONDA_ENV_ROOT}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "DreamZero WAM inference server"
echo "  Checkpoint     : $MODEL_PATH"
echo "  Port           : $PORT"
echo "  GPUs           : $N_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_DEVICES)"
echo "  Python         : $(command -v python)"
echo "  Default prompt : $DEFAULT_PROMPT"
echo "  Denoise steps  : $WAM_NUM_INFERENCE_STEPS"
echo "  Protocol       : openpi-style msgpack/websocket; clients must set chunk_size=24"
echo "  World-model    : ${SAVE_VIDEO_DIR:-./world_model_videos (default)}"
echo "=========================================="

cd "$REPO_ROOT"

EXTRA_ARGS=()
[[ -n "$SAVE_VIDEO_DIR" ]] && EXTRA_ARGS+=(--save-video-dir "$SAVE_VIDEO_DIR")

torchrun \
    --standalone \
    --nproc_per_node="$N_GPUS" \
    serve_wam.py \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --enable-dit-cache \
    --default-prompt "$DEFAULT_PROMPT" \
    "${EXTRA_ARGS[@]}"
