#!/usr/bin/env bash
# Launch the Adam bimanual real-robot CLIENT (deploy_adam.py).
#
# Runs on the machine wired to the two xArm 6 arms + the ZMQ camera feeds, and connects
# to a running serve_wam.py policy server. The client needs the xArm SDK + zmq + websockets
# (the 'dreamzero' conda env has them); it does NOT use the GPU.
#
# Usage:
#   bash scripts/inference/deploy_adam.sh                      # live, with the defaults below
#   MOTION=0 bash scripts/inference/deploy_adam.sh             # DRY-RUN (no motion)
#   PROMPT="stack the cubes" bash scripts/inference/deploy_adam.sh
#   REANCHOR_SKIP=4 CHUNK_TAIL_SKIP=4 bash scripts/inference/deploy_adam.sh
#   FORCE_STOP=1 ESTOP_TORQUE=12 bash scripts/inference/deploy_adam.sh   # enable force guard
#
# NOTE: even when MOTION=1 (live), deploy_adam.py still pauses with a "Press Enter to start"
# safety gate before moving.
set -euo pipefail

# ── Tunables (env-overridable; defaults match the validated async setup) ─────────────
PROMPT="${PROMPT:-place the bowl on the corresponding plate}"
POLICY_HOST="${POLICY_HOST:-localhost}"
POLICY_PORT="${POLICY_PORT:-5000}"
ZMQ_HOST="${ZMQ_HOST:-192.222.10.10}"
INFERENCE_FREQ="${INFERENCE_FREQ:-30}"
REANCHOR_SKIP="${REANCHOR_SKIP:-6}"
CHUNK_TAIL_SKIP="${CHUNK_TAIL_SKIP:-8}"
MOTION="${MOTION:-1}"              # 1 = live (--no-dry-run), 0 = dry-run (log only)
FORCE_STOP="${FORCE_STOP:-0}"      # 1 = enable joint-torque force guard
ESTOP_TORQUE="${ESTOP_TORQUE:-20}" # N·m excess; soft guard trips at HALF this. CALIBRATE.

# ── Python env (no GPU needed for the client) ────────────────────────────────────────
CONDA_ENV_ROOT="${CONDA_ENV_ROOT:-/home/thematrix/miniconda3/envs/dreamzero}"
PY="${CONDA_ENV_ROOT}/bin/python"
[[ -x "$PY" ]] || { echo "Error: python not found at $PY (override CONDA_ENV_ROOT)." >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Assemble optional flag groups.
DRY=( --no-dry-run ); [[ "$MOTION" == "0" ]] && DRY=( --dry-run )
FORCE=(); [[ "$FORCE_STOP" == "1" ]] && FORCE=( --force-stop --estop-torque "$ESTOP_TORQUE" )

echo "=========================================="
echo "Adam deploy client"
echo "  prompt        : $PROMPT"
echo "  policy server : ${POLICY_HOST}:${POLICY_PORT}"
echo "  ZMQ cameras   : $ZMQ_HOST"
echo "  control       : ${INFERENCE_FREQ}Hz  async  reanchor-skip=${REANCHOR_SKIP}  tail-skip=${CHUNK_TAIL_SKIP}"
echo "  motion        : $([[ "$MOTION" == "0" ]] && echo 'DRY-RUN (no motion)' || echo 'LIVE')"
echo "  force guard   : $([[ "$FORCE_STOP" == "1" ]] && echo "ON (e-stop ${ESTOP_TORQUE} Nm, soft $((ESTOP_TORQUE/2)))" || echo OFF)"
echo "=========================================="

exec "$PY" deploy_adam.py \
    "${DRY[@]}" \
    --prompt "$PROMPT" \
    --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" \
    --zmq-host "$ZMQ_HOST" \
    --inference-freq "$INFERENCE_FREQ" \
    --async-prefetch --reanchor \
    --reanchor-skip "$REANCHOR_SKIP" \
    --chunk-tail-skip "$CHUNK_TAIL_SKIP" \
    "${FORCE[@]}"
