#!/bin/bash
# Push offline wandb runs to the dashboard periodically, from the Tillicum LOGIN node
# (which has reliable internet). Training runs WANDB_MODE=offline because compute-node
# wandb.init() intermittently times out (90s CommError) and kills the slice at step 0;
# offline writes metrics locally and this loop syncs them up -> near-live dashboard with
# no risk to the run.
#
# Usage (on the login node):
#   nohup bash scripts/slurm/wandb_sync_loop.sh 600 \
#     >/gpfs/scrubbed/$USER/dreamzero/logs/wandb_sync.log 2>&1 &
#
# Metrics are ALSO always in OUTPUT_DIR/loss_log.jsonl regardless of wandb.
set -uo pipefail
module load conda >/dev/null 2>&1
conda activate /gpfs/projects/macsvlarobotics/env 2>/dev/null
# The trainer overrides WANDB_DIR to OUTPUT_DIR (base.py:626), so offline runs land in
# OUTPUT_DIR/wandb/offline-run-*. Each chained slice creates a new one (same run_id -> same
# dashboard run). Sync them all.
WBDIR=/gpfs/scrubbed/$USER/dreamzero/adam_stage_a_full/wandb
INTERVAL=${1:-600}
echo "wandb sync loop: $WBDIR every ${INTERVAL}s ($(date))"
while true; do
  shopt -s nullglob
  for d in "$WBDIR"/offline-run-*; do
    [ -d "$d" ] || continue
    wandb sync "$d" 2>&1 | tail -1
  done
  sleep "$INTERVAL"
done
