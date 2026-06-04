#!/bin/bash
# Submit a chain of identical 24h Tillicum slices for Adam Stage A.
#
# Why a chain: normal QOS caps walltime at 24h. Each slice auto-resumes from the
# latest checkpoint (groot/vla/experiment/base.py:644), so N slices = one long run.
# We use afterany so the next slice starts even if the previous is killed at the
# walltime boundary. The slice that finds training complete exits in seconds, and
# all later slices likewise no-op — harmless and cheap.
#
# Usage:  bash scripts/slurm/submit_chain.sh [N]      (default N=4 → up to 96 GPU·node·h)
#
# Before the first paid chain: run scripts/slurm/smoke_adam_tillicum.slurm (mode B,
# 8-GPU, ~2 GPU·h) to validate the DeepSpeed/NCCL path inside the free 100 GPU·h tier.
#
# Cost check: 8 GPUs × $0.90/GPU·h = $7.20/h → ~$173 per full 24h slice. The free
# 100 GPU·h tier ≈ 12.5 wall-clock h on a full node — gone before one slice finishes.
#
# Maintenance is the 2nd Tuesday of each month: a slice spanning that window may be
# killed. That's safe here (10k-step checkpoints + auto-resume lose < one save
# interval), but avoid launching a fresh slice you know will straddle it.

set -euo pipefail
N=${1:-4}
# Optional 2nd arg = the SLURM wrapper to chain (default LoRA). For full-FT pass
# train_adam_full_tillicum.slurm:  bash scripts/slurm/submit_chain.sh 4 train_adam_full_tillicum.slurm
SCRIPT="$(dirname "$0")/${2:-train_adam_tillicum.slurm}"
[ -f "$SCRIPT" ] || { echo "no such SLURM script: $SCRIPT"; exit 1; }
echo "chaining: $SCRIPT"

prev=""
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable "$SCRIPT")
  else
    jid=$(sbatch --parsable --dependency=afterany:"$prev" "$SCRIPT")
  fi
  echo "slice $i: job $jid${prev:+ (after $prev)}"
  prev=$jid
done
echo "Submitted $N chained slices. Monitor: squeue -u \$USER"
