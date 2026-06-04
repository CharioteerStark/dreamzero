#!/bin/bash
# Prepare the new WAM dataset for DreamZero/Adam training ON TILLICUM.
#
# Best-upload rationale: the bulk of a LeRobot v2 dataset is parquet + mp4, which the
# GEAR converter never touches (it only ADDS small meta/*.json). So we transfer the
# 24 GB tarball ONCE to Tillicum scrubbed, extract there, and run the (metadata-only)
# conversion on the login node — no double transfer, no local re-tar.
#
# Pipeline:
#   1. extract WAM.tar.gz  -> $WORK/WAM_raw/<dataset-root>
#   2. locate the dataset root (the dir that contains meta/info.json)
#   3. sanity-check it matches the Adam format (14-dim state/action; cams top/left_wrist/right_wrist)
#   4. run scripts/data/convert_lerobot_to_gear.py with the Adam mapping (same as data_merged)
#   5. validate the GEAR meta files were written
#
# Run on the Tillicum login node:
#   bash scripts/data/prepare_wam_tillicum.sh /gpfs/scrubbed/$USER/dreamzero/WAM.tar.gz
# Then point training at the printed GEAR dataset path (ADAM_DATA_ROOT).

set -euo pipefail
TARBALL=${1:?usage: prepare_wam_tillicum.sh <path/to/WAM.tar.gz> [work_dir]}
WORK=${2:-/gpfs/scrubbed/$USER/dreamzero/wam}
REPO=/gpfs/projects/macsvlarobotics/dreamzero
ENVDIR=/gpfs/projects/macsvlarobotics/env

module load conda
conda activate "$ENVDIR"

mkdir -p "$WORK/extract"
echo "=== [1/5] extracting $(du -h "$TARBALL" | cut -f1) tarball -> $WORK/extract ==="
tar -xzf "$TARBALL" -C "$WORK/extract"

echo "=== [2/5] locating dataset root (dir containing meta/info.json) ==="
ROOT=$(dirname "$(find "$WORK/extract" -maxdepth 3 -path '*/meta/info.json' | head -1)")
ROOT=$(dirname "$ROOT")   # strip the trailing /meta
echo "dataset root: $ROOT"
if [ ! -f "$ROOT/meta/info.json" ]; then
  echo "ERROR: no meta/info.json under $WORK/extract — is this a LeRobot v2 dataset?"; exit 1
fi

echo "=== [3/5] inspecting format ==="
python - "$ROOT" <<'PY'
import json, sys
root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
f = info.get("features", {})
st = f.get("observation.state", {}).get("shape")
ac = f.get("action", {}).get("shape")
cams = [k.replace("observation.images.", "") for k in f if k.startswith("observation.images")]
print(f"  episodes={info.get('total_episodes')} frames={info.get('total_frames')} fps={info.get('fps')} robot={info.get('robot_type')}")
print(f"  state_shape={st} action_shape={ac} cameras={sorted(cams)}")
print(f"  annotation_keys={[k for k in f if k.startswith('annotation')]}")
ok = (st == [14] and ac == [14] and set(cams) == {"top", "left_wrist", "right_wrist"})
if not ok:
    print("\n  !! MISMATCH vs the Adam format (expected state/action [14] and cams {top,left_wrist,right_wrist}).")
    print("     Stop and adjust --state-keys/--action-keys/cameras before converting.")
    sys.exit(2)
print("  format matches Adam — safe to convert.")
PY

echo "=== [4/5] converting LeRobot v2 -> GEAR (adam) ==="
cd "$REPO"
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path "$ROOT" \
  --embodiment-tag adam \
  --state-keys  '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --action-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key annotation.task

echo "=== [5/5] validating GEAR metadata ==="
for fjson in modality.json embodiment.json stats.json relative_stats_dreamzero.json tasks.jsonl episodes.jsonl; do
  if [ -f "$ROOT/meta/$fjson" ]; then echo "  ok  meta/$fjson"; else echo "  MISSING meta/$fjson"; fi
done
echo
echo "GEAR dataset ready at: $ROOT"
echo "Point training at it:  ADAM_DATA_ROOT=$ROOT"
