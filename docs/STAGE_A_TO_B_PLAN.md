# Two-Stage Fine-Tuning Plan — Stage A (Embodiment) → Stage B (Unseen Tasks)

> Target: deploy DreamZero on a **custom bimanual robot (~14-DoF, 2 grippers, 3 cameras)** for **tasks DreamZero has never seen**, by running two strictly sequential post-training stages on top of the `DreamZero-AgiBot` checkpoint.
>
> Companion document: [`DATA_COLLECTION_STAGES_zh.md`](DATA_COLLECTION_STAGES_zh.md) — the data-team-facing specification of what to capture for each stage.

---

## 1. Why two stages

The paper validates two different things, never strung together as one workflow:

| Paper experiment | Section | What it shows |
|---|---|---|
| Few-shot embodiment adaptation (AgiBot → YAM, 30 min play data, 55 traj, 11 tasks) | §5 Q5, Fig 12, footnote 11 | A new bimanual robot can be acquired with ~30 min of language-rich short demos while keeping zero-shot generalization |
| Post-training on AgiBot for downstream tasks (shirt folding 33 h, fruit packing 12 h, table bussing 40 h) | §4.2, §5 Q3, Fig 10 | After tens of hours per task with heavy randomization, the model matches/beats VLA baselines and retains environment generalization |

Our pipeline = (embodiment adaptation) → (post-training on top), which is an extrapolation but follows directly from the two validated recipes.

```
DreamZero-AgiBot
       │  Stage A — embodiment LoRA  (≈30–60 min, language-rich primitives)
       ▼
Stage-A weights  ──merge──▶  Stage-A-merged base
                                   │  Stage B — task LoRA  (hours per unseen task)
                                   ▼
                            Stage-B checkpoint ──▶ inference
```

---

## 2. Stage A — Embodiment Adaptation

**Goal.** Teach the model how *this* robot moves, how its 14-D state/action space maps to motion primitives, and how natural-language instructions tie to those motions. **Not** a task-success stage.

### 2.1 Data

See §2 of [`DATA_COLLECTION_STAGES_zh.md`](DATA_COLLECTION_STAGES_zh.md). Headline numbers (paper-aligned):

| Knob | Target |
|---|---|
| Total wall time (post idle-trim) | ≥ **30 min** (paper); capture ~45–60 min for safety |
| Episodes | ~50–80 short clips, ~20–40 s each |
| Tasks | **10–12 distinct short primitives** (paper used 11) |
| Trajectories per task | ≥ 4–6 (paper averaged 55 / 11 ≈ 5) |
| Language | ≥ 1 clean English sentence per episode; ≥ 2 paraphrases per task recommended (the paper explicitly flagged single-string-per-task as a limitation) |
| Cameras | 3 views (top + L-wrist + R-wrist), fixed extrinsics, ordering identical to YAM |
| Action representation | Absolute joint positions; relative deltas computed by `convert_lerobot_to_gear.py` |

### 2.2 One-time embodiment registration (code edits, before any training)

| File | Change |
|---|---|
| `groot/vla/data/schema/embodiment_tags.py` | Add `MYROBOT = "myrobot"` to `EmbodimentTag` enum |
| `scripts/data/convert_lerobot_to_gear.py` | Add `"myrobot"` to `VALID_EMBODIMENT_TAGS` |
| `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | Clone the four `yam` entries (`modality_configs`, `transforms`, `metadata_versions`, `fps`) under the `myrobot` key; swap camera names to match your conversion (e.g. `video.cam_top`, `video.cam_left`, `video.cam_right`) |
| `groot/vla/configs/data/dreamzero/myrobot_relative.yaml` | New file. Copy `yam_relative.yaml`, rename `yam_data_root` → `myrobot_data_root`, change the embodiment key in `mixture_spec.dataset_path` |
| `scripts/train/myrobot_stage_a.sh` | New file. Copy `scripts/train/yam_training.sh` and parameterize per §2.4 |

YAM-equivalent values to keep verbatim: `num_views=3`, `action_horizon=24`, `num_frames=33`, `image_resolution_width=320`, `image_resolution_height=176`, LoRA `rank=4`, `alpha=4` (default in `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml`).

Camera order must follow `[top, left_wrist, right_wrist]` to match the YAM cross-view stitch in `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:357-379` (the paper attributes part of YAM's transfer efficiency to layout similarity with AgiBot).

### 2.3 LeRobot v2 conversion + GEAR metadata

LeRobot v2 directory layout is defined in [`DATA_COLLECTION_STAGES_zh.md`](DATA_COLLECTION_STAGES_zh.md) §4. Once the data team delivers, run:

```bash
conda activate dreamzero
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path ./data/myrobot_stage_a_lerobot \
  --embodiment-tag myrobot \
  --state-keys '{"left_joint_pos":[0,6], "left_gripper_pos":[6,7], "right_joint_pos":[7,13], "right_gripper_pos":[13,14]}' \
  --action-keys '{"left_joint_pos":[0,6], "left_gripper_pos":[6,7], "right_joint_pos":[7,13], "right_gripper_pos":[13,14]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key annotation.task
```

Adjust the index ranges if your 14-D vector is laid out differently (e.g. `[left_arm_7, right_arm_7]` with gripper interleaved).

### 2.4 Train Stage A LoRA

`scripts/train/myrobot_stage_a.sh` overrides on top of `yam_training.sh`:

```
output_dir=./checkpoints/myrobot_stage_a_lora
pretrained_model_path=./checkpoints/DreamZero-AgiBot
max_steps=5000
save_steps=1000
save_total_limit=3
per_device_train_batch_size=1
training_args.learning_rate=1e-5
save_lora_only=true
```

With 3× RTX PRO 6000 (96 GB each) set `NUM_GPUS=3`. Keep `save_lora_only=true` to keep checkpoints small (~50 MB each); we run an explicit merge step before Stage B.

### 2.5 Stage A exit criteria

1. **Smoke test (`max_steps=10`)** completes without errors — verifies data + embodiment registration.
2. **Loss curve** — action MSE decreases monotonically across the first 1000 steps. Instant plateau ⇒ state/action dim mis-mapped or wrong `relative_action_keys`.
3. **Behavioral check** — load Stage A LoRA into the inference server, issue **one of the Stage A training-set captions verbatim**, and confirm the trajectory is qualitatively correct in joint space. We are *not* yet checking task success — just that the policy moves the robot meaningfully toward the captioned target.

---

## 3. Transition — merge Stage A LoRA into the base

Stage B will inject a fresh rank-4 LoRA, so it needs a clean dense base to start from. Pick the best Stage A checkpoint (lowest val loss / best behavioral check) and merge:

```bash
python scripts/utils/merge_lora.py \
  --base ./checkpoints/DreamZero-AgiBot \
  --lora ./checkpoints/myrobot_stage_a_lora/checkpoint-XXXX \
  --output ./checkpoints/myrobot_stage_a_merged
```

This script (new file, ~30 lines) loads the base model + the LoRA adapter, calls PEFT's `merge_and_unload()`, and writes the merged state-dict. The merged directory then serves as `pretrained_model_path` for Stage B.

> **Note.** The paper's own post-training is **full-parameter** (`§4.2: "we update all parameters except the text encoder, image encoder, and VAE"`) and explicitly tried LoRA during pretraining with suboptimal results (footnote 7). Our two-stage LoRA pipeline is an engineering compromise for the 3-GPU box. If you have the disk/compute, switch Stage A to full-FT, skip the merge, and use the resulting checkpoint directly as Stage B's pretrained path.

---

## 4. Stage B — Unseen Task Specialization

**Goal.** Acquire one or more **new tasks** that DreamZero-AgiBot has never seen, on top of the embodiment-grounded Stage A base.

### 4.1 Data

See §3 of [`DATA_COLLECTION_STAGES_zh.md`](DATA_COLLECTION_STAGES_zh.md). Headline numbers (paper-aligned):

| Knob | Target |
|---|---|
| Per-task budget | **10–40 hours** for complex new motions (paper: shirt 33 h, fruit packing 12 h, bussing 40 h). For *mild* variants of Stage A primitives, 1–3 h may suffice |
| Randomization | Object identity, count, pose, scene layout, lighting — varied **every episode**. The paper randomizes 5 trash + 5 dishware types/combinations/positions in 40 h of bussing data |
| Multi-stage tasks | Encouraged (paper's shirt fold = 5 sequential stages; candy task = place + close + color-match) |
| Language | One clear English sentence per episode. For long-horizon tasks, sub-segment with per-phase captions if your annotation tool supports it |
| Cameras / kinematics | **Identical** to Stage A — same extrinsics, same camera order, same joint layout, same units, same FPS. Any drift here silently destroys performance |

### 4.2 Conversion

Same `convert_lerobot_to_gear.py` invocation as Stage A, with `--dataset-path ./data/myrobot_stage_b_lerobot/<task_name>`. **No code or YAML changes**; the `myrobot` embodiment registration covers both stages.

### 4.3 Train Stage B LoRA

`scripts/train/myrobot_stage_b.sh` (copy of `myrobot_stage_a.sh`):

```
output_dir=./checkpoints/myrobot_stage_b_lora/<task_name>
pretrained_model_path=./checkpoints/myrobot_stage_a_merged
myrobot_data_root=./data/myrobot_stage_b_lerobot/<task_name>
max_steps=50000                          # paper's per-task post-training budget
save_steps=2000
save_total_limit=5
per_device_train_batch_size=1
training_args.learning_rate=5e-6         # half of Stage A for stability
save_lora_only=true
++action_head_cfg.config.skip_component_loading=true
++action_head_cfg.config.defer_lora_injection=true
```

The two `action_head_cfg` flags are the deferred-LoRA-injection mechanism documented at `groot/vla/experiment/base.py:703-734` — they load the Stage-A-merged weights first, then inject a fresh rank-4 LoRA, which Stage B trains.

### 4.4 Stage B exit criteria

1. **Smoke test (`max_steps=10`)** — confirms `pretrained_model_path` resolves and the deferred LoRA injection succeeds.
2. **Behavioral check** — run the inference server on the Stage B checkpoint and issue an unseen-task command. Score success following the paper's protocol (partial task progress on a 0–1.0 scale, ≥ 10 rollouts per task, randomized initial conditions, image-overlay anchor like Barreiros et al. 2025).

---

## 5. Inference

```bash
conda activate dreamzero
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/myrobot_stage_b_lora/<task_name>/checkpoint-final
```

The server resolves the LoRA's `base_model_name_or_path` automatically; just ensure `./checkpoints/myrobot_stage_a_merged/` is present.

---

## 6. Decision checklist (resolve before kicking off Stage A capture)

| Decision | Default | Why you might change it |
|---|---|---|
| LoRA (rank 4) vs full-FT | LoRA both stages | Full-FT is paper-aligned but needs ~30 GB per checkpoint and longer training. Switch if Stage B underperforms or you have disk headroom |
| Stage A budget | 30–60 min, 10–12 tasks | Push to 90 min if your tasks need bimanual coordination or precision (paper's YAM also covered insertion + folding) |
| Stage B per-task budget | 10–40 hours of varied demos | Drop to 1–3 hours only if the target task is a mild object/pose variant of a Stage A primitive |
| Stage B per task vs joint | One LoRA per task | Joint multi-task LoRA possible but untested in our setup; default to per-task |
| Camera placement | Top + L-wrist + R-wrist, fixed extrinsics | Match YAM; do not move cameras between Stage A and Stage B |

---

## 7. Critical files

**Reference (do not edit):**
- `docs/DATASET_TO_GEAR_AND_TRAIN.md` — canonical single-stage recipe; both stages reuse it
- `docs/DATA_COLLECTION_STAGES_zh.md` — data team's per-stage capture spec
- `scripts/train/yam_training.sh` — template both stage scripts copy from
- `groot/vla/data/schema/lerobot.py` — Pydantic schema for `modality.json` validation
- `groot/vla/experiment/base.py:703-734` — `pretrained_model_path` loading + deferred LoRA injection
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:325-365` — what gets frozen vs trained
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:357-379` — camera view ordering

**To create:**
- `groot/vla/configs/data/dreamzero/myrobot_relative.yaml`
- `scripts/train/myrobot_stage_a.sh`
- `scripts/train/myrobot_stage_b.sh`
- `scripts/utils/merge_lora.py`

**To edit:**
- `groot/vla/data/schema/embodiment_tags.py` (add `MYROBOT` enum entry)
- `scripts/data/convert_lerobot_to_gear.py` (add `"myrobot"` to `VALID_EMBODIMENT_TAGS`)
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` (add modality + transform blocks)

---

## 8. Risks

- **Stage B data volume.** The paper's per-task budget is 10–40 hours, not "20–50 demos". For genuinely new motions (folding, insertion, multi-stage tasks), plan accordingly.
- **Camera / kinematics drift between stages** silently kills Stage B. Lock the cell after Stage A capture.
- **State/action layout.** The example above splits 14-D as `[L-arm 6, L-grip 1, R-arm 6, R-grip 1]`. Confirm your robot's actual exposure before running `convert_lerobot_to_gear.py`.
- **Language diversity.** Paper footnote 12 (p.17): single-string-per-task caused YAM transfer to plateau. Use 2–3 paraphrases per task in Stage A; one clear sentence per episode in Stage B.
- **Idle frames.** The pretraining recipe filters idle actions; replicate this when delivering Stage A data or the embodiment LoRA will overfit to "do-nothing" frames.
