# Stage A Fine-Tuning Plan — Embodiment Adaptation

> Target: adapt DreamZero to a **custom bimanual robot (Adam: ~14-DoF, 2 grippers, 3 cameras)** by LoRA post-training on top of the `DreamZero-AgiBot` checkpoint, mirroring the YAM embodiment-adaptation recipe.
>
> Companion document: [`DATA_COLLECTION_STAGES_zh.md`](DATA_COLLECTION_STAGES_zh.md) — the data-team-facing capture specification.
>
> **Scope note:** This document covers **Stage A (embodiment adaptation) only.** Task-specialization ("Stage B") is deferred for now.

---

## 1. Why embodiment adaptation

The paper's few-shot embodiment adaptation result (§5 Q5, Fig 12, footnote 11) shows a new bimanual robot can be acquired from `DreamZero-AgiBot` with ~30 min of language-rich short demos while keeping zero-shot generalization:

| Paper experiment | Section | What it shows |
|---|---|---|
| Few-shot embodiment adaptation (AgiBot → YAM, 30 min play data, 55 traj, 11 tasks) | §5 Q5, Fig 12, footnote 11 | A new bimanual robot can be acquired with ~30 min of language-rich short demos while keeping zero-shot generalization |

```
DreamZero-AgiBot
       │  Stage A — embodiment LoRA  (≈30–60 min data, language-rich primitives)
       ▼
Stage-A LoRA checkpoint ──▶ inference (base + LoRA)
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
| Language | ≥ 1 clean English sentence per episode; ≥ 2 paraphrases per task recommended (the paper flagged single-string-per-task as a limitation) |
| Cameras | 3 views (top + L-wrist + R-wrist), fixed extrinsics, ordering identical to YAM |
| Action representation | Absolute joint positions; relative deltas computed by `convert_lerobot_to_gear.py` |

### 2.2 One-time embodiment registration (code edits, before any training)

| File | Change |
|---|---|
| `groot/vla/data/schema/embodiment_tags.py` | Add `MYROBOT = "myrobot"` to `EmbodimentTag` enum |
| `scripts/data/convert_lerobot_to_gear.py` | Add `"myrobot"` to `VALID_EMBODIMENT_TAGS` |
| `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | Clone the four `yam` entries (`modality_configs`, `transforms`, `metadata_versions`, `fps`) under the `myrobot` key; swap camera names to match your conversion |
| `groot/vla/configs/data/dreamzero/myrobot_relative.yaml` | New file. Copy `yam_relative.yaml`, rename `yam_data_root` → `myrobot_data_root`, change the embodiment key in `mixture_spec.dataset_path` |
| `scripts/train/myrobot_stage_a.sh` | New file. Copy `scripts/train/adam_stage_a.sh` (which mirrors `yam_training.sh`) and parameterize per §2.4 |

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

Adjust the index ranges if your 14-D vector is laid out differently.

### 2.4 Train Stage A LoRA

`scripts/train/adam_stage_a.sh` mirrors **YAM's recipe verbatim** (`scripts/train/yam_training.sh`), changing only the embodiment-specific data config:

```
output_dir=./checkpoints/adam_stage_a_lora
pretrained_model_path=./checkpoints/DreamZero-AgiBot
train_architecture=lora
max_steps=100000                # matches yam_training.sh
save_steps=10000                # matches yam_training.sh
save_total_limit=10
per_device_train_batch_size=4
training_args.learning_rate=1e-5
training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json
dataloader_pin_memory=false
dataloader_num_workers=1
save_lora_only=true
++action_head_cfg.config.skip_component_loading=true
++action_head_cfg.config.defer_lora_injection=true
```

> **Why 100k steps and not 5k?** A 5k-step attempt (`max_steps=5000, batch_size=1`) left the gripper output head collapsed to a per-head constant ≈ the training mean — offline open-loop eval reported joint MSE 0.001–0.004 rad² (healthy) but gripper MSE ≈294 (left) / 4.6 (right). YAM uses 400k effective sample-updates (100k × bs=4) — ~80× more gripper signal — which is empirically enough for the gripper head to absorb the absolute→relative semantic flip that `relative_action_keys` introduces. See §6 (Risks).
>
> If your dataset is much smaller than YAM's, you can lower `max_steps` proportionally, but keep `per_device_train_batch_size=4` and the LR.

> **Paper vs repo recipe.** The *paper* reports full-parameter fine-tuning and notes LoRA gave suboptimal results (footnote 7). The released *repo* (`yam_training.sh`) uses **LoRA** for YAM embodiment adaptation, and YAM has been validated on a real robot. **The current Adam run uses full fine-tuning** (paper-aligned) on Tillicum: `scripts/train/adam_stage_a_full.sh` + `scripts/slurm/train_adam_full_tillicum.slurm`, `train_architecture=full`, `save_lora_only=false`, DeepSpeed **`zero2_offload`** (CPU Adam; `zero3` breaks the frozen tiled VAE, plain `zero2` OOMs), measured **bs=4 / LR 2e-5 / 7500 steps / save_steps 1000** on 8× H200 (~31.5 s/it). See the README "Adam Stage A on UW Tillicum" tables and `docs/TILLICUM_SETUP.md` for the full measured config and the throughput sweep (bs 1–5 fit, bs=6 OOMs). The LoRA recipe below remains the lighter-weight alternative.

With 3× RTX PRO 6000 (96 GB each) set `NUM_GPUS=3`. `save_lora_only=true` keeps checkpoints small (~200 MB each); the inference server loads base + LoRA directly.

### 2.5 Stage A exit criteria

1. **Smoke test (`MAX_STEPS=10 bash scripts/train/adam_stage_a.sh`)** completes without errors — verifies data + embodiment registration.
2. **Loss curve** — action MSE decreases over the first 1000 steps. Instant plateau ⇒ state/action dim mis-mapped or wrong `relative_action_keys`.
3. **Quantitative open-loop check** — run `scripts/open_loop_adam.py` on the Stage A checkpoint against held-out frames from the training dataset:

   ```bash
   python scripts/open_loop_adam.py \
     --model_path ./checkpoints/adam_stage_a_lora/checkpoint-XXXX \
     --dataset_path ./data \
     --num_samples 200 --use_dataset_prompt
   ```

   Per-key MSE thresholds (rough, embodiment-dependent):

   | Key | Healthy | Under-trained / mis-mapped |
   |---|---|---|
   | `*_joint_pos` | ≲ 0.01 rad² | ≳ 0.1 rad² → check `relative_action_keys` + state slicing |
   | `*_gripper_pos` | ≲ (range)² × 1e−3 | stuck at training mean → keep training; if it plateaus, see §6 |

4. **Behavioral check** — load the Stage A LoRA into the inference server, issue **one of the Stage A training-set captions verbatim**, and confirm the trajectory is qualitatively correct in joint space. Not checking task success yet — just that the policy moves meaningfully toward the captioned target.

---

## 3. Inference

Serve the Stage A checkpoint with `serve_wam.py` and drive the real robot with `deploy_adam.py`. See [`../INFERENCE_COMMANDS.md`](../INFERENCE_COMMANDS.md) for the full commands. In short:

```bash
# Server (H100): base + Stage A LoRA, msgpack websocket on :5000
bash scripts/inference/serve_wam.sh ./checkpoints/adam_stage_a_lora/checkpoint-XXXX 5000 2 0,1

# Real robot client (reads ZED cameras over ZMQ + xArm state, sends to server):
python deploy_adam.py --no-dry-run --prompt "<one of the Stage A training captions>"
```

The server loads `base_model_name_or_path` automatically; ensure `./checkpoints/DreamZero-AgiBot/` is present.

---

## 4. Decision checklist (resolve before kicking off Stage A capture)

| Decision | Default | Why you might change it |
|---|---|---|
| LoRA (rank 4) vs full-FT | **Full-FT, paper-aligned** is the current Adam run (`train_architecture=full`, `zero2_offload`, bs=4, LR 2e-5, 7500 steps on 8× H200). LoRA (`max_steps=100000, bs=4, zero2`) is the YAM-aligned alternative | Full-FT needs `zero2_offload` (CPU RAM ~1.8 TB for the offloaded Adam state) and **~230 GB per checkpoint** (incl. optimizer state) → write to scrubbed, not projects. Do **not** shorten Stage A below ~50k effective sample-updates — see §6 (gripper convergence) |
| Stage A budget | 30–60 min, 10–12 tasks | Push to 90 min if your tasks need bimanual coordination or precision |
| Camera placement | Top + L-wrist + R-wrist, fixed extrinsics | Match YAM; lock the cell once capture starts |

---

## 5. Critical files

**Reference (do not edit):**
- `docs/DATASET_TO_GEAR_AND_TRAIN.md` — canonical single-stage recipe
- `docs/DATA_COLLECTION_STAGES_zh.md` — data team's capture spec
- `scripts/train/yam_training.sh` — template the Adam Stage A script mirrors
- `scripts/train/adam_stage_a.sh` — concrete Adam Stage A launcher (mirrors YAM exactly)
- `scripts/open_loop_adam.py` — offline open-loop checker (no server); §2.5 quantitative exit criterion
- `INFERENCE_COMMANDS.md` — serve_wam.py + deploy_adam.py real-robot inference
- `groot/vla/data/schema/lerobot.py` — Pydantic schema for `modality.json` validation
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:325-365` — what gets frozen vs trained
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:357-379` — camera view ordering

**To create / edit for a new embodiment:**
- `groot/vla/configs/data/dreamzero/myrobot_relative.yaml` (new)
- `scripts/train/myrobot_stage_a.sh` (new, copy of `adam_stage_a.sh`)
- `groot/vla/data/schema/embodiment_tags.py` (add enum entry)
- `scripts/data/convert_lerobot_to_gear.py` (add to `VALID_EMBODIMENT_TAGS`)
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` (add modality + transform blocks)

---

## 6. Risks

- **Camera / kinematics drift.** Confirm and lock camera extrinsics, camera order, joint layout, units, and FPS before capture. Drift silently destroys performance.
- **State/action layout.** The example splits 14-D as `[L-arm 6, L-grip 1, R-arm 6, R-grip 1]`. Confirm your robot's actual exposure before running `convert_lerobot_to_gear.py`.
- **Language diversity.** Paper footnote 12 (p.17): single-string-per-task caused YAM transfer to plateau. Use 2–3 paraphrases per task.
- **Idle frames.** The pretraining recipe filters idle actions; replicate this when delivering Stage A data or the embodiment LoRA will overfit to "do-nothing" frames.
- **Gripper head convergence (slow).** `DreamZero-AgiBot`'s gripper output head was pretrained to emit **absolute** gripper values (its `relative_action_keys` covers joints/head/waist only, not effector). `adam_relative.yaml` (like `yam_relative.yaml`) *does* include gripper in `relative_action_keys`, which silently re-frames gripper as a state-anchored delta. The flow-matching head needs ≳ 50k effective sample-updates (e.g. 12.5k steps × bs=4) to absorb that semantic flip; below that, both gripper heads collapse to a per-head constant ≈ the training mean and `open_loop_adam.py` reports gripper MSE ~ (range)². YAM's 100k × bs=4 = 400k clears this threshold by a wide margin. If you must train shorter, the cheap mitigation is to drop the two gripper keys from `relative_action_keys` so the gripper head retains the pretrain's absolute semantics.
