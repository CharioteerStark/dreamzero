# Tillicum Setup Guide — DreamZero / Adam Stage A (terminal / SSH)

UW Hyak **Tillicum** is a GPU-only SLURM cluster: NVIDIA **H200 (141 GB)**, 8 per node,
~8 CPU + ~200 GB RAM per GPU. No `sudo`/`apt` (shared cluster) — software comes from
`module load` + conda. Account: **macsvlarobotics**. Billing: 100 free GPU-hr, then
`$0.90/GPU-hr` (worktag PG225985). This guide is the **terminal/SSH** path; the Open
OnDemand/Jupyter web path (needs the VPN) is in §11.

> H200 = **Hopper** (sm_90), same family as H100. Follow the **H100 path** in the repo
> README — **skip the GB200-only** transformer_engine / TensorRT blocks.

Build once on `/gpfs` (shared), use from every node. Sequence:
**access → storage → env → repo → build → stage weights/data → verify → smoke → train → monitor → exfiltrate.**

---

## 0. Access (no VPN needed for SSH)

SSH works directly, off-campus, with Duo 2FA — the VPN is **only** for the OnDemand web
portal (§11).

```bash
ssh <UWNetID>@tillicum.hyak.uw.edu      # approve the Duo push / enter passcode
hostname                                 # confirms a login node, e.g. tillicum-login01
```

> Repeated failed logins → ~1-hour IP ban. Make sure Duo 2FA is enrolled at identity.uw.edu first.
> Login nodes are for setup, staging, and `sbatch` submission only — never run training on them.

---

## 1. Storage layout

| Path | Quota / policy | Use for |
|---|---|---|
| `/gpfs/home/<netid>` | 10 GB, backed up | dotfiles only — **not** env/data |
| `/gpfs/projects/macsvlarobotics` | 1 TB, backed up daily, **PURGED at project end** | repo, conda env, weights, checkpoints |
| `/gpfs/scrubbed/<netid>` | large, **no backup, purged after 60d idle** | dataset, logs, scratch |

```bash
ALLOC=/gpfs/projects/macsvlarobotics
SCRUBBED=/gpfs/scrubbed/$USER/dreamzero
mkdir -p $ALLOC $SCRUBBED/logs $SCRUBBED/data_merged
```

Keep conda off the 10 GB home dir — create `~/.condarc`:

```yaml
envs_dirs:
  - /gpfs/projects/macsvlarobotics/conda/envs
pkgs_dirs:
  - /gpfs/projects/macsvlarobotics/conda/pkgs
```

---

## 2. Conda environment (on `/gpfs`, sibling of the repo)

```bash
module load conda
conda create --prefix $ALLOC/env python=3.11 -y     # repo requires Python 3.11
conda activate $ALLOC/env
```

The training scripts find this env via `CONDA_ENV=$ALLOC/env` (already wired into the
SLURM scripts) — no `~/miniconda3` like the Nebius VM.

---

## 3. Clone the repo

```bash
cd $ALLOC
git clone <repo-url> dreamzero      # => /gpfs/projects/macsvlarobotics/dreamzero
cd dreamzero
```

Layout that the SLURM scripts assume: repo at `$ALLOC/dreamzero`, env at `$ALLOC/env`
(sibling), checkpoints at `$ALLOC/dreamzero/checkpoints` (gitignored).

---

## 4. Build dependencies — in an interactive GPU session

`flash-attn` compiles from source and needs `nvcc` + a GPU-class toolchain. Don't do this
on the login node — grab a cheap interactive GPU session (counts against the free 100 GPU-hr):

```bash
salloc --account=macsvlarobotics --qos=interactive --gpus=1 --time=02:00:00
# ---- now on a compute node ----
module load conda
conda activate /gpfs/projects/macsvlarobotics/env
cd /gpfs/projects/macsvlarobotics/dreamzero

# nvcc / CUDA toolkit (12.9+ to match the cu129 wheels). Check what's available:
module avail cuda 2>&1 | head        # then load the closest 12.9+, e.g.:
module load cuda/12.9                 # (adjust to the exact module name shown)
export CUDA_HOME=$CUDA_HOME           # set by the module; flash-attn build reads this

# system libs without sudo: get ffmpeg/OpenGL via conda
conda install -c conda-forge ffmpeg -y

# 1) package + deps (PyTorch 2.8+ / CUDA 12.9)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# 2) flash attention — install the PREBUILT wheel, do NOT compile from source.
#    Source build does os.rename() across /gpfs <-> node-local tmp -> "[Errno 18] cross-device link".
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# H200 is Hopper => SKIP the GB200-only transformer_engine / TensorRT blocks from the README.

# verify, then release the session
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
exit    # frees the GPU
```

> If `import cv2` later complains about `libGL`, `conda install -c conda-forge opencv` (or
> swap to `opencv-python-headless`) — there's no system `libgl1` to apt-install here.

> **CUDA: build vs runtime differ (important).** The `cuda` module is fine for *building*
> (flash-attn / cpu_adam need `nvcc`). But for *runtime*, do **not** load the `cuda` module —
> its bundled cuDNN 9.14 shadows torch's and crashes at the first conv3d (Wan VAE) with
> `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. The SLURM scripts therefore use **`module load
> gcc` + manual `export CUDA_HOME=/gpfs/software/cuda/12.9.1`** (gives deepspeed `nvcc` while
> leaving torch's cuDNN on the path). **Full-FT also needs `gcc/11.5.0` (not 13.4)** — gcc 13's
> `-march=native` emits an AVX512-FP16 op the assembler rejects when building DeepSpeed's
> `cpu_adam` — plus `DS_SKIP_CUDA_CHECK=1` and a single-process `cpu_adam` pre-build (8 ranks
> racing to JIT-build it lose the `.so`). All of this is already wired into the SLURM wrappers.

---

## 5. Stage weights and dataset (login node — it has internet)

Compute nodes may have no outbound internet, so download on the login node into `/gpfs`.

```bash
exit   # back to the login node if still in salloc
module load conda && conda activate /gpfs/projects/macsvlarobotics/env
cd /gpfs/projects/macsvlarobotics/dreamzero
export HF_TOKEN=<your-hf-token>

# Wan2.1 backbone (~28 GB), umt5-xxl tokenizer, AgiBot base (~45 GB) — into the repo's checkpoints/
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P
hf download google/umt5-xxl            --local-dir checkpoints/umt5-xxl
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir checkpoints/DreamZero-AgiBot
```

Datasets → **scrubbed** (large, no backup):

- **`wam_geer_us`** — the **current full-FT training set** (new WAM capture: 84 ep / 97 800 frames / `adam` / 30 fps, in GEAR format). Built by decoding the raw ZED `.svo2` + teleop JSON (`scripts/data/convert_wam_raw_to_lerobot.py` → `convert_lerobot_to_gear.py`) on Nebius, published to HF `RichtechRD/wa-adam-geer-us` (private), then pulled on the login node:
  ```bash
  hf download RichtechRD/wa-adam-geer-us --repo-type dataset \
    --local-dir /gpfs/scrubbed/$USER/dreamzero/wam_geer_us
  # verify: meta/{modality,embodiment,info}.json present, embodiment_tag = adam
  ```
- **`data_merged`** (322 ep) — older merged set, used only for LoRA smoke/throughput probes. From the Nebius FS (Globus beats scp at this size):
  ```bash
  rsync -avz /mnt/filesystem-c5/dreamzero/data_merged/ \
    <netid>@tillicum.hyak.uw.edu:/gpfs/scrubbed/<netid>/dreamzero/data_merged/
  ```

> ⚠️ Base checkpoint must be **DreamZero-AgiBot**, not raw Wan2.1 — training the LoRA on the
> wrong base is the known corruption bug. The SLURM script points `PRETRAINED_MODEL_PATH` here.

Expected layout:
```
/gpfs/projects/macsvlarobotics/
├── env/                                   # conda --prefix env
└── dreamzero/                             # repo (= $PROJECT in the SLURM scripts)
    └── checkpoints/{Wan2.1-I2V-14B-480P, umt5-xxl, DreamZero-AgiBot, adam_stage_a_lora/}
/gpfs/scrubbed/<netid>/dreamzero/
├── data_merged/                           # dataset
└── logs/                                  # SLURM logs
```

---

## 6. Verify (quick, login node)

```bash
ls checkpoints/DreamZero-AgiBot && ls /gpfs/scrubbed/$USER/dreamzero/data_merged/meta/
# data_merged must contain meta/modality.json + meta/embodiment.json (GEAR metadata)
```

---

## 7. Smoke test — before paying for the full node

Validates env + data + the DeepSpeed/NCCL path on a few steps, within the free tier.

```bash
cd /gpfs/projects/macsvlarobotics/dreamzero
sbatch scripts/slurm/smoke_adam_tillicum.slurm     # mode A: 1 GPU, debug QOS, MAX_STEPS=20
# then edit the file's commented directives to mode B (8 GPU) and re-run to exercise multi-GPU NCCL
squeue -u $USER
tail -f /gpfs/scrubbed/$USER/dreamzero/logs/adam_smoke_*.out
```

---

## 8. Train — chained 24h slices

`normal` QOS caps at 24 h. The trainer auto-resumes from `OUTPUT_DIR`
(`groot/vla/experiment/base.py:644`), so identical slices chain into one long run.
`submit_chain.sh` takes the slice count and (optionally) the SLURM wrapper to chain:

```bash
# Full fine-tune (current run): bs=4, zero2_offload, 7500 steps, ckpts -> scrubbed
bash scripts/slurm/submit_chain.sh 4 train_adam_full_tillicum.slurm

# LoRA (YAM-exact alternative): defaults to train_adam_tillicum.slurm
bash scripts/slurm/submit_chain.sh 4

squeue -u $USER
```

> ⚠️ **`save_steps` MUST fit inside the 24 h walltime.** At full-FT bs=4 (~31.5 s/it) only
> ~2700 steps complete per slice, so `save_steps=3000` would **never** write a checkpoint —
> the `afterany` next slice then finds nothing and restarts from step 0 *forever*, burning
> ~$173/slice for zero progress. Rule: `save_steps < (24h − startup) / s-per-it`, with margin
> for the ~230 GB save to flush. The full-FT wrapper uses **`save_steps=1000`** (≈ every 8.75 h).

To run uninterrupted instead of chaining: email help@uw.edu for the **`long`** QOS, set
`--qos=long` in the wrapper, and submit it once with `sbatch`.

---

## 9. Monitor & cost

```bash
hyakusage                       # GPU-hr + cost this billing cycle, by account + QOS
squeue -u $USER                 # queue / running
seff <jobid>                    # per-job GPU/mem efficiency (after completion)
```

- Full node = **8 GPU-hr per wall-clock hour** → free 100 GPU-hr ≈ **12.5 h** on a full node.
- One 24h slice ≈ **$173**. Set an **enforced monthly budget** via help@uw.edu (subject "Tillicum")
  so a runaway chain can't silently bill PG225985.
- **Maintenance: 2nd Tuesday/month** — a slice may be killed; checkpoints + auto-resume cover it.

---

## 10. Exfiltrate results (storage is purged)

Both checkpoint locations are temporary: `/gpfs/projects` is backed up but **deleted when the
project closes**; `/gpfs/scrubbed` has **no backup and is purged after 60 days idle**. Copy
checkpoints off-cluster before either happens.

```bash
# Full-FT checkpoints (~230 GB each) live on SCRUBBED:
rsync -avz <netid>@tillicum.hyak.uw.edu:/gpfs/scrubbed/<netid>/dreamzero/adam_stage_a_full/ \
  /mnt/filesystem-c5/dreamzero/checkpoints/adam_stage_a_full/

# LoRA adapters (~200 MB each) live on PROJECTS:
rsync -avz <netid>@tillicum.hyak.uw.edu:/gpfs/projects/macsvlarobotics/dreamzero/checkpoints/adam_stage_a_lora/ \
  /mnt/filesystem-c5/dreamzero/checkpoints/adam_stage_a_lora/
```

> A full-FT checkpoint includes the fp32 optimizer state. To serve it, load the model weights
> only (or strip optimizer shards) — the inference server doesn't need the Adam state.

---

## 11. Later: Open OnDemand / Jupyter (needs the VPN)

For interactive dev/eval/visualization (not the 8-GPU training run):

1. Connect **Husky OnNet** VPN (F5 BIG-IP Edge Client; NetID + Duo; "UW Campus Network
   Traffic Only" profile is enough). Required off-campus for OnDemand.
2. Open **https://tillicum-ood.hyak.uw.edu/** → Interactive Apps → Jupyter.
3. Form: Account `macsvlarobotics`, QOS `interactive` (≤8 h, ≤2 GPU), **1 GPU**, modest walltime.
4. Make your env a kernel (one-time):
   ```bash
   module load conda && conda activate /gpfs/projects/macsvlarobotics/env
   conda install ipykernel -y
   python -m ipykernel install --user --name dreamzero --display-name "Python (dreamzero)"
   ```
   Pick "Python (dreamzero)" in JupyterLab.
5. **Click Delete on the session card when done** — an idle session still bills GPU-hr.

---

## Reference: key paths & commands

| Item | Value |
|---|---|
| SSH | `ssh <netid>@tillicum.hyak.uw.edu` |
| Account | `macsvlarobotics` (`#SBATCH --account=macsvlarobotics`) |
| Conda env | `/gpfs/projects/macsvlarobotics/env` (`module load conda` first) |
| Repo | `/gpfs/projects/macsvlarobotics/dreamzero` |
| Dataset (full-FT) | `/gpfs/scrubbed/<netid>/dreamzero/wam_geer_us` |
| Dataset (LoRA probes) | `/gpfs/scrubbed/<netid>/dreamzero/data_merged` |
| Smoke | `sbatch scripts/slurm/smoke_adam_tillicum.slurm` |
| Train (full-FT) | `bash scripts/slurm/submit_chain.sh 4 train_adam_full_tillicum.slurm` |
| Train (LoRA) | `bash scripts/slurm/submit_chain.sh 4` |
| Full-FT ckpts | `/gpfs/scrubbed/<netid>/dreamzero/adam_stage_a_full` (~230 GB each) |
| Monitor | `hyakusage`, `squeue -u $USER`, `seff <jobid>` |
| OnDemand | `https://tillicum-ood.hyak.uw.edu/` (VPN required off-campus) |
