# VM Setup Guide: DreamZero / WAM Environment

Start here when provisioning a fresh VM. Follow sections in order.

---

## Step 1: Install Claude Code

Claude Code is the AI CLI you'll use to continue setup interactively on the VM.

```bash
# Install Node.js (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Set your Anthropic API key (get it from console.anthropic.com)
export ANTHROPIC_API_KEY=<your-key>

# Verify
claude --version
```

To launch an interactive session:
```bash
claude
```

---

## Step 2: System Prerequisites

```bash
sudo apt-get update && sudo apt-get install -y \
    git git-lfs \
    curl wget \
    build-essential \
    libssl-dev libffi-dev \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0
```

Verify CUDA 12.9+ is available:
```bash
nvcc --version
nvidia-smi
```

If CUDA is missing, install the CUDA 12.9 toolkit from the NVIDIA developer portal before proceeding.

---

## Step 3: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
conda init
```

---

## Step 4: Create the `dreamzero` Conda Environment

The training and inference scripts expect the conda env at `~/miniconda3/envs/dreamzero`.

```bash
conda create -n dreamzero python=3.11 -y
conda activate dreamzero
```

---

## Step 5: Clone the Repository

```bash
git clone <repo-url> ~/tony/dreamzero
cd ~/tony/dreamzero
```

Replace `<repo-url>` with the actual repo URL. If using a private repo, set up your SSH key or personal access token first.

---

## Step 6: Install Python Dependencies

Run these in order — each step must complete before the next.

```bash
cd ~/tony/dreamzero

# Install the package and all dependencies (PyTorch 2.8 + CUDA 12.9)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Install flash attention (this takes ~10–20 min to compile)
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

**GB200 only** — skip this block on H100:
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

**GB200 + TensorRT only** — skip on H100:
```bash
pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 \
    tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 \
    transformer_engine_torch==2.10.0
```

Verify the install:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed; print(deepspeed.__version__)"
```

---

## Step 7: Download Base Model Weights

These are required for both training and inference.

```bash
pip install "huggingface_hub[cli]"

# Set your HuggingFace token if needed
export HF_TOKEN=<your-hf-token>

# Wan2.1 video model backbone (~28 GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ~/tony/dreamzero/checkpoints/Wan2.1-I2V-14B-480P

# umt5-xxl text tokenizer
hf download google/umt5-xxl \
    --local-dir ~/tony/dreamzero/checkpoints/umt5-xxl
```

---

## Step 8: Download the DreamZero-AgiBot Pretrained Checkpoint

This is the starting point for fine-tuning on Adam / new embodiments (~45 GB).

```bash
hf download GEAR-Dreams/DreamZero-AgiBot \
    --repo-type model \
    --local-dir ~/tony/dreamzero/checkpoints/DreamZero-AgiBot
```

---

## Step 9: (Optional) Download DreamZero-DROID for Inference

Only needed if running inference with the DROID-trained checkpoint.

```bash
hf download GEAR-Dreams/DreamZero-DROID \
    --repo-type model \
    --local-dir ~/tony/dreamzero/checkpoints/DreamZero-DROID
```

---

## Step 10: Verify Checkpoint Layout

```
checkpoints/
├── Wan2.1-I2V-14B-480P/       # base video model
├── umt5-xxl/                  # text tokenizer
├── DreamZero-AgiBot/          # pretrain checkpoint for fine-tuning
└── DreamZero-DROID/           # (optional) DROID inference checkpoint
```

---

## Quick Smoke Test

Run the Adam inference server against a test checkpoint (replace path as needed):

```bash
cd ~/tony/dreamzero
bash scripts/inference/serve_wam.sh \
    ./checkpoints/DreamZero-AgiBot 5000 2 0,1
```

In a second terminal:
```bash
cd ~/tony/dreamzero
python test_client_AR.py --port 5000
```

First few inferences take a few minutes to warm up. After warm-up: ~3s on H100, ~0.6s on GB200.

---

## Launching Claude on This Repo

Once Claude Code is installed, launch it from the repo root so it has full project context:

```bash
cd ~/tony/dreamzero
claude
```

From there you can ask Claude to help with training scripts, dataset conversion, config changes, etc.

---

## Reference: Key Paths

| Path | Purpose |
|---|---|
| `~/miniconda3/envs/dreamzero` | Conda env used by all training/inference scripts |
| `checkpoints/DreamZero-AgiBot` | Base checkpoint for Adam fine-tuning |
| `checkpoints/Wan2.1-I2V-14B-480P` | WAN video backbone weights |
| `checkpoints/umt5-xxl` | Text tokenizer |
| `scripts/train/adam_stage_a.sh` | Stage A full fine-tune launch |
| `scripts/train/adam_stage_b.sh` | Stage B task-specific LoRA launch |
| `scripts/inference/serve_wam.sh` | WAM (Adam) inference server launcher |
| `docs/STAGE_A_TO_B_PLAN.md` | Two-stage fine-tune plan |
| `docs/DATASET_TO_GEAR_AND_TRAIN.md` | New embodiment onboarding guide |
