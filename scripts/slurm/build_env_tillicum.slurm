#!/bin/bash
# One-shot: build the DreamZero conda env on a Tillicum H200 node (has nvcc + internet).
# Compute nodes have nvcc (via modules) and outbound internet; the login node has
# neither nvcc nor GPUs, so the flash-attn compile must run here, not on login.
#
# Prereqs: conda env created at /gpfs/projects/macsvlarobotics/env, repo cloned at
#          /gpfs/projects/macsvlarobotics/dreamzero.
# Run:    sbatch scripts/slurm/build_env_tillicum.slurm
# Watch:  tail -f /gpfs/scrubbed/$USER/dreamzero/logs/build_<jobid>.out
#
# H200 = Hopper (sm_90) => H100 path: SKIP the GB200-only transformer_engine / TensorRT
# blocks from the repo README.

#SBATCH --job-name=build_env
#SBATCH --account=macsvlarobotics
#SBATCH --qos=interactive          # 8h cap, ≤2 GPU; 1 GPU is plenty for a build
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/build_%j.out

set -euo pipefail
ALLOC=/gpfs/projects/macsvlarobotics

module load conda
conda activate "$ALLOC/env"
module load gcc/13.4.0 cuda/12.9.1          # gcc FIRST (cuda is hierarchical under it)
export TORCH_CUDA_ARCH_LIST="9.0"           # restrict flash-attn build to Hopper => faster

cd "$ALLOC/dreamzero"
echo "host=$(hostname) python=$(python --version 2>&1) CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2

echo "=== [1/3] pip install -e . (PyTorch cu129 + deps) ==="
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

echo "=== [2/3] flash-attn (PREBUILT wheel) ==="
# flash-attn's source build downloads a prebuilt wheel then os.rename()s it into the
# pip cache, which throws EXDEV ([Errno 18] cross-device link) on GPFS. Install the
# matching prebuilt wheel directly (torch2.8 / cu12 / cp311 / cxx11abiTRUE) to skip that.
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

echo "=== [3/3] verify ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'cap', torch.cuda.get_device_capability())"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "BUILD_DONE"
