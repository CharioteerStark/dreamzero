#!/bin/bash
# DreamZero Adam — Stage A FULL fine-tune (vs the LoRA recipe in adam_stage_a.sh).
#
# Full-parameter finetune of DreamZero-AgiBot (paper's approach, footnote 7). Validated on
# an 8x RTX PRO 6000 (96GB) box and adapted for Tillicum 8x H200.
#
# Key differences from adam_stage_a.sh (LoRA):
#   train_architecture=full, save_lora_only=false, DeepSpeed=zero2_offload (CPU Adam),
#   per_device_bs=1 (bs>1 OOMs on the multi-view grid). NOT zero3 (zero3 shards the FROZEN
#   VAE and breaks its tiled encode: "Sizes of tensors must match ... 384 vs 96"). Plain
#   zero2 (no offload) OOMs VRAM (full bf16 ~16.5B weights replicate per GPU); zero2_offload
#   fits ~67-68 GB/GPU at bs1 with ~198 GB fp32 Adam state offloaded to CPU.
#
# IMPORTANT env (set by the SLURM wrapper, not here):
#   DS_SKIP_CUDA_CHECK=1  -- else DeepSpeedCPUAdam JIT fails (CUDAMismatchException) when the
#                            system/toolkit CUDA != torch's bundled CUDA.
#   module load gcc; CUDA_HOME=<toolkit>  -- do NOT load the cuda module's libs (cuDNN clash).

export HYDRA_FULL_ERROR=1

CONDA_ENV=${CONDA_ENV:-/home/thematrix/miniconda3/envs/dreamzero}
if [ -x "$CONDA_ENV/bin/torchrun" ]; then
  TORCHRUN=$CONDA_ENV/bin/torchrun
  unset VIRTUAL_ENV; export PATH="$CONDA_ENV/bin:$PATH"
  export CONDA_PREFIX=$CONDA_ENV; export CONDA_DEFAULT_ENV=$(basename "$CONDA_ENV")
else
  TORCHRUN=torchrun
fi

# ============ CHANGE THESE ============
ADAM_DATA_ROOT=${ADAM_DATA_ROOT:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/adam_stage_a_full"}   # full-FT ckpts ~230 GB each!
if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-8}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-AgiBot"}

# Full-FT budget (mirrors YAM LR; 30k steps per the validated setup).
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_STEPS=${SAVE_STEPS:-5000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}     # repo floor is 5 ("standardized eval"); ckpts go to scrubbed
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}           # bs>1 OOMs on the multi-view grid
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}
# =====================================

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl not found at $TOKENIZER_DIR. Downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -f "$ADAM_DATA_ROOT/meta/modality.json" ] || [ ! -f "$ADAM_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: $ADAM_DATA_ROOT missing GEAR metadata. Run convert_lerobot_to_gear.py --embodiment-tag adam first."; exit 1
fi
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: base checkpoint not found at $PRETRAINED_MODEL_PATH"; exit 1
fi

$TORCHRUN --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/adam_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=$SAVE_TOTAL_LIMIT \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    adam_data_root=$ADAM_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.skip_component_loading=true
