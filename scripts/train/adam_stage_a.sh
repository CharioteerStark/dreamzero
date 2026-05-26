#!/bin/bash
# DreamZero Adam — Stage A: Embodiment Adaptation (LoRA of DreamZero-AgiBot)
#
# Purpose:
#   Teach the model how the Adam robot (14-DoF bimanual, 3 cameras) moves,
#   how its 14-D state/action maps to motion primitives, and how natural-language
#   instructions tie to those motions. NOT a task-success stage.
#
# Recipe matches scripts/train/yam_training.sh (LoRA, 100k steps, bs=4 per device,
# LR=1e-5, DeepSpeed zero2, save_lora_only=true). Only save_steps stays low so we
# get frequent checkpoints early in training.
#
# Reference: docs/STAGE_A_TO_B_PLAN.md §2; scripts/train/yam_training.sh
#
# Prerequisites:
#   - Adam dataset converted via scripts/data/convert_lerobot_to_gear.py with
#     --embodiment-tag adam at ADAM_DATA_ROOT. meta/embodiment.json must have
#     "embodiment_tag": "adam". Videos in observation.images.{top,left_wrist,right_wrist}.
#   - Wan2.1-I2V-14B-480P weights at $WAN_CKPT_DIR (auto-downloaded if missing).
#   - umt5-xxl tokenizer at $TOKENIZER_DIR (auto-downloaded if missing).
#   - DreamZero-AgiBot pretrained checkpoint at ./checkpoints/DreamZero-AgiBot.

export HYDRA_FULL_ERROR=1

# Pin to the conda dreamzero env (pinned torch 2.8.0+cu129 / transformers 4.51.3 /
# deepspeed 0.19.0 / peft 0.5.0 + nvcc). Use absolute paths so any active venv
# (e.g. RLinf) cannot leak into PATH resolution.
CONDA_ENV=${CONDA_ENV:-/home/thematrix/miniconda3/envs/dreamzero}
TORCHRUN=$CONDA_ENV/bin/torchrun
unset VIRTUAL_ENV
export PATH="$CONDA_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CONDA_PREFIX=$CONDA_ENV
export CONDA_DEFAULT_ENV=dreamzero

# ============ CHANGE THESE VARIABLES ============
# Dataset path (Adam in LeRobot v2 + GEAR metadata format).
ADAM_DATA_ROOT=${ADAM_DATA_ROOT:-"./data"}

# Output directory for Stage A LoRA checkpoints (~200 MB each with save_lora_only=true).
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/adam_stage_a_lora"}

# Number of GPUs (default: all visible GPUs).
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# Model weight paths (download from HuggingFace if not already present).
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Base pretrained checkpoint to fine-tune from.
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-AgiBot"}

# Training budget — matches yam_training.sh except SAVE_STEPS (kept low for frequent checkpoints).
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
# ================================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists and is converted (has GEAR metadata).
if [ ! -d "$ADAM_DATA_ROOT" ]; then
    echo "ERROR: Adam dataset not found at $ADAM_DATA_ROOT"
    echo "Run: python scripts/data/convert_lerobot_to_gear.py --dataset-path $ADAM_DATA_ROOT --embodiment-tag adam ..."
    exit 1
fi
if [ ! -f "$ADAM_DATA_ROOT/meta/modality.json" ] || [ ! -f "$ADAM_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: $ADAM_DATA_ROOT is missing GEAR metadata (meta/modality.json or meta/embodiment.json)."
    echo "Run scripts/data/convert_lerobot_to_gear.py with --embodiment-tag adam first."
    exit 1
fi

# Validate the base pretrained checkpoint.
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: Base checkpoint not found at $PRETRAINED_MODEL_PATH"
    echo "Clone: git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot $PRETRAINED_MODEL_PATH"
    exit 1
fi

$TORCHRUN --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/adam_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
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
    save_lora_only=true \
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
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
