#!/bin/bash
# DreamZero Adam — Stage B: Unseen-Task Specialization (LoRA on top of Stage A merged base)
#
# Purpose:
#   Acquire one or more new tasks DreamZero-AgiBot has never seen, on top of
#   the embodiment-grounded Stage A base.
#
# Reference: docs/STAGE_A_TO_B_PLAN.md §4
#
# Prerequisites:
#   - Stage A LoRA trained and merged into the base via scripts/utils/merge_lora.py,
#     producing ./checkpoints/adam_stage_a_merged.
#   - Stage B dataset converted via scripts/data/convert_lerobot_to_gear.py with
#     --embodiment-tag adam at ADAM_DATA_ROOT. SAME camera extrinsics, joint
#     layout, units, and FPS as Stage A (any drift silently destroys performance).
#   - Wan2.1-I2V-14B-480P + umt5-xxl tokenizer present (or auto-download).

export HYDRA_FULL_ERROR=1

# ============ CHANGE THESE VARIABLES ============
# Task name — used in the output directory only.
TASK_NAME=${TASK_NAME:-"task_a"}

# Stage B dataset path for this task (LeRobot v2 + GEAR metadata, embodiment_tag=adam).
ADAM_DATA_ROOT=${ADAM_DATA_ROOT:-"./data/adam_stage_b/${TASK_NAME}"}

# Output directory for Stage B LoRA checkpoints (per-task).
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/adam_stage_b_lora/${TASK_NAME}"}

# Number of GPUs (default: all visible GPUs).
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-3}

# Model weight paths.
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Stage-A-merged base (output of scripts/utils/merge_lora.py).
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/adam_stage_a_merged"}

# Stage B training budget (paper: 10-40h per task; for our setup 50k steps is the
# rough equivalent of the paper's per-task post-training. Half the Stage A LR
# for stability on top of a tuned base.)
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
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

# Validate dataset exists.
if [ ! -d "$ADAM_DATA_ROOT" ]; then
    echo "ERROR: Stage B dataset not found at $ADAM_DATA_ROOT"
    echo "Run: python scripts/data/convert_lerobot_to_gear.py --dataset-path $ADAM_DATA_ROOT --embodiment-tag adam ..."
    exit 1
fi
if [ ! -f "$ADAM_DATA_ROOT/meta/modality.json" ] || [ ! -f "$ADAM_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: $ADAM_DATA_ROOT is missing GEAR metadata."
    exit 1
fi

# Validate the Stage-A-merged base.
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "ERROR: Stage-A-merged base not found at $PRETRAINED_MODEL_PATH"
    echo "Run: python scripts/utils/merge_lora.py --base ./checkpoints/DreamZero-AgiBot --lora ./checkpoints/adam_stage_a_lora/checkpoint-<best> --output $PRETRAINED_MODEL_PATH"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
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
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
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
