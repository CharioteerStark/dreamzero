#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --account=macsvlarobotics
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/%u/dreamzero/logs/timing_%j.out
set -uo pipefail
ALLOC=/gpfs/projects/macsvlarobotics
module load conda; conda activate $ALLOC/env
module load gcc/11.5.0; export CUDA_HOME=/gpfs/software/cuda/12.9.1; export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN DS_SKIP_CUDA_CHECK=1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $ALLOC/dreamzero
ARCH=${ARCH:?}; BS=${BS:?}; DS=${DS:?}; STEPS=${STEPS:-12}
W=$ALLOC/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
COMMON="report_to=wandb data=dreamzero/adam_relative wandb_project=dreamzero num_frames=33 action_horizon=24 num_views=3 model=dreamzero/vla model/dreamzero/action_head=wan_flow_matching_action_tf model/dreamzero/transform=dreamzero_cotrain num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 seed=42 training_args.learning_rate=1e-5 training_args.deepspeed=groot/vla/configs/deepspeed/${DS}.json save_steps=100000 training_args.warmup_ratio=0.05 output_dir=/gpfs/scrubbed/$USER/dreamzero/timing/${ARCH}_${DS}_bs${BS}_$SLURM_JOB_ID per_device_train_batch_size=$BS max_steps=$STEPS weight_decay=1e-5 save_total_limit=5 upload_checkpoints=false bf16=true tf32=true eval_bf16=true dataloader_pin_memory=false dataloader_num_workers=1 image_resolution_width=320 image_resolution_height=176 max_chunk_size=4 frame_seqlen=880 save_strategy=steps adam_data_root=/gpfs/scrubbed/$USER/dreamzero/data_merged dit_version=$W text_encoder_pretrained_path=$W/models_t5_umt5-xxl-enc-bf16.pth image_encoder_pretrained_path=$W/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth vae_pretrained_path=$W/Wan2.1_VAE.pth tokenizer_path=$ALLOC/dreamzero/checkpoints/umt5-xxl pretrained_model_path=$ALLOC/dreamzero/checkpoints/DreamZero-AgiBot"
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load(); print(chr(99))"
echo "===== TIMING arch=$ARCH ds=$DS bs=$BS steps=$STEPS host=$(hostname) ====="
if [ "$ARCH" = "lora" ]; then
  torchrun --nproc_per_node 8 --standalone groot/vla/experiment/experiment.py $COMMON train_architecture=lora save_lora_only=true ++action_head_cfg.config.skip_component_loading=true ++action_head_cfg.config.defer_lora_injection=true && echo "===== DONE arch=lora bs=$BS ====="
else
  torchrun --nproc_per_node 8 --standalone groot/vla/experiment/experiment.py $COMMON train_architecture=full save_lora_only=false ++action_head_cfg.config.skip_component_loading=true && echo "===== DONE arch=full bs=$BS ====="
fi
