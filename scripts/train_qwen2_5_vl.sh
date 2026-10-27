#!/bin/bash
# LoRA SFT of Qwen2.5-VL-7B-Instruct on RefCOCO. Effective batch = NUM_GPUS x BATCH_SIZE x GRAD_ACCUM.
export WANDB_DISABLED=true

NUM_GPUS=4
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
DATA_PATH=./data/refcoco_train.json
IMAGE_FOLDER=./data/coco
OUTPUT_DIR=./checkpoints/qwen2.5-vl-7b_sft

BATCH_SIZE=16
GRAD_ACCUM=4
LR=5e-5
EPOCHS=2
LORA_R=16
LORA_ALPHA=32

torchrun --nproc_per_node $NUM_GPUS train.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --deepspeed ds_configs/zero2.json \
    --bf16 True \
    --tf32 True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to none
