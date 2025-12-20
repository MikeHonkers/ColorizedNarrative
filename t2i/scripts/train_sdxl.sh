#!/bin/bash

set -e

CONFIG_NAME=${1:-base}
CONFIG_FILE="config/training/sdxl_${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

MODEL_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['model']['pretrained_model_path'])")
VAE_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['model']['pretrained_vae_path'])")
DATASET_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['dataset']['path'])")
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['output_dir'])")

export MODEL_NAME="$MODEL_PATH"
export VAE_NAME="$VAE_PATH"
export DATASET_NAME="$DATASET_PATH"

IMAGE_COL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['dataset']['image_column'])")
CAPTION_COL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['dataset']['caption_column'])")

ACCELERATE_CMD="accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"
ACCELERATE_CMD="$ACCELERATE_CMD --pretrained_model_name_or_path=\"\$MODEL_NAME\""
ACCELERATE_CMD="$ACCELERATE_CMD --pretrained_vae_model_name_or_path=\"\$VAE_NAME\""
ACCELERATE_CMD="$ACCELERATE_CMD --dataset_name=\"\$DATASET_NAME\""
ACCELERATE_CMD="$ACCELERATE_CMD --image_column=\"$IMAGE_COL\""
ACCELERATE_CMD="$ACCELERATE_CMD --caption_column=\"$CAPTION_COL\""

RESOLUTION=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['dataset']['resolution'])")
BATCH_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['train_batch_size'])")
EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['num_train_epochs'])")
LR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['learning_rate'])")
MIXED_PREC=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['mixed_precision'])")
WORKERS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['dataloader_num_workers'])")
CHECKPOINT_STEPS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['checkpointing_steps'])")
LOGGING_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['logging_dir'])")
REPORT_TO=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['report_to'])")
SEED=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['seed'])")

RANDOM_FLIP=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training'].get('random_flip', False))" 2>/dev/null || echo "False")
USE_8BIT_ADAM=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training'].get('use_8bit_adam', False))" 2>/dev/null || echo "False")
VALIDATION_PROMPT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['training'].get('validation_prompt', ''))" 2>/dev/null || echo "")
NUM_VALIDATION_IMAGES=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['training'].get('num_validation_images', 0))" 2>/dev/null || echo "0")
VALIDATION_EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['training'].get('validation_epochs', 0))" 2>/dev/null || echo "0")

ACCELERATE_CMD="$ACCELERATE_CMD --resolution=$RESOLUTION"
if [ "$RANDOM_FLIP" = "True" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --random_flip"
fi
ACCELERATE_CMD="$ACCELERATE_CMD --train_batch_size=$BATCH_SIZE"
ACCELERATE_CMD="$ACCELERATE_CMD --gradient_accumulation_steps=1"
ACCELERATE_CMD="$ACCELERATE_CMD --mixed_precision=\"$MIXED_PREC\""
ACCELERATE_CMD="$ACCELERATE_CMD --allow_tf32"
if [ "$USE_8BIT_ADAM" = "True" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --use_8bit_adam"
fi
ACCELERATE_CMD="$ACCELERATE_CMD --gradient_checkpointing"
ACCELERATE_CMD="$ACCELERATE_CMD --dataloader_num_workers=$WORKERS"
ACCELERATE_CMD="$ACCELERATE_CMD --learning_rate=$LR"
ACCELERATE_CMD="$ACCELERATE_CMD --lr_scheduler=constant"
ACCELERATE_CMD="$ACCELERATE_CMD --lr_warmup_steps=0"
ACCELERATE_CMD="$ACCELERATE_CMD --num_train_epochs=$EPOCHS"
if [ -n "$VALIDATION_PROMPT" ] && [ "$VALIDATION_PROMPT" != "" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --validation_prompt=\"$VALIDATION_PROMPT\""
    if [ "$NUM_VALIDATION_IMAGES" != "0" ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --num_validation_images=$NUM_VALIDATION_IMAGES"
    fi
    if [ "$VALIDATION_EPOCHS" != "0" ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --validation_epochs=$VALIDATION_EPOCHS"
    fi
fi
ACCELERATE_CMD="$ACCELERATE_CMD --checkpointing_steps=$CHECKPOINT_STEPS"
ACCELERATE_CMD="$ACCELERATE_CMD --seed=$SEED"
ACCELERATE_CMD="$ACCELERATE_CMD --report_to=\"$REPORT_TO\""
ACCELERATE_CMD="$ACCELERATE_CMD --logging_dir=\"$LOGGING_DIR\""

TRAIN_TEXT_ENC=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['training'].get('train_text_encoder', False))" 2>/dev/null || echo "False")
if [ "$TRAIN_TEXT_ENC" = "True" ]; then
    TEXT_ENC_LR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['text_encoder_lr'])")
    ACCELERATE_CMD="$ACCELERATE_CMD --train_text_encoder"
    ACCELERATE_CMD="$ACCELERATE_CMD --text_encoder_lr=$TEXT_ENC_LR"
fi

ACCELERATE_CMD="$ACCELERATE_CMD --output_dir=\"$OUTPUT_DIR\""

echo "Training SDXL LoRA with config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

eval $ACCELERATE_CMD

