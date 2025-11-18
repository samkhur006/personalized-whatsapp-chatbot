#!/bin/bash

# Docker-based Llama 3.1 Training with Megatron-LM
# Uses config.yaml for configuration

set -e

echo "🐳 Docker Megatron-LM Training"
echo "=============================="

# Configuration
DOCKER_IMAGE=${DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:24.07-py3"}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/training/config.yaml"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Load configuration using Python
echo "📄 Loading configuration from config.yaml..."
eval $(python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/training')
from config_loader import load_config, config_to_env_vars
config = load_config('$CONFIG_FILE')
env_vars = config_to_env_vars(config)
for key, value in env_vars.items():
    print(f'export {key}=\"{value}\"')
")

# Override with command line arguments if provided
CHECKPOINT_DIR=${1:-"$PROJECT_ROOT/training/$CHECKPOINT_DIR"}
TENSORBOARD_DIR=${2:-"$PROJECT_ROOT/training/$TENSORBOARD_DIR"}
TOKENIZER_MODEL=${3:-"$TOKENIZER_MODEL"}
DATA_PREFIX=${4:-"/workspace/$DATA_PREFIX"}

echo "Using configuration:"
echo "  Model: $BASE_MODEL"
echo "  Data: $DATA_PREFIX"
echo "  Batch Size: $MICRO_BATCH_SIZE micro, $GLOBAL_BATCH_SIZE global"
echo "  Learning Rate: $LR"
echo "  Training Steps: $TRAIN_STEPS"

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_DIR")"
mkdir -p "$(dirname "$TENSORBOARD_DIR")"
mkdir -p "$DATA_DIR/processed"

# Extract data directory from config
DATA_DIR="$PROJECT_ROOT/$(dirname "${DATA_PREFIX#/workspace/}")"
mkdir -p "$DATA_DIR"

# Check if preprocessed data exists
DATA_BIN_FILE="${DATA_PREFIX#/workspace/}_text_document.bin"
DATA_BIN_FILE="$PROJECT_ROOT/${DATA_BIN_FILE#/}"

if [[ ! -f "$DATA_BIN_FILE" ]]; then
    echo "📊 Preprocessing data..."
    docker run --rm --gpus all \
        -v "$PROJECT_ROOT:/workspace" \
        -w "/workspace/training" \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        $DOCKER_IMAGE \
        python preprocess_hinglish_data.py --use-hinglish-top \
            --output-dir "/workspace/$(dirname "${DATA_PREFIX#/workspace/}")" \
            --output-prefix "$(basename "${DATA_PREFIX#/workspace/}")" \
            --tokenizer-model "$TOKENIZER_MODEL"
    echo "✅ Data preprocessing completed"
else
    echo "✅ Preprocessed data exists: $DATA_BIN_FILE"
fi

# Launch training
echo "🚀 Starting training..."
docker run --rm --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    -v "$CHECKPOINT_DIR:/workspace/training/checkpoints/llama31_docker" \
    -v "$TENSORBOARD_DIR:/workspace/training/tensorboard_logs/llama31_docker" \
    -w "/workspace/training" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
    -e GPUS_PER_NODE=${GPUS_PER_NODE:-4} \
    -e TP_SIZE="$TP_SIZE" \
    -e PP_SIZE="$PP_SIZE" \
    -e CP_SIZE="$CP_SIZE" \
    -e MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    -e GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    -e SEQ_LENGTH="$SEQ_LENGTH" \
    -e MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS" \
    -e LR="$LR" \
    -e MIN_LR="$MIN_LR" \
    -e WARMUP_STEPS="$WARMUP_STEPS" \
    -e TRAIN_STEPS="$TRAIN_STEPS" \
    -e EVAL_INTERVAL="$EVAL_INTERVAL" \
    -e SAVE_INTERVAL="$SAVE_INTERVAL" \
    -e WEIGHT_DECAY="$WEIGHT_DECAY" \
    -e GRAD_CLIP="$GRAD_CLIP" \
    -e USE_BF16="$USE_BF16" \
    -e USE_FP8="$USE_FP8" \
    -e LOG_INTERVAL="$LOG_INTERVAL" \
    -e VOCAB_SIZE="$VOCAB_SIZE" \
    -e BASE_MODEL="$BASE_MODEL" \
    -e HF_TOKEN=${HF_TOKEN} \
    --name llama31_training_$(date +%Y%m%d_%H%M%S) \
    $DOCKER_IMAGE \
    bash train_llama31_instruct_8b.sh \
        "/workspace/training/checkpoints/llama31_docker" \
        "/workspace/training/tensorboard_logs/llama31_docker" \
        "$TOKENIZER_MODEL" \
        "$DATA_PREFIX"

echo "🎉 Training completed!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
echo "Configuration used: $CONFIG_FILE"
