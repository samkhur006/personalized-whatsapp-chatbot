#!/bin/bash

# Llama 3.1 Instruct 8B Training Script with Megatron-LM
# Uses config.yaml for configuration

set -e

# =============================================================================
# Load Configuration
# =============================================================================

# Check if config.yaml exists
CONFIG_FILE="./config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Load configuration from YAML
echo "📄 Loading configuration from config.yaml..."

# Print configuration summary
python3 -c "
import sys
sys.path.append('.')
from config_loader import load_config, print_config_summary
config = load_config('$CONFIG_FILE')
print_config_summary(config)
"

# Load environment variables
eval $(python3 -c "
import sys
sys.path.append('.')
from config_loader import load_config, config_to_env_vars
config = load_config('$CONFIG_FILE')
env_vars = config_to_env_vars(config)
for key, value in env_vars.items():
    print(f'export {key}=\"{value}\"')
")

# Paths
MEGATRON_PATH="../Megatron-LM"
CHECKPOINT_PATH=${1:-"$CHECKPOINT_DIR"}
TENSORBOARD_LOGS_PATH=${2:-"$TENSORBOARD_DIR"}
TOKENIZER_MODEL=${3:-"$TOKENIZER_MODEL"}
DATA_PREFIX=${4:-"$DATA_PREFIX"}

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "./data/processed"

# Environment variables for performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

# =============================================================================
# Training Configuration
# =============================================================================

# Distributed training setup
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Model parallelism (loaded from config.yaml)
# TP_SIZE, PP_SIZE, CP_SIZE are set from config

# Training hyperparameters (loaded from config.yaml)
# MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE, SEQ_LENGTH, etc. are set from config
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}

# Learning rate and optimization (loaded from config.yaml)
# LR, MIN_LR, WEIGHT_DECAY, WARMUP_STEPS, TRAIN_STEPS are set from config

# Precision (loaded from config.yaml)
# USE_FP8, USE_BF16 are set from config

# Data cache
DATA_CACHE_PATH="./cache/llama31_instruct_8b"
mkdir -p "$DATA_CACHE_PATH"

# =============================================================================
# Argument Arrays
# =============================================================================

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --kv-channels "$KV_CHANNELS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type "$POSITION_EMBEDDING_TYPE"
    --rotary-base "$ROTARY_BASE"
    --rotary-percent "$ROTARY_PERCENT"
    --attention-dropout "$ATTENTION_DROPOUT"
    --hidden-dropout "$HIDDEN_DROPOUT"
    --init-method-std "$INIT_METHOD_STD"
    --attention-backend "$ATTENTION_BACKEND"
    --normalization "$NORMALIZATION"
)

if [[ "$USE_SWIGLU" == "true" ]]; then
    MODEL_ARGS+=(--swiglu)
fi

if [[ "$APPLY_LAYERNORM_1P" == "true" ]]; then
    MODEL_ARGS+=(--apply-layernorm-1p)
fi

if [[ "$UNTIE_EMBEDDINGS_AND_OUTPUT_WEIGHTS" == "true" ]]; then
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi

if [[ "$DISABLE_BIAS_LINEAR" == "true" ]]; then
    MODEL_ARGS+=(--disable-bias-linear)
fi

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_STEPS
    --lr-warmup-iters $WARMUP_STEPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay $WEIGHT_DECAY
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

# Precision arguments
PRECISION_ARGS=()
if [[ "$USE_FP8" == "true" ]]; then
    PRECISION_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
elif [[ "$USE_BF16" == "true" ]]; then
    PRECISION_ARGS+=(
        "--bf16"
        "--grad-reduce-in-bf16"
    )
else
    PRECISION_ARGS+=(
        "--fp16"
    )
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
)

if [[ $PP_SIZE -gt 1 ]]; then
    MODEL_PARALLEL_ARGS+=(--pipeline-model-parallel-size $PP_SIZE)
fi

if [[ $TP_SIZE -gt 1 ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

# DDP arguments for multi-GPU training
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Data arguments
DATA_ARGS=(
    --data-path "$DATA_PREFIX"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --data-cache-path "$DATA_CACHE_PATH"
    --split "98,1,1"  # train, validation, test splits
    --no-create-attention-mask-in-dataloader
    --vocab-size 128256
    --num-workers 4
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-iters 10
    --eval-interval 100
    --save-interval 500
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --tensorboard-queue-size 5
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

# Optional: Load from checkpoint
if [[ -d "$CHECKPOINT_PATH" ]] && [[ -n "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
    EVAL_AND_LOGGING_ARGS+=(--load "$CHECKPOINT_PATH")
    echo "Found existing checkpoint, will resume training from: $CHECKPOINT_PATH"
fi

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "=========================================="
echo "Llama 3.1 Instruct 8B Training Setup"
echo "=========================================="
echo "Megatron-LM Path: $MEGATRON_PATH"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Tensorboard Logs: $TENSORBOARD_LOGS_PATH"
echo "Tokenizer Model: $TOKENIZER_MODEL"
echo "Data Prefix: $DATA_PREFIX"
echo "World Size: $WORLD_SIZE"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Learning Rate: $LR"
echo "Training Steps: $TRAIN_STEPS"
echo "TP/PP/CP: $TP_SIZE/$PP_SIZE/$CP_SIZE"
echo "Precision: $(if [[ "$USE_FP8" == "true" ]]; then echo "FP8"; elif [[ "$USE_BF16" == "true" ]]; then echo "BF16"; else echo "FP16"; fi)"
echo "=========================================="

# Check if Megatron-LM exists
if [[ ! -d "$MEGATRON_PATH" ]]; then
    echo "Error: Megatron-LM not found at $MEGATRON_PATH"
    echo "Please ensure Megatron-LM is properly installed."
    exit 1
fi

# Check if pretrain_gpt.py exists
PRETRAIN_SCRIPT="$MEGATRON_PATH/pretrain_gpt.py"
if [[ ! -f "$PRETRAIN_SCRIPT" ]]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT"
    exit 1
fi

# Check if using mock data or real data
if [[ "$DATA_PREFIX" == "MOCK" ]] || [[ "$DATA_PREFIX" == "/workspace/MOCK" ]]; then
    echo "🎭 Using mock data for testing..."
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size $VOCAB_SIZE
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --num-workers 1
    )
    echo "✅ Mock data configured."
elif [[ ! -f "${DATA_PREFIX}_text_document.bin" ]] && [[ ! -f "${DATA_PREFIX}.bin" ]]; then
    echo "❌ Error: Training data not found at $DATA_PREFIX"
    echo "Expected files:"
    echo "  - ${DATA_PREFIX}_text_document.bin"
    echo "  - ${DATA_PREFIX}_text_document.idx"
    echo ""
    echo "Solutions:"
    echo "1. Run data preprocessing first"
    echo "2. Use mock data by setting data_prefix: 'MOCK' in config.yaml"
    exit 1
else
    echo "✅ Found training data at $DATA_PREFIX"
    # Use the DATA_ARGS already defined above for real data
fi

# =============================================================================
# Launch Training
# =============================================================================

echo "Starting training at $(date)"
echo "Command: torchrun ${DISTRIBUTED_ARGS[*]} $PRETRAIN_SCRIPT ..."

cd "$MEGATRON_PATH"

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$PRETRAIN_SCRIPT" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PRECISION_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DDP_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"

echo "Training completed at $(date)"
