#!/bin/bash

# Llama 3.1 Instruct 8B Training Script with Megatron-LM
# Optimized for instruction tuning and fine-tuning

set -e

# =============================================================================
# Configuration Parameters
# =============================================================================

# Paths
MEGATRON_PATH="../Megatron-LM"
CHECKPOINT_PATH=${1:-"./checkpoints/llama31_instruct_8b"}
TENSORBOARD_LOGS_PATH=${2:-"./tensorboard_logs/llama31_instruct_8b"}
TOKENIZER_MODEL=${3:-"meta-llama/Llama-3.1-8B-Instruct"}  # HuggingFace model path
DATA_PREFIX=${4:-"./data/processed/hinglish_instruct"}      # Preprocessed data prefix

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
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Model parallelism
TP_SIZE=${TP_SIZE:-1}        # Tensor parallel size
PP_SIZE=${PP_SIZE:-1}        # Pipeline parallel size
CP_SIZE=${CP_SIZE:-1}        # Context parallel size

# Training hyperparameters
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}  # Smaller for instruction tuning
SEQ_LENGTH=${SEQ_LENGTH:-4096}              # Reduced for instruction data
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}

# Learning rate and optimization
LR=${LR:-5e-5}              # Lower LR for instruction tuning
MIN_LR=${MIN_LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-100}
TRAIN_STEPS=${TRAIN_STEPS:-5000}  # Adjust based on your dataset size

# Precision
USE_FP8=${USE_FP8:-false}
USE_BF16=${USE_BF16:-true}

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
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --normalization RMSNorm
)

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
    --log-batch-size-to-tensorboard
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

# Check if data exists (unless using mock data)
if [[ ! -f "${DATA_PREFIX}_text_document.bin" ]] && [[ ! -f "${DATA_PREFIX}.bin" ]]; then
    echo "Warning: Training data not found at $DATA_PREFIX"
    echo "Please run the data preprocessing script first, or use mock data for testing."
    echo "To use mock data, set DATA_PREFIX to 'MOCK'"
    
    if [[ "$DATA_PREFIX" != "MOCK" ]]; then
        read -p "Continue with mock data? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        # Switch to mock data
        DATA_ARGS=(
            --mock-data
            --tokenizer-type NullTokenizer
            --vocab-size 128256
            --data-cache-path "$DATA_CACHE_PATH"
            --split "99,1,0"
            --no-create-attention-mask-in-dataloader
            --num-workers 1
        )
        echo "Switched to mock data for testing."
    fi
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
