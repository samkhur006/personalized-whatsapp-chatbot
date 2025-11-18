#!/bin/bash

# Docker-based Llama 3.1 Instruct 8B Training with Megatron-LM
# This script sets up and runs training inside a Docker container

set -e

echo "🐳 Setting up Docker-based Megatron-LM Training"
echo "=============================================="

# =============================================================================
# Configuration
# =============================================================================

# Docker image (use NVIDIA's PyTorch container with CUDA support)
DOCKER_IMAGE=${DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:24.07-py3"}

# Project paths (absolute paths on host)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$PROJECT_ROOT/training"
MEGATRON_DIR="$PROJECT_ROOT/Megatron-LM"
DATA_DIR="$PROJECT_ROOT/data"

# Container paths
CONTAINER_WORKSPACE="/workspace"
CONTAINER_MEGATRON="$CONTAINER_WORKSPACE/Megatron-LM"
CONTAINER_TRAINING="$CONTAINER_WORKSPACE/training"
CONTAINER_DATA="$CONTAINER_WORKSPACE/data"

# Training configuration
CHECKPOINT_DIR=${1:-"$TRAINING_DIR/checkpoints/llama31_docker"}
TENSORBOARD_DIR=${2:-"$TRAINING_DIR/tensorboard_logs/llama31_docker"}
TOKENIZER_MODEL=${3:-"meta-llama/Llama-3.1-8B-Instruct"}
DATA_PREFIX=${4:-"$CONTAINER_DATA/processed/hinglish_pretrain"}

# GPU configuration
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}

# Create necessary directories
mkdir -p "$(dirname "$CHECKPOINT_DIR")"
mkdir -p "$(dirname "$TENSORBOARD_DIR")"
mkdir -p "$DATA_DIR/processed"

echo "Project Root: $PROJECT_ROOT"
echo "Megatron Dir: $MEGATRON_DIR"
echo "Training Dir: $TRAINING_DIR"
echo "Data Dir: $DATA_DIR"
echo "Docker Image: $DOCKER_IMAGE"

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "Running pre-flight checks..."

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not available. Please install nvidia-container-toolkit."
    exit 1
fi

# Check if Megatron-LM exists
if [[ ! -d "$MEGATRON_DIR" ]]; then
    echo "❌ Megatron-LM not found at $MEGATRON_DIR"
    exit 1
fi

echo "✅ All checks passed!"

# =============================================================================
# Data Preprocessing (if needed)
# =============================================================================

echo "Checking if preprocessed data exists..."

if [[ ! -f "$DATA_DIR/processed/hinglish_pretrain_text_document.bin" ]]; then
    echo "📊 Preprocessing data in Docker container..."
    
    docker run --rm --gpus all \
        -v "$PROJECT_ROOT:$CONTAINER_WORKSPACE" \
        -w "$CONTAINER_TRAINING" \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        $DOCKER_IMAGE \
        bash -c "
            pip install datasets pandas requests transformers tokenizers && \
            python preprocess_hinglish_data.py --use-hinglish-top
        "
    
    echo "✅ Data preprocessing completed in Docker"
else
    echo "✅ Preprocessed data already exists"
fi

# =============================================================================
# Training Script for Container
# =============================================================================

# Create training script that will run inside the container
cat > "$TRAINING_DIR/docker_train_internal.sh" << 'EOF'
#!/bin/bash

set -e

echo "🚀 Starting Llama 3.1 Training inside Docker container"
echo "======================================================"

# Environment variables for performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

# Training parameters (passed from host)
CHECKPOINT_PATH=${1:-"/workspace/training/checkpoints/llama31_docker"}
TENSORBOARD_PATH=${2:-"/workspace/training/tensorboard_logs/llama31_docker"}
TOKENIZER_MODEL=${3:-"meta-llama/Llama-3.1-8B-Instruct"}
DATA_PREFIX=${4:-"/workspace/data/processed/hinglish_pretrain"}

# Distributed training setup
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# Model and training parameters
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
LR=${LR:-5e-5}
TRAIN_STEPS=${TRAIN_STEPS:-5000}

echo "Starting training with:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Tensorboard: $TENSORBOARD_PATH"
echo "  Data: $DATA_PREFIX"
echo "  GPUs: $GPUS_PER_NODE"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Learning Rate: $LR"

# Install additional dependencies
pip install transformer-engine[pytorch] datasets pandas requests

# Change to Megatron directory
cd /workspace/Megatron-LM

# Install Megatron in development mode
pip install -e .

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_PATH")"

# Convert HuggingFace model to Megatron format (if needed)
CONVERTED_MODEL_PATH="/workspace/converted_models/llama31_8b_megatron"
mkdir -p "$(dirname "$CONVERTED_MODEL_PATH")"

if [[ ! -d "$CONVERTED_MODEL_PATH" ]]; then
    echo "Converting HuggingFace model to Megatron format..."
    python tools/checkpoint_util.py \
        --model-type GPT \
        --loader llama_mistral \
        --saver megatron \
        --load-dir "$TOKENIZER_MODEL" \
        --save-dir "$CONVERTED_MODEL_PATH" \
        --tokenizer-model "$TOKENIZER_MODEL" \
        --target-tensor-parallel-size $TP_SIZE \
        --target-pipeline-parallel-size $PP_SIZE \
        --bf16
    echo "✅ Model conversion completed"
fi

# Run training (continuing from pretrained Llama 3.1)
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_gpt.py \
    --load "$CONVERTED_MODEL_PATH" \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --init-method-std 0.0134 \
    --attention-backend fused \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --normalization RMSNorm \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_STEPS \
    --lr-warmup-iters 100 \
    --lr $LR \
    --min-lr 5e-6 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --bf16 \
    --grad-reduce-in-bf16 \
    --cross-entropy-loss-fusion \
    --calculate-per-token-loss \
    --manual-gc \
    --empty-unused-memory-level 1 \
    --tensor-model-parallel-size $TP_SIZE \
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --data-path "$DATA_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --vocab-size 128256 \
    --split "98,1,1" \
    --no-create-attention-mask-in-dataloader \
    --num-workers 4 \
    --log-interval 10 \
    --eval-iters 10 \
    --eval-interval 100 \
    --save-interval 500 \
    --log-throughput \
    --ckpt-format torch_dist \
    --distributed-timeout-minutes 60 \
    --save "$CHECKPOINT_PATH" \
    --tensorboard-dir "$TENSORBOARD_PATH" \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard

echo "✅ Training completed!"
EOF

chmod +x "$TRAINING_DIR/docker_train_internal.sh"

# =============================================================================
# Launch Training Container
# =============================================================================

echo "🚀 Launching training container..."

# Pull the latest image
echo "Pulling Docker image: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE

# Run training in Docker container
docker run --rm --gpus all \
    -v "$PROJECT_ROOT:$CONTAINER_WORKSPACE" \
    -v "$CHECKPOINT_DIR:$CONTAINER_WORKSPACE/training/checkpoints/llama31_docker" \
    -v "$TENSORBOARD_DIR:$CONTAINER_WORKSPACE/training/tensorboard_logs/llama31_docker" \
    -w "$CONTAINER_WORKSPACE" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=32g \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e NUM_NODES=$NUM_NODES \
    -e TP_SIZE=${TP_SIZE:-1} \
    -e PP_SIZE=${PP_SIZE:-1} \
    -e CP_SIZE=${CP_SIZE:-1} \
    -e MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1} \
    -e GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32} \
    -e SEQ_LENGTH=${SEQ_LENGTH:-4096} \
    -e LR=${LR:-5e-5} \
    -e TRAIN_STEPS=${TRAIN_STEPS:-5000} \
    -e HF_TOKEN=${HF_TOKEN} \
    --name llama31_training_$(date +%Y%m%d_%H%M%S) \
    $DOCKER_IMAGE \
    bash training/docker_train_internal.sh \
        "$CONTAINER_WORKSPACE/training/checkpoints/llama31_docker" \
        "$CONTAINER_WORKSPACE/training/tensorboard_logs/llama31_docker" \
        "$TOKENIZER_MODEL" \
        "$DATA_PREFIX"

echo "🎉 Docker training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "TensorBoard logs: $TENSORBOARD_DIR"
