#!/bin/bash

# Environment Setup Script for Llama 3.1 Instruct 8B Training with Megatron-LM
# This script sets up the necessary environment and dependencies

set -e

echo "=========================================="
echo "Setting up Llama 3.1 Training Environment"
echo "=========================================="

# =============================================================================
# Configuration
# =============================================================================

PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
CUDA_VERSION=${CUDA_VERSION:-"12.1"}
PYTORCH_VERSION=${PYTORCH_VERSION:-"2.1.0"}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGATRON_PATH="$PROJECT_ROOT/Megatron-LM"
TRAINING_PATH="$PROJECT_ROOT/training"

echo "Project Root: $PROJECT_ROOT"
echo "Megatron-LM Path: $MEGATRON_PATH"
echo "Training Path: $TRAINING_PATH"

# =============================================================================
# Check Prerequisites
# =============================================================================

echo "Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VER"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected. Training will use CPU (not recommended)"
fi

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# =============================================================================
# Install Dependencies
# =============================================================================

echo "Installing Python dependencies..."

# Core dependencies
pip install --upgrade pip setuptools wheel

# PyTorch with CUDA support
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Transformer Engine (for FP8 support)
echo "Installing Transformer Engine..."
pip install transformer-engine[pytorch]

# HuggingFace libraries
echo "Installing HuggingFace libraries..."
pip install transformers datasets tokenizers accelerate

# Data processing libraries
echo "Installing data processing libraries..."
pip install pandas numpy scipy scikit-learn

# Monitoring and logging
echo "Installing monitoring libraries..."
pip install tensorboard wandb

# Other utilities
echo "Installing utility libraries..."
pip install tqdm requests psutil

# Development tools
echo "Installing development tools..."
pip install black isort flake8 pytest

# =============================================================================
# Setup Megatron-LM
# =============================================================================

echo "Setting up Megatron-LM..."

if [[ ! -d "$MEGATRON_PATH" ]]; then
    echo "❌ Megatron-LM not found at $MEGATRON_PATH"
    echo "Please ensure Megatron-LM is cloned as a submodule"
    exit 1
fi

cd "$MEGATRON_PATH"

# Install Megatron-LM in development mode
echo "Installing Megatron-LM..."
pip install -e .

# Install additional Megatron dependencies
pip install nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# =============================================================================
# Create Directory Structure
# =============================================================================

echo "Creating directory structure..."

cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p training/checkpoints
mkdir -p training/tensorboard_logs
mkdir -p training/cache
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs

echo "✅ Directory structure created"

# =============================================================================
# Setup Configuration Files
# =============================================================================

echo "Creating configuration files..."

# Create requirements.txt
cat > "$TRAINING_PATH/requirements.txt" << EOF
# Core ML libraries
torch>=2.1.0
torchvision
torchaudio
transformer-engine[pytorch]

# HuggingFace ecosystem
transformers>=4.35.0
datasets>=2.14.0
tokenizers>=0.14.0
accelerate>=0.24.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Monitoring and logging
tensorboard>=2.14.0
wandb>=0.15.0

# Utilities
tqdm>=4.65.0
requests>=2.31.0
psutil>=5.9.0

# Development
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
pytest>=7.4.0

# NLP utilities
nltk>=3.8.0
EOF

# Create .env template
cat > "$TRAINING_PATH/.env.template" << EOF
# Environment variables for training
# Copy this to .env and fill in your values

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=llama31-hinglish-training

# HuggingFace Hub (for model downloads)
HF_TOKEN=your_huggingface_token_here

# Training configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=6000

# Performance tuning
CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_IB_TIMEOUT=19
NVTE_FWD_LAYERNORM_SM_MARGIN=16
NVTE_BWD_LAYERNORM_SM_MARGIN=16
EOF

# Create training config
cat > "$TRAINING_PATH/config.yaml" << EOF
# Llama 3.1 Instruct 8B Training Configuration

model:
  name: "llama31_instruct_8b"
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  
training:
  # Batch sizes
  micro_batch_size: 1
  global_batch_size: 32
  
  # Sequence length
  seq_length: 4096
  max_position_embeddings: 4096
  
  # Learning rate
  learning_rate: 5e-5
  min_learning_rate: 5e-6
  warmup_steps: 100
  
  # Training steps
  train_steps: 5000
  eval_interval: 100
  save_interval: 500
  
  # Optimization
  weight_decay: 0.01
  grad_clip: 1.0
  
  # Precision
  use_bf16: true
  use_fp8: false

parallelism:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  context_parallel_size: 1
  
data:
  data_prefix: "./data/processed/hinglish_instruct"
  tokenizer_model: "meta-llama/Llama-3.1-8B-Instruct"
  vocab_size: 128256
  split: "98,1,1"  # train, validation, test

logging:
  log_interval: 10
  tensorboard_dir: "./tensorboard_logs"
  checkpoint_dir: "./checkpoints"
EOF

echo "✅ Configuration files created"

# =============================================================================
# Verify Installation
# =============================================================================

echo "Verifying installation..."

# Test PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Test Transformer Engine
python3 -c "import transformer_engine; print('Transformer Engine: OK')" 2>/dev/null || echo "⚠️  Transformer Engine not available"

# Test HuggingFace
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Test Megatron
cd "$MEGATRON_PATH"
python3 -c "import megatron; print('Megatron-LM: OK')" 2>/dev/null || echo "⚠️  Megatron-LM import failed"

# =============================================================================
# Create Helper Scripts
# =============================================================================

echo "Creating helper scripts..."

cd "$TRAINING_PATH"

# Make scripts executable
chmod +x train_llama31_instruct_8b.sh
chmod +x preprocess_hinglish_data.py
chmod +x setup_environment.sh

# Create quick start script
cat > quick_start.sh << 'EOF'
#!/bin/bash

# Quick Start Script for Llama 3.1 Instruct 8B Training

echo "🚀 Llama 3.1 Instruct 8B Training - Quick Start"
echo "=============================================="

# Step 1: Preprocess data
echo "Step 1: Preprocessing data..."
python preprocess_hinglish_data.py \
    --use-hinglish-top \
    --output-dir ./data/processed \
    --output-prefix hinglish_instruct \
    --tokenizer-model meta-llama/Llama-3.1-8B-Instruct

# Step 2: Start training
echo "Step 2: Starting training..."
./train_llama31_instruct_8b.sh \
    ./checkpoints/llama31_instruct_8b \
    ./tensorboard_logs/llama31_instruct_8b \
    meta-llama/Llama-3.1-8B-Instruct \
    ./data/processed/hinglish_instruct

echo "✅ Training started! Monitor progress with:"
echo "tensorboard --logdir ./tensorboard_logs"
EOF

chmod +x quick_start.sh

# =============================================================================
# Final Instructions
# =============================================================================

echo ""
echo "=========================================="
echo "✅ Environment setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate your virtual environment (if not already active)"
echo "2. Set up your HuggingFace token: huggingface-cli login"
echo "3. Preprocess your data:"
echo "   cd $TRAINING_PATH"
echo "   python preprocess_hinglish_data.py --use-hinglish-top"
echo ""
echo "4. Start training:"
echo "   ./train_llama31_instruct_8b.sh"
echo ""
echo "5. Monitor training:"
echo "   tensorboard --logdir ./tensorboard_logs"
echo ""
echo "For quick start: ./quick_start.sh"
echo ""
echo "Configuration files:"
echo "- requirements.txt: Python dependencies"
echo "- config.yaml: Training configuration"
echo "- .env.template: Environment variables template"
echo ""
echo "Happy training! 🎉"
