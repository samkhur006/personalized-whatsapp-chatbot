#!/bin/bash

# Convert Megatron checkpoint to HuggingFace format and run evaluation
# This script automates the entire process from trained model to evaluation results

set -e

echo "🔄 Megatron to HuggingFace Conversion & Evaluation Pipeline"
echo "=========================================================="

# =============================================================================
# Configuration
# =============================================================================

# Paths
TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$TRAINING_DIR")"
EVAL_DIR="$PROJECT_ROOT/eval"

# Default paths (can be overridden)
MEGATRON_CHECKPOINT_DIR=${1:-"$TRAINING_DIR/checkpoints/llama31_instruct_8b"}
HF_OUTPUT_DIR=${2:-"$TRAINING_DIR/converted_models/llama31_hinglish_hf"}
BASE_MODEL=${3:-"meta-llama/Llama-3.1-8B-Instruct"}

# Model parallelism settings (should match training config)
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

# Evaluation settings
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"$PROJECT_ROOT/eval_cache"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"$PROJECT_ROOT/eval_results"}

echo "Megatron Checkpoint: $MEGATRON_CHECKPOINT_DIR"
echo "HF Output: $HF_OUTPUT_DIR"
echo "Base Model: $BASE_MODEL"
echo "Eval Cache: $EVAL_CACHE_DIR"
echo "Eval Output: $EVAL_OUTPUT_DIR"

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "Running pre-flight checks..."

# Check if checkpoint directory exists
if [[ ! -d "$MEGATRON_CHECKPOINT_DIR" ]]; then
    echo "❌ Megatron checkpoint directory not found: $MEGATRON_CHECKPOINT_DIR"
    echo "Available checkpoints:"
    ls -la "$TRAINING_DIR/checkpoints/" 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Check if evaluation script exists
if [[ ! -f "$EVAL_DIR/evaluate_hinglish_models.sh" ]]; then
    echo "❌ Evaluation script not found: $EVAL_DIR/evaluate_hinglish_models.sh"
    exit 1
fi

# Check if Megatron-LM exists
if [[ ! -d "$PROJECT_ROOT/Megatron-LM" ]]; then
    echo "❌ Megatron-LM not found at $PROJECT_ROOT/Megatron-LM"
    exit 1
fi

echo "✅ All checks passed!"

# =============================================================================
# Step 1: Convert Megatron Checkpoint to HuggingFace Format
# =============================================================================

echo ""
echo "🔄 Step 1: Converting Megatron checkpoint to HuggingFace format..."

if [[ -d "$HF_OUTPUT_DIR" ]] && [[ -f "$HF_OUTPUT_DIR/config.json" ]]; then
    echo "✅ HuggingFace model already exists at: $HF_OUTPUT_DIR"
    read -p "Do you want to reconvert? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$HF_OUTPUT_DIR"
    else
        echo "Skipping conversion, using existing model."
    fi
fi

if [[ ! -f "$HF_OUTPUT_DIR/config.json" ]]; then
    echo "Converting checkpoint..."
    
    python "$TRAINING_DIR/convert_megatron_to_hf.py" \
        --megatron-checkpoint "$MEGATRON_CHECKPOINT_DIR" \
        --hf-output "$HF_OUTPUT_DIR" \
        --base-model "$BASE_MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --pipeline-parallel-size $PP_SIZE \
        --auto-find-latest
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Conversion completed successfully!"
    else
        echo "❌ Conversion failed!"
        exit 1
    fi
else
    echo "✅ Using existing HuggingFace model"
fi

# =============================================================================
# Step 2: Update Evaluation Script with New Model
# =============================================================================

echo ""
echo "📝 Step 2: Updating evaluation script with trained model..."

# Create a custom evaluation script for the trained model
CUSTOM_EVAL_SCRIPT="$EVAL_DIR/evaluate_trained_model.sh"

# Copy the original script and modify it
cp "$EVAL_DIR/evaluate_hinglish_models.sh" "$CUSTOM_EVAL_SCRIPT"

# Update the models array to include our trained model
sed -i.bak "s|models=(|models=(\n    \"$HF_OUTPUT_DIR\"|" "$CUSTOM_EVAL_SCRIPT"

# Update cache and output directories to avoid conflicts
sed -i.bak "s|CACHE_DIR=\".*\"|CACHE_DIR=\"$EVAL_CACHE_DIR\"|" "$CUSTOM_EVAL_SCRIPT"
sed -i.bak "s|OUTPUT_DIR=\".*\"|OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"|" "$CUSTOM_EVAL_SCRIPT"

# Make executable
chmod +x "$CUSTOM_EVAL_SCRIPT"

echo "✅ Created custom evaluation script: $CUSTOM_EVAL_SCRIPT"

# =============================================================================
# Step 3: Run Evaluation
# =============================================================================

echo ""
echo "🚀 Step 3: Running evaluation on trained model..."

# Create output directories
mkdir -p "$EVAL_CACHE_DIR"
mkdir -p "$EVAL_OUTPUT_DIR"

# Change to eval directory (required for custom tasks)
cd "$EVAL_DIR"

echo "Starting evaluation..."
echo "Model: $HF_OUTPUT_DIR"
echo "Tasks: hinglish_translation, hinglish_perplexity"
echo "Output: $EVAL_OUTPUT_DIR"

# Run evaluation
python -m lm_eval \
    --model hf \
    --model_args pretrained="$HF_OUTPUT_DIR",trust_remote_code=True,device_map=auto,cache_dir="$EVAL_CACHE_DIR" \
    --tasks "hinglish_translation,hinglish_perplexity" \
    --batch_size 1 \
    --log_samples \
    --limit 1000 \
    --output_path "$EVAL_OUTPUT_DIR/trained_model_$(date +%Y%m%d_%H%M%S)" \
    --include_path "$(pwd)"

if [[ $? -eq 0 ]]; then
    echo "✅ Evaluation completed successfully!"
else
    echo "❌ Evaluation failed!"
    exit 1
fi

# =============================================================================
# Step 4: Compare Results (Optional)
# =============================================================================

echo ""
echo "📊 Step 4: Evaluation completed!"
echo "Results saved to: $EVAL_OUTPUT_DIR"

# List all result files
echo ""
echo "Available evaluation results:"
ls -la "$EVAL_OUTPUT_DIR"

echo ""
echo "🎉 Pipeline completed successfully!"
echo ""
echo "Summary:"
echo "  ✅ Converted Megatron checkpoint to HuggingFace format"
echo "  ✅ Model saved to: $HF_OUTPUT_DIR"
echo "  ✅ Evaluation completed"
echo "  ✅ Results saved to: $EVAL_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check evaluation results in: $EVAL_OUTPUT_DIR"
echo "  2. Compare with baseline models"
echo "  3. Use converted model: $HF_OUTPUT_DIR"
