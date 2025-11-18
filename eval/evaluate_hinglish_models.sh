#!/bin/bash

# Hinglish Model Evaluation Script
# Evaluates multiple models on hinglish_translation and hinglish_perplexity tasks

# Set common parameters
CACHE_DIR="/home/ubuntu/hf_models"
OUTPUT_DIR="/home/ubuntu/eval_output"
BATCH_SIZE=1
TASKS="hinglish_translation,hinglish_perplexity"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of models to evaluate
models=(
    "Qwen/Qwen2.5-14B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    # Add your trained model path here:
    # "../training/converted_models/llama31_hinglish_hf"
)

# Function to run evaluation for a single model
evaluate_model() {
    local model=$1
    local model_name=$(echo "$model" | sed 's/\//_/g')  # Replace / with _ for filename
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_path="${OUTPUT_DIR}/${model_name}_${timestamp}"
    
    echo "=========================================="
    echo "Starting evaluation for: $model"
    echo "Output path: $output_path"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run the evaluation
    python -m lm_eval \
        --model hf \
        --model_args pretrained="$model",trust_remote_code=True,device_map=auto,cache_dir="$CACHE_DIR" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --limit 1000 \
        --output_path "$output_path" \
        --include_path "$(dirname "$0")"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $model evaluation completed"
        echo "Results saved to: $output_path"
    else
        echo "❌ FAILED: $model evaluation failed with exit code $exit_code"
    fi
    
    echo "Finished: $model at $(date)"
    echo ""
}

# Main execution
echo "🚀 Starting Hinglish Model Evaluation Suite"
echo "Models to evaluate: ${#models[@]}"
echo "Tasks: $TASKS"
echo "Cache directory: $CACHE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

# Evaluate each model
for model in "${models[@]}"; do
    evaluate_model "$model"
    
    # Add a small delay between models to prevent potential issues
    echo "Waiting 30 seconds before next model..."
    sleep 30
done

echo "🎉 All model evaluations completed!"
echo "End time: $(date)"
echo "Check results in: $OUTPUT_DIR"
