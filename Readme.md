# Personalized WhatsApp Chatbot - Hinglish Model Training & Evaluation

This repository contains a complete pipeline for training and evaluating Llama 3.1 models on Hinglish data using Megatron-LM, along with comprehensive evaluation tools.

## 🚀 Quick Start

### **Prerequisites**
- Python 3.10+
- NVIDIA GPUs with CUDA support
- Docker (recommended for training)
- HuggingFace account with Llama 3.1 access

### **Setup**
```bash
git clone --recursive git@github.com:samkhur006/personalized-whatsapp-chatbot.git
cd personalized-whatsapp-chatbot

# Install dependencies
pip install --no-build-isolation megatron-core[mlm,dev]
cd Megatron-LM && pip install --no-build-isolation .[mlm,dev] && cd ..
cd lm-evaluation-harness && pip install -e . && cd ..
```

## 📊 Evaluation Only

If you just want to evaluate existing models on Hinglish tasks:

### **1. Setup Evaluation Environment**
```bash
cd eval
# Dependencies should already be installed from main setup
```

### **2. Run Evaluation**
```bash
# Evaluate multiple models on Hinglish tasks
./evaluate_hinglish_models.sh
```

**What it evaluates:**
- **Models**: Qwen2.5-14B, Qwen3-14B, Qwen3-8B, GLM-4-9B variants
- **Tasks**: `hinglish_translation`, `hinglish_perplexity`
- **Dataset**: Hinglish TOP dataset (auto-downloaded)
- **Output**: Results saved to `/home/ubuntu/eval_output/`

### **3. Monitor Results**
```bash
# Check evaluation progress
ls -la /home/ubuntu/eval_output/

# View specific results
cat /home/ubuntu/eval_output/MODEL_NAME_TIMESTAMP/results.json
```

## 🏋️ Training

Train your own Llama 3.1 model on Hinglish data:

### **Method 1: Docker Training (Recommended)**

#### **Setup**
```bash
cd training
chmod +x docker_train.sh

# Set your HuggingFace token
export HF_TOKEN=your_huggingface_token_here
```

#### **Configure Training**
```bash
# Edit config.yaml to customize training parameters
nano config.yaml

# Example modifications:
# - Change batch sizes, learning rate, training steps
# - Use different base model or checkpoint
# - Adjust parallelism settings
```

#### **Start Training**
```bash
# Training uses config.yaml automatically
./docker_train.sh

# Override with environment variables if needed
export GPUS_PER_NODE=8
./docker_train.sh
```

#### **Monitor Training**
```bash
# Check training logs
docker logs -f llama31_training_TIMESTAMP

# TensorBoard (if using docker-compose)
docker-compose up tensorboard
# Access: http://localhost:6006
```

### **Method 2: Native Training**

#### **Setup Environment**
```bash
cd training
./setup_environment.sh  # Creates directories only
huggingface-cli login
```

#### **Preprocess Data**
```bash
python preprocess_hinglish_data.py --use-hinglish-top
```

#### **Configure and Start Training**
```bash
# Edit config.yaml for your training parameters
nano config.yaml

# Start training (uses config.yaml automatically)
chmod +x train_llama31_instruct_8b.sh
./train_llama31_instruct_8b.sh
```

#### **Monitor Training**
```bash
# Real-time monitoring
python monitor_training.py --mode watch

# TensorBoard
tensorboard --logdir ./tensorboard_logs
```

### **Training Configuration**

**Default Settings:**
- **Model**: Llama 3.1 8B Instruct (pretrained base)
- **Data**: Hinglish TOP dataset (English-Hinglish pairs)
- **Format**: Pretraining format (concatenated text)
- **Batch Size**: Global 32, Micro 1
- **Learning Rate**: 5e-5 with cosine decay
- **Steps**: 5000 (adjustable)
- **Precision**: BF16
- **Parallelism**: TP=1, PP=1, CP=1 (configurable)

**Customization via config.yaml:**
```yaml
# Edit training/config.yaml
training:
  micro_batch_size: 1
  global_batch_size: 64
  learning_rate: 3e-5
  train_steps: 10000
  seq_length: 8192

parallelism:
  tensor_parallel_size: 2
  pipeline_parallel_size: 1

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  # Or use a checkpoint:
  # base_model: "./checkpoints/my_previous_training"
  # load_from_checkpoint: true
```

## 🔄 Evaluate Trained Model

After training, evaluate your custom model:

### **Method 1: Automated Pipeline (Recommended)**
```bash
cd training

# Convert Megatron checkpoint to HuggingFace format + Run evaluation
./convert_and_eval.sh

# Custom paths
./convert_and_eval.sh \
  ./checkpoints/my_model \
  ./converted_models/my_model_hf \
  meta-llama/Llama-3.1-8B-Instruct
```

### **Method 2: Manual Process**

#### **Step 1: Convert Model Format**
```bash
cd training

# Convert Megatron → HuggingFace
python convert_megatron_to_hf.py \
  --megatron-checkpoint ./checkpoints/llama31_instruct_8b \
  --hf-output ./converted_models/llama31_hinglish_hf \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --auto-find-latest
```

#### **Step 2: Add to Evaluation Script**
```bash
cd eval

# Edit evaluate_hinglish_models.sh
# Uncomment line 23 and update path:
# "../training/converted_models/llama31_hinglish_hf"
```

#### **Step 3: Run Evaluation**
```bash
./evaluate_hinglish_models.sh
```

### **Quick Single Model Evaluation**
```bash
cd eval
python -m lm_eval \
  --model hf \
  --model_args pretrained="../training/converted_models/llama31_hinglish_hf",trust_remote_code=True,device_map=auto \
  --tasks "hinglish_translation,hinglish_perplexity" \
  --batch_size 1 \
  --log_samples \
  --limit 1000 \
  --output_path "../eval_results/my_trained_model" \
  --include_path "$(pwd)"
```

## 📁 Directory Structure

```
personalized-whatsapp-chatbot/
├── eval/                           # Evaluation tools
│   ├── evaluate_hinglish_models.sh # Main evaluation script
│   ├── hinglish_translation.yaml   # Translation task definition
│   ├── hinglish_perplexity.yaml   # Perplexity task definition
│   └── utils.py                    # Dataset loading utilities
├── training/                       # Training tools
│   ├── config.yaml                # Training configuration
│   ├── docker_train.sh            # Docker training script
│   ├── train_llama31_instruct_8b.sh # Native training script
│   ├── preprocess_hinglish_data.py # Data preprocessing
│   ├── convert_megatron_to_hf.py  # Model format conversion
│   ├── convert_and_eval.sh        # Automated conversion + eval
│   ├── monitor_training.py        # Training monitoring
│   ├── setup_environment.sh       # Directory setup
│   ├── docker-compose.yml         # Docker Compose config
│   └── Dockerfile                 # Custom Docker image
├── Megatron-LM/                   # Megatron-LM submodule
├── lm-evaluation-harness/         # LM Eval submodule
└── data/                          # Training data
    ├── raw/                       # Raw datasets
    └── processed/                 # Preprocessed data
```

## 🎯 Evaluation Tasks

### **Hinglish Translation**
- **Task**: English → Hinglish translation
- **Dataset**: Hinglish TOP dataset
- **Metrics**: BLEU, Exact Match
- **Examples**:
  - English: "Turn off the lights"
  - Hinglish: "lights band kar do"

### **Hinglish Perplexity**
- **Task**: Language modeling on Hinglish text
- **Dataset**: Hinglish TOP dataset
- **Metrics**: Perplexity, Log-likelihood
- **Purpose**: Measure fluency in Hinglish

## 🔧 Configuration Files

### **Training Config** (`training/config.yaml`)
This file contains all training parameters and can be modified as needed:

```yaml
model:
  name: "llama31_instruct_8b"
  base_model: "meta-llama/Llama-3.1-8B-Instruct"

training:
  micro_batch_size: 1
  global_batch_size: 32
  seq_length: 4096
  learning_rate: 5e-5
  train_steps: 5000
  eval_interval: 100
  save_interval: 500

parallelism:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  context_parallel_size: 1

data:
  data_prefix: "./data/processed/hinglish_pretrain"
  tokenizer_model: "meta-llama/Llama-3.1-8B-Instruct"
  vocab_size: 128256

model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  # Use checkpoint: "./checkpoints/my_model"
  load_from_checkpoint: false
```

### **Docker Compose** (`training/docker-compose.yml`)
- **Training container**: NVIDIA PyTorch with Megatron-LM
- **TensorBoard**: Monitoring at `http://localhost:6006`
- **Jupyter**: Analysis at `http://localhost:8888`

## 📈 Monitoring & Logging

### **Training Monitoring**
```bash
# Real-time system metrics
python monitor_training.py --mode watch

# Generate training report
python monitor_training.py --mode report

# Create progress plots
python monitor_training.py --mode plot

# TensorBoard
tensorboard --logdir ./tensorboard_logs
```

### **Key Metrics Tracked**
- **Training Loss**: Model learning progress
- **Learning Rate**: Schedule visualization
- **Throughput**: Tokens/second per GPU
- **GPU Utilization**: Memory and compute usage
- **System Resources**: CPU, memory, disk usage

## 🚨 Troubleshooting

### **Common Issues**

#### **Training Issues**
```bash
# Out of memory
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=16

# Slow training
python monitor_training.py --mode report
# Check GPU utilization and adjust batch sizes

# Data preprocessing fails
python preprocess_hinglish_data.py --skip-megatron-preprocessing
# Then run Megatron preprocessing manually
```

#### **Evaluation Issues**
```bash
# Custom tasks not found
cd eval
python -m lm_eval --tasks list | grep hinglish

# Model loading fails
# Ensure model is in HuggingFace format
ls converted_models/llama31_hinglish_hf/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

#### **Docker Issues**
```bash
# GPU not accessible
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Container out of memory
# Increase shared memory: --shm-size=32g
# Or edit docker-compose.yml: shm_size: 32gb
```

## 🔗 Dependencies

### **Required Installation Commands**
```bash
# 1. Install Megatron Core with required dependencies
pip install --no-build-isolation megatron-core[mlm,dev]

# 2. Install Megatron-LM from submodule
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
cd ..

# 3. Install LM Evaluation Harness from submodule
cd lm-evaluation-harness
pip install -e .
cd ..
```

### **What gets installed:**
- **Megatron-LM**: Training framework
- **PyTorch**: 2.1.0+ with CUDA support
- **Transformers**: 4.35.0+ (HuggingFace)
- **LM Evaluation Harness**: Model evaluation
- **All dependencies**: Automatically resolved

### **Optional**
- **Docker**: Containerized training (recommended)

## 📊 Results & Benchmarks

### **Expected Performance**
- **Training Speed**: ~13K tokens/sec/GPU (A100 80GB)
- **Memory Usage**: ~60GB per GPU
- **Convergence**: 2000-5000 steps for fine-tuning

### **Evaluation Baselines**
Run evaluation to compare your trained model against:
- Qwen2.5-14B
- Qwen3-14B/8B
- GLM-4-9B variants

## 🤝 Contributing

1. **Test with mock data**: `DATA_PREFIX=MOCK ./docker_train.sh`
2. **Validate preprocessing**: Check JSONL output format
3. **Monitor training**: Use provided monitoring tools
4. **Document changes**: Update relevant README sections

## 📄 License

This project follows the licenses of the underlying models and frameworks:
- **Llama 3.1**: Meta's custom license
- **Megatron-LM**: Apache 2.0
- **LM Evaluation Harness**: MIT

---

## 🎉 Quick Commands Summary

```bash
# Evaluation only
cd eval && ./evaluate_hinglish_models.sh

# Training (Docker)
cd training && export HF_TOKEN=xxx && ./docker_train.sh

# Training (Native)
cd training && ./setup_environment.sh && ./train_llama31_instruct_8b.sh

# Evaluate trained model
cd training && ./convert_and_eval.sh

# Monitor training
python monitor_training.py --mode watch
```
