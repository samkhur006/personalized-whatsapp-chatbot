# Llama 3.1 Instruct 8B Training with Megatron-LM

This directory contains everything needed to train Llama 3.1 Instruct 8B using Megatron-LM for Hinglish instruction tuning.

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

2. **Preprocess Data**:
   ```bash
   python preprocess_hinglish_data.py --use-hinglish-top
   ```

3. **Start Training**:
   ```bash
   ./train_llama31_instruct_8b.sh
   ```

4. **Monitor Training**:
   ```bash
   python monitor_training.py --mode watch
   ```

## 📁 Files Overview

### Core Scripts
- **`train_llama31_instruct_8b.sh`** - Main training script with optimized configuration for Llama 3.1 Instruct 8B
- **`preprocess_hinglish_data.py`** - Data preprocessing script that converts Hinglish data to Megatron format
- **`setup_environment.sh`** - Environment setup and dependency installation
- **`monitor_training.py`** - Training monitoring and visualization utilities

### Configuration Files
- **`config.yaml`** - Training configuration parameters
- **`requirements.txt`** - Python dependencies
- **`.env.template`** - Environment variables template

### Helper Scripts
- **`quick_start.sh`** - One-command training pipeline

## 🔧 Configuration

### Training Parameters

The training script is optimized for instruction tuning with these key parameters:

- **Model**: Llama 3.1 Instruct 8B (32 layers, 4096 hidden size)
- **Batch Size**: Global batch size of 32 (adjustable)
- **Sequence Length**: 4096 tokens
- **Learning Rate**: 5e-5 with cosine decay
- **Precision**: BF16 (FP8 optional for H100)
- **Parallelism**: Configurable TP/PP/CP

### Hardware Requirements

**Minimum**:
- 8x NVIDIA V100 (32GB) or equivalent
- 256GB system RAM
- 2TB NVMe storage

**Recommended**:
- 8x NVIDIA A100 (80GB) or H100
- 512GB system RAM
- 4TB NVMe storage

## 📊 Data Preprocessing

### Supported Data Formats

1. **Hinglish TOP Dataset** (automatic download):
   ```bash
   python preprocess_hinglish_data.py --use-hinglish-top
   ```

2. **Custom JSONL** with English-Hinglish pairs:
   ```json
   {"english": "Turn off the lights", "hinglish": "lights band kar do"}
   ```

3. **Conversation Format**:
   ```json
   {
     "conversations": [
       {"from": "user", "value": "Translate: Hello"},
       {"from": "assistant", "value": "Namaste"}
     ]
   }
   ```

### Data Processing Options

```bash
# Use Hinglish TOP dataset
python preprocess_hinglish_data.py --use-hinglish-top

# Process custom data
python preprocess_hinglish_data.py --input-file my_data.jsonl

# Combine both
python preprocess_hinglish_data.py --use-hinglish-top --input-file my_data.jsonl

# Skip Megatron preprocessing (create JSONL only)
python preprocess_hinglish_data.py --skip-megatron-preprocessing
```

## 🏃‍♂️ Training

### Basic Training

```bash
# Default configuration
./train_llama31_instruct_8b.sh

# Custom paths
./train_llama31_instruct_8b.sh \
  ./checkpoints/my_model \
  ./tensorboard_logs/my_model \
  meta-llama/Llama-3.1-8B-Instruct \
  ./data/processed/my_data
```

### Environment Variables

```bash
# Multi-node training
export NUM_NODES=2
export MASTER_ADDR=node1.cluster
export NODE_RANK=0

# Performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19

# Model parallelism
export TP_SIZE=2
export PP_SIZE=2
export CP_SIZE=1

# Training parameters
export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=1
export LR=3e-5
export TRAIN_STEPS=10000
```

### Mock Data Testing

For testing without real data:

```bash
DATA_PREFIX=MOCK ./train_llama31_instruct_8b.sh
```

## 📈 Monitoring

### Real-time Monitoring

```bash
# Watch mode (updates every 60 seconds)
python monitor_training.py --mode watch

# Generate report
python monitor_training.py --mode report

# Create plots
python monitor_training.py --mode plot

# Save metrics
python monitor_training.py --mode metrics
```

### TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs
```

### Monitoring Features

- **System Resources**: CPU, memory, disk usage
- **GPU Metrics**: Utilization, memory, temperature
- **Training Progress**: Loss, learning rate, throughput
- **Checkpoint Status**: Count, sizes, latest checkpoint
- **Automated Recommendations**: Performance optimization suggestions

## 🔄 Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
# Will automatically detect and load from checkpoint
./train_llama31_instruct_8b.sh ./checkpoints/existing_model
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `MICRO_BATCH_SIZE` or `GLOBAL_BATCH_SIZE`
   - Enable gradient checkpointing
   - Increase model parallelism

2. **Slow Training**:
   - Check GPU utilization with monitoring script
   - Optimize data loading (`--num-workers`)
   - Verify network bandwidth for multi-node

3. **Loss Not Decreasing**:
   - Check learning rate schedule
   - Verify data quality and format
   - Monitor gradient norms

4. **Checkpoint Issues**:
   - Ensure sufficient disk space
   - Check file permissions
   - Verify checkpoint format compatibility

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor system resources
python monitor_training.py --mode watch --interval 10

# Validate data preprocessing
python preprocess_hinglish_data.py --input-file data.jsonl --skip-megatron-preprocessing

# Test with mock data
DATA_PREFIX=MOCK TRAIN_STEPS=10 ./train_llama31_instruct_8b.sh
```

## 📋 Performance Benchmarks

### Expected Performance (8x A100 80GB)

| Metric | Value |
|--------|-------|
| Throughput | ~13,000 tokens/sec/GPU |
| Memory Usage | ~60GB per GPU |
| Training Speed | ~100 steps/hour |
| Convergence | ~2000-5000 steps |

### Scaling Guidelines

| GPUs | TP | PP | CP | Global BS | Micro BS |
|------|----|----|----|-----------|---------| 
| 8    | 1  | 1  | 1  | 32        | 1       |
| 16   | 2  | 1  | 1  | 64        | 1       |
| 32   | 4  | 2  | 1  | 128       | 1       |
| 64   | 8  | 4  | 1  | 256       | 1       |

## 🤝 Contributing

1. Test changes with mock data first
2. Update configuration documentation
3. Add monitoring for new features
4. Validate on multiple GPU configurations

## 📚 References

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Hinglish TOP Dataset](https://github.com/google-research-datasets/Hinglish-TOP-Dataset)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

## 📄 License

This training setup follows the same license as the underlying models and frameworks used.
