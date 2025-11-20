#!/usr/bin/env python3
"""
Configuration loader for YAML-based training configuration.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    
    if config_path is None:
        # Default to config.yaml in the same directory as this script
        config_path = Path(__file__).parent / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def config_to_env_vars(config: Dict[str, Any]) -> Dict[str, str]:
    """Convert config to environment variables for shell scripts."""
    
    env_vars = {}
    
    # Model configuration
    if 'model' in config:
        model = config['model']
        env_vars['BASE_MODEL'] = model.get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')
        env_vars['MODEL_NAME'] = model.get('name', 'llama31_instruct_8b')
        env_vars['LOAD_FROM_CHECKPOINT'] = str(model.get('load_from_checkpoint', False)).lower()

        # Model architecture
        env_vars['NUM_LAYERS'] = str(model.get('num_layers', 16))
        env_vars['HIDDEN_SIZE'] = str(model.get('hidden_size', 2048))
        env_vars['FFN_HIDDEN_SIZE'] = str(model.get('ffn_hidden_size', 8192))
        env_vars['NUM_ATTENTION_HEADS'] = str(model.get('num_attention_heads', 32))
        env_vars['NUM_QUERY_GROUPS'] = str(model.get('num_query_groups', 8))
        env_vars['KV_CHANNELS'] = str(model.get('kv_channels', 64))
        env_vars['POSITION_EMBEDDING_TYPE'] = model.get('position_embedding_type', 'rope')
        env_vars['ROTARY_BASE'] = str(model.get('rotary_base', 500000))
        env_vars['ROTARY_PERCENT'] = str(model.get('rotary_percent', 1.0))
        env_vars['ATTENTION_DROPOUT'] = str(model.get('attention_dropout', 0.0))
        env_vars['HIDDEN_DROPOUT'] = str(model.get('hidden_dropout', 0.0))
        env_vars['USE_SWIGLU'] = str(model.get('swiglu', True)).lower()
        env_vars['INIT_METHOD_STD'] = str(model.get('init_method_std', 0.0134))
        env_vars['ATTENTION_BACKEND'] = model.get('attention_backend', 'fused')
        env_vars['APPLY_LAYERNORM_1P'] = str(model.get('apply_layernorm_1p', True)).lower()
        env_vars['UNTIE_EMBEDDINGS_AND_OUTPUT_WEIGHTS'] = str(model.get('untie_embeddings_and_output_weights', True)).lower()
        env_vars['DISABLE_BIAS_LINEAR'] = str(model.get('disable_bias_linear', True)).lower()
        env_vars['NORMALIZATION'] = model.get('normalization', 'RMSNorm')
    
    # Training configuration
    if 'training' in config:
        training = config['training']
        env_vars['MICRO_BATCH_SIZE'] = str(training.get('micro_batch_size', 1))
        env_vars['GLOBAL_BATCH_SIZE'] = str(training.get('global_batch_size', 32))
        env_vars['SEQ_LENGTH'] = str(training.get('seq_length', 4096))
        env_vars['MAX_POSITION_EMBEDDINGS'] = str(training.get('max_position_embeddings', 4096))
        env_vars['LR'] = str(training.get('learning_rate', 5e-5))
        env_vars['MIN_LR'] = str(training.get('min_learning_rate', 5e-6))
        env_vars['WARMUP_STEPS'] = str(training.get('warmup_steps', 100))
        env_vars['TRAIN_STEPS'] = str(training.get('train_steps', 5000))
        env_vars['EVAL_INTERVAL'] = str(training.get('eval_interval', 100))
        env_vars['SAVE_INTERVAL'] = str(training.get('save_interval', 500))
        env_vars['WEIGHT_DECAY'] = str(training.get('weight_decay', 0.01))
        env_vars['GRAD_CLIP'] = str(training.get('grad_clip', 1.0))
        env_vars['USE_BF16'] = str(training.get('use_bf16', True)).lower()
        env_vars['USE_FP8'] = str(training.get('use_fp8', False)).lower()
    
    # Parallelism configuration
    if 'parallelism' in config:
        parallelism = config['parallelism']
        env_vars['TP_SIZE'] = str(parallelism.get('tensor_parallel_size', 1))
        env_vars['PP_SIZE'] = str(parallelism.get('pipeline_parallel_size', 1))
        env_vars['CP_SIZE'] = str(parallelism.get('context_parallel_size', 1))
    
    # Data configuration
    if 'data' in config:
        data = config['data']
        env_vars['DATA_PREFIX'] = data.get('data_prefix', './data/processed/hinglish_pretrain')
        env_vars['TOKENIZER_MODEL'] = data.get('tokenizer_model', 'meta-llama/Llama-3.1-8B-Instruct')
        env_vars['VOCAB_SIZE'] = str(data.get('vocab_size', 128256))
    
    # Logging configuration
    if 'logging' in config:
        logging = config['logging']
        env_vars['LOG_INTERVAL'] = str(logging.get('log_interval', 10))
        env_vars['TENSORBOARD_DIR'] = logging.get('tensorboard_dir', './tensorboard_logs')
        env_vars['CHECKPOINT_DIR'] = logging.get('checkpoint_dir', './checkpoints')
    
    return env_vars

def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the loaded configuration."""
    
    print("📋 Training Configuration Summary:")
    print("=" * 50)
    
    if 'model' in config:
        print(f"🤖 Model: {config['model'].get('name', 'N/A')}")
        print(f"📦 Base Model: {config['model'].get('base_model', 'N/A')}")
    
    if 'training' in config:
        training = config['training']
        print(f"🔢 Batch Size: {training.get('micro_batch_size', 1)} micro, {training.get('global_batch_size', 32)} global")
        print(f"📏 Sequence Length: {training.get('seq_length', 4096)}")
        print(f"📈 Learning Rate: {training.get('learning_rate', 5e-5)}")
        print(f"🔄 Training Steps: {training.get('train_steps', 5000)}")
        print(f"💾 Save Interval: {training.get('save_interval', 500)}")
        print(f"🎯 Precision: {'BF16' if training.get('use_bf16', True) else 'FP16'}")
    
    if 'parallelism' in config:
        parallelism = config['parallelism']
        print(f"⚡ Parallelism: TP={parallelism.get('tensor_parallel_size', 1)}, PP={parallelism.get('pipeline_parallel_size', 1)}, CP={parallelism.get('context_parallel_size', 1)}")
    
    if 'data' in config:
        data = config['data']
        print(f"📊 Data Prefix: {data.get('data_prefix', 'N/A')}")
        print(f"🔤 Tokenizer: {data.get('tokenizer_model', 'N/A')}")
    
    print("=" * 50)

if __name__ == "__main__":
    # Test the configuration loader
    config = load_config()
    print_config_summary(config)
    
    print("\n🔧 Environment Variables:")
    env_vars = config_to_env_vars(config)
    for key, value in env_vars.items():
        print(f"export {key}='{value}'")
