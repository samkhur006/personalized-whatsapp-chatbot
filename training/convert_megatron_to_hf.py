#!/usr/bin/env python3
"""
Convert Megatron-LM checkpoint to HuggingFace format for evaluation.
This script converts your trained Llama 3.1 model back to HF format.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add Megatron-LM to path
MEGATRON_PATH = Path(__file__).parent.parent / "Megatron-LM"
sys.path.append(str(MEGATRON_PATH))
sys.path.append(str(MEGATRON_PATH / "tools"))

def convert_megatron_to_hf(
    megatron_checkpoint_path: str,
    hf_output_path: str,
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1
):
    """Convert Megatron checkpoint to HuggingFace format."""
    
    print(f"Converting Megatron checkpoint to HuggingFace format...")
    print(f"Input: {megatron_checkpoint_path}")
    print(f"Output: {hf_output_path}")
    print(f"Base model: {base_model_name}")
    
    # Create output directory
    os.makedirs(hf_output_path, exist_ok=True)
    
    try:
        # Import Megatron conversion tools
        from checkpoint_util import main as checkpoint_main
        
        # Prepare arguments for conversion
        sys.argv = [
            'checkpoint_util.py',
            '--model-type', 'GPT',
            '--loader', 'megatron',
            '--saver', 'llama_mistral',
            '--load-dir', megatron_checkpoint_path,
            '--save-dir', hf_output_path,
            '--tokenizer-model', base_model_name,
            '--target-tensor-parallel-size', '1',
            '--target-pipeline-parallel-size', '1',
            '--bf16'
        ]
        
        # Run conversion
        checkpoint_main()
        
        print(f"✅ Conversion completed successfully!")
        print(f"HuggingFace model saved to: {hf_output_path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing Megatron tools: {e}")
        print("Please ensure Megatron-LM is properly installed.")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in the directory."""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for iteration directories
    iter_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith('iter_')]
    
    if not iter_dirs:
        raise FileNotFoundError(f"No checkpoint iterations found in {checkpoint_dir}")
    
    # Sort by iteration number and get the latest
    iter_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    latest_checkpoint = iter_dirs[-1]
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)

def main():
    parser = argparse.ArgumentParser(description="Convert Megatron-LM checkpoint to HuggingFace format")
    
    parser.add_argument(
        '--megatron-checkpoint',
        type=str,
        required=True,
        help='Path to Megatron checkpoint directory'
    )
    
    parser.add_argument(
        '--hf-output',
        type=str,
        required=True,
        help='Output path for HuggingFace model'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Base model name for tokenizer and config'
    )
    
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Tensor parallel size used during training'
    )
    
    parser.add_argument(
        '--pipeline-parallel-size',
        type=int,
        default=1,
        help='Pipeline parallel size used during training'
    )
    
    parser.add_argument(
        '--auto-find-latest',
        action='store_true',
        help='Automatically find the latest checkpoint'
    )
    
    args = parser.parse_args()
    
    # Find latest checkpoint if requested
    if args.auto_find_latest:
        try:
            checkpoint_path = find_latest_checkpoint(args.megatron_checkpoint)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    else:
        checkpoint_path = args.megatron_checkpoint
    
    # Convert checkpoint
    success = convert_megatron_to_hf(
        checkpoint_path,
        args.hf_output,
        args.base_model,
        args.tensor_parallel_size,
        args.pipeline_parallel_size
    )
    
    if success:
        print(f"\n🎉 Model conversion completed!")
        print(f"You can now use the model for evaluation:")
        print(f"  Model path: {args.hf_output}")
        print(f"  Add to eval script: '{args.hf_output}'")
    else:
        print(f"\n❌ Conversion failed. Check the logs above.")

if __name__ == "__main__":
    main()
