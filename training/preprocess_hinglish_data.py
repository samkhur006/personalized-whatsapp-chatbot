#!/usr/bin/env python3
"""
Data preprocessing script for Hinglish instruction tuning data.
Converts raw Hinglish conversation data into Megatron-LM compatible format.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset, load_dataset
import requests
import io

# Add Megatron-LM tools to path
MEGATRON_PATH = Path(__file__).parent.parent / "Megatron-LM"
sys.path.append(str(MEGATRON_PATH / "tools"))

def load_hinglish_top_dataset() -> Dataset:
    """Load the Hinglish TOP dataset from GitHub."""
    print("Loading Hinglish TOP dataset...")
    
    # URLs for the dataset
    train_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/train.tsv"
    val_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/validation.tsv"
    test_url = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/Human%20Annotated%20Data/test.tsv"
    
    def load_split(url: str) -> pd.DataFrame:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep='\t')
        df.columns = df.columns.str.strip()
        return df[['en_query', 'cs_query']].dropna()
    
    # Load all splits
    train_df = load_split(train_url)
    val_df = load_split(val_url)
    # test_df = load_split(test_url)
    
    # Combine all data for training
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Loaded {len(all_df)} examples from Hinglish TOP dataset")
    return Dataset.from_pandas(all_df)

def create_pretraining_format(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert English-Hinglish pairs into simple pretraining format."""
    
    english_text = example['en_query'].strip()
    hinglish_text = example['cs_query'].strip()
    
    # Create separate examples for English and Hinglish text
    examples = []
    
    # Add English text as one example
    
    examples.append({"text": english_text+ " "+ hinglish_text})
    
    return examples


def process_custom_data(input_file: str) -> List[Dict[str, str]]:
    """Process custom JSON/JSONL data file."""
    
    print(f"Processing custom data from {input_file}")
    
    data = []
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    processed_examples = []
    
    for example in data:
        if 'english' in example and 'hinglish' in example:
            # Use pretraining format
            simple_examples = create_pretraining_format({
                'en_query': example['english'],
                'cs_query': example['hinglish']
            })
            processed_examples.extend(simple_examples)
        elif 'text' in example:
            # Already formatted text
            processed_examples.append({"text": example['text']})
        elif 'conversations' in example:
            # Extract just the text content from conversations for pretraining
            conversation_text = ""
            for turn in example['conversations']:
                content = turn.get('value', '').strip()
                if content:
                    conversation_text += content + " "
            
            if conversation_text.strip():
                processed_examples.append({"text": conversation_text.strip()})
    
    print(f"Processed {len(processed_examples)} examples from custom data")
    return processed_examples

def save_jsonl(data: List[Dict[str, str]], output_file: str):
    """Save data in JSONL format."""
    
    print(f"Saving {len(data)} examples to {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def run_megatron_preprocessing(
    input_file: str,
    output_prefix: str,
    tokenizer_model: str,
    vocab_size: int = 128256,
    workers: int = 4
):
    """Run Megatron-LM preprocessing on the JSONL data."""
    
    print(f"Running Megatron-LM preprocessing...")
    print(f"Input: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Tokenizer: {tokenizer_model}")
    
    # Import Megatron preprocessing tools
    try:
        from preprocess_data import main as preprocess_main
        import sys
        
        # Prepare arguments for Megatron preprocessing
        sys.argv = [
            'preprocess_data.py',
            '--input', input_file,
            '--output-prefix', output_prefix,
            '--tokenizer-type', 'HuggingFaceTokenizer',
            '--tokenizer-model', tokenizer_model,
            '--vocab-size', str(vocab_size),
            '--workers', str(workers),
            '--append-eod'
        ]
        
        # Run preprocessing
        preprocess_main()
        
        print(f"Preprocessing completed. Files created:")
        print(f"  - {output_prefix}_text_document.bin")
        print(f"  - {output_prefix}_text_document.idx")
        
    except ImportError as e:
        print(f"Error importing Megatron preprocessing tools: {e}")
        print("Please ensure Megatron-LM is properly installed and accessible.")
        print(f"You can manually run preprocessing with:")
        print(f"cd {MEGATRON_PATH}")
        print(f"python tools/preprocess_data.py \\")
        print(f"  --input {input_file} \\")
        print(f"  --output-prefix {output_prefix} \\")
        print(f"  --tokenizer-type HuggingFaceTokenizer \\")
        print(f"  --tokenizer-model {tokenizer_model} \\")
        print(f"  --vocab-size {vocab_size} \\")
        print(f"  --workers {workers} \\")
        print(f"  --append-eod")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess Hinglish data for Megatron-LM training")
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input JSON/JSONL file with custom data (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='hinglish_pretrain',
        help='Prefix for output files'
    )
    
    parser.add_argument(
        '--tokenizer-model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='HuggingFace tokenizer model'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=128256,
        help='Vocabulary size'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker processes'
    )
    
    parser.add_argument(
        '--use-hinglish-top',
        action='store_true',
        help='Include Hinglish TOP dataset'
    )
    
    parser.add_argument(
        '--skip-megatron-preprocessing',
        action='store_true',
        help='Skip Megatron preprocessing step (only create JSONL)'
    )
    
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_examples = []
    
    # Load Hinglish TOP dataset if requested
    if args.use_hinglish_top:
        hinglish_dataset = load_hinglish_top_dataset()
        for example in hinglish_dataset:
            # Use pretraining format (concatenated English + Hinglish)
            simple_examples = create_pretraining_format(example)
            all_examples.extend(simple_examples)
    
    # Load custom data if provided
    if args.input_file and os.path.exists(args.input_file):
        custom_examples = process_custom_data(args.input_file)
        all_examples.extend(custom_examples)
    
    if not all_examples:
        print("No data to process. Please provide --input-file or use --use-hinglish-top")
        return
    
    # Save as JSONL
    jsonl_file = os.path.join(args.output_dir, f"{args.output_prefix}.jsonl")
    save_jsonl(all_examples, jsonl_file)
    
    # Run Megatron preprocessing
    if not args.skip_megatron_preprocessing:
        output_prefix = os.path.join(args.output_dir, args.output_prefix)
        success = run_megatron_preprocessing(
            jsonl_file,
            output_prefix,
            args.tokenizer_model,
            args.vocab_size,
            args.workers
        )
        
        if success:
            print(f"\n✅ Data preprocessing completed successfully!")
            print(f"Use this data prefix in training: {output_prefix}")
        else:
            print(f"\n⚠️  JSONL file created at: {jsonl_file}")
            print(f"Please run Megatron preprocessing manually.")
    else:
        print(f"\n✅ JSONL file created at: {jsonl_file}")
        print(f"Run Megatron preprocessing manually when ready.")

if __name__ == "__main__":
    main()
