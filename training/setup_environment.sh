#!/bin/bash

# Simple Directory Setup Script

set -e

echo "Creating directory structure..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p training/checkpoints
mkdir -p training/tensorboard_logs
mkdir -p training/cache
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs

echo "✅ Directory structure created"
echo "✅ Setup completed! Use config.yaml for training configuration."
