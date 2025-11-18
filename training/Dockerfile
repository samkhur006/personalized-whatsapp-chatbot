# Custom Dockerfile for Llama 3.1 Training with Megatron-LM
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformer-engine[pytorch] \
    datasets \
    pandas \
    requests \
    transformers \
    tokenizers \
    accelerate \
    tensorboard \
    wandb \
    tqdm \
    psutil \
    GPUtil \
    matplotlib \
    seaborn \
    jupyter \
    ipywidgets

# Set environment variables for performance
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_IB_TIMEOUT=19
ENV NVTE_FWD_LAYERNORM_SM_MARGIN=16
ENV NVTE_BWD_LAYERNORM_SM_MARGIN=16

# Create directories
RUN mkdir -p /workspace/training/checkpoints \
    && mkdir -p /workspace/training/tensorboard_logs \
    && mkdir -p /workspace/data/processed

# Copy training scripts (will be mounted at runtime)
# COPY training/ /workspace/training/
# COPY Megatron-LM/ /workspace/Megatron-LM/

# Install Megatron-LM (will be done at runtime when source is mounted)
# RUN cd /workspace/Megatron-LM && pip install -e .

# Expose ports for TensorBoard and Jupyter
EXPOSE 6006 8888

# Default command
CMD ["bash"]
