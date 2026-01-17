# MMTU Evaluation Docker Image
# Includes dependencies for LLaDA (diffusion LLM) and QWEN evaluation

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    # Editors and terminal tools
    vim \
    nano \
    screen \
    tmux \
    htop \
    # Text processing
    jq \
    less \
    tree \
    gawk \
    ripgrep \
    # Cloud tools
    s3fs \
    awscli \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    safetensors \
    sentencepiece \
    tiktoken \
    datasets \
    huggingface_hub \
    pandas \
    numpy \
    openpyxl \
    xlsxwriter \
    tabulate \
    tqdm \
    tenacity \
    openai \
    boto3 \
    azure-ai-inference \
    azure-core

# Create workspace
WORKDIR /workspace

# Copy MMTU code (includes llada submodule)
COPY . /workspace/MMTU

# LLaDA is included as a submodule at /workspace/MMTU/llada
# Symlink for compatibility with LLADA_CODE_PATH
RUN ln -s /workspace/MMTU/llada /workspace/LLaDA

# Set up MMTU
WORKDIR /workspace/MMTU
RUN pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Pre-download QWEN tokenizer (small, speeds up first run)
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct', trust_remote_code=True)" || true

# Create models directory
RUN mkdir -p /models

# Environment variables
ENV PYTHONPATH=/workspace/MMTU:/workspace/LLaDA:$PYTHONPATH
ENV LLADA_CODE_PATH=/workspace/LLaDA
ENV LLADA_MODEL_PATH=/models/llada-8b-instruct
ENV QWEN_MODEL_PATH=/models/qwen2-7b-instruct
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Copy and set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
