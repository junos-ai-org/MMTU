#!/bin/bash
set -e

echo "========================================"
echo "MMTU Evaluation Container Starting"
echo "========================================"

# LLaDA is included as a submodule - verify it exists
if [ -d "/workspace/MMTU/llada" ]; then
    echo "LLaDA submodule found at /workspace/MMTU/llada"
elif [ -d "/workspace/LLaDA" ]; then
    echo "LLaDA found at /workspace/LLaDA"
else
    echo "Warning: LLaDA not found. If running outside Docker, run:"
    echo "  git submodule update --init"
fi

# Mount S3 bucket with model weights (if AWS credentials provided)
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$S3_MODEL_BUCKET" ]; then
    echo "Setting up S3 model access..."

    # Option 1: Mount with s3fs (slower but transparent)
    if [ "$USE_S3FS" = "true" ]; then
        echo "Mounting S3 bucket via s3fs..."
        mkdir -p /models
        echo "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" > /tmp/passwd-s3fs
        chmod 600 /tmp/passwd-s3fs
        s3fs "$S3_MODEL_BUCKET" /models \
            -o passwd_file=/tmp/passwd-s3fs \
            -o url=https://s3.amazonaws.com \
            -o use_path_request_style \
            -o allow_other
        rm /tmp/passwd-s3fs
        echo "S3 bucket mounted at /models"
    else
        # Option 2: Download models (faster for repeated inference)
        echo "Downloading models from S3..."

        if [ ! -d "/models/llada-8b-instruct" ] && [ -n "$LLADA_S3_PATH" ]; then
            echo "Downloading LLaDA weights..."
            aws s3 sync "s3://$S3_MODEL_BUCKET/$LLADA_S3_PATH" /models/llada-8b-instruct
        fi

        if [ ! -d "/models/qwen2-7b-instruct" ] && [ -n "$QWEN_S3_PATH" ]; then
            echo "Downloading QWEN weights..."
            aws s3 sync "s3://$S3_MODEL_BUCKET/$QWEN_S3_PATH" /models/qwen2-7b-instruct
        fi
    fi
fi

# Check for HuggingFace token
if [ -n "$HF_TOKEN" ]; then
    echo "Setting up HuggingFace authentication..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Print environment info
echo ""
echo "Environment:"
echo "  LLADA_MODEL_PATH: $LLADA_MODEL_PATH"
echo "  QWEN_MODEL_PATH: $QWEN_MODEL_PATH"
echo "  LLADA_CODE_PATH: $LLADA_CODE_PATH"
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "  No GPU detected"
echo ""

# Check if models are available
echo "Model Status:"
if [ -d "$LLADA_MODEL_PATH" ]; then
    echo "  LLaDA: Available at $LLADA_MODEL_PATH"
else
    echo "  LLaDA: NOT FOUND at $LLADA_MODEL_PATH"
fi

if [ -d "$QWEN_MODEL_PATH" ]; then
    echo "  QWEN: Available at $QWEN_MODEL_PATH"
else
    echo "  QWEN: NOT FOUND at $QWEN_MODEL_PATH"
    echo "  (Will download from HuggingFace on first use)"
fi
echo ""

echo "========================================"
echo "Container ready!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  # Create samples"
echo "  python sample_data.py --both"
echo ""
echo "  # Run verification"
echo "  python run_evaluation.py --model qwen --input verification_sample.jsonl --output results/qwen_verify.jsonl"
echo ""
echo "  # Evaluate results"
echo "  python evaluate.py results/qwen_verify.jsonl"
echo ""

# Execute command
exec "$@"
