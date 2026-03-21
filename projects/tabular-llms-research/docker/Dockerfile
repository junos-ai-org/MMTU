FROM vllm/vllm-openai:latest

# System tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    screen vim grep htop curl net-tools procps git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — shared across all backends
# vLLM image already includes: torch, transformers, accelerate, sentencepiece, vllm
RUN pip install --no-cache-dir \
    pandas tabulate tiktoken tqdm xlsxwriter tenacity datasets openai azure-ai-inference \
    pyyaml pytest protobuf

# MMTU code — install outside /workspace (RunPod mounts a volume there)
COPY . /opt/MMTU

COPY projects/tabular-llms-research/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT []
CMD ["/entrypoint.sh"]
