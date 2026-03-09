"""LLaDA data-parallel backend — runs replicas across multiple GPUs."""

import torch
from typing import List

from backends.llada_backend import LLaDABackend


class LLaDADataParallelBackend(LLaDABackend):
    """Loads one LLaDA model replica per GPU for data-parallel inference.

    Each batch is split across GPUs, each replica processes its shard,
    and results are gathered back.
    """

    def __init__(self):
        super().__init__()
        self.replicas: list[torch.nn.Module] = []
        self.devices: list[str] = []

    def load(self, config: dict) -> None:
        model_cfg = config["model"]
        inference_cfg = config["inference"]

        model_path = model_cfg["model_path"]
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for data-parallel backend")

        self.devices = [f"cuda:{i}" for i in range(n_gpus)]

        # Load tokenizer once (CPU-only, shared)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

        # Load one replica per GPU
        print(f"Loading {n_gpus} LLaDA replicas: {model_path} (dtype={dtype})...")
        for i, device in enumerate(self.devices):
            print(f"  Loading replica {i} on {device}...")
            replica = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device).eval()
            self.replicas.append(replica)

        # Keep self.model pointing to first replica for health_check / single generate
        self.model = self.replicas[0]
        self.device = self.devices[0]

        # Apply inference config
        self.gen_length = inference_cfg.get("max_output_tokens", 512)
        self.steps = inference_cfg.get("steps", 128)
        self.block_length = inference_cfg.get("block_length", 128)
        self.temperature = inference_cfg.get("temperature", 0.0)
        self.cfg_scale = inference_cfg.get("cfg_scale", 0.0)
        self.remasking = inference_cfg.get("remasking", "low_confidence")

        print(f"  LLaDA DP ready ({n_gpus} GPUs). gen_length={self.gen_length}, "
              f"steps={self.steps}, block_length={self.block_length}, "
              f"remasking={self.remasking}")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        from backends.llada_generate import generate

        n_gpus = len(self.replicas)

        # Tokenize all prompts
        all_input_ids = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            ids = self.tokenizer(text)["input_ids"]
            all_input_ids.append(ids)

        # Split prompts across GPUs (round-robin to balance)
        gpu_shards: list[list[tuple[int, list[int]]]] = [[] for _ in range(n_gpus)]
        for idx, ids in enumerate(all_input_ids):
            gpu_shards[idx % n_gpus].append((idx, ids))

        # Run each shard on its GPU
        results: list[tuple[int, str]] = []
        for gpu_idx, shard in enumerate(gpu_shards):
            if not shard:
                continue

            device = self.devices[gpu_idx]
            model = self.replicas[gpu_idx]

            indices = [s[0] for s in shard]
            shard_ids = [s[1] for s in shard]

            # Left-pad to longest in this shard
            max_len = max(len(ids) for ids in shard_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = []
            for ids in shard_ids:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)

            batch_tensor = torch.tensor(padded, device=device)

            output = generate(
                model,
                batch_tensor,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.MASK_ID,
            )

            for i, orig_idx in enumerate(indices):
                generated_ids = output[i, max_len:]
                response = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                results.append((orig_idx, response.strip()))

        # Reassemble in original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
