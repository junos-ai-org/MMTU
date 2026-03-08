"""Qwen inference backend — standard autoregressive generation via transformers."""

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backends.base import InferenceBackend


class QwenBackend(InferenceBackend):
    """Runs inference using Qwen's autoregressive generate()."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        # Generation defaults (overridden by config)
        self.max_new_tokens = 512
        self.temperature = 0.0

    def load(self, config: dict) -> None:
        model_cfg = config["model"]
        inference_cfg = config["inference"]

        model_path = model_cfg["model_path"]
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))

        print(f"Loading Qwen model: {model_path} (dtype={dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = inference_cfg.get("max_output_tokens", 512)
        self.temperature = inference_cfg.get("temperature", 0.0)

        print(f"  Qwen ready. max_new_tokens={self.max_new_tokens}, "
              f"temperature={self.temperature}")

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(self, prompt: str) -> str:
        text = self._apply_chat_template(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        texts = [self._apply_chat_template(p) for p in prompts]

        # Left-pad for batched generation
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        # Decode only the generated portion for each sample
        prompt_len = inputs["input_ids"].shape[1]
        responses = []
        for i in range(len(prompts)):
            generated_ids = output_ids[i, prompt_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def health_check(self) -> bool:
        return self.model is not None
