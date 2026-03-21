"""T5Gemma inference backend — HuggingFace transformers (encoder-decoder)."""

import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

from backends.base import InferenceBackend


class T5GemmaBackend(InferenceBackend):
    """Runs inference using T5Gemma via HuggingFace transformers."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.max_new_tokens = 512
        self.temperature = 0.0

    def load(self, config: dict) -> None:
        model_cfg = config["model"]
        inference_cfg = config["inference"]

        model_path = model_cfg["model_path"]
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        self.max_new_tokens = inference_cfg.get("max_output_tokens", 512)
        self.temperature = inference_cfg.get("temperature", 0.0)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading T5Gemma processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)

        print(f"Loading T5Gemma model from {model_path} (dtype={dtype})...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        print(f"  T5Gemma ready on {self.device}. max_new_tokens={self.max_new_tokens}")

    def generate(self, prompt: str) -> str:
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        )
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated tokens (skip input for encoder-decoder)
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # T5Gemma supports batched generation via padding
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def health_check(self) -> bool:
        return self.model is not None and self.processor is not None
