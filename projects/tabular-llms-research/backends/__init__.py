"""Inference backend registry."""

from backends.base import InferenceBackend
from backends.qwen3_backend import Qwen3Backend
from backends.t5gemma_backend import T5GemmaBackend

BACKENDS = {
    "qwen3": Qwen3Backend,
    "t5gemma": T5GemmaBackend,
}


def get_backend(name: str) -> type[InferenceBackend]:
    """Look up a backend class by name."""
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return BACKENDS[name]
