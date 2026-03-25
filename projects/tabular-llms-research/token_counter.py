"""Token counter supporting both tiktoken (OpenAI) and HuggingFace tokenizers.

Project-local module for tabular-llms-research. Keeps the shared
``utils/count_token.py`` free of heavy HuggingFace dependencies.
"""

from __future__ import annotations

import functools
from multiprocessing import cpu_count

import tiktoken


# ---------------------------------------------------------------------------
# Tokenizer caching
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _get_tiktoken_encoder(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


@functools.lru_cache(maxsize=4)
def _get_hf_tokenizer(model_id: str):
    """Load and cache a HuggingFace tokenizer (lazy import)."""
    from transformers import AutoTokenizer  # noqa: F811 – lazy to avoid import cost
    return AutoTokenizer.from_pretrained(model_id)


def _is_tiktoken_model(name: str) -> bool:
    """Return True if *name* can be resolved by tiktoken."""
    try:
        tiktoken.encoding_for_model(name)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_tokens(
    prompts: list[str],
    tokenizer: str = "gpt-4o",
    num_threads: int | None = None,
) -> list[int]:
    """Return per-prompt token counts using the specified tokenizer.

    Parameters
    ----------
    prompts:
        List of text strings to tokenize.
    tokenizer:
        Either a tiktoken model name (e.g. ``"gpt-4o"``) or a HuggingFace
        model ID (e.g. ``"google/t5gemma-9b-9b-ul2-it"``).
    num_threads:
        Thread count for tiktoken batch encoding. Ignored for HF tokenizers.
        Defaults to ``cpu_count()``.
    """
    if not prompts:
        return []

    if num_threads is None:
        num_threads = cpu_count()

    if _is_tiktoken_model(tokenizer):
        encoder = _get_tiktoken_encoder(tokenizer)
        tokens = encoder.encode_batch(prompts, num_threads=num_threads)
        return [len(t) for t in tokens]

    # HuggingFace tokenizer path
    hf_tok = _get_hf_tokenizer(tokenizer)
    encoded = hf_tok(prompts, add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]
