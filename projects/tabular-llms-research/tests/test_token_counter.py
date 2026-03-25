"""Tests for token_counter.py — dual-backend token counting."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so ``import token_counter`` works.
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from token_counter import count_tokens, _is_tiktoken_model


# ---------------------------------------------------------------------------
# tiktoken path (mocked — tiktoken needs network to download encodings)
# ---------------------------------------------------------------------------

def _make_mock_tiktoken_encoder():
    """Return a mock tiktoken encoder that splits on whitespace."""
    enc = MagicMock()
    enc.encode_batch.side_effect = lambda prompts, num_threads=1: [
        p.split() for p in prompts
    ]
    return enc


class TestTiktokenPath:
    @patch("token_counter._get_tiktoken_encoder")
    @patch("token_counter._is_tiktoken_model", return_value=True)
    def test_basic_counting(self, _mock_is_tik, mock_get_enc):
        mock_get_enc.return_value = _make_mock_tiktoken_encoder()
        counts = count_tokens(["hello world", "foo bar baz"], tokenizer="gpt-4o")
        assert counts == [2, 3]

    def test_empty_input(self):
        assert count_tokens([], tokenizer="gpt-4o") == []

    @patch("token_counter._get_tiktoken_encoder")
    @patch("token_counter._is_tiktoken_model", return_value=True)
    def test_single_token(self, _mock_is_tik, mock_get_enc):
        mock_get_enc.return_value = _make_mock_tiktoken_encoder()
        counts = count_tokens(["hello"], tokenizer="gpt-4o")
        assert counts == [1]

    def test_is_tiktoken_model_known(self):
        # These should not need network; they just check tiktoken's model map
        assert _is_tiktoken_model("google/t5gemma-9b-9b-ul2-it") is False
        assert _is_tiktoken_model("not-a-real-model") is False


# ---------------------------------------------------------------------------
# HuggingFace path (mocked to avoid network / heavy deps)
# ---------------------------------------------------------------------------

class TestHuggingFacePath:
    def _make_mock_tokenizer(self):
        """Return a mock that behaves like a HuggingFace tokenizer __call__."""
        tok = MagicMock()
        # Simulate tokenizer("hello world") → {"input_ids": [[1, 2, 3]]}
        tok.side_effect = lambda prompts, add_special_tokens=True: {
            "input_ids": [[i] * (len(p.split()) + 1) for i, p in enumerate(prompts)]
        }
        return tok

    @patch("token_counter._get_hf_tokenizer")
    @patch("token_counter._is_tiktoken_model", return_value=False)
    def test_hf_counting(self, _mock_is_tik, mock_get_hf):
        mock_tok = self._make_mock_tokenizer()
        mock_get_hf.return_value = mock_tok

        counts = count_tokens(
            ["hello world", "one two three four"],
            tokenizer="google/t5gemma-9b-9b-ul2-it",
        )
        assert len(counts) == 2
        # "hello world" → 3 tokens (2 words + 1), "one two three four" → 5 tokens
        assert counts[0] == 3
        assert counts[1] == 5
        mock_get_hf.assert_called_once_with("google/t5gemma-9b-9b-ul2-it")

    @patch("token_counter._get_hf_tokenizer")
    def test_hf_empty_input(self, mock_get_hf):
        # Empty list should short-circuit without calling the tokenizer
        counts = count_tokens([], tokenizer="google/t5gemma-9b-9b-ul2-it")
        assert counts == []
        mock_get_hf.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: build_dataset uses token_counter
# ---------------------------------------------------------------------------

class TestBuildDatasetIntegration:
    """Verify that build_dataset.py imports from token_counter (not utils.count_token)."""

    def test_import_path(self):
        """build_dataset should reference token_counter, not the old shared utility."""
        build_dataset_path = _project_dir / "build_dataset.py"
        source = build_dataset_path.read_text()
        assert "from token_counter import count_tokens" in source
        assert "from utils.count_token import" not in source
