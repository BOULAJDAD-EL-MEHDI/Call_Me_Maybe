"""Thin wrapper around the llm_sdk.Small_LLM_Model.

This module is the ONLY place that talks to the SDK directly.
Everything else (decoder.py, model.py) only calls methods on LLMWrapper.
"""

import json
import importlib
Small_LLM_Model = importlib.import_module("llm_sdk").Small_LLM_Model


class LLMWrapper:
    """Loads the model once and exposes what the decoder needs.

    Attributes:
        model: The underlying Small_LLM_Model instance from the SDK.
        id_to_token: Mapping from token id -> token string, built once
            from the vocab file (avoids re-reading the file every call).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = Small_LLM_Model(model_name)
        self.id_to_token = self._build_id_to_token()

    def _build_id_to_token(self) -> dict[int, str]:
        """Build an id -> token string map from the SDK's vocab file."""
        vocab_path = self.model.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r") as file:
                vocab = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not load vocab file: {vocab_path}") from error
        return {token_id: token_str for token_str, token_id in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encode text into a flat list of token ids."""
        ids: list[int] = self.model.encode(text).tolist()[0]
        return ids

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return the logits for the next token, given input_ids so far."""
        logits: list[float] = self.model.get_logits_from_input_ids(input_ids)
        return logits

    def clean_token(self, token_id: int) -> str:
        """Return the human-readable text a token id represents.

        BPE tokenizers (like Qwen's) mark a leading space with 'Ġ'.
        We convert that into a real space so we can build real text.
        """
        return self.id_to_token.get(token_id, "").replace("Ġ", " ")
