"""Reference Token Counter for the Token Optimiser gateway.

Calculates exact token counts using OpenAI's `tiktoken` BPE tokenizer (`cl100k_base` or `o200k_base`)
when installed, gracefully falling back to a deterministic whitespace heuristic for offline CPU environments.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("gateway.tokenizer")

_WS = re.compile(r"\s+")


class TokenCounter:
    """Calculates prompt and completion tokens with exact BPE subword segmentation.

    Attributes:
        encoding: Tiktoken encoding name (default 'cl100k_base').
        _enc: Loaded tiktoken Encoding instance or None if unavailable.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.encoding = encoding
        self._enc = self._load_encoding(encoding)

    @staticmethod
    def _load_encoding(encoding: str) -> Any | None:
        """Attempts to load the specified tiktoken encoding."""
        try:
            import tiktoken

            return tiktoken.get_encoding(encoding)
        except Exception as exc:
            logger.debug("Tiktoken encoding '%s' not loaded, using whitespace heuristic: %s", encoding, exc)
            return None

    @property
    def backend(self) -> str:
        """Returns the active tokenization backend identifier."""
        return f"tiktoken_{self.encoding}" if self._enc is not None else "whitespace"

    def count(self, text: str) -> int:
        """Counts the number of tokens in the given text string.

        Args:
            text: Raw input text.

        Returns:
            Integer token count.
        """
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return len(_WS.sub(" ", text).strip().split()) if text.strip() else 0
