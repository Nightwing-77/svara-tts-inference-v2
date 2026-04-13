
from __future__ import annotations
import re
from typing import List, Optional, Iterable

_TOKEN_RE = re.compile(r"<\|codebook_(\d+)\|>")

def extract_codebook_token_numbers(text: str):
    """Extract codebook token numbers from a text string.
    
    Each codebook token is represented by a <|codebook_N|> tag in the text, where N is the token number.
    This function extracts all the token numbers from the text and yields them one by one.
    
    Args:
        text: The text string to extract codebook token numbers from.
        
    Yields:
        int: The codebook token number.
    """
    for m in _TOKEN_RE.findall(text or ""):
        try:
            n = int(m)
            if n >= 0:
                yield n
        except Exception:
            continue

def raw_to_code_id(raw_num: int) -> int:
    """Convert a raw number to a code id.
    
    For NeuCodec, we simply return the raw number as the code ID.
    
    Args:
        raw_num: The raw number to convert.
    """
    return raw_num

class NeuCodecMapper:
    """
    Aggregates NeuCodec codebook token ids.
    Emits tokens as they arrive for immediate decoding.

    Args:
        buffer_size: Number of tokens to buffer before emitting. Default 100.
    """
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.codes: List[int] = []

    def feed_raw(self, raw: int) -> Optional[List[int]]:
        """Feed a raw token number. Returns buffered tokens when ready, else None."""
        code = raw_to_code_id(raw)
        if code < 0:
            return None
        self.codes.append(code)
        
        # Return buffered tokens when we have enough
        if len(self.codes) >= self.buffer_size:
            tokens_to_emit = self.codes[:self.buffer_size]
            self.codes = self.codes[self.buffer_size:]
            return tokens_to_emit
        return None

    def feed_text(self, token_text: str) -> List[List[int]]:
        """Return zero or more ready token buffers from a token_text."""
        out: List[List[int]] = []
        for n in extract_codebook_token_numbers(token_text):
            tokens = self.feed_raw(n)
            if tokens is not None:
                out.append(tokens)
        return out
    
    def flush(self) -> List[int]:
        """Flush any remaining tokens."""
        remaining = self.codes
        self.codes = []
        return remaining
