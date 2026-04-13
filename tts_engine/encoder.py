import torch
from typing import List, Optional, Union

from .constants import (
    BOS_TOKEN,
    END_OF_TURN,
    START_OF_HUMAN,
    END_OF_HUMAN,
    START_OF_AI,
    END_OF_AI,
    START_OF_SPEECH,
    END_OF_SPEECH,
    AUDIO_TOKEN,
)

# Optional: pre-create scalar token tensors (1,1) to avoid re-alloc each call
BOS_ID             = torch.tensor([[BOS_TOKEN]],        dtype=torch.int64)
START_OF_HUMAN_ID  = torch.tensor([[START_OF_HUMAN]],   dtype=torch.int64)
END_OF_HUMAN_ID    = torch.tensor([[END_OF_HUMAN]],     dtype=torch.int64)
START_OF_AI_ID     = torch.tensor([[START_OF_AI]],      dtype=torch.int64)
END_OF_AI_ID       = torch.tensor([[END_OF_AI]],        dtype=torch.int64)
START_OF_SPEECH_ID = torch.tensor([[START_OF_SPEECH]],  dtype=torch.int64)
END_OF_SPEECH_ID   = torch.tensor([[END_OF_SPEECH]],    dtype=torch.int64)
END_OF_TURN_ID     = torch.tensor([[END_OF_TURN]],      dtype=torch.int64)
AUDIO_TOKEN_ID     = torch.tensor([[AUDIO_TOKEN]],      dtype=torch.int64)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shape (1, seq_len)."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _human_turn(text_ids: torch.Tensor) -> torch.Tensor:
    """
    Build a human text block:
    START_OF_HUMAN, AUDIO_TOKEN, text_ids, END_OF_HUMAN, END_OF_TURN
    """
    text_ids = _ensure_2d(text_ids)
    return torch.cat(
        [
            START_OF_HUMAN_ID,
            AUDIO_TOKEN_ID,
            text_ids,
            END_OF_HUMAN_ID,
            END_OF_TURN_ID,
        ],
        dim=1,
    )


def _audio_turn(audio_ids: torch.Tensor) -> torch.Tensor:
    """
    Build an AI audio reference block:
    START_OF_AI, START_OF_SPEECH, audio_ids, END_OF_SPEECH, END_OF_AI, END_OF_TURN
    """
    audio_ids = _ensure_2d(audio_ids)
    return torch.cat(
        [
            START_OF_AI_ID,
            START_OF_SPEECH_ID,
            audio_ids,
            END_OF_SPEECH_ID,
            END_OF_AI_ID,
            END_OF_TURN_ID,
        ],
        dim=1,
    )


def _final_generation_prefix() -> torch.Tensor:
    """
    Final tail to start generation of speech tokens:
    START_OF_AI, START_OF_SPEECH
    (no END_OF_SPEECH / END_OF_AI here; model generates them)
    """
    return torch.cat(
        [
            START_OF_AI_ID,
            START_OF_SPEECH_ID,
        ],
        dim=1,
    )



def simple_text_to_tokens(
    text: str,
    tokenizer = None,
    return_decoded: bool = False,
) -> Union[List[int], str]:
    """
    Build simple prompt for NeuCodec-based TTS model.
    Just tokenizes the text directly without complex formatting.

    Args:
        text: Target text to synthesize.
        tokenizer: Tokenizer used to convert text to IDs.
        return_decoded: 
            - False (default): return List[int] of token IDs.
            - True: return decoded string prompt.

    Returns:
        List[int] if return_decoded=False
        str      if return_decoded=True
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for simple_text_to_tokens")
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    # Simple tokenization - just convert text to token IDs
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,  # Add BOS and other special tokens
    )
    
    input_ids = inputs.input_ids.view(-1).tolist()

    if return_decoded:
        return tokenizer.decode(input_ids, skip_special_tokens=False)

    return input_ids