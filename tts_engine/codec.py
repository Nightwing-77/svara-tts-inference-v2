# tts_engine/codec.py
from neucodec import NeuCodec
from typing import List, Optional
import logging
import numpy as np
import torch
import os
from transformers import AutoTokenizer
from .utils import resample_audio

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading NeuCodec model for each instance
_NEUCODEC_MODEL_CACHE: dict[str, NeuCodec] = {}

# Global tokenizer cache to avoid reloading tokenizer for each request
_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def _get_or_load_neucodec_model(device: str, model_name: str = "neuphonic/neucodec") -> NeuCodec:
    """
    Get cached NeuCodec model or load it if not cached.

    This prevents repeated model loading when creating multiple codec instances.
    Models are cached per device to handle multi-GPU scenarios.
    """
    cache_key = f"{model_name}_{device}"

    if cache_key not in _NEUCODEC_MODEL_CACHE:
        logger.info(f"Loading NeuCodec model: {model_name} on device: {device}")
        model = NeuCodec.from_pretrained(model_name).eval().to(device)
        _NEUCODEC_MODEL_CACHE[cache_key] = model

    return _NEUCODEC_MODEL_CACHE[cache_key]


def get_or_load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Get cached tokenizer or load it if not cached.

    Automatically uses HF_TOKEN environment variable if available for private models.
    """
    if model_name not in _TOKENIZER_CACHE:
        logger.info(f"Loading tokenizer: {model_name}")
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded (vocab_size={len(tokenizer)})")
        _TOKENIZER_CACHE[model_name] = tokenizer

    return _TOKENIZER_CACHE[model_name]


class NeuCodecWrapper:
    """
    Unified NeuCodec wrapper for decoding tokens to audio.
    
    Supports:
    - Decoding: NeuCodec tokens → PCM16 audio (for TTS synthesis)
    
    Uses a global model cache to avoid reloading the NeuCodec model when creating
    multiple instances, which significantly improves initialization time.
    """
    
    def __init__(self, device: Optional[str] = None, model_name: str = "neuphonic/neucodec"):
        """
        Initialize NeuCodec wrapper.
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            model_name: HuggingFace model identifier for NeuCodec
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.sample_rate = 24000  # NeuCodec 24kHz model
        
        # Get or load model from cache
        self.model = _get_or_load_neucodec_model(device, model_name)

    def encode_audio(
        self,
        audio: torch.Tensor,
        input_sample_rate: int = 24000
    ) -> List[int]:
        """
        Encode audio waveform to NeuCodec tokens.
        
        This method takes audio and converts it to a sequence of tokens.
        Automatically resamples audio to 24kHz if needed.
        
        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,).
            input_sample_rate: Sample rate of the input audio in Hz. If not 24000,
                              audio will be automatically resampled to 24kHz.
        
        Returns:
            List of token IDs.
        """
        # Resample to 24kHz if needed
        if input_sample_rate != self.sample_rate:            
            audio = resample_audio(audio, input_sample_rate, self.sample_rate, self.device)
        
        logger.debug(f"Audio shape after resample: {audio.shape}")
        
        # Ensure proper shape for NeuCodec
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, samples)
        
        # Move to device and ensure float32
        audio = audio.to(dtype=torch.float32, device=self.device)
        
        logger.debug(f"Audio shape going into NeuCodec encode: {audio.shape}")
        
        # Encode with NeuCodec
        with torch.inference_mode():
            codes = self.model.encode(audio)
        
        return codes.tolist()
    
    def decode_tokens(self, tokens: List[int]) -> bytes:
        """
        Decode NeuCodec tokens into PCM16 bytes.

        Args:
            tokens: List of NeuCodec token IDs.

        Returns:
            PCM16 mono bytes; empty bytes if invalid input.
        """
        if not tokens:
            return b""

        # Convert to tensor and reshape for NeuCodec
        codec_tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        # Add batch and sequence dimensions if needed
        if codec_tokens.dim() == 1:
            codec_tokens = codec_tokens.unsqueeze(0)  # [1, seq_len]
        
        logger.debug(f"Token shape for NeuCodec decode: {codec_tokens.shape}")

        with torch.inference_mode():
            audio = self.model.decode_code(codec_tokens)

        # Convert to PCM16
        audio = audio.squeeze().cpu().numpy()
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()

