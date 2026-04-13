#!/usr/bin/env python3
"""
Simple FastAPI server for NeuCodec Qwen TTS using direct model loading.
This bypasses vLLM and uses the model directly like your original inference code.
"""

from __future__ import annotations
import os
import sys
import logging
import asyncio
import re
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = os.getenv("VLLM_MODEL", "kenpath/qwen3.5-0.8b-stage5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Global instances
model = None
tokenizer = None
codec = None

# ============================================================================
# Simple TTS Engine (Direct Model Loading)
# ============================================================================

class SimpleTTSEngine:
    """Simple TTS engine using direct model loading (no vLLM)."""
    
    def __init__(self):
        self.model_name = MODEL_NAME
        self.device = DEVICE
        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(self.device)
        
    def initialize(self):
        """Initialize model and tokenizer."""
        global model, tokenizer
        
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Login to HuggingFace if token is provided
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Logged in to HuggingFace")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        logger.info("Model and tokenizer loaded successfully")
        
    def generate_speech(self, text: str) -> bytes:
        """Generate speech from text using the NeuCodec model."""
        if model is None or tokenizer is None:
            raise RuntimeError("Model not initialized")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2000,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Extract codebook tokens from generated text
        generated_ids = output[0]
        decoded = tokenizer.decode(generated_ids)
        
        # Extract codebook tokens using regex
        ids = list(map(int, re.findall(r"<\|codebook_(\d+)\|>", decoded)))
        
        if not ids:
            logger.warning("No codebook tokens found in generated text")
            return b""
        
        logger.info(f"Generated {len(ids)} codebook tokens")
        
        # Convert to tensor and decode audio
        codec_tokens = torch.tensor(ids, dtype=torch.long).to(self.device)
        codec_tokens = codec_tokens.unsqueeze(0).unsqueeze(1)
        
        with torch.no_grad():
            audio = self.codec.decode_code(codec_tokens)
        
        # Convert to PCM bytes
        audio = audio.squeeze().cpu().numpy()
        pcm16 = (audio * 32767.0).astype(np.int16)
        return pcm16.tobytes()

# Global TTS engine instance
tts_engine = SimpleTTSEngine()

# ============================================================================
# FastAPI Server
# ============================================================================

app = FastAPI(
    title="NeuCodec Qwen TTS API",
    description="Simple TTS API using NeuCodec Qwen model",
    version="1.0.0"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Initializing NeuCodec Qwen TTS API...")
    tts_engine.initialize()
    yield
    logger.info("Shutting down NeuCodec Qwen TTS API...")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/v1/audio/speech")
async def speech_endpoint(request: dict):
    """Generate speech from text."""
    try:
        text = request.get("input", "")
        if not text:
            raise HTTPException(status_code=400, detail="Input text is required")
        
        response_format = request.get("response_format", "wav")
        
        logger.info(f"Generating speech for: {text[:50]}...")
        
        # Generate audio
        audio_bytes = tts_engine.generate_speech(text)
        
        if response_format == "wav":
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        else:
            # For other formats, you'd need ffmpeg conversion
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
            
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, lifespan=lifespan)
