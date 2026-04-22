#!/usr/bin/env python3
"""
FastAPI server for NeuCodec Qwen TTS using direct model loading.
"""

from __future__ import annotations
import os
import re
import logging
import base64
from pathlib import Path
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch
import time
import numpy as np
import soundfile as sf
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("VLLM_MODEL", "kenpath/qwen3.5-0.8b-stage5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SPEAKER_ID = os.getenv("SPEAKER_ID", "a1e51fd5")

# Global instances
model = None
tokenizer = None
codec = None

def initialize():
    """Load model, tokenizer, and codec."""
    global model, tokenizer, codec

    logger.info(f"Loading model: {MODEL_NAME} on device: {DEVICE}")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec = codec.to(model.device)

    logger.info("Model, tokenizer, and codec loaded successfully")


def generate_tts(
    text: str,
    speaker_id: str = DEFAULT_SPEAKER_ID,
    max_new_tokens: int = 2000,
) -> bytes:
    """Generate speech audio bytes from input text."""
    if model is None or tokenizer is None or codec is None:
        raise RuntimeError("Model not initialized")

    formatted_text = (
        f"<|tts|><tts_text_bos_single>{speaker_id}: "
        f"{text}<tts_text_eod><|audio_start|>"
    )

    inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=[248071, 248044],
        )

    decoded = tokenizer.decode(output[0])
    ids = list(map(int, re.findall(r"<\|codebook_(\d+)\|>", decoded)))

    if not ids:
        raise ValueError("No codec tokens generated. Check prompt or model.")

    logger.info(f"Generated {len(ids)} codebook tokens")

    codec_tokens = torch.tensor(ids, dtype=torch.long).to(model.device)
    codec_tokens = codec_tokens.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        audio = codec.decode_code(codec_tokens)

    audio = audio.squeeze().cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=24000, format="WAV")
    buf.seek(0)
    return buf.read()


def generate_tts_with_timing(
    text: str,
    speaker_id: str = DEFAULT_SPEAKER_ID,
    max_new_tokens: int = 2000,
) -> tuple[bytes, dict]:
    """Generate speech with timing metrics (TTFB and total time)."""
    if model is None or tokenizer is None or codec is None:
        raise RuntimeError("Model not initialized")

    metrics = {
        "start_time": time.time(),
        "ttfb_ms": None,
        "total_time_ms": None,
        "tokens_generated": 0,
        "input_text": text,
        "speaker_id": speaker_id,
    }

    formatted_text = (
        f"<|tts|><tts_text_bos_single>{speaker_id}: "
        f"{text}<tts_text_eod><|audio_start|>"
    )

    inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=[248071, 248044],
        )

    first_token_time = time.time()
    metrics["ttfb_ms"] = round((first_token_time - metrics["start_time"]) * 1000, 2)

    decoded = tokenizer.decode(output[0])
    ids = list(map(int, re.findall(r"<\|codebook_(\d+)\|>", decoded)))

    if not ids:
        raise ValueError("No codec tokens generated. Check prompt or model.")

    metrics["tokens_generated"] = len(ids)

    codec_tokens = torch.tensor(ids, dtype=torch.long).to(model.device)
    codec_tokens = codec_tokens.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        audio = codec.decode_code(codec_tokens)

    audio = audio.squeeze().cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=24000, format="WAV")
    buf.seek(0)
    audio_bytes = buf.read()

    metrics["total_time_ms"] = round((time.time() - metrics["start_time"]) * 1000, 2)

    return audio_bytes, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuCodec Qwen TTS API...")
    initialize()
    yield
    logger.info("Shutting down NeuCodec Qwen TTS API...")


app = FastAPI(
    title="NeuCodec Qwen TTS API",
    description="TTS API using NeuCodec Qwen model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}


class SpeechRequest(BaseModel):
    input: str
    speaker_id: str = DEFAULT_SPEAKER_ID
    response_format: str = "wav"
    max_new_tokens: int = 2000


@app.post("/v1/audio/speech")
async def speech_endpoint(request: SpeechRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input text is required")

    logger.info(f"Generating speech for: {request.input[:80]}...")

    try:
        wav_bytes = generate_tts(
            text=request.input,
            speaker_id=request.speaker_id,
            max_new_tokens=request.max_new_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


class MetricsResponse(BaseModel):
    audio: str
    metrics: dict


@app.post("/v1/audio/speech/metrics")
async def speech_metrics_endpoint(request: SpeechRequest):
    """Generate speech and return with TTFB and timing metrics."""
    if not request.input:
        raise HTTPException(status_code=400, detail="Input text is required")

    logger.info(f"Generating speech with metrics for: {request.input[:80]}...")

    try:
        wav_bytes, metrics = generate_tts_with_timing(
            text=request.input,
            speaker_id=request.speaker_id,
            max_new_tokens=request.max_new_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    return JSONResponse({
        "audio": audio_b64,
        "metrics": metrics
    })


# Load demo HTML template on startup
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DEMO_HTML = (_TEMPLATE_DIR / "demo.html").read_text(encoding="utf-8")


@app.get("/demo", response_class=HTMLResponse)
async def demo_endpoint():
    """Serve the TTS GUI demo page for inference benchmarking."""
    return _DEMO_HTML


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)