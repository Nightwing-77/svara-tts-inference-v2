# Running NeuCodec Qwen TTS Inference

## Quick Setup Steps

### 1. Clone and Setup
```bash
git clone https://github.com/Nightwing-77/svara-tts-inference-v2
cd svara-tts-inference-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` file if needed (defaults should work):
```
VLLM_MODEL=kenpath/qwen3.5-0.8b-stage5
DEVICE=cuda
NEUCODEC_BUFFER_SIZE=100
```

### 4. Login to Hugging Face (if model is private)
```python
from huggingface_hub import login
login(token="your_hf_token")
```
Or set in `.env`:
```
HF_TOKEN=your_hf_token
```

### 5. Run the Server

**Option A: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option B: Direct Python**
```bash
cd api && python server.py
```

### 6. Test the API

**Health Check:**
```bash
curl http://localhost:8080/health
```

**Generate Speech:**
```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test of the NeuCodec Qwen TTS model",
    "response_format": "wav"
  }' \
  --output test_output.wav
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "input": "Hello world, this is NeuCodec TTS speaking!",
        "response_format": "mp3"
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## What's Different from Original

- **Model**: Uses `kenpath/qwen3.5-0.8b-stage5` instead of SNAC-based model
- **Codec**: NeuCodec instead of SNAC
- **Tokens**: Uses `<|codebook_N|>` format instead of `<custom_token_N>`
- **Simpler**: No complex conversational prompt format, just direct text input

## Troubleshooting

**ModuleNotFoundError: No module named 'neucodec'**
```bash
pip install neucodec
```

**CUDA out of memory:**
- Reduce `VLLM_GPU_MEMORY_UTILIZATION` in `.env`
- Use smaller model or quantization

**Slow inference:**
- Ensure `DEVICE=cuda` in `.env`
- Check GPU is being used

## API Endpoints

- `GET /health` - Health check
- `POST /v1/audio/speech` - Generate speech (OpenAI-compatible)
- `GET /v1/voices` - List available voices (if configured)

The server runs on port 8080 by default.
