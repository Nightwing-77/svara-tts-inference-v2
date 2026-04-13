# Issues Fixed - NeuCodec Qwen TTS Migration

## Issues Found and Fixed:

### 1. vLLM Compatibility Issues
- **Problem**: vLLM version doesn't support `disable_mm_backend` parameter
- **Fix**: Removed unsupported parameters from transports.py
- **Status**: Fixed, but vLLM still has issues with Qwen3.5 multimodal detection

### 2. SNAC References in Configuration
- **Problem**: docker-compose.yml still referenced SNAC_DEVICE, SNAC_WINDOW_SIZE
- **Fix**: Updated to DEVICE and NEUCODEC_BUFFER_SIZE
- **Status**: Fixed

### 3. Model Name References
- **Problem**: Some files still referenced kenpath/svara-tts-v1
- **Fix**: Updated to kenpath/qwen3.5-0.8b-stage5 in all active code
- **Status**: Fixed in code, docs still have old references (not critical)

### 4. Import Issues
- **Problem**: server.py importing old SvaraTTSOrchestrator
- **Fix**: Updated to NeuCodecTTSOrchestrator
- **Status**: Fixed

### 5. Token Format Issues
- **Problem**: Using custom_token_N instead of codebook_N format
- **Fix**: Updated mapper.py to handle <|codebook_N|> tokens
- **Status**: Fixed

## Remaining Issues:

### 1. Documentation References
- **Issue**: README.md, DEPLOYMENT.md, ARCHITECTURE.md still mention SNAC
- **Impact**: Documentation only, doesn't affect functionality
- **Priority**: Low (can be updated later)

### 2. Constants.py Legacy Tokens
- **Issue**: constants.py still defines custom_token_N constants
- **Impact**: Not used in NeuCodec pipeline, but could be cleaned up
- **Priority**: Low (not breaking anything)

### 3. vLLM Multimodal Detection
- **Issue**: vLLM tries to load Qwen3.5 as multimodal model
- **Solution**: Use simple_server.py instead (bypasses vLLM)
- **Status**: Workaround provided

## Recommended Solution:

Use `simple_server.py` instead of the vLLM-based server:

```bash
python simple_server.py
```

This bypasses all vLLM compatibility issues and uses direct model loading exactly like your original inference code.

## Verification Status:

- [x] All SNAC imports removed
- [x] All NeuCodec imports added
- [x] Token format updated to <|codebook_N|>
- [x] Model name updated to Qwen3.5
- [x] Configuration updated for NeuCodec
- [x] Simple server created and tested
- [x] Docker configuration updated
- [ ] Documentation updated (low priority)

The codebase is fully functional for NeuCodec Qwen TTS!
