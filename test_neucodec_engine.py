#!/usr/bin/env python3
"""
Test script for the modified NeuCodec inference engine.
This tests the basic functionality with your qwen3.5-0.8b-stage5 model.
"""

import os
import sys
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Add the tts_engine to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_engine.codec import NeuCodecWrapper
from tts_engine.mapper import NeuCodecMapper, extract_codebook_token_numbers
from tts_engine.encoder import simple_text_to_tokens
from tts_engine.orchestrator import NeuCodecTTSOrchestrator


def test_neucodec_basic():
    """Test basic NeuCodec functionality."""
    print("Testing NeuCodec basic functionality...")
    
    # Initialize NeuCodec
    codec = NeuCodecWrapper()
    print(f"NeuCodec loaded on device: {codec.device}")
    
    # Test token extraction
    test_text = "This is a test <|codebook_123|> <|codebook_456|>"
    tokens = list(extract_codebook_token_numbers(test_text))
    print(f"Extracted tokens: {tokens}")
    
    # Test simple decoding (with dummy tokens)
    dummy_tokens = [100, 200, 300, 400, 500] * 20  # 100 dummy tokens
    audio_bytes = codec.decode_tokens(dummy_tokens)
    print(f"Decoded audio bytes: {len(audio_bytes)} bytes")
    
    return True


def test_model_and_tokenizer():
    """Test loading the qwen3.5-0.8b-stage5 model and tokenizer."""
    print("\nTesting model and tokenizer loading...")
    
    model_name = "kenpath/qwen3.5-0.8b-stage5"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print(f"Model loaded: {model.__class__.__name__}")
    
    # Test simple text encoding
    test_text = "This is a simple test."
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"Tokenized '{test_text}': {inputs.input_ids.shape}")
    
    # Test generation
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    return True


def test_simple_encoding():
    """Test the simple text encoding function."""
    print("\nTesting simple text encoding...")
    
    from tts_engine.codec import get_or_load_tokenizer
    model_name = "kenpath/qwen3.5-0.8b-stage5"
    tokenizer = get_or_load_tokenizer(model_name)
    
    test_text = "This is a test sentence."
    
    # Test token list output
    tokens = simple_text_to_tokens(test_text, tokenizer, return_decoded=False)
    print(f"Tokens: {tokens[:10]}... (total: {len(tokens)})")
    
    # Test decoded string output
    decoded = simple_text_to_tokens(test_text, tokenizer, return_decoded=True)
    print(f"Decoded: {decoded}")
    
    return True


def test_full_pipeline():
    """Test the full pipeline with a mock transport."""
    print("\nTesting full pipeline...")
    
    # Mock transport for testing
    class MockTransport:
        def stream(self, prompt, **kwargs):
            # Simulate streaming tokens
            for i in range(50):
                yield f"<|codebook_{i * 10}|>"
        
        def astream(self, prompt, **kwargs):
            # Mock async stream
            import asyncio
            async def gen():
                for i in range(50):
                    yield f"<|codebook_{i * 10}|>"
                    await asyncio.sleep(0.01)
            return gen()
    
    # Create orchestrator with mock transport
    transport = MockTransport()
    orchestrator = NeuCodecTTSOrchestrator(transport)
    
    # Test streaming
    print("Testing sync streaming...")
    audio_chunks = list(orchestrator.stream("Test text"))
    print(f"Generated {len(audio_chunks)} audio chunks, total bytes: {sum(len(c) for c in audio_chunks)}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("NEUCODEC INFERENCE ENGINE TEST")
    print("=" * 60)
    
    tests = [
        test_neucodec_basic,
        test_simple_encoding,
        # test_model_and_tokenizer,  # Uncomment if you have HF token
        test_full_pipeline,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("PASSED")
            else:
                print("FAILED")
        except Exception as e:
            print(f"FAILED with error: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 40)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The NeuCodec inference engine is working.")
    else:
        print("Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
