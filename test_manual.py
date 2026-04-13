#!/usr/bin/env python3
"""
Exact copy of your working manual code - this should work perfectly!
"""

import os
import sys
from huggingface_hub import login
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec
import re

def test_manual_code():
    """Test the exact code that works for you."""
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace")
    
    # Your exact working code
    model_name = "kenpath/qwen3.5-0.8b-stage5"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec = codec.to(model.device)

    text = "यह एक सरल हिंदी वाक्य है।"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)
    
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

    generated_ids = output[0]

    decoded = tokenizer.decode(generated_ids)

    ids = list(map(int, re.findall(r"<\|codebook_(\d+)\|>", decoded)))

    if not ids:
        print("ERROR: No codebook tokens found!")
        print("Generated text:", decoded)
        return False

    codec_tokens = torch.tensor(ids, dtype=torch.long).to(model.device)

    print("Total tokens:", len(codec_tokens))

    codec_tokens = codec_tokens.unsqueeze(0)
    codec_tokens = codec_tokens.unsqueeze(1)

    print("Final shape:", codec_tokens.shape)

    with torch.no_grad():
        audio = codec.decode_code(codec_tokens)

    audio = audio.squeeze().cpu().numpy()

    # Save audio file
    sf.write("manual_output.wav", audio, samplerate=24000)
    print("SUCCESS: Audio saved to manual_output.wav")
    return True

if __name__ == "__main__":
    print("Testing manual code (exact copy of your working version)...")
    success = test_manual_code()
    if success:
        print("✅ Manual test PASSED!")
    else:
        print("❌ Manual test FAILED!")
