#!/usr/bin/env python3
"""
Simple test script to verify the NeuCodec server works.
"""

import requests
import json

def test_server():
    """Test the NeuCodec TTS server."""
    base_url = "http://localhost:8080"
    
    print("Testing NeuCodec TTS Server...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test voices endpoint
    try:
        response = requests.get(f"{base_url}/v1/voices")
        print(f"✅ Voices check: {response.status_code}")
        if response.status_code == 200:
            voices = response.json()
            print(f"   Available voices: {len(voices.get('voices', []))}")
    except Exception as e:
        print(f"❌ Voices check failed: {e}")
    
    # Test TTS generation
    try:
        tts_request = {
            "input": "Hello world, this is the NeuCodec Qwen TTS model speaking!",
            "response_format": "wav"
        }
        
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json=tts_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"✅ TTS generation: {response.status_code}")
        
        if response.status_code == 200:
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print(f"   Audio saved to: test_output.wav ({len(response.content)} bytes)")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return False
    
    print("\n🎉 Server test completed!")
    return True

if __name__ == "__main__":
    test_server()
