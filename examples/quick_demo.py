#!/usr/bin/env python3
"""
Quick demonstration of Adaptive Speculative Decoding
"""

import requests
import json
import time

def test_adaptive_decoding():
    """Test the adaptive speculative decoding system"""
    
    # API endpoint
    url = "http://localhost:8000/generate"
    
    # Test prompts with varying difficulty
    test_prompts = [
        {
            "name": "Simple",
            "prompt": "What is the capital of France?",
            "expected_stage": 0  # Should stop early
        },
        {
            "name": "Medium",
            "prompt": "Explain the process of photosynthesis in plants.",
            "expected_stage": 1  # Middle stages
        },
        {
            "name": "Complex",
            "prompt": "Write a detailed implementation of the quicksort algorithm in Python with comprehensive error handling and optimization for small arrays.",
            "expected_stage": 3  # Should reach final stage
        }
    ]
    
    print("Testing Adaptive Speculative Decoding System\n")
    print("=" * 60)
    
    for test in test_prompts:
        print(f"\nTest: {test['name']}")
        print(f"Prompt: {test['prompt'][:50]}...")
        
        # Make request
        response = requests.post(
            url,
            json={
                "prompt": test['prompt'],
                "max_tokens": 256,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Stopped at stage: {result['stopped_at_stage']} (expected: {test['expected_stage']})")
            print(f"Latency: {result['latency_ms']:.2f}ms")
            print(f"Stage probabilities: {[f'{p:.3f}' for p in result['stage_probabilities']]}")
            print(f"Output preview: {result['output'][:100]}...")
            
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        print("-" * 60)
        time.sleep(1)  # Avoid overwhelming the server

def check_server_health():
    """Check if the server is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Server is healthy")
            return True
    except:
        pass
    
    print("✗ Server is not running. Start it with: python -m src.serving.server")
    return False

def display_statistics():
    """Display server statistics"""
    try:
        response = requests.get("http://localhost:8000/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\nServer Statistics:")
            print(f"Total requests: {stats['total_requests']}")
            print(f"Average latency: {stats['avg_latency']:.2f}ms")
            print(f"Stage distribution: {stats['stage_stops']}")
    except:
        print("Could not fetch statistics")

if __name__ == "__main__":
    print("Adaptive Speculative Decoding Demo")
    print("==================================\n")
    
    if check_server_health():
        test_adaptive_decoding()
        display_statistics()
    else:
        print("\nPlease start the server first with:")
        print("  python -m src.serving.server --config configs/serving.yaml")