#!/usr/bin/env python3
"""
Test 7B model only to verify basic functionality.
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('.')

def test_7b_model():
    """Test 7B model with simple prompts."""
    model_path = "/raid/sasaki/adaptive-speculative-decoding/models/qwen3-7b"
    
    print("=== Testing Qwen3-7B Model ===")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("Loading model...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "The capital of Japan is",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of renewable energy?"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"\\nTest {i+1}: {prompt}")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = len(inputs["input_ids"][0])
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            inference_time = time.time() - start_time
            
            # Decode
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate metrics
            tokens_per_sec = len(generated_ids) / inference_time if inference_time > 0 else 0
            
            print(f"Generated: {generated_text[:200]}...")
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Tokens generated: {len(generated_ids)}")
            print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": tokens_per_sec
            })
        
        # Summary
        avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_time = sum(r["inference_time"] for r in results) / len(results)
        
        print(f"\\n=== Summary ===")
        print(f"Average throughput: {avg_throughput:.2f} tokens/sec")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Model device: {device}")
        
        # Save results
        results_path = "/raid/sasaki/adaptive-speculative-decoding/results/7b_test_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "model_name": "Qwen3-7B",
                "load_time": load_time,
                "avg_throughput": avg_throughput,
                "avg_inference_time": avg_time,
                "device": str(device),
                "test_results": results
            }, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_mmlu_sample():
    """Test with MMLU dataset sample."""
    print("\\n=== Testing with MMLU Sample ===")
    
    dataset_path = "/raid/sasaki/adaptive-speculative-decoding/datasets/mmlu_test.json"
    if not Path(dataset_path).exists():
        print("MMLU dataset not found")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Take first 10 samples
    samples = data[:10]
    
    model_path = "/raid/sasaki/adaptive-speculative-decoding/models/qwen3-7b"
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        device = next(model.parameters()).device
        results = []
        
        for i, sample in enumerate(samples):
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            print(f"Sample {i+1}: {prompt[:100]}...")
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = len(inputs["input_ids"][0])
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            tokens_per_sec = len(generated_ids) / inference_time if inference_time > 0 else 0
            
            print(f"  Generated: {generated_text[:100]}...")
            print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "tokens_per_second": tokens_per_sec
            })
        
        avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
        print(f"\\nMMLU Average throughput: {avg_throughput:.2f} tokens/sec")
        
    except Exception as e:
        print(f"MMLU test error: {e}")

if __name__ == "__main__":
    success = test_7b_model()
    if success:
        run_mmlu_sample()
        print("\\n=== 7B Model Test Complete ===")
    else:
        print("\\n=== 7B Model Test Failed ===")