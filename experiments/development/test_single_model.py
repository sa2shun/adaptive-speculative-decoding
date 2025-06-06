#!/usr/bin/env python3
"""
Test single model performance to verify model loading and basic functionality.
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('.')

def test_model_loading(model_path: str, model_name: str):
    """Test loading and basic inference with a single model."""
    print(f"\n=== Testing {model_name} ===")
    print(f"Model path: {model_path}")
    
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
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test inference
        test_prompts = [
            "The capital of France is",
            "What is 2 + 2?",
            "Explain the concept of machine learning in one sentence."
        ]
        
        results = []
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: {prompt}")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"Response: {generated_text}")
            print(f"Inference time: {inference_time:.3f}s")
            
            results.append({
                "prompt": prompt,
                "response": generated_text,
                "inference_time": inference_time,
                "total_tokens": len(outputs[0]),
                "new_tokens": len(outputs[0]) - len(inputs.input_ids[0])
            })
        
        # Calculate throughput
        total_new_tokens = sum(r["new_tokens"] for r in results)
        total_time = sum(r["inference_time"] for r in results)
        throughput = total_new_tokens / total_time if total_time > 0 else 0
        
        print(f"\n=== {model_name} Summary ===")
        print(f"Average inference time: {total_time/len(results):.3f}s")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Model loading time: {load_time:.2f}s")
        
        return {
            "model_name": model_name,
            "model_path": model_path,
            "total_params": total_params,
            "load_time": load_time,
            "throughput": throughput,
            "test_results": results
        }
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return {
            "model_name": model_name,
            "model_path": model_path,
            "error": str(e)
        }

def main():
    """Test all available models."""
    base_model_dir = Path("/raid/sasaki/adaptive-speculative-decoding/models")
    
    # Check available models
    available_models = []
    
    model_configs = [
        ("qwen3-7b", "Qwen3-7B"),
        ("qwen3-14b", "Qwen3-14B"), 
        ("qwen3-32b", "Qwen3-32B"),
        ("qwen3-72b", "Qwen3-72B")
    ]
    
    for model_dir, model_name in model_configs:
        model_path = base_model_dir / model_dir
        if model_path.exists() and (model_path / "config.json").exists():
            available_models.append((str(model_path), model_name))
            print(f"✓ Found {model_name} at {model_path}")
        else:
            print(f"✗ {model_name} not ready at {model_path}")
    
    if not available_models:
        print("No models available for testing!")
        return
    
    # Test each available model
    all_results = []
    for model_path, model_name in available_models:
        result = test_model_loading(model_path, model_name)
        all_results.append(result)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        time.sleep(2)
    
    # Save results
    results_path = "/raid/sasaki/adaptive-speculative-decoding/results/model_test_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== All Results Saved ===")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print(f"\n=== Summary ===")
    for result in all_results:
        if "error" not in result:
            print(f"{result['model_name']}: {result['throughput']:.2f} tokens/sec")
        else:
            print(f"{result['model_name']}: ERROR - {result['error']}")

if __name__ == "__main__":
    main()