#!/usr/bin/env python3
"""
Simplified real cost measurement for Qwen2.5 models
Measures actual inference latencies to replace theoretical parameter-based costs
"""

import time
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_model_latency(model_path: str, model_name: str, samples: int = 20) -> Dict:
    """Measure latency for a single model with simple prompts."""
    
    try:
        from vllm import LLM, SamplingParams
        
        logger.info(f"Loading model: {model_name}")
        
        # Configure based on model size
        if "7b" in model_name:
            tensor_parallel_size = 1
        elif "14b" in model_name:
            tensor_parallel_size = 1
        elif "32b" in model_name:
            tensor_parallel_size = 2
        else:  # 72b
            tensor_parallel_size = 4
        
        # Load model
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            trust_remote_code=False,
            max_model_len=2048,
        )
        
        # Simple test prompts
        test_prompts = [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Write a short poem about nature.",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "Describe the water cycle.",
            "What is artificial intelligence?",
            "Explain the theory of relativity.",
            "How do computers work?",
            "What is climate change?"
        ]
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,  # Short responses for consistency
        )
        
        latencies = []
        
        # Warmup
        logger.info(f"Warming up {model_name}...")
        for _ in range(5):
            llm.generate(test_prompts[:2], sampling_params)
        
        # Actual measurements
        logger.info(f"Measuring {model_name} latency...")
        for i in range(samples):
            prompt = test_prompts[i % len(test_prompts)]
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = llm.generate([prompt], sampling_params)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  {i + 1}/{samples} completed")
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        median_latency = np.median(latencies)
        
        logger.info(f"{model_name} results:")
        logger.info(f"  Mean latency: {mean_latency:.3f}s")
        logger.info(f"  Std latency: {std_latency:.3f}s")
        logger.info(f"  Median latency: {median_latency:.3f}s")
        
        return {
            "model_name": model_name,
            "model_path": model_path,
            "samples": samples,
            "mean_latency": mean_latency,
            "std_latency": std_latency,
            "median_latency": median_latency,
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "all_latencies": latencies,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to measure {model_name}: {e}")
        return {
            "model_name": model_name,
            "model_path": model_path,
            "error": str(e),
            "success": False
        }

def main():
    """Main measurement function."""
    
    # Model configurations
    models = [
        {
            "name": "qwen2.5-7b",
            "path": "/raid/sasaki/adaptive-sd-models/qwen3-7b"
        },
        {
            "name": "qwen2.5-14b", 
            "path": "/raid/sasaki/adaptive-sd-models/qwen3-14b"
        },
        {
            "name": "qwen2.5-32b",
            "path": "/raid/sasaki/adaptive-sd-models/qwen3-32b"
        },
        {
            "name": "qwen2.5-72b",
            "path": "/raid/sasaki/adaptive-sd-models/qwen3-72b"
        }
    ]
    
    results = {}
    
    logger.info("Starting real cost measurement for Qwen2.5 models...")
    
    # Measure each model sequentially to avoid GPU conflicts
    for model_config in models:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Measuring {model_name}")
        logger.info(f"{'='*50}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure latency
        result = measure_model_latency(model_path, model_name, samples=30)
        results[model_name] = result
        
        # Save intermediate results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "real_cost_measurements.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        if not result["success"]:
            logger.warning(f"Skipping {model_name} due to error")
            continue
    
    # Calculate relative costs (normalize by smallest model)
    successful_results = {k: v for k, v in results.items() if v["success"]}
    
    if successful_results:
        # Use median latency for cost calculation
        latencies = {name: result["median_latency"] for name, result in successful_results.items()}
        min_latency = min(latencies.values())
        
        # Normalize costs (smallest model = 1.0)
        normalized_costs = {name: latency / min_latency for name, latency in latencies.items()}
        
        logger.info(f"\n{'='*50}")
        logger.info("FINAL COST RESULTS")
        logger.info(f"{'='*50}")
        
        for name, cost in normalized_costs.items():
            actual_latency = latencies[name]
            logger.info(f"{name:12}: {cost:.2f}x cost ({actual_latency:.3f}s)")
        
        # Update cost results
        for name in successful_results:
            results[name]["normalized_cost"] = normalized_costs[name]
        
        # Save final results
        with open(output_dir / "real_cost_measurements.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate cost configuration
        cost_config = {
            "measurement_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_method": "real_inference_latency",
            "normalization": "smallest_model_equals_1.0",
            "costs": normalized_costs,
            "raw_latencies": latencies
        }
        
        with open(output_dir / "cost_model.yaml", "w") as f:
            yaml.dump(cost_config, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}/")
        logger.info("Cost measurement completed!")
        
        return cost_config
    
    else:
        logger.error("No successful measurements obtained!")
        return None

if __name__ == "__main__":
    main()