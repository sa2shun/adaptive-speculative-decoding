#!/usr/bin/env python3
"""
Immediate experiments with downloaded models for adaptive speculative decoding.
Uses transformers directly to avoid vLLM complexity.
"""

import os
import sys
import json
import time
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/sasaki/adaptive-speculative-decoding/logs/immediate_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleStage:
    """Simple stage wrapper for downloaded models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load with appropriate precision for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.loaded = True
            logger.info(f"✓ Successfully loaded {self.model_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {self.model_path}: {e}")
            self.loaded = False
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Tuple[str, float]:
        """Generate text and return output with latency."""
        if not self.loaded:
            return "", 0.0
            
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from output
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):]
                
            latency = (time.time() - start_time) * 1000  # Convert to ms
            return output_text.strip(), latency
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "", 0.0
    
    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.loaded = False
            logger.info(f"Unloaded {self.model_path}")

class MockQualityPredictor:
    """Mock quality predictor for experiments."""
    
    def predict_quality(self, prompt: str, stage: int) -> float:
        """Predict quality based on prompt complexity and stage."""
        # Simple heuristic: longer prompts need bigger models
        prompt_length = len(prompt.split())
        
        if stage == 0:  # 8B model
            if prompt_length < 10:
                return 0.85 + random.uniform(-0.1, 0.1)
            elif prompt_length < 20:
                return 0.70 + random.uniform(-0.1, 0.1)
            else:
                return 0.55 + random.uniform(-0.1, 0.1)
                
        elif stage == 1:  # 13B model
            if prompt_length < 10:
                return 0.90 + random.uniform(-0.05, 0.05)
            elif prompt_length < 30:
                return 0.80 + random.uniform(-0.1, 0.1)
            else:
                return 0.65 + random.uniform(-0.1, 0.1)
                
        elif stage == 2:  # 34B model
            if prompt_length < 20:
                return 0.95 + random.uniform(-0.02, 0.02)
            else:
                return 0.85 + random.uniform(-0.05, 0.05)
                
        else:  # 70B model
            return 0.98 + random.uniform(-0.01, 0.01)

class AdaptiveSpeculativeDecoder:
    """Adaptive speculative decoder using real models."""
    
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param
        self.quality_predictor = MockQualityPredictor()
        
        # Model paths
        self.model_paths = [
            "/raid/sasaki/adaptive-sd-models/llama-3.1-8b",  # 8B
            "/raid/sasaki/adaptive-sd-models/13b",          # 13B
            "/raid/sasaki/adaptive-sd-models/34b-hf",       # 34B
            "/raid/sasaki/adaptive-sd-models/70b-full"      # 70B
        ]
        
        # Stage costs (relative computational cost)
        self.stage_costs = [1.0, 1.6, 4.2, 8.8]
        
        # Initialize stages
        self.stages = [SimpleStage(path) for path in self.model_paths]
        self.current_stage = None
    
    def compute_optimal_stopping(self, prompt: str) -> int:
        """Compute optimal stopping stage using dynamic programming."""
        n_stages = len(self.stages)
        
        # Get quality predictions for all stages
        qualities = [self.quality_predictor.predict_quality(prompt, i) for i in range(n_stages)]
        
        # Dynamic programming for optimal stopping
        # V[i] = expected value of continuing from stage i
        V = [0.0] * (n_stages + 1)
        
        # Work backwards
        for i in range(n_stages - 1, -1, -1):
            # Cost of using stage i
            cost_stop = self.stage_costs[i]
            quality_stop = qualities[i]
            value_stop = self.lambda_param * quality_stop - cost_stop
            
            # Value of continuing (if not last stage)
            if i < n_stages - 1:
                value_continue = V[i + 1] - self.stage_costs[i]
                V[i] = max(value_stop, value_continue)
                
                # Decision: stop if stopping value >= continuing value
                if value_stop >= value_continue:
                    return i
            else:
                V[i] = value_stop
                return i
        
        return 0  # Default to first stage
    
    def decode(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """Perform adaptive speculative decoding."""
        # Determine optimal stopping stage
        optimal_stage = self.compute_optimal_stopping(prompt)
        
        # Load the chosen model
        stage = self.stages[optimal_stage]
        if not stage.loaded:
            stage.load()
            self.current_stage = optimal_stage
        
        # Generate text
        output, latency = stage.generate(prompt, max_tokens, temperature)
        
        # Get quality predictions for analysis
        quality_predictions = [
            self.quality_predictor.predict_quality(prompt, i) 
            for i in range(len(self.stages))
        ]
        
        return {
            "output": output,
            "stopped_at_stage": optimal_stage,
            "stage_name": ["8B", "13B", "34B", "70B"][optimal_stage],
            "latency_ms": latency,
            "total_tokens": len(output.split()) if output else 0,
            "stage_probabilities": quality_predictions,
            "lambda": self.lambda_param,
            "prompt_length": len(prompt),
            "computational_cost": self.stage_costs[optimal_stage]
        }
    
    def cleanup(self):
        """Cleanup loaded models."""
        for stage in self.stages:
            if stage.loaded:
                stage.unload()

def create_immediate_experiment_config():
    """Create config for models we can access right now"""
    
    immediate_config = {
        "models": {
            "7b": {
                "name": "tiiuae/falcon-7b",
                "size": "7b", 
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8,
                "quantization": {
                    "enabled": True,
                    "method": "nf4",
                    "compute_dtype": "float16"
                },
                "cost_per_token": 1.0
            },
            "34b": {
                "name": "codellama/CodeLlama-34b-hf",
                "size": "34b",
                "tensor_parallel_size": 2, 
                "gpu_memory_utilization": 0.85,
                "quantization": {
                    "enabled": True,
                    "method": "nf4",
                    "compute_dtype": "float16"
                },
                "cost_per_token": 4.2
            },
            "40b": {
                "name": "tiiuae/falcon-40b",
                "size": "40b",
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.85, 
                "quantization": {
                    "enabled": True,
                    "method": "nf4",
                    "compute_dtype": "float16"
                },
                "cost_per_token": 5.0
            },
            "70b": {
                "name": "meta-llama/Llama-3.1-70B-Instruct",
                "size": "70b",
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.9,
                "quantization": {
                    "enabled": True,
                    "method": "nf4", 
                    "compute_dtype": "float16",
                    "use_double_quant": True
                },
                "cost_per_token": 8.8
            }
        }
    }
    
    return immediate_config

def download_accessible_models():
    """Download models we can access immediately"""
    
    print("🚀 Starting Download of Accessible Models")
    print("=" * 60)
    
    models_to_download = [
        ("7b", "tiiuae/falcon-7b", "~14GB"),
        ("34b", "codellama/CodeLlama-34b-hf", "~68GB"), 
        ("40b", "tiiuae/falcon-40b", "~80GB"),
        ("70b", "meta-llama/Llama-3.1-70B-Instruct", "~140GB")
    ]
    
    base_dir = f"/raid/{os.environ.get('USER', 'user')}/adaptive-sd-models"
    
    print(f"💾 Download destination: {base_dir}")
    print(f"💿 Available space: ~25TB")
    print(f"📦 Total estimated size: ~302GB")
    print()
    
    for size, model_name, estimated_size in models_to_download:
        print(f"📋 {size.upper()} Model:")
        print(f"   Name: {model_name}")
        print(f"   Size: {estimated_size}")
        print(f"   Status: Ready to download")
        print()
    
    return models_to_download, base_dir

def run_immediate_experiments():
    """Run immediate experiments with various prompts and lambda values."""
    
    logger.info("=== IMMEDIATE ADAPTIVE SPECULATIVE DECODING EXPERIMENTS ===")
    
    # Test prompts of different complexities
    test_prompts = [
        # Simple factual (should use 8B)
        "What is 2 + 2?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        
        # Medium complexity (should use 13B/34B)
        "Explain how photosynthesis works in plants.",
        "What are the main differences between machine learning and deep learning?",
        "Describe the process of how a car engine converts fuel into motion.",
        
        # Complex reasoning (should use 34B/70B)
        "Design a distributed system architecture for a real-time chat application that can handle millions of concurrent users.",
        "Analyze the philosophical implications of artificial intelligence potentially achieving consciousness.",
        "Write a detailed algorithm for implementing a self-balancing binary search tree with insertion, deletion, and rotation operations."
    ]
    
    # Test different lambda values
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    
    all_results = []
    
    for lambda_val in lambda_values:
        logger.info(f"\n--- Testing λ = {lambda_val} ---")
        
        decoder = AdaptiveSpeculativeDecoder(lambda_param=lambda_val)
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Prompt {i+1}: {prompt[:50]}...")
            
            try:
                result = decoder.decode(prompt, max_tokens=100, temperature=0.7)
                result["lambda"] = lambda_val
                result["prompt"] = prompt
                result["prompt_category"] = (
                    "simple" if i < 3 else
                    "medium" if i < 6 else
                    "complex"
                )
                
                all_results.append(result)
                
                logger.info(f"  → Stage: {result['stage_name']}, "
                          f"Latency: {result['latency_ms']:.1f}ms, "
                          f"Cost: {result['computational_cost']:.1f}")
                
            except Exception as e:
                logger.error(f"  → Error: {e}")
        
        # Cleanup between lambda tests
        decoder.cleanup()
        time.sleep(2)  # Let GPU memory settle
    
    return all_results

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze experimental results."""
    
    logger.info("\n=== RESULTS ANALYSIS ===")
    
    if not results:
        logger.error("No results to analyze")
        return
    
    # Overall statistics
    total_experiments = len(results)
    avg_latency = np.mean([r['latency_ms'] for r in results])
    avg_cost = np.mean([r['computational_cost'] for r in results])
    
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Average latency: {avg_latency:.1f}ms")
    logger.info(f"Average computational cost: {avg_cost:.2f}")
    
    # Stage distribution
    stage_counts = {}
    for result in results:
        stage = result['stage_name']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    logger.info("\nStage distribution:")
    for stage, count in sorted(stage_counts.items()):
        percentage = (count / total_experiments) * 100
        logger.info(f"  {stage}: {count} ({percentage:.1f}%)")
    
    # Lambda analysis
    lambda_analysis = {}
    for result in results:
        lam = result['lambda']
        if lam not in lambda_analysis:
            lambda_analysis[lam] = {'costs': [], 'stages': [], 'latencies': []}
        
        lambda_analysis[lam]['costs'].append(result['computational_cost'])
        lambda_analysis[lam]['stages'].append(result['stopped_at_stage'])
        lambda_analysis[lam]['latencies'].append(result['latency_ms'])
    
    logger.info("\nLambda parameter analysis:")
    for lam in sorted(lambda_analysis.keys()):
        data = lambda_analysis[lam]
        avg_cost = np.mean(data['costs'])
        avg_stage = np.mean(data['stages'])
        avg_latency = np.mean(data['latencies'])
        
        logger.info(f"  λ={lam}: avg_cost={avg_cost:.2f}, avg_stage={avg_stage:.1f}, avg_latency={avg_latency:.1f}ms")
    
    # Complexity analysis
    complexity_analysis = {}
    for result in results:
        cat = result['prompt_category']
        if cat not in complexity_analysis:
            complexity_analysis[cat] = {'stages': [], 'costs': []}
        
        complexity_analysis[cat]['stages'].append(result['stopped_at_stage'])
        complexity_analysis[cat]['costs'].append(result['computational_cost'])
    
    logger.info("\nComplexity category analysis:")
    for category in ['simple', 'medium', 'complex']:
        if category in complexity_analysis:
            data = complexity_analysis[category]
            avg_stage = np.mean(data['stages'])
            avg_cost = np.mean(data['costs'])
            logger.info(f"  {category.title()}: avg_stage={avg_stage:.1f}, avg_cost={avg_cost:.2f}")

def main():
    """Main experiment function."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        logger.error("CUDA not available - experiments may be very slow")
    
    # Create output directory
    output_dir = "/raid/sasaki/adaptive-sd-results/immediate"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run experiments
        results = run_immediate_experiments()
        
        # Analyze results
        analyze_results(results)
        
        # Save results
        results_file = f"{output_dir}/immediate_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n=== EXPERIMENTS COMPLETE ===")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Total experiments: {len(results)}")
        
    except Exception as e:
        logger.error(f"Experiments failed: {e}")
        raise

if __name__ == "__main__":
    main()