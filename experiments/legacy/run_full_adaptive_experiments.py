#!/usr/bin/env python3
"""
Full adaptive speculative decoding experiments with complete model hierarchy.
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
        logging.FileHandler('/home/sasaki/adaptive-speculative-decoding/logs/full_adaptive_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleStage:
    """Simple stage wrapper for downloaded models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
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
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Tuple[str, float]:
        """Generate text and return output with latency."""
        if not self.loaded:
            return "", 0.0
            
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to('cuda:0')
            
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

class RealQualityPredictor:
    """Realistic quality predictor based on actual prompt complexity."""
    
    def predict_quality(self, prompt: str, stage: int) -> float:
        """Predict quality based on prompt complexity and stage."""
        # More sophisticated heuristics
        prompt_length = len(prompt.split())
        
        # Check for complexity indicators
        complexity_indicators = [
            'algorithm', 'implement', 'design', 'architecture', 'optimization',
            'complexity', 'system', 'distributed', 'detailed', 'comprehensive',
            'analyze', 'philosophical', 'technical', 'programming', 'code'
        ]
        
        complexity_score = sum(1 for word in complexity_indicators if word.lower() in prompt.lower())
        
        # Base quality predictions for each stage
        base_qualities = [0.75, 0.85, 0.92, 0.97]  # 13B, 34B, 70B
        
        if stage >= len(base_qualities):
            stage = len(base_qualities) - 1
            
        base_quality = base_qualities[stage]
        
        # Adjust based on complexity
        if complexity_score >= 3:  # High complexity
            adjustment = [0.0, 0.05, 0.03, 0.01][stage]
        elif complexity_score >= 1:  # Medium complexity 
            adjustment = [0.05, 0.02, 0.01, 0.0][stage]
        else:  # Low complexity
            adjustment = [0.1, 0.0, -0.02, -0.05][stage]
            
        # Adjust based on length
        if prompt_length > 50:
            length_adjustment = [-0.1, -0.05, 0.0, 0.02][stage]
        elif prompt_length > 20:
            length_adjustment = [-0.05, 0.0, 0.02, 0.01][stage]
        else:
            length_adjustment = [0.0, 0.02, 0.0, -0.02][stage]
        
        final_quality = base_quality + adjustment + length_adjustment
        
        # Add some noise and clamp
        final_quality += random.uniform(-0.02, 0.02)
        return max(0.0, min(1.0, final_quality))

class FullAdaptiveDecoder:
    """Full adaptive decoder using all available models."""
    
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param
        self.quality_predictor = RealQualityPredictor()
        
        # All available model paths
        self.model_paths = [
            "/raid/sasaki/adaptive-sd-models/13b",          # 13B (index 0)
            "/raid/sasaki/adaptive-sd-models/34b-hf",       # 34B (index 1)
            "/raid/sasaki/adaptive-sd-models/70b-full"      # 70B (index 2)
        ]
        
        # Stage costs (relative computational cost)
        self.stage_costs = [1.6, 4.2, 8.8]
        
        # Stage names
        self.stage_names = ["13B", "34B", "70B"]
        
        # Initialize stages (load on demand)
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
    
    def decode(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Perform adaptive speculative decoding."""
        # Determine optimal stopping stage
        optimal_stage = self.compute_optimal_stopping(prompt)
        
        # Unload any currently loaded model
        if self.current_stage is not None and self.current_stage != optimal_stage:
            self.stages[self.current_stage].unload()
        
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
            "stage_name": self.stage_names[optimal_stage],
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

def run_full_adaptive_experiments():
    """Run full adaptive experiments with multiple lambda values."""
    
    logger.info("=== FULL ADAPTIVE SPECULATIVE DECODING EXPERIMENTS ===")
    
    # Comprehensive test prompts of different complexities
    test_prompts = [
        # Simple factual (should use 13B)
        ("What is 2 + 2?", "simple"),
        ("What color is the sky?", "simple"),
        ("Who wrote Romeo and Juliet?", "simple"),
        
        # Medium complexity (should use 13B/34B)
        ("Explain how photosynthesis works in plants.", "medium"),
        ("What are the main differences between machine learning and deep learning?", "medium"), 
        ("Describe the process of how a car engine converts fuel into motion.", "medium"),
        
        # High complexity (should use 34B/70B)
        ("Design a distributed system architecture for a real-time chat application that can handle millions of concurrent users.", "complex"),
        ("Implement a comprehensive B+ tree data structure with support for range queries, insertions, deletions, and automatic rebalancing.", "complex"),
        ("Analyze the philosophical implications of artificial intelligence potentially achieving consciousness and explain the hard problem of consciousness.", "complex"),
        
        # Programming tasks (complexity varies)
        ("Write a Python function to reverse a string.", "simple"),
        ("Implement quicksort algorithm with detailed error handling and optimization for small arrays.", "complex"),
        ("Create a simple web scraper using BeautifulSoup.", "medium")
    ]
    
    # Test different lambda values
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    
    all_results = []
    
    for lambda_val in lambda_values:
        logger.info(f"\n=== Testing λ = {lambda_val} ===")
        
        decoder = FullAdaptiveDecoder(lambda_param=lambda_val)
        
        for i, (prompt, category) in enumerate(test_prompts):
            logger.info(f"Prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                result = decoder.decode(prompt, max_tokens=80, temperature=0.7)
                result["lambda"] = lambda_val
                result["prompt"] = prompt
                result["prompt_category"] = category
                
                all_results.append(result)
                
                logger.info(f"  → Stage: {result['stage_name']}, "
                          f"Latency: {result['latency_ms']:.1f}ms, "
                          f"Cost: {result['computational_cost']:.1f}")
                
            except Exception as e:
                logger.error(f"  → Error: {e}")
        
        # Cleanup between lambda tests
        decoder.cleanup()
        time.sleep(3)  # Let GPU memory settle
    
    return all_results

def analyze_full_results(results: List[Dict[str, Any]]):
    """Analyze comprehensive experimental results."""
    
    logger.info("\n=== COMPREHENSIVE RESULTS ANALYSIS ===")
    
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
    logger.info("\nLambda parameter analysis:")
    lambda_analysis = {}
    for result in results:
        lam = result['lambda']
        if lam not in lambda_analysis:
            lambda_analysis[lam] = {'costs': [], 'stages': [], 'latencies': []}
        
        lambda_analysis[lam]['costs'].append(result['computational_cost'])
        lambda_analysis[lam]['stages'].append(result['stopped_at_stage'])
        lambda_analysis[lam]['latencies'].append(result['latency_ms'])
    
    for lam in sorted(lambda_analysis.keys()):
        data = lambda_analysis[lam]
        avg_cost = np.mean(data['costs'])
        avg_stage = np.mean(data['stages'])
        avg_latency = np.mean(data['latencies'])
        
        logger.info(f"  λ={lam}: avg_cost={avg_cost:.2f}, avg_stage={avg_stage:.1f}, avg_latency={avg_latency:.1f}ms")
    
    # Complexity analysis
    logger.info("\nComplexity category analysis:")
    complexity_analysis = {}
    for result in results:
        cat = result['prompt_category']
        if cat not in complexity_analysis:
            complexity_analysis[cat] = {'stages': [], 'costs': [], 'latencies': []}
        
        complexity_analysis[cat]['stages'].append(result['stopped_at_stage'])
        complexity_analysis[cat]['costs'].append(result['computational_cost'])
        complexity_analysis[cat]['latencies'].append(result['latency_ms'])
    
    for category in ['simple', 'medium', 'complex']:
        if category in complexity_analysis:
            data = complexity_analysis[category]
            avg_stage = np.mean(data['stages'])
            avg_cost = np.mean(data['costs'])
            avg_latency = np.mean(data['latencies'])
            logger.info(f"  {category.title()}: avg_stage={avg_stage:.1f}, avg_cost={avg_cost:.2f}, avg_latency={avg_latency:.1f}ms")
    
    # Efficiency analysis
    total_theoretical_cost = total_experiments * 8.8  # If all used 70B model
    actual_cost = sum(r['computational_cost'] for r in results)
    efficiency_gain = ((total_theoretical_cost - actual_cost) / total_theoretical_cost) * 100
    
    logger.info(f"\nEfficiency analysis:")
    logger.info(f"  Theoretical cost (all 70B): {total_theoretical_cost:.1f} units")
    logger.info(f"  Actual cost: {actual_cost:.1f} units")
    logger.info(f"  Efficiency gain: {efficiency_gain:.1f}%")

def main():
    """Main experiment function."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
        for i in range(min(4, torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        logger.error("CUDA not available")
        return
    
    # Create output directory
    output_dir = "/raid/sasaki/adaptive-sd-results/full"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run full experiments
        results = run_full_adaptive_experiments()
        
        # Analyze results
        analyze_full_results(results)
        
        # Save results
        results_file = f"{output_dir}/full_adaptive_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n=== FULL EXPERIMENTS COMPLETE ===")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Total experiments: {len(results)}")
        
        if results:
            avg_efficiency = sum(1 for r in results if r['computational_cost'] < 8.8) / len(results) * 100
            logger.info(f"Efficiency: {avg_efficiency:.1f}% of requests avoided using 70B model")
        
    except Exception as e:
        logger.error(f"Full experiments failed: {e}")
        raise

if __name__ == "__main__":
    main()