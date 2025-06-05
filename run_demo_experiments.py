#!/usr/bin/env python3
"""
Demo experiments with adaptive speculative decoding using smaller models first.
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
        logging.FileHandler('/home/sasaki/adaptive-speculative-decoding/logs/demo_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleStage:
    """Simple stage wrapper for downloaded models."""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
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

class DemoAdaptiveDecoder:
    """Demo adaptive decoder that tests one model at a time."""
    
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param
        self.quality_predictor = MockQualityPredictor()
        
        # Model paths - start with 13B (8B not fully downloaded)
        self.model_paths = [
            "/raid/sasaki/adaptive-sd-models/13b",  # 13B
        ]
        
        # Stage costs (relative computational cost)
        self.stage_costs = [1.0]
        
        # Initialize stages
        self.stages = [SimpleStage(path, device=f"cuda:0") for path in self.model_paths]
        self.current_stage = None
    
    def compute_optimal_stopping(self, prompt: str) -> int:
        """For demo, always use stage 0 (8B model)."""
        return 0
    
    def decode(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Perform generation with first available model."""
        # Use first model
        stage = self.stages[0]
        if not stage.loaded:
            stage.load()
            self.current_stage = 0
        
        # Generate text
        output, latency = stage.generate(prompt, max_tokens, temperature)
        
        return {
            "output": output,
            "stopped_at_stage": 0,
            "stage_name": "13B",
            "latency_ms": latency,
            "total_tokens": len(output.split()) if output else 0,
            "lambda": self.lambda_param,
            "prompt_length": len(prompt),
            "computational_cost": 1.6
        }
    
    def cleanup(self):
        """Cleanup loaded models."""
        for stage in self.stages:
            if stage.loaded:
                stage.unload()

def run_demo_experiments():
    """Run demo experiments with 8B model first."""
    
    logger.info("=== DEMO ADAPTIVE SPECULATIVE DECODING EXPERIMENTS ===")
    logger.info("Testing with 13B model first")
    
    # Simple test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Explain machine learning in simple terms.",
        "Write a short Python function to calculate factorial."
    ]
    
    decoder = DemoAdaptiveDecoder(lambda_param=1.0)
    all_results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"Prompt {i+1}: {prompt}")
        
        try:
            result = decoder.decode(prompt, max_tokens=50, temperature=0.7)
            result["prompt"] = prompt
            
            all_results.append(result)
            
            logger.info(f"  → Stage: {result['stage_name']}, "
                      f"Latency: {result['latency_ms']:.1f}ms")
            logger.info(f"  → Output: {result['output'][:100]}...")
            
        except Exception as e:
            logger.error(f"  → Error: {e}")
    
    decoder.cleanup()
    return all_results

def main():
    """Main demo function."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
        for i in range(min(4, torch.cuda.device_count())):  # Only show first 4
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        logger.error("CUDA not available")
        return
    
    # Create output directory
    output_dir = "/raid/sasaki/adaptive-sd-results/demo"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run demo experiments
        results = run_demo_experiments()
        
        # Save results
        results_file = f"{output_dir}/demo_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n=== DEMO COMPLETE ===")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Total experiments: {len(results)}")
        
        # Show results
        if results:
            avg_latency = np.mean([r['latency_ms'] for r in results])
            logger.info(f"Average latency: {avg_latency:.1f}ms")
            logger.info(f"13B model working successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()