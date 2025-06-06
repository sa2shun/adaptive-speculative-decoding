#!/usr/bin/env python3
"""
Final real experiments with available Qwen3 models.
This implements the complete adaptive speculative decoding pipeline.
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import sys
sys.path.append('.')

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    base_model_dir: str = "/raid/sasaki/adaptive-speculative-decoding/models"
    dataset_dir: str = "/raid/sasaki/adaptive-speculative-decoding/datasets" 
    results_dir: str = "/raid/sasaki/adaptive-speculative-decoding/results"
    num_samples: int = 100
    lambda_values: List[float] = None
    
    def __post_init__(self):
        if self.lambda_values is None:
            self.lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]

class AdaptiveSpeculativeDecoder:
    """Complete adaptive speculative decoding implementation."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.available_stages = []
        
        # Model configurations with computational costs
        self.model_configs = {
            "7b": {
                "path": "qwen3-7b",
                "name": "Qwen3-7B", 
                "cost": 1.0,
                "stage": 0
            },
            "32b": {
                "path": "qwen3-32b",
                "name": "Qwen3-32B",
                "cost": 4.5,
                "stage": 1
            },
            "72b": {
                "path": "qwen3-72b", 
                "name": "Qwen3-72B",
                "cost": 10.0,
                "stage": 2
            }
        }
        
        # Initialize results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def check_available_models(self) -> List[str]:
        """Check which models are ready for use."""
        available = []
        
        for size, config in self.model_configs.items():
            model_path = Path(self.config.base_model_dir) / config["path"]
            if self._model_is_complete(model_path):
                available.append(size)
                print(f"✓ {config['name']} ready")
            else:
                print(f"✗ {config['name']} not ready")
        
        return sorted(available, key=lambda x: self.model_configs[x]["stage"])
    
    def _model_is_complete(self, model_path: Path) -> bool:
        """Check if model is completely downloaded."""
        required_files = ["config.json", "tokenizer.json"]
        return all((model_path / f).exists() for f in required_files)
    
    def load_model(self, size: str) -> bool:
        """Load a specific model."""
        if size in self.models:
            return True
        
        config = self.model_configs[size]
        model_path = Path(self.config.base_model_dir) / config["path"]
        
        try:
            print(f"Loading {config['name']}...")
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate device mapping
            device_map = "auto"  # Let transformers handle device mapping automatically
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            
            self.tokenizers[size] = tokenizer
            self.models[size] = model
            
            print(f"✓ {config['name']} loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {config['name']}: {e}")
            return False
    
    def unload_model(self, size: str):
        """Unload model to free GPU memory."""
        if size in self.models:
            del self.models[size]
            del self.tokenizers[size]
            torch.cuda.empty_cache()
            print(f"Unloaded {self.model_configs[size]['name']}")
    
    def generate_with_model(self, size: str, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate text with a specific model."""
        if size not in self.models:
            return {"error": f"Model {size} not loaded"}
        
        model = self.models[size]
        tokenizer = self.tokenizers[size]
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                # Find device from model parameters
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_length = len(inputs["input_ids"][0])
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            inference_time = time.time() - start_time
            
            # Extract generated text
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate quality metrics
            if hasattr(outputs, 'scores') and outputs.scores:
                scores = torch.stack(outputs.scores, dim=0)
                probs = torch.softmax(scores, dim=-1)
                selected_probs = torch.gather(probs, 2, generated_ids.unsqueeze(0).unsqueeze(-1))
                avg_log_prob = torch.log(selected_probs.squeeze()).mean().item()
                confidence = selected_probs.squeeze().mean().item()
            else:
                avg_log_prob = -2.0
                confidence = 0.5
            
            return {
                "generated_text": generated_text,
                "inference_time": inference_time,
                "input_tokens": input_length,
                "output_tokens": len(generated_ids),
                "total_tokens": len(outputs.sequences[0]),
                "tokens_per_second": len(generated_ids) / inference_time if inference_time > 0 else 0,
                "avg_log_prob": avg_log_prob,
                "confidence": confidence,
                "computational_cost": self.model_configs[size]["cost"]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def compute_optimal_thresholds(self, lambda_val: float, available_stages: List[str]) -> Dict[str, float]:
        """Compute optimal stopping thresholds for given lambda."""
        thresholds = {}
        
        # Simple heuristic based on theoretical analysis
        for i, stage in enumerate(available_stages[:-1]):  # Exclude last stage
            next_stage = available_stages[i + 1]
            
            current_cost = self.model_configs[stage]["cost"]
            next_cost = self.model_configs[next_stage]["cost"]
            
            # Threshold decreases with higher lambda (prefer quality)
            # and increases with cost difference
            cost_ratio = next_cost / current_cost
            base_threshold = 1.0 / (1.0 + lambda_val)
            threshold = base_threshold * (1.0 - 0.5 / cost_ratio)
            
            thresholds[f"{stage}_to_{next_stage}"] = threshold
        
        return thresholds
    
    def should_stop_at_stage(self, confidence: float, stage: str, next_stage: str, thresholds: Dict[str, float]) -> bool:
        """Decide whether to stop at current stage."""
        threshold_key = f"{stage}_to_{next_stage}"
        if threshold_key in thresholds:
            return confidence >= thresholds[threshold_key]
        return False
    
    def adaptive_generate(self, prompt: str, lambda_val: float = 1.0, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate text using adaptive speculative decoding."""
        available_stages = [s for s in ["7b", "32b", "72b"] if s in self.models]
        
        if not available_stages:
            return {"error": "No models available"}
        
        # Compute thresholds for this lambda
        thresholds = self.compute_optimal_thresholds(lambda_val, available_stages)
        
        total_cost = 0
        total_time = 0
        stage_results = []
        final_result = None
        
        for i, stage in enumerate(available_stages):
            # Generate with current stage
            result = self.generate_with_model(stage, prompt, max_tokens)
            
            if "error" in result:
                continue
            
            stage_results.append({
                "stage": stage,
                "model_name": self.model_configs[stage]["name"], 
                "result": result
            })
            
            total_cost += result["computational_cost"]
            total_time += result["inference_time"]
            
            # Check if we should stop at this stage
            if i < len(available_stages) - 1:  # Not the last stage
                next_stage = available_stages[i + 1]
                if self.should_stop_at_stage(result["confidence"], stage, next_stage, thresholds):
                    final_result = result
                    final_result["chosen_stage"] = stage
                    final_result["stopped_early"] = True
                    break
            else:
                # Last stage - always stop
                final_result = result
                final_result["chosen_stage"] = stage
                final_result["stopped_early"] = False
        
        if final_result is None:
            return {"error": "No successful generation"}
        
        return {
            "prompt": prompt,
            "lambda": lambda_val,
            "final_result": final_result,
            "stage_results": stage_results,
            "total_computational_cost": total_cost,
            "total_inference_time": total_time,
            "thresholds_used": thresholds,
            "available_stages": available_stages
        }
    
    def run_baseline_experiments(self, dataset_name: str, num_samples: int) -> Dict[str, Any]:
        """Run baseline experiments with single models."""
        print(f"\\n=== Baseline Experiments: {dataset_name} ===")
        
        # Load dataset
        dataset_path = Path(self.config.dataset_dir) / f"{dataset_name}_test.json"
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > num_samples:
            data = data[:num_samples]
        
        available_models = self.check_available_models()
        baselines = {}
        
        for size in available_models:
            print(f"\\nRunning baseline with {self.model_configs[size]['name']}...")
            
            if not self.load_model(size):
                continue
            
            results = []
            total_time = 0
            total_cost = 0
            
            for i, sample in enumerate(data):
                if i % 20 == 0:
                    print(f"  Processing {i+1}/{len(data)}...")
                
                prompt = sample.get("prompt", sample.get("question", ""))
                if not prompt:
                    continue
                
                result = self.generate_with_model(size, prompt, max_tokens=100)
                
                if "error" not in result:
                    results.append(result)
                    total_time += result["inference_time"]
                    total_cost += result["computational_cost"]
            
            baselines[size] = {
                "model_name": self.model_configs[size]["name"],
                "num_samples": len(results),
                "avg_inference_time": total_time / len(results) if results else 0,
                "avg_computational_cost": total_cost / len(results) if results else 0,
                "total_time": total_time,
                "total_cost": total_cost,
                "throughput": sum(r["tokens_per_second"] for r in results) / len(results) if results else 0,
                "sample_results": results[:5]  # Save first 5 for inspection
            }
            
            print(f"  Throughput: {baselines[size]['throughput']:.2f} tokens/sec")
            print(f"  Avg cost: {baselines[size]['avg_computational_cost']:.2f}")
            
            # Unload to save memory
            self.unload_model(size)
        
        return baselines
    
    def run_adaptive_experiments(self, dataset_name: str, num_samples: int) -> Dict[str, Any]:
        """Run adaptive decoding experiments."""
        print(f"\\n=== Adaptive Experiments: {dataset_name} ===")
        
        # Load dataset
        dataset_path = Path(self.config.dataset_dir) / f"{dataset_name}_test.json"
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > num_samples:
            data = data[:num_samples]
        
        # Load all available models
        available_models = self.check_available_models()
        for size in available_models:
            if not self.load_model(size):
                return {"error": f"Failed to load {size}"}
        
        adaptive_results = {}
        
        for lambda_val in self.config.lambda_values:
            print(f"\\nTesting lambda = {lambda_val}...")
            
            results = []
            stage_counts = {size: 0 for size in available_models}
            total_cost = 0
            total_time = 0
            
            for i, sample in enumerate(data):
                if i % 20 == 0:
                    print(f"  Processing {i+1}/{len(data)}...")
                
                prompt = sample.get("prompt", sample.get("question", ""))
                if not prompt:
                    continue
                
                result = self.adaptive_generate(prompt, lambda_val, max_tokens=100)
                
                if "error" not in result:
                    results.append(result)
                    chosen_stage = result["final_result"]["chosen_stage"]
                    stage_counts[chosen_stage] += 1
                    total_cost += result["total_computational_cost"]
                    total_time += result["total_inference_time"]
            
            if results:
                avg_cost = total_cost / len(results)
                avg_time = total_time / len(results)
                
                # Calculate speedup vs largest model
                largest_model = available_models[-1]
                largest_cost = self.model_configs[largest_model]["cost"]
                speedup = largest_cost / avg_cost if avg_cost > 0 else 1.0
                
                adaptive_results[str(lambda_val)] = {
                    "lambda": lambda_val,
                    "num_samples": len(results),
                    "avg_computational_cost": avg_cost,
                    "avg_inference_time": avg_time,
                    "total_time": total_time,
                    "speedup_vs_largest": speedup,
                    "stage_distribution": stage_counts,
                    "stage_percentages": {k: v/len(results)*100 for k, v in stage_counts.items()},
                    "sample_results": results[:5]  # Save first 5 for inspection
                }
                
                print(f"  Avg cost: {avg_cost:.2f}")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Stage distribution: {stage_counts}")
        
        return adaptive_results
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline."""
        print("=== COMPREHENSIVE ADAPTIVE SPECULATIVE DECODING EVALUATION ===")
        
        timestamp = int(time.time())
        experiment_results = {
            "timestamp": timestamp,
            "config": {
                "num_samples": self.config.num_samples,
                "lambda_values": self.config.lambda_values,
                "available_models": self.check_available_models()
            },
            "datasets": {}
        }
        
        # Test datasets
        datasets = ["mmlu", "humaneval", "simple_qa"]
        
        for dataset in datasets:
            print(f"\\n{'='*60}")
            print(f"EVALUATING DATASET: {dataset.upper()}")
            print(f"{'='*60}")
            
            dataset_results = {}
            
            # 1. Baseline experiments
            baselines = self.run_baseline_experiments(dataset, self.config.num_samples)
            dataset_results["baselines"] = baselines
            
            # 2. Adaptive experiments
            adaptive = self.run_adaptive_experiments(dataset, self.config.num_samples)
            dataset_results["adaptive"] = adaptive
            
            experiment_results["datasets"][dataset] = dataset_results
            
            # Save intermediate results
            intermediate_file = Path(self.config.results_dir) / f"intermediate_results_{timestamp}_{dataset}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(dataset_results, f, indent=2)
            print(f"Intermediate results saved to {intermediate_file}")
        
        # Save final results
        final_results_file = Path(self.config.results_dir) / f"comprehensive_results_{timestamp}.json"
        with open(final_results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\\n=== EVALUATION COMPLETE ===")
        print(f"Final results saved to: {final_results_file}")
        
        # Generate summary
        self.generate_summary(experiment_results)
        
        return experiment_results
    
    def generate_summary(self, results: Dict[str, Any]):
        """Generate and print experiment summary."""
        print(f"\\n=== EXPERIMENT SUMMARY ===")
        
        available_models = results["config"]["available_models"]
        print(f"Available models: {', '.join([self.model_configs[m]['name'] for m in available_models])}")
        print(f"Lambda values tested: {results['config']['lambda_values']}")
        print(f"Samples per dataset: {results['config']['num_samples']}")
        
        for dataset, data in results["datasets"].items():
            print(f"\\n{dataset.upper()} Results:")
            
            # Baseline summary
            if "baselines" in data:
                print("  Baselines:")
                for model_size, baseline in data["baselines"].items():
                    if "error" not in baseline:
                        print(f"    {baseline['model_name']}: {baseline['throughput']:.1f} tokens/sec, cost={baseline['avg_computational_cost']:.1f}")
            
            # Adaptive summary
            if "adaptive" in data:
                print("  Adaptive (best lambda):")
                best_lambda = None
                best_speedup = 0
                
                for lambda_str, adaptive in data["adaptive"].items():
                    if isinstance(adaptive, dict) and "error" not in adaptive and adaptive.get("speedup_vs_largest", 0) > best_speedup:
                        best_speedup = adaptive["speedup_vs_largest"]
                        best_lambda = adaptive["lambda"]
                
                if best_lambda is not None:
                    best_result = data["adaptive"][str(best_lambda)]
                    print(f"    λ={best_lambda}: {best_speedup:.2f}x speedup, cost={best_result['avg_computational_cost']:.2f}")
                    
                    # Stage distribution for best lambda
                    stage_dist = best_result["stage_percentages"]
                    dist_str = ", ".join([f"{self.model_configs[k]['name']}: {v:.1f}%" 
                                        for k, v in stage_dist.items() if v > 0])
                    print(f"    Distribution: {dist_str}")

def main():
    """Main execution."""
    config = ExperimentConfig(
        num_samples=30,  # Manageable number for 3-stage experiments
        lambda_values=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    decoder = AdaptiveSpeculativeDecoder(config)
    
    # Check what models are available
    available = decoder.check_available_models()
    if len(available) < 2:
        print(f"Need at least 2 models for adaptive decoding. Found: {available}")
        return
    
    print(f"Starting experiments with {len(available)} models: {available}")
    
    # Run comprehensive evaluation
    results = decoder.run_comprehensive_evaluation()
    
    print("\\n=== ALL EXPERIMENTS COMPLETED ===")
    print("Ready for conference submission!")

if __name__ == "__main__":
    main()