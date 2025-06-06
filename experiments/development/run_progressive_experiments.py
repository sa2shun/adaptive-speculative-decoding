#!/usr/bin/env python3
"""
Progressive experiments that adapt to available models.
Start with single model, expand as more models become available.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('.')

class ProgressiveExperimentRunner:
    def __init__(self, base_model_dir: str, dataset_dir: str, results_dir: str):
        self.base_model_dir = Path(base_model_dir)
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.tokenizers = {}
        self.available_stages = []
        
        # Expected model configurations
        self.model_configs = {
            "7b": {"path": "qwen3-7b", "name": "Qwen3-7B", "stage": 0},
            "14b": {"path": "qwen3-14b", "name": "Qwen3-14B", "stage": 1},
            "32b": {"path": "qwen3-32b", "name": "Qwen3-32B", "stage": 2},
            "72b": {"path": "qwen3-72b", "name": "Qwen3-72B", "stage": 3}
        }
    
    def check_available_models(self) -> List[str]:
        """Check which models are ready for use."""
        available = []
        
        for size, config in self.model_configs.items():
            model_path = self.base_model_dir / config["path"]
            tokenizer_path = model_path / "tokenizer.json"
            config_path = model_path / "config.json"
            
            if model_path.exists() and tokenizer_path.exists() and config_path.exists():
                available.append(size)
                print(f"✓ {config['name']} ready at {model_path}")
            else:
                print(f"✗ {config['name']} not ready (missing files)")
        
        return sorted(available, key=lambda x: self.model_configs[x]["stage"])
    
    def load_model(self, size: str) -> bool:
        """Load a specific model size."""
        if size in self.models:
            return True
            
        config = self.model_configs[size]
        model_path = self.base_model_dir / config["path"]
        
        try:
            print(f"Loading {config['name']}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizers[size] = tokenizer
            self.models[size] = model
            self.available_stages.append(config["stage"])
            
            print(f"✓ {config['name']} loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {config['name']}: {e}")
            return False
    
    def unload_model(self, size: str):
        """Unload a model to free memory."""
        if size in self.models:
            del self.models[size]
            del self.tokenizers[size]
            torch.cuda.empty_cache()
            print(f"Unloaded {self.model_configs[size]['name']}")
    
    def generate_text(self, size: str, prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
        """Generate text with a specific model."""
        if size not in self.models:
            return {"error": f"Model {size} not loaded"}
        
        model = self.models[size]
        tokenizer = self.tokenizers[size]
        
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = len(inputs.input_ids[0])
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            # Decode
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_text[len(prompt):].strip()
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
                "inference_time": inference_time,
                "input_tokens": input_length,
                "total_tokens": len(outputs[0]),
                "new_tokens": len(outputs[0]) - input_length,
                "tokens_per_second": (len(outputs[0]) - input_length) / inference_time if inference_time > 0 else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def load_dataset(self, dataset_name: str, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        dataset_path = self.dataset_dir / f"{dataset_name}_test.json"
        
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found at {dataset_path}")
            return []
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Sample if too large
        if len(data) > max_samples:
            import random
            random.seed(42)
            data = random.sample(data, max_samples)
        
        print(f"Loaded {len(data)} samples from {dataset_name}")
        return data
    
    def run_single_model_baseline(self, size: str, dataset_name: str, max_samples: int = 50) -> Dict[str, Any]:
        """Run baseline with single model."""
        print(f"\n=== Baseline: {self.model_configs[size]['name']} on {dataset_name} ===")
        
        # Load dataset
        data = self.load_dataset(dataset_name, max_samples)
        if not data:
            return {"error": "No data loaded"}
        
        # Load model
        if not self.load_model(size):
            return {"error": f"Failed to load model {size}"}
        
        results = []
        total_time = 0
        total_tokens = 0
        
        for i, sample in enumerate(data):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(data)}...")
            
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            result = self.generate_text(size, prompt, max_tokens=100)
            if "error" not in result:
                results.append(result)
                total_time += result["inference_time"]
                total_tokens += result["new_tokens"]
        
        # Calculate metrics
        avg_time = total_time / len(results) if results else 0
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        baseline_result = {
            "model_size": size,
            "model_name": self.model_configs[size]["name"],
            "dataset": dataset_name,
            "num_samples": len(results),
            "avg_inference_time": avg_time,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "results": results[:10]  # Save first 10 for inspection
        }
        
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Avg time per sample: {avg_time:.3f}s")
        
        return baseline_result
    
    def simulate_adaptive_decoding(self, dataset_name: str, max_samples: int = 50) -> Dict[str, Any]:
        """Simulate adaptive decoding with available models."""
        print(f"\n=== Adaptive Decoding Simulation on {dataset_name} ===")
        
        available_sizes = self.check_available_models()
        if not available_sizes:
            return {"error": "No models available"}
        
        # Load dataset
        data = self.load_dataset(dataset_name, max_samples)
        if not data:
            return {"error": "No data loaded"}
        
        # For simulation, assign difficulty scores randomly but consistently
        np.random.seed(42)
        difficulty_scores = np.random.beta(2, 5, len(data))  # Skewed toward easier tasks
        
        results = []
        stage_counts = {size: 0 for size in available_sizes}
        total_cost = 0
        total_time = 0
        
        for i, (sample, difficulty) in enumerate(zip(data, difficulty_scores)):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(data)}...")
            
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            # Determine which model to use based on difficulty
            # Lower difficulty -> use smaller model
            if difficulty < 0.3 and "7b" in available_sizes:
                chosen_size = "7b"
                cost = 1.0
            elif difficulty < 0.6 and "14b" in available_sizes:
                chosen_size = "14b" 
                cost = 2.0
            elif difficulty < 0.8 and "32b" in available_sizes:
                chosen_size = "32b"
                cost = 4.5
            elif "72b" in available_sizes:
                chosen_size = "72b"
                cost = 10.0
            else:
                # Fall back to largest available
                chosen_size = available_sizes[-1]
                cost = {"7b": 1.0, "14b": 2.0, "32b": 4.5, "72b": 10.0}[chosen_size]
            
            # Load model if needed
            if not self.load_model(chosen_size):
                continue
            
            # Generate
            result = self.generate_text(chosen_size, prompt, max_tokens=100)
            if "error" not in result:
                result["chosen_model"] = chosen_size
                result["difficulty_score"] = difficulty
                result["computational_cost"] = cost
                results.append(result)
                
                stage_counts[chosen_size] += 1
                total_cost += cost
                total_time += result["inference_time"]
        
        # Calculate metrics
        avg_cost = total_cost / len(results) if results else 0
        avg_time = total_time / len(results) if results else 0
        
        # Calculate speedup vs always using largest model
        largest_size = available_sizes[-1]
        largest_cost = {"7b": 1.0, "14b": 2.0, "32b": 4.5, "72b": 10.0}[largest_size]
        speedup = largest_cost / avg_cost if avg_cost > 0 else 1.0
        
        adaptive_result = {
            "dataset": dataset_name,
            "num_samples": len(results),
            "available_models": available_sizes,
            "stage_distribution": stage_counts,
            "avg_computational_cost": avg_cost,
            "avg_inference_time": avg_time,
            "total_time": total_time,
            "speedup_vs_largest": speedup,
            "results": results[:10]  # Save first 10 for inspection
        }
        
        print(f"Average cost: {avg_cost:.2f}")
        print(f"Speedup vs {largest_size}: {speedup:.2f}x")
        print("Stage distribution:", {k: f"{v}/{len(results)}" for k, v in stage_counts.items() if v > 0})
        
        return adaptive_result
    
    def run_progressive_experiments(self):
        """Run experiments progressively as models become available."""
        print("=== PROGRESSIVE ADAPTIVE SPECULATIVE DECODING EXPERIMENTS ===\n")
        
        # Check initial availability
        available_sizes = self.check_available_models()
        
        if not available_sizes:
            print("No models available yet. Waiting for downloads...")
            return
        
        all_results = {"timestamp": time.time(), "experiments": []}
        
        # Dataset names to test
        datasets = ["mmlu", "humaneval", "simple_qa"]
        
        # 1. Single model baselines
        print("\n1. SINGLE MODEL BASELINES")
        print("=" * 50)
        
        baselines = {}
        for size in available_sizes:
            for dataset in datasets:
                baseline = self.run_single_model_baseline(size, dataset, max_samples=30)
                if "error" not in baseline:
                    baselines[f"{size}_{dataset}"] = baseline
                
                # Unload model to save memory for next test
                self.unload_model(size)
        
        all_results["baselines"] = baselines
        
        # 2. Adaptive decoding simulation
        print("\n2. ADAPTIVE DECODING SIMULATION")
        print("=" * 50)
        
        adaptive_results = {}
        for dataset in datasets:
            adaptive = self.simulate_adaptive_decoding(dataset, max_samples=30)
            if "error" not in adaptive:
                adaptive_results[dataset] = adaptive
        
        all_results["adaptive_results"] = adaptive_results
        
        # 3. Save results
        timestamp = int(time.time())
        results_file = self.results_dir / f"progressive_experiments_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n=== RESULTS SAVED ===")
        print(f"Results saved to: {results_file}")
        
        # 4. Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print experiment summary."""
        print(f"\n=== EXPERIMENT SUMMARY ===")
        
        if "baselines" in results:
            print("\nBaseline Performance:")
            for key, baseline in results["baselines"].items():
                model = baseline.get("model_name", "Unknown")
                dataset = baseline.get("dataset", "Unknown")
                throughput = baseline.get("throughput", 0)
                print(f"  {model} on {dataset}: {throughput:.2f} tokens/sec")
        
        if "adaptive_results" in results:
            print("\nAdaptive Decoding Performance:")
            for dataset, adaptive in results["adaptive_results"].items():
                avg_cost = adaptive.get("avg_computational_cost", 0)
                speedup = adaptive.get("speedup_vs_largest", 1)
                print(f"  {dataset}: Cost={avg_cost:.2f}, Speedup={speedup:.2f}x")
                
                # Stage distribution
                stage_dist = adaptive.get("stage_distribution", {})
                total = sum(stage_dist.values())
                if total > 0:
                    dist_str = ", ".join([f"{k}:{v}" for k, v in stage_dist.items() if v > 0])
                    print(f"    Distribution: {dist_str}")

def main():
    """Main experimental pipeline."""
    runner = ProgressiveExperimentRunner(
        base_model_dir="/raid/sasaki/adaptive-speculative-decoding/models",
        dataset_dir="/raid/sasaki/adaptive-speculative-decoding/datasets",
        results_dir="/raid/sasaki/adaptive-speculative-decoding/results"
    )
    
    results = runner.run_progressive_experiments()
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print("Results include baseline and adaptive performance.")
    print("As more models download, rerun this script for expanded experiments.")

if __name__ == "__main__":
    main()