#!/usr/bin/env python3
"""
Run experiments with 7B model only, providing baseline performance.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('.')

class SevenBOnlyExperiments:
    """Experiments with 7B model only."""
    
    def __init__(self):
        self.model_path = "/raid/sasaki/adaptive-speculative-decoding/models/qwen3-7b"
        self.dataset_dir = "/raid/sasaki/adaptive-speculative-decoding/datasets"
        self.results_dir = "/raid/sasaki/adaptive-speculative-decoding/results"
        
        self.model = None
        self.tokenizer = None
        
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load 7B model."""
        print("Loading Qwen3-7B model...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        device = next(self.model.parameters()).device
        
        print(f"✓ Model loaded in {load_time:.2f}s on {device}")
        return True
    
    def generate_text(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate text with the model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = len(inputs["input_ids"][0])
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            inference_time = time.time() - start_time
            
            # Decode
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Quality metrics
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
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "input_tokens": input_length,
                "output_tokens": len(generated_ids),
                "tokens_per_second": len(generated_ids) / inference_time if inference_time > 0 else 0,
                "avg_log_prob": avg_log_prob,
                "confidence": confidence,
                "computational_cost": 1.0  # 7B model cost
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_dataset(self, dataset_name: str, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on a specific dataset."""
        print(f"\\n=== Evaluating {dataset_name.upper()} ===")
        
        # Load dataset
        dataset_path = Path(self.dataset_dir) / f"{dataset_name}_test.json"
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > num_samples:
            data = data[:num_samples]
        
        results = []
        total_time = 0
        total_tokens = 0
        
        for i, sample in enumerate(data):
            if i % 20 == 0:
                print(f"Processing {i+1}/{len(data)}...")
            
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            result = self.generate_text(prompt, max_tokens=100)
            
            if "error" not in result:
                results.append(result)
                total_time += result["inference_time"]
                total_tokens += result["output_tokens"]
        
        if not results:
            return {"error": "No successful generations"}
        
        # Calculate metrics
        avg_time = total_time / len(results)
        avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_log_prob = sum(r["avg_log_prob"] for r in results) / len(results)
        
        return {
            "dataset": dataset_name,
            "model": "Qwen3-7B",
            "num_samples": len(results),
            "avg_inference_time": avg_time,
            "avg_throughput": avg_throughput,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "avg_confidence": avg_confidence,
            "avg_log_prob": avg_log_prob,
            "computational_cost": 1.0,
            "sample_results": results[:10]  # Save first 10
        }
    
    def run_quality_analysis(self, dataset_name: str, num_samples: int = 50) -> Dict[str, Any]:
        """Analyze quality distribution for threshold setting."""
        print(f"\\n=== Quality Analysis: {dataset_name.upper()} ===")
        
        # Load dataset
        dataset_path = Path(self.dataset_dir) / f"{dataset_name}_test.json"
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > num_samples:
            data = data[:num_samples]
        
        confidences = []
        log_probs = []
        
        for i, sample in enumerate(data):
            if i % 10 == 0:
                print(f"Analyzing {i+1}/{len(data)}...")
            
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            result = self.generate_text(prompt, max_tokens=50)  # Shorter for analysis
            
            if "error" not in result:
                confidences.append(result["confidence"])
                log_probs.append(result["avg_log_prob"])
        
        if not confidences:
            return {"error": "No successful generations"}
        
        # Statistical analysis
        confidence_stats = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "percentiles": {
                "10": np.percentile(confidences, 10),
                "25": np.percentile(confidences, 25),
                "50": np.percentile(confidences, 50),
                "75": np.percentile(confidences, 75),
                "90": np.percentile(confidences, 90)
            }
        }
        
        log_prob_stats = {
            "mean": np.mean(log_probs),
            "std": np.std(log_probs),
            "min": np.min(log_probs),
            "max": np.max(log_probs),
            "percentiles": {
                "10": np.percentile(log_probs, 10),
                "25": np.percentile(log_probs, 25),
                "50": np.percentile(log_probs, 50),
                "75": np.percentile(log_probs, 75),
                "90": np.percentile(log_probs, 90)
            }
        }
        
        # Optimal thresholds for hypothetical multi-stage system
        optimal_thresholds = {
            "conservative": confidence_stats["percentiles"]["75"],  # High confidence threshold
            "balanced": confidence_stats["percentiles"]["50"],     # Median threshold
            "aggressive": confidence_stats["percentiles"]["25"]    # Low confidence threshold
        }
        
        return {
            "dataset": dataset_name,
            "num_samples": len(confidences),
            "confidence_statistics": confidence_stats,
            "log_prob_statistics": log_prob_stats,
            "optimal_thresholds": optimal_thresholds,
            "raw_confidences": confidences,
            "raw_log_probs": log_probs
        }
    
    def simulate_adaptive_performance(self, dataset_name: str, num_samples: int = 100) -> Dict[str, Any]:
        """Simulate adaptive performance assuming larger models were available."""
        print(f"\\n=== Simulating Adaptive Performance: {dataset_name.upper()} ===")
        
        # Load dataset
        dataset_path = Path(self.dataset_dir) / f"{dataset_name}_test.json"
        if not dataset_path.exists():
            return {"error": f"Dataset {dataset_name} not found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if len(data) > num_samples:
            data = data[:num_samples]
        
        # Simulate different lambda values
        lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        simulation_results = {}
        
        for lambda_val in lambda_values:
            print(f"  Simulating λ = {lambda_val}...")
            
            stage_counts = {"7b": 0, "14b": 0, "32b": 0, "72b": 0}
            total_cost = 0
            total_time = 0
            results = []
            
            for i, sample in enumerate(data):
                prompt = sample.get("prompt", sample.get("question", ""))
                if not prompt:
                    continue
                
                # Generate with 7B model to get confidence
                result = self.generate_text(prompt, max_tokens=100)
                
                if "error" not in result:
                    confidence = result["confidence"]
                    
                    # Simulate decision based on confidence and lambda
                    # Higher lambda = prefer quality = higher threshold = more likely to continue
                    base_threshold = 0.5
                    adjusted_threshold = base_threshold + (lambda_val - 1.0) * 0.1
                    
                    if confidence >= adjusted_threshold + 0.3:
                        # Very confident - stop at 7B
                        chosen_stage = "7b"
                        cost = 1.0
                        time_penalty = 1.0
                    elif confidence >= adjusted_threshold:
                        # Moderately confident - would stop at 14B if available
                        chosen_stage = "14b"
                        cost = 2.0
                        time_penalty = 1.5
                    elif confidence >= adjusted_threshold - 0.2:
                        # Less confident - would stop at 32B if available
                        chosen_stage = "32b"
                        cost = 4.5
                        time_penalty = 2.0
                    else:
                        # Low confidence - would use 72B if available
                        chosen_stage = "72b"
                        cost = 10.0
                        time_penalty = 3.0
                    
                    stage_counts[chosen_stage] += 1
                    total_cost += cost
                    total_time += result["inference_time"] * time_penalty
                    
                    results.append({
                        "prompt": prompt,
                        "confidence": confidence,
                        "chosen_stage": chosen_stage,
                        "cost": cost,
                        "simulated_time": result["inference_time"] * time_penalty
                    })
            
            if results:
                avg_cost = total_cost / len(results)
                avg_time = total_time / len(results)
                
                # Calculate speedup vs always using 72B
                speedup_vs_72b = 10.0 / avg_cost if avg_cost > 0 else 1.0
                
                simulation_results[str(lambda_val)] = {
                    "lambda": lambda_val,
                    "num_samples": len(results),
                    "avg_computational_cost": avg_cost,
                    "avg_simulated_time": avg_time,
                    "speedup_vs_72b": speedup_vs_72b,
                    "stage_distribution": stage_counts,
                    "stage_percentages": {k: v/len(results)*100 for k, v in stage_counts.items()},
                    "sample_results": results[:5]
                }
        
        return simulation_results
    
    def run_comprehensive_experiments(self):
        """Run all experiments with 7B model."""
        print("=== COMPREHENSIVE 7B MODEL EXPERIMENTS ===")
        
        # Load model
        if not self.load_model():
            print("Failed to load model!")
            return
        
        timestamp = int(time.time())
        all_results = {
            "timestamp": timestamp,
            "model": "Qwen3-7B",
            "experiments": {}
        }
        
        datasets = ["mmlu", "humaneval", "simple_qa"]
        
        for dataset in datasets:
            print(f"\\n{'='*60}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*60}")
            
            dataset_results = {}
            
            # 1. Basic evaluation
            eval_result = self.evaluate_dataset(dataset, num_samples=100)
            dataset_results["evaluation"] = eval_result
            
            # 2. Quality analysis
            quality_result = self.run_quality_analysis(dataset, num_samples=50)
            dataset_results["quality_analysis"] = quality_result
            
            # 3. Adaptive simulation
            simulation_result = self.simulate_adaptive_performance(dataset, num_samples=100)
            dataset_results["adaptive_simulation"] = simulation_result
            
            all_results["experiments"][dataset] = dataset_results
        
        # Save results
        results_file = Path(self.results_dir) / f"7b_comprehensive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\\n=== RESULTS SAVED ===")
        print(f"Results saved to: {results_file}")
        
        # Generate summary
        self.generate_summary(all_results)
        
        return all_results
    
    def generate_summary(self, results: Dict[str, Any]):
        """Generate experiment summary."""
        print(f"\\n=== EXPERIMENT SUMMARY ===")
        print(f"Model: {results['model']}")
        
        for dataset, data in results["experiments"].items():
            print(f"\\n{dataset.upper()} Results:")
            
            # Evaluation summary
            if "evaluation" in data and "error" not in data["evaluation"]:
                eval_data = data["evaluation"]
                print(f"  Baseline (7B only): {eval_data['avg_throughput']:.1f} tokens/sec")
                print(f"  Average confidence: {eval_data['avg_confidence']:.3f}")
            
            # Quality analysis summary
            if "quality_analysis" in data and "error" not in data["quality_analysis"]:
                qual_data = data["quality_analysis"]
                thresholds = qual_data["optimal_thresholds"]
                print(f"  Optimal thresholds: Conservative={thresholds['conservative']:.3f}, Balanced={thresholds['balanced']:.3f}")
            
            # Simulation summary (best lambda)
            if "adaptive_simulation" in data:
                sim_data = data["adaptive_simulation"]
                best_speedup = 0
                best_lambda = None
                
                for lambda_str, sim_result in sim_data.items():
                    if isinstance(sim_result, dict) and sim_result.get("speedup_vs_72b", 0) > best_speedup:
                        best_speedup = sim_result["speedup_vs_72b"]
                        best_lambda = sim_result["lambda"]
                
                if best_lambda is not None:
                    best_result = sim_data[str(best_lambda)]
                    print(f"  Best adaptive (λ={best_lambda}): {best_speedup:.2f}x speedup vs 72B")
                    
                    # Stage distribution
                    stage_dist = best_result["stage_percentages"]
                    dist_parts = [f"{stage}: {pct:.1f}%" for stage, pct in stage_dist.items() if pct > 0]
                    print(f"  Distribution: {', '.join(dist_parts)}")

def main():
    """Main execution."""
    experiments = SevenBOnlyExperiments()
    results = experiments.run_comprehensive_experiments()
    
    print("\\n=== 7B EXPERIMENTS COMPLETE ===")
    print("This provides baseline performance and simulated adaptive behavior.")
    print("When larger models download, we can run real adaptive experiments.")

if __name__ == "__main__":
    main()