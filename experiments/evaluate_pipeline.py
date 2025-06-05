#!/usr/bin/env python3
"""
Pipeline evaluation script
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import requests
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Evaluator for the adaptive pipeline"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        
        # Load evaluation metrics
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        
        # Verify server connection
        self._check_server()
    
    def _check_server(self):
        """Check if server is accessible"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("✓ Server connection verified")
            else:
                raise Exception(f"Server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to server: {e}")
            raise
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        num_samples: int = 1000,
        lambda_value: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline on a specific dataset
        
        Args:
            dataset_name: Name of dataset to evaluate
            num_samples: Number of samples to evaluate
            lambda_value: Lambda parameter value
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {dataset_name} with {num_samples} samples, lambda={lambda_value}")
        
        # Load dataset
        samples = self._load_dataset(dataset_name, num_samples)
        
        # Evaluation metrics
        results = {
            "dataset": dataset_name,
            "lambda": lambda_value,
            "num_samples": len(samples),
            "latencies": [],
            "qualities": [],
            "stage_distributions": [0] * 4,
            "tokens_per_second": [],
            "cache_hits": [],
            "individual_results": []
        }
        
        # Process each sample
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
            try:
                result = self._evaluate_sample(sample, i)
                
                # Aggregate metrics
                results["latencies"].append(result["latency_ms"])
                results["qualities"].append(result["quality"])
                results["stage_distributions"][result["stopped_at_stage"]] += 1
                results["tokens_per_second"].append(result["tokens_per_second"])
                results["cache_hits"].append(result["cache_hits"])
                
                # Store individual result
                results["individual_results"].append({
                    "sample_id": i,
                    "prompt": sample["prompt"][:100] + "..." if len(sample["prompt"]) > 100 else sample["prompt"],
                    "output": result["output"][:200] + "..." if len(result["output"]) > 200 else result["output"],
                    "latency_ms": result["latency_ms"],
                    "quality": result["quality"],
                    "stopped_at_stage": result["stopped_at_stage"],
                    "tokens_per_second": result["tokens_per_second"]
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample {i}: {e}")
                continue
        
        # Compute aggregate statistics
        results["stats"] = self._compute_statistics(results)
        
        return results
    
    def _load_dataset(self, dataset_name: str, num_samples: int) -> List[Dict]:
        """Load dataset samples"""
        logger.info(f"Loading {dataset_name} dataset...")
        
        samples = []
        
        try:
            if dataset_name == "mmlu":
                dataset = load_dataset("cais/mmlu", "all", split="test")
                samples = [
                    {
                        "prompt": f"Question: {sample['question']}\nAnswer:",
                        "reference": sample.get("answer", "")
                    }
                    for sample in dataset[:num_samples]
                ]
                
            elif dataset_name == "humaneval":
                dataset = load_dataset("openai_humaneval", split="test")
                samples = [
                    {
                        "prompt": sample["prompt"],
                        "reference": sample.get("canonical_solution", "")
                    }
                    for sample in dataset[:num_samples]
                ]
                
            elif dataset_name == "hotpotqa":
                dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
                samples = [
                    {
                        "prompt": f"Question: {sample['question']}\nAnswer:",
                        "reference": sample.get("answer", "")
                    }
                    for sample in dataset[:num_samples]
                ]
                
            elif dataset_name == "alpacaeval":
                # Simple test prompts for demo
                test_prompts = [
                    "Explain the concept of machine learning in simple terms.",
                    "Write a short poem about artificial intelligence.",
                    "What are the main advantages of renewable energy?",
                    "Describe the process of photosynthesis.",
                    "How do neural networks work?",
                    "What is the difference between AI and machine learning?",
                    "Explain quantum computing to a 10-year-old.",
                    "What are the ethical implications of AI?",
                    "How does natural language processing work?",
                    "What is the future of autonomous vehicles?"
                ]
                
                samples = [
                    {"prompt": prompt, "reference": ""}
                    for prompt in (test_prompts * (num_samples // len(test_prompts) + 1))[:num_samples]
                ]
                
            elif dataset_name == "longbench":
                # Long context test prompts
                long_prompt = "Summarize the following document: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
                samples = [
                    {"prompt": long_prompt, "reference": ""}
                    for _ in range(num_samples)
                ]
                
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def _evaluate_sample(self, sample: Dict, sample_id: int) -> Dict:
        """Evaluate a single sample"""
        prompt = sample["prompt"]
        reference = sample.get("reference", "")
        
        # Generate with pipeline
        start_time = time.time()
        
        response = requests.post(
            f"{self.server_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Compute quality score
        quality = self._compute_quality(result["output"], reference)
        
        return {
            "sample_id": sample_id,
            "output": result["output"],
            "latency_ms": result["latency_ms"],
            "quality": quality,
            "stopped_at_stage": result["stopped_at_stage"],
            "stage_probabilities": result["stage_probabilities"],
            "tokens_per_second": result["tokens_per_second"],
            "cache_hits": result["cache_hits"]
        }
    
    def _compute_quality(self, generated: str, reference: str) -> float:
        """Compute quality score for generated text"""
        if not reference:
            # Use heuristic quality measures when no reference
            words = generated.split()
            if len(words) < 5:
                return 0.1  # Too short
            
            # Check for repetition
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                return 0.3  # Too repetitive
            
            # Length-based quality (prefer reasonable length)
            if len(words) < 10:
                return 0.5
            elif len(words) < 50:
                return 0.8
            else:
                return min(1.0, 100 / len(words))  # Penalize very long outputs
        
        # Use BLEU and ROUGE with reference
        try:
            bleu_score = self.bleu.compute(
                predictions=[generated],
                references=[[reference]]
            )["bleu"]
            
            rouge_score = self.rouge.compute(
                predictions=[generated],
                references=[reference]
            )["rougeL"]
            
            # Weighted combination
            quality = 0.5 * bleu_score + 0.5 * rouge_score
            return quality
            
        except Exception as e:
            logger.warning(f"Quality computation failed: {e}")
            return 0.5  # Default quality
    
    def _compute_statistics(self, results: Dict) -> Dict:
        """Compute aggregate statistics"""
        latencies = results["latencies"]
        qualities = results["qualities"]
        stage_counts = results["stage_distributions"]
        
        if not latencies:
            return {}
        
        total_samples = len(latencies)
        
        stats = {
            # Latency statistics
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            
            # Quality statistics
            "quality_mean": np.mean(qualities),
            "quality_std": np.std(qualities),
            "quality_p25": np.percentile(qualities, 25),
            "quality_p75": np.percentile(qualities, 75),
            
            # Stage distribution
            "stage_distribution": [count / total_samples for count in stage_counts],
            "early_stop_rate": sum(stage_counts[:3]) / total_samples,  # Stop before stage 3
            
            # Throughput
            "avg_tokens_per_second": np.mean(results["tokens_per_second"]),
            
            # Cache efficiency
            "avg_cache_hits": np.mean(results["cache_hits"]) if results["cache_hits"] else 0,
            
            # Efficiency metrics
            "samples_per_second": total_samples / sum(latencies) * 1000 if latencies else 0
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive speculative decoding pipeline")
    parser.add_argument("--lambda", type=float, default=1.0, help="Lambda parameter value")
    parser.add_argument("--datasets", default="mmlu", help="Comma-separated list of datasets")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples per dataset")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--output-file", required=True, help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = PipelineEvaluator(args.server_url)
    
    # Update lambda on server
    logger.info(f"Setting lambda to {args.lambda}")
    response = requests.post(
        f"{args.server_url}/update_lambda",
        json={"lambda_value": args.lambda}
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to update lambda: {response.text}")
        return
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n=== Evaluating {dataset} ===")
        
        try:
            results = evaluator.evaluate_dataset(
                dataset_name=dataset,
                num_samples=args.num_samples,
                lambda_value=args.lambda
            )
            
            all_results[dataset] = results
            
            # Log key metrics
            stats = results["stats"]
            logger.info(f"Results for {dataset}:")
            logger.info(f"  Avg latency: {stats['latency_mean']:.1f}ms (p95: {stats['latency_p95']:.1f}ms)")
            logger.info(f"  Avg quality: {stats['quality_mean']:.3f}")
            logger.info(f"  Stage distribution: {[f'{d:.2f}' for d in stats['stage_distribution']]}")
            logger.info(f"  Early stop rate: {stats['early_stop_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset}: {e}")
            continue
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Summary
    if all_results:
        logger.info("\n=== Summary ===")
        for dataset, results in all_results.items():
            stats = results["stats"]
            logger.info(f"{dataset}: {stats['latency_mean']:.1f}ms, quality={stats['quality_mean']:.3f}")


if __name__ == "__main__":
    main()