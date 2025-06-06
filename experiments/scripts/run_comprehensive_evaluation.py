#!/usr/bin/env python3
"""
Comprehensive evaluation script for adaptive speculative decoding.
Runs large-scale experiments across all lambda values and datasets.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse
import subprocess
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.serving.pipeline import AdaptiveSpeculativeDecodingPipeline
from src.models.stage import StageManager
from src.models.predictor import QualityPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/sasaki/adaptive-speculative-decoding/logs/comprehensive_eval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            mem_allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"GPU {i}: {mem_total:.1f}GB total, {mem_reserved:.1f}GB reserved, {mem_allocated:.1f}GB allocated")
    else:
        logger.warning("CUDA not available")

def run_baseline_experiments(datasets: List[str], output_dir: str):
    """Run baseline single-model experiments."""
    logger.info("Starting baseline experiments...")
    
    baseline_models = [
        "/raid/sasaki/adaptive-sd-models/llama-3.1-8b",
        "/raid/sasaki/adaptive-sd-models/13b", 
        "/raid/sasaki/adaptive-sd-models/34b-hf",
        "/raid/sasaki/adaptive-sd-models/70b-full"
    ]
    
    results = {}
    
    for model_path in baseline_models:
        model_name = model_path.split('/')[-1]
        logger.info(f"Running baseline for {model_name}...")
        
        for dataset in datasets:
            logger.info(f"  Dataset: {dataset}")
            try:
                # Run baseline evaluation
                cmd = [
                    "python", "experiments/evaluate_pipeline.py",
                    "--baseline-only",
                    "--model-path", model_path,
                    "--datasets", dataset,
                    "--output-dir", f"{output_dir}/baseline_{model_name}_{dataset}",
                    "--num-samples", "500"  # Reasonable sample size for baselines
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    logger.info(f"    ✓ Completed {model_name} on {dataset}")
                    results[f"{model_name}_{dataset}"] = "completed"
                else:
                    logger.error(f"    ✗ Failed {model_name} on {dataset}: {result.stderr}")
                    results[f"{model_name}_{dataset}"] = "failed"
                    
            except subprocess.TimeoutExpired:
                logger.error(f"    ✗ Timeout for {model_name} on {dataset}")
                results[f"{model_name}_{dataset}"] = "timeout"
            except Exception as e:
                logger.error(f"    ✗ Error for {model_name} on {dataset}: {e}")
                results[f"{model_name}_{dataset}"] = "error"
    
    return results

def run_adaptive_experiments(datasets: List[str], lambda_values: List[float], output_dir: str):
    """Run adaptive speculative decoding experiments."""
    logger.info("Starting adaptive experiments...")
    
    results = {}
    
    for lambda_val in lambda_values:
        logger.info(f"Lambda = {lambda_val}")
        
        for dataset in datasets:
            logger.info(f"  Dataset: {dataset}")
            
            try:
                # Run adaptive evaluation
                cmd = [
                    "python", "experiments/evaluate_pipeline.py",
                    "--datasets", dataset,
                    "--lambda", str(lambda_val),
                    "--output-dir", f"{output_dir}/adaptive_lambda{lambda_val}_{dataset}",
                    "--num-samples", "1000"  # Full evaluation samples
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
                if result.returncode == 0:
                    logger.info(f"    ✓ Completed λ={lambda_val} on {dataset}")
                    results[f"lambda{lambda_val}_{dataset}"] = "completed"
                else:
                    logger.error(f"    ✗ Failed λ={lambda_val} on {dataset}: {result.stderr}")
                    results[f"lambda{lambda_val}_{dataset}"] = "failed"
                    
            except subprocess.TimeoutExpired:
                logger.error(f"    ✗ Timeout for λ={lambda_val} on {dataset}")
                results[f"lambda{lambda_val}_{dataset}"] = "timeout"
            except Exception as e:
                logger.error(f"    ✗ Error for λ={lambda_val} on {dataset}: {e}")
                results[f"lambda{lambda_val}_{dataset}"] = "error"
    
    return results

def compile_results(output_dir: str):
    """Compile all experimental results into summary."""
    logger.info("Compiling results...")
    
    summary = {
        "experiment_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": ["8B", "13B", "34B", "70B"],
            "datasets": ["mmlu", "humaneval", "gsm8k", "truthfulqa"],
            "lambda_values": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        },
        "baseline_results": {},
        "adaptive_results": {},
        "comparison_metrics": {}
    }
    
    # Collect results from individual experiment outputs
    for result_file in Path(output_dir).rglob("results.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                
            if "baseline" in str(result_file):
                summary["baseline_results"][result_file.parent.name] = data
            else:
                summary["adaptive_results"][result_file.parent.name] = data
                
        except Exception as e:
            logger.error(f"Failed to load {result_file}: {e}")
    
    # Save comprehensive summary
    summary_file = f"{output_dir}/comprehensive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results compiled to {summary_file}")
    return summary

def evaluate_different_scenarios():
    """Test the system with different types of prompts and complexity levels"""
    
    test_scenarios = [
        # Simple factual questions
        {
            "category": "Simple Factual",
            "prompts": [
                "What is 2 + 2?",
                "What color is the sky?", 
                "Who wrote Romeo and Juliet?",
                "What is the capital of Japan?",
                "How many days are in a week?"
            ],
            "expected_complexity": "low"
        },
        
        # Medium explanations
        {
            "category": "Medium Explanations", 
            "prompts": [
                "Explain how photosynthesis works.",
                "What is machine learning?",
                "How does a car engine work?",
                "Describe the water cycle.",
                "What causes earthquakes?"
            ],
            "expected_complexity": "medium"
        },
        
        # Complex technical tasks
        {
            "category": "Complex Technical",
            "prompts": [
                "Implement a binary search tree with insert, delete, and search operations in Python.",
                "Explain the differences between supervised, unsupervised, and reinforcement learning with examples.",
                "Design a distributed system architecture for a real-time chat application that can handle millions of users.",
                "Write a comprehensive guide on optimizing database queries for large-scale applications.",
                "Analyze the time and space complexity of various graph algorithms and their use cases."
            ],
            "expected_complexity": "high"
        }
    ]
    
    print("🔬 Comprehensive Evaluation of Adaptive Speculative Decoding")
    print("=" * 80)
    
    all_results = []
    base_url = "http://localhost:8001"
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['category']}")
        print(f"Expected complexity: {scenario['expected_complexity']}")
        print("-" * 40)
        
        scenario_results = []
        
        for i, prompt in enumerate(scenario['prompts']):
            print(f"  {i+1}. {prompt[:50]}...")
            
            try:
                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": 256,
                        "temperature": 0.7
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    stage_names = ["8B", "13B", "34B", "70B"]
                    
                    result_data = {
                        "category": scenario['category'],
                        "expected_complexity": scenario['expected_complexity'],
                        "prompt": prompt,
                        "prompt_length": len(prompt),
                        "word_count": len(prompt.split()),
                        "stopped_at_stage": result['stopped_at_stage'],
                        "stage_name": stage_names[result['stopped_at_stage']],
                        "latency_ms": result['latency_ms'],
                        "total_tokens": result['total_tokens'],
                        "stage_probabilities": result['stage_probabilities'],
                        "output_length": len(result['output'])
                    }
                    
                    scenario_results.append(result_data)
                    all_results.append(result_data)
                    
                    print(f"     Stage: {stage_names[result['stopped_at_stage']]}, "
                          f"Latency: {result['latency_ms']:.1f}ms, "
                          f"Tokens: {result['total_tokens']}")
                
                else:
                    print(f"     ❌ Error: {response.status_code}")
                
            except Exception as e:
                print(f"     ❌ Exception: {e}")
            
            time.sleep(0.2)  # Small delay between requests
        
        # Analyze scenario results
        if scenario_results:
            df = pd.DataFrame(scenario_results)
            avg_stage = df['stopped_at_stage'].mean()
            avg_latency = df['latency_ms'].mean()
            avg_tokens = df['total_tokens'].mean()
            
            print(f"\n  📊 Scenario Summary:")
            print(f"     Average stage: {avg_stage:.1f}")
            print(f"     Average latency: {avg_latency:.1f}ms")
            print(f"     Average tokens: {avg_tokens:.1f}")
    
    return all_results


def analyze_comprehensive_results(results: List[Dict[str, Any]]):
    """Analyze the comprehensive evaluation results"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    df = pd.DataFrame(results)
    
    print("\n📈 Comprehensive Analysis Results")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"   Total prompts tested: {len(results)}")
    print(f"   Average latency: {df['latency_ms'].mean():.2f}ms")
    print(f"   Latency std dev: {df['latency_ms'].std():.2f}ms")
    print(f"   Average tokens generated: {df['total_tokens'].mean():.1f}")
    print(f"   Average prompt length: {df['prompt_length'].mean():.1f} chars")
    
    # Stage distribution analysis
    print(f"\n🎯 Stage Distribution Analysis:")
    stage_counts = df['stopped_at_stage'].value_counts().sort_index()
    stage_names = ["8B", "13B", "34B", "70B"]
    
    for stage, count in stage_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   {stage_names[stage]}: {count} requests ({percentage:.1f}%)")
    
    # Category-wise analysis
    print(f"\n📝 Analysis by Category:")
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        
        print(f"\n   {category}:")
        print(f"     Requests: {len(cat_df)}")
        print(f"     Avg stage: {cat_df['stopped_at_stage'].mean():.2f}")
        print(f"     Avg latency: {cat_df['latency_ms'].mean():.1f}ms")
        print(f"     Avg tokens: {cat_df['total_tokens'].mean():.1f}")
        print(f"     Avg prompt length: {cat_df['prompt_length'].mean():.1f} chars")
        
        # Stage distribution for this category
        cat_stage_counts = cat_df['stopped_at_stage'].value_counts().sort_index()
        stage_dist = [cat_stage_counts.get(i, 0) for i in range(4)]
        print(f"     Stage distribution: {stage_dist}")
    
    # Complexity vs Stage analysis
    print(f"\n🔍 Complexity vs Stage Analysis:")
    complexity_mapping = {"low": 0, "medium": 1, "high": 2}
    
    for complexity in ["low", "medium", "high"]:
        comp_df = df[df['expected_complexity'] == complexity]
        if len(comp_df) > 0:
            avg_stage = comp_df['stopped_at_stage'].mean()
            print(f"   {complexity.title()} complexity: Avg stage {avg_stage:.2f}")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    print(f"   Fastest response: {df['latency_ms'].min():.1f}ms")
    print(f"   Slowest response: {df['latency_ms'].max():.1f}ms")
    print(f"   Median latency: {df['latency_ms'].median():.1f}ms")
    
    # Efficiency analysis
    total_theoretical_cost = len(results) * 8.8  # If all used 70B model
    actual_cost = sum(
        row['total_tokens'] * [1.0, 1.6, 4.2, 8.8][row['stopped_at_stage']]
        for _, row in df.iterrows()
    )
    efficiency_gain = ((total_theoretical_cost - actual_cost) / total_theoretical_cost) * 100
    
    print(f"\n💰 Cost Efficiency Analysis:")
    print(f"   Theoretical cost (all 70B): {total_theoretical_cost:.1f} units")
    print(f"   Actual cost: {actual_cost:.1f} units")
    print(f"   Efficiency gain: {efficiency_gain:.1f}%")


def test_edge_cases():
    """Test edge cases and unusual inputs"""
    
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    edge_cases = [
        {"name": "Empty prompt", "prompt": ""},
        {"name": "Single word", "prompt": "Hello"},
        {"name": "Very long prompt", "prompt": "A" * 1000},
        {"name": "Numbers only", "prompt": "123 456 789"},
        {"name": "Special characters", "prompt": "!@#$%^&*()"},
        {"name": "Mixed languages", "prompt": "Hello 世界 Bonjour мир"},
    ]
    
    base_url = "http://localhost:8001"
    results = []
    
    for case in edge_cases:
        print(f"  Testing: {case['name']}")
        
        try:
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "prompt": case['prompt'],
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                stage_names = ["8B", "13B", "34B", "70B"]
                print(f"    ✅ Stage: {stage_names[result['stopped_at_stage']]}, "
                      f"Latency: {result['latency_ms']:.1f}ms")
                results.append({
                    "case": case['name'],
                    "prompt": case['prompt'],
                    "stage": result['stopped_at_stage'],
                    "latency": result['latency_ms']
                })
            else:
                print(f"    ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ Exception: {e}")
        
        time.sleep(0.1)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--datasets", nargs="+", 
                       default=["mmlu", "humaneval", "gsm8k", "truthfulqa"],
                       help="Datasets to evaluate")
    parser.add_argument("--lambda-values", nargs="+", type=float,
                       default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                       help="Lambda values to test")
    parser.add_argument("--output-dir", 
                       default="/raid/sasaki/adaptive-sd-results/comprehensive",
                       help="Output directory")
    parser.add_argument("--skip-baselines", action="store_true",
                       help="Skip baseline experiments")
    parser.add_argument("--skip-adaptive", action="store_true", 
                       help="Skip adaptive experiments")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=== COMPREHENSIVE ADAPTIVE SPECULATIVE DECODING EVALUATION ===")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Lambda values: {args.lambda_values}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check system resources
    check_gpu_memory()
    
    results = {}
    
    try:
        # Run baseline experiments
        if not args.skip_baselines:
            baseline_results = run_baseline_experiments(args.datasets, args.output_dir)
            results["baselines"] = baseline_results
        
        # Run adaptive experiments  
        if not args.skip_adaptive:
            adaptive_results = run_adaptive_experiments(args.datasets, args.lambda_values, args.output_dir)
            results["adaptive"] = adaptive_results
        
        # Compile final results
        summary = compile_results(args.output_dir)
        
        logger.info("=== EVALUATION COMPLETE ===")
        logger.info(f"Total experiments: {len(results.get('baselines', {})) + len(results.get('adaptive', {}))}")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()