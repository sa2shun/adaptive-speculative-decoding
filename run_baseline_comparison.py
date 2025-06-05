#!/usr/bin/env python3
"""
Comprehensive baseline comparison for Adaptive Speculative Decoding research.
Compares against single-model baselines and fixed pipeline strategies.
"""

import logging
import time
import json
import numpy as np
from pathlib import Path
from src.models.stage import StageManager
from src.models.predictor import QualityPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineComparator:
    """Comprehensive baseline comparison for research evaluation."""
    
    def __init__(self):
        # Model paths
        self.model_paths = [
            "/raid/sasaki/adaptive-sd-models/13b",       # 13B (index 0)
            "/raid/sasaki/adaptive-sd-models/34b-hf",    # 34B (index 1) 
            "/raid/sasaki/adaptive-sd-models/70b-full"   # 70B (index 2)
        ]
        
        self.stage_names = ["13B", "34B", "70B"]
        self.stage_costs = [1.0, 1.3, 1.8]
        
        # Initialize stage manager and quality predictor
        self.stage_manager = StageManager(self.model_paths)
        self.quality_predictor = QualityPredictor()
        
    def run_single_model_baseline(self, prompts, model_idx):
        """Run baseline using only a single model for all prompts."""
        model_name = self.stage_names[model_idx]
        logger.info(f"\n=== {model_name} Only Baseline ===")
        
        results = []
        
        for i, (prompt, category) in enumerate(prompts):
            logger.info(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            result = self.stage_manager.generate_at_stage(prompt, model_idx)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Record result
            result_data = {
                'prompt': prompt,
                'prompt_category': category,
                'method': f'{model_name}_Only',
                'model_used': model_name,
                'stage_idx': model_idx,
                'output': result['text'],
                'latency_ms': latency_ms,
                'computational_cost': self.stage_costs[model_idx],
                'tokens_generated': len(result['text'].split()),
                'quality_proxy': 1.0 if model_idx == 2 else 0.85 if model_idx == 1 else 0.75  # Fixed quality estimates
            }
            
            results.append(result_data)
            
            logger.info(f"  → Model: {model_name}, Latency: {latency_ms:.1f}ms, "
                       f"Cost: {self.stage_costs[model_idx]:.1f}")
        
        return results
    
    def run_fixed_pipeline_baseline(self, prompts, stages_to_use):
        """Run fixed pipeline baseline (e.g., always try 13B then 34B)."""
        pipeline_name = "→".join([self.stage_names[i] for i in stages_to_use])
        logger.info(f"\n=== Fixed Pipeline: {pipeline_name} ===")
        
        results = []
        
        for i, (prompt, category) in enumerate(prompts):
            logger.info(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            # Fixed pipeline logic: try stages in order, stop at first "acceptable" result
            final_stage = stages_to_use[0]  # Default to first stage
            total_cost = 0
            
            for stage_idx in stages_to_use:
                # Simulate quality check (in practice, this would be a more sophisticated check)
                quality_estimate = self.quality_predictor.predict_quality(prompt, stage_idx)
                total_cost += self.stage_costs[stage_idx]
                
                # Fixed threshold: if quality > 0.8, stop here
                if quality_estimate > 0.8 or stage_idx == stages_to_use[-1]:
                    final_stage = stage_idx
                    break
            
            # Generate with the selected stage
            result = self.stage_manager.generate_at_stage(prompt, final_stage)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            result_data = {
                'prompt': prompt,
                'prompt_category': category, 
                'method': f'Fixed_{pipeline_name}',
                'model_used': self.stage_names[final_stage],
                'stage_idx': final_stage,
                'output': result['text'],
                'latency_ms': latency_ms,
                'computational_cost': total_cost,
                'tokens_generated': len(result['text'].split()),
                'quality_proxy': self.quality_predictor.predict_quality(prompt, final_stage)
            }
            
            results.append(result_data)
            
            logger.info(f"  → Final Model: {self.stage_names[final_stage]}, "
                       f"Latency: {latency_ms:.1f}ms, Cost: {total_cost:.1f}")
        
        return results
    
    def run_adaptive_baseline(self, prompts, lambda_val=5.0):
        """Run our adaptive method for comparison."""
        logger.info(f"\n=== Adaptive (Ours) λ={lambda_val} ===")
        
        results = []
        
        for i, (prompt, category) in enumerate(prompts):
            logger.info(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            # Adaptive optimal stopping
            qualities = [self.quality_predictor.predict_quality(prompt, j) for j in range(3)]
            values = [lambda_val * q - c for q, c in zip(qualities, self.stage_costs)]
            best_stage = values.index(max(values))
            
            result = self.stage_manager.generate_at_stage(prompt, best_stage)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            result_data = {
                'prompt': prompt,
                'prompt_category': category,
                'method': 'Adaptive_Ours',
                'model_used': self.stage_names[best_stage],
                'stage_idx': best_stage,
                'output': result['text'],
                'latency_ms': latency_ms,
                'computational_cost': self.stage_costs[best_stage],
                'tokens_generated': len(result['text'].split()),
                'quality_proxy': qualities[best_stage],
                'lambda': lambda_val,
                'all_qualities': qualities,
                'all_values': values
            }
            
            results.append(result_data)
            
            logger.info(f"  → Model: {self.stage_names[best_stage]}, "
                       f"Latency: {latency_ms:.1f}ms, Cost: {self.stage_costs[best_stage]:.1f}")
        
        return results
    
    def analyze_results(self, all_results):
        """Analyze and compare all baseline results."""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE BASELINE COMPARISON ANALYSIS")
        logger.info("="*60)
        
        # Group results by method
        method_results = {}
        for result in all_results:
            method = result['method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        # Calculate metrics for each method
        comparison_data = {}
        for method, results in method_results.items():
            latencies = [r['latency_ms'] for r in results]
            costs = [r['computational_cost'] for r in results]
            qualities = [r['quality_proxy'] for r in results]
            
            comparison_data[method] = {
                'avg_latency': np.mean(latencies),
                'std_latency': np.std(latencies),
                'avg_cost': np.mean(costs),
                'avg_quality': np.mean(qualities),
                'total_experiments': len(results)
            }
        
        # Print comparison table
        logger.info("\nMethod Comparison:")
        logger.info(f"{'Method':<20} {'Latency(ms)':<12} {'Cost':<8} {'Quality':<8} {'Experiments':<12}")
        logger.info("-" * 70)
        
        for method, data in comparison_data.items():
            logger.info(f"{method:<20} {data['avg_latency']:<12.1f} "
                       f"{data['avg_cost']:<8.2f} {data['avg_quality']:<8.3f} "
                       f"{data['total_experiments']:<12}")
        
        # Calculate improvements
        if 'Adaptive_Ours' in comparison_data:
            ours = comparison_data['Adaptive_Ours']
            logger.info(f"\nAdaptive Method Performance:")
            
            for method, data in comparison_data.items():
                if method != 'Adaptive_Ours':
                    latency_improvement = ((data['avg_latency'] - ours['avg_latency']) / data['avg_latency']) * 100
                    cost_improvement = ((data['avg_cost'] - ours['avg_cost']) / data['avg_cost']) * 100
                    quality_change = ((ours['avg_quality'] - data['avg_quality']) / data['avg_quality']) * 100
                    
                    logger.info(f"  vs {method}:")
                    logger.info(f"    Latency: {latency_improvement:+.1f}% ({'faster' if latency_improvement > 0 else 'slower'})")
                    logger.info(f"    Cost: {cost_improvement:+.1f}% ({'cheaper' if cost_improvement > 0 else 'expensive'})")
                    logger.info(f"    Quality: {quality_change:+.1f}% ({'better' if quality_change > 0 else 'worse'})")
        
        return comparison_data

def main():
    """Run comprehensive baseline comparison experiments."""
    logger.info("🎯 COMPREHENSIVE BASELINE COMPARISON")
    logger.info("=" * 60)
    
    # Test prompts across complexity levels
    test_prompts = [
        ("What is 2 + 2?", "simple"),
        ("What color is the sky?", "simple"),
        ("Who wrote Romeo and Juliet?", "simple"),
        ("What is the capital of France?", "simple"),
        ("Explain photosynthesis.", "medium"),
        ("How does a car work?", "medium"),
        ("Describe machine learning.", "medium"),
        ("Design a distributed system architecture for real-time data processing.", "complex"),
        ("Implement a B+ tree data structure with balancing algorithms.", "complex"),
        ("Create a comprehensive machine learning pipeline for fraud detection.", "complex")
    ]
    
    comparator = BaselineComparator()
    all_results = []
    
    try:
        # 1. Single model baselines
        for model_idx in range(3):  # 13B, 34B, 70B
            results = comparator.run_single_model_baseline(test_prompts, model_idx)
            all_results.extend(results)
        
        # 2. Fixed pipeline baselines
        fixed_pipelines = [
            [0, 1],      # 13B → 34B
            [0, 2],      # 13B → 70B  
            [1, 2],      # 34B → 70B
            [0, 1, 2]    # 13B → 34B → 70B
        ]
        
        for pipeline in fixed_pipelines:
            results = comparator.run_fixed_pipeline_baseline(test_prompts, pipeline)
            all_results.extend(results)
        
        # 3. Our adaptive method
        adaptive_results = comparator.run_adaptive_baseline(test_prompts, lambda_val=5.0)
        all_results.extend(adaptive_results)
        
        # 4. Analysis
        comparison_data = comparator.analyze_results(all_results)
        
        # 5. Save results
        results_dir = Path("/raid/sasaki/adaptive-sd-results/baseline_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"baseline_comparison_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'comparison_data': comparison_data,
                'experiment_info': {
                    'total_methods': len(set(r['method'] for r in all_results)),
                    'total_experiments': len(all_results),
                    'prompt_count': len(test_prompts)
                }
            }, f, indent=2)
        
        logger.info(f"\n✅ Baseline comparison complete!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📊 Total experiments: {len(all_results)}")
        logger.info(f"🔬 Methods compared: {len(set(r['method'] for r in all_results))}")
        
    except Exception as e:
        logger.error(f"❌ Baseline comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()