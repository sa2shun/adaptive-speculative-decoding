#!/usr/bin/env python3
"""
Quick validation experiment for adaptive speculative decoding
Tests the core algorithm with simulated models for rapid prototyping
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""
    n_stages: int = 3
    n_queries: int = 100
    lambda_values: List[float] = None
    cost_ratios: List[float] = None
    quality_bounds: List[float] = None
    
    def __post_init__(self):
        if self.lambda_values is None:
            self.lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        if self.cost_ratios is None:
            self.cost_ratios = [1.0, 4.5, 10.0]
        if self.quality_bounds is None:
            self.quality_bounds = [0.7, 0.85, 0.95]


class MockLLMStage:
    """Simulated LLM stage for validation."""
    
    def __init__(self, stage_id: int, cost: float, base_quality: float):
        self.stage_id = stage_id
        self.cost = cost
        self.base_quality = base_quality
        
    def generate(self, query_complexity: float) -> Tuple[str, float, float]:
        """Simulate generation with quality dependent on complexity."""
        # Simulate processing time proportional to cost
        time.sleep(self.cost * 0.001)  # 1ms per cost unit
        
        # Quality varies with query complexity and stage capability
        quality = min(0.99, self.base_quality + (1 - query_complexity) * 0.1)
        quality += np.random.normal(0, 0.02)  # Add noise
        quality = max(0.1, min(0.99, quality))
        
        output = f"Stage-{self.stage_id} response (quality: {quality:.3f})"
        return output, quality, self.cost


class QualityPredictor:
    """Simple quality predictor for validation."""
    
    def __init__(self, n_stages: int):
        self.n_stages = n_stages
        # Simple heuristic: harder queries need more stages
        
    def predict(self, query_complexity: float, stage: int) -> float:
        """Predict if stopping at stage will give acceptable quality."""
        # Probability that current stage is sufficient
        base_prob = min(0.9, (stage + 1) / self.n_stages + 0.2)
        complexity_penalty = query_complexity * 0.3
        confidence = max(0.1, base_prob - complexity_penalty)
        
        # Add some noise to simulate predictor uncertainty
        confidence += np.random.normal(0, 0.05)
        return max(0.1, min(0.9, confidence))


class AdaptiveDecodingValidator:
    """Validation framework for adaptive decoding."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.stages = []
        self.predictor = QualityPredictor(config.n_stages)
        
        # Initialize stages
        for i in range(config.n_stages):
            stage = MockLLMStage(
                stage_id=i,
                cost=config.cost_ratios[i],
                base_quality=config.quality_bounds[i]
            )
            self.stages.append(stage)
    
    def compute_threshold(self, stage: int, lambda_param: float) -> float:
        """Compute optimal stopping threshold."""
        if stage >= len(self.stages) - 1:
            return 0.0  # Always stop at final stage
        
        next_cost = self.config.cost_ratios[stage + 1]
        expected_gain = 0.1  # Simplified quality improvement estimate
        
        base_threshold = next_cost / (next_cost + lambda_param)
        return base_threshold * (1.0 - expected_gain)
    
    def adaptive_decode(self, query_complexity: float, lambda_param: float) -> Dict:
        """Run adaptive decoding for a single query."""
        total_cost = 0
        outputs = []
        decisions = []
        
        for stage_idx in range(len(self.stages)):
            stage = self.stages[stage_idx]
            
            # Generate with current stage
            output, quality, cost = stage.generate(query_complexity)
            total_cost += cost
            outputs.append((stage_idx, output, quality, cost))
            
            # Check if we should stop
            if stage_idx < len(self.stages) - 1:
                confidence = self.predictor.predict(query_complexity, stage_idx)
                threshold = self.compute_threshold(stage_idx, lambda_param)
                
                decision = {
                    'stage': stage_idx,
                    'confidence': confidence,
                    'threshold': threshold,
                    'stop': confidence >= threshold
                }
                decisions.append(decision)
                
                if confidence >= threshold:
                    break
            else:
                # Final stage - always stop
                decisions.append({
                    'stage': stage_idx,
                    'confidence': 1.0,
                    'threshold': 0.0,
                    'stop': True
                })
        
        final_stage = len(outputs) - 1
        final_quality = outputs[-1][2]
        
        return {
            'query_complexity': query_complexity,
            'lambda': lambda_param,
            'stopping_stage': final_stage,
            'total_cost': total_cost,
            'final_quality': final_quality,
            'outputs': outputs,
            'decisions': decisions
        }
    
    def run_validation_experiment(self) -> Dict:
        """Run complete validation experiment."""
        print("=== ADAPTIVE SPECULATIVE DECODING VALIDATION ===")
        print(f"Configuration: {self.config.n_stages} stages, {self.config.n_queries} queries")
        print(f"Lambda values: {self.config.lambda_values}")
        print()
        
        results = {
            'config': self.config.__dict__,
            'experiments': {},
            'baselines': {},
            'summary': {}
        }
        
        # Generate diverse query complexities
        query_complexities = np.random.beta(2, 2, self.config.n_queries)
        
        # Run experiments for each lambda
        for lambda_val in self.config.lambda_values:
            print(f"Running experiments for λ = {lambda_val}...")
            
            lambda_results = []
            for query_complexity in query_complexities:
                result = self.adaptive_decode(query_complexity, lambda_val)
                lambda_results.append(result)
            
            results['experiments'][lambda_val] = lambda_results
            
            # Compute summary statistics
            avg_cost = np.mean([r['total_cost'] for r in lambda_results])
            avg_quality = np.mean([r['final_quality'] for r in lambda_results])
            stage_distribution = [0] * self.config.n_stages
            for r in lambda_results:
                stage_distribution[r['stopping_stage']] += 1
            stage_distribution = [x / len(lambda_results) for x in stage_distribution]
            
            print(f"  Average cost: {avg_cost:.2f}")
            print(f"  Average quality: {avg_quality:.3f}")
            print(f"  Stage distribution: {[f'{x:.1%}' for x in stage_distribution]}")
            print()
        
        # Run baseline comparisons
        print("Running baseline comparisons...")
        for stage_idx in range(self.config.n_stages):
            print(f"  Baseline: Always use stage {stage_idx}")
            
            baseline_results = []
            for query_complexity in query_complexities:
                stage = self.stages[stage_idx]
                output, quality, cost = stage.generate(query_complexity)
                
                baseline_results.append({
                    'query_complexity': query_complexity,
                    'stage': stage_idx,
                    'cost': cost,
                    'quality': quality
                })
            
            results['baselines'][f'stage_{stage_idx}'] = baseline_results
            
            avg_cost = np.mean([r['cost'] for r in baseline_results])
            avg_quality = np.mean([r['quality'] for r in baseline_results])
            print(f"    Average cost: {avg_cost:.2f}, Average quality: {avg_quality:.3f}")
        
        print()
        
        # Generate analysis
        self._analyze_results(results)
        
        return results
    
    def _analyze_results(self, results: Dict):
        """Analyze and summarize results."""
        print("=== ANALYSIS ===")
        
        # Compare against baselines
        baseline_costs = {
            name: np.mean([r['cost'] for r in data])
            for name, data in results['baselines'].items()
        }
        baseline_qualities = {
            name: np.mean([r['quality'] for r in data])
            for name, data in results['baselines'].items()
        }
        
        print("Baseline Performance:")
        for name in baseline_costs:
            print(f"  {name}: Cost = {baseline_costs[name]:.2f}, Quality = {baseline_qualities[name]:.3f}")
        print()
        
        print("Adaptive Method Performance:")
        best_lambda = None
        best_efficiency = 0
        
        for lambda_val, data in results['experiments'].items():
            avg_cost = np.mean([r['total_cost'] for r in data])
            avg_quality = np.mean([r['final_quality'] for r in data])
            
            # Compare to most expensive baseline (highest quality)
            highest_baseline_cost = max(baseline_costs.values())
            speedup = highest_baseline_cost / avg_cost
            
            efficiency = avg_quality / avg_cost
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_lambda = lambda_val
            
            print(f"  λ = {lambda_val}: Cost = {avg_cost:.2f}, Quality = {avg_quality:.3f}, Speedup = {speedup:.2f}x")
        
        print()
        print(f"Best λ = {best_lambda} (highest quality/cost ratio)")
        
        # Stage utilization analysis
        print("\nStage Utilization by λ:")
        for lambda_val, data in results['experiments'].items():
            stage_counts = [0] * self.config.n_stages
            for r in data:
                stage_counts[r['stopping_stage']] += 1
            stage_percentages = [100 * x / len(data) for x in stage_counts]
            print(f"  λ = {lambda_val}: " + " | ".join([f"Stage {i}: {p:.1f}%" for i, p in enumerate(stage_percentages)]))


def main():
    """Run validation experiment."""
    config = ValidationConfig(
        n_stages=3,
        n_queries=100,
        lambda_values=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    validator = AdaptiveDecodingValidator(config)
    results = validator.run_validation_experiment()
    
    # Save results
    results_path = "results/validation_results.json"
    Path("results").mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def clean_for_json(data):
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_for_json(item) for item in data]
        else:
            return convert_numpy(data)
    
    with open(results_path, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()