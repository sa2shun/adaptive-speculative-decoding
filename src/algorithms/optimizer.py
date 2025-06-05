"""
Lambda parameter optimization for quality-speed tradeoff
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
import logging
from scipy.optimize import minimize_scalar, minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of lambda optimization"""
    optimal_lambda: float
    achieved_latency: float
    achieved_quality: float
    constraint_satisfied: bool
    iterations: int


class LambdaOptimizer:
    """
    Optimizer for finding optimal lambda parameter
    """
    
    def __init__(
        self,
        latency_constraint: Optional[float] = None,
        quality_constraint: Optional[float] = None,
        lambda_bounds: Tuple[float, float] = (0.01, 100.0)
    ):
        """
        Initialize optimizer
        
        Args:
            latency_constraint: Maximum allowed latency (ms)
            quality_constraint: Minimum required quality (0-1)
            lambda_bounds: (min_lambda, max_lambda)
        """
        self.latency_constraint = latency_constraint
        self.quality_constraint = quality_constraint
        self.lambda_bounds = lambda_bounds
        
    def optimize_for_latency_constraint(
        self,
        evaluate_function: Callable[[float], Tuple[float, float]],
        tolerance: float = 1e-3,
        max_iterations: int = 50
    ) -> OptimizationResult:
        """
        Find optimal lambda that satisfies latency constraint
        
        Args:
            evaluate_function: Function that takes lambda and returns (latency, quality)
            tolerance: Convergence tolerance
            max_iterations: Maximum optimization iterations
            
        Returns:
            OptimizationResult with optimal lambda
        """
        if self.latency_constraint is None:
            raise ValueError("Latency constraint must be set")
        
        logger.info(f"Optimizing lambda for latency constraint: {self.latency_constraint}ms")
        
        def objective(lam: float) -> float:
            """Objective: minimize negative quality subject to latency constraint"""
            latency, quality = evaluate_function(lam)
            
            # Penalty for violating constraint
            if latency > self.latency_constraint:
                penalty = 1000 * (latency - self.latency_constraint)
                return -quality + penalty
            
            return -quality  # Maximize quality
        
        # Binary search for constraint satisfaction
        low_lam, high_lam = self.lambda_bounds
        best_lambda = low_lam
        best_quality = 0.0
        iterations = 0
        
        for iteration in range(max_iterations):
            iterations = iteration + 1
            
            # Try middle value
            mid_lam = (low_lam + high_lam) / 2
            latency, quality = evaluate_function(mid_lam)
            
            logger.debug(f"Iteration {iteration}: lambda={mid_lam:.3f}, "
                        f"latency={latency:.1f}ms, quality={quality:.3f}")
            
            if latency <= self.latency_constraint:
                # Constraint satisfied, try lower lambda (better quality)
                best_lambda = mid_lam
                best_quality = quality
                high_lam = mid_lam
            else:
                # Constraint violated, try higher lambda (lower latency)
                low_lam = mid_lam
            
            # Check convergence
            if high_lam - low_lam < tolerance:
                break
        
        # Final evaluation
        final_latency, final_quality = evaluate_function(best_lambda)
        constraint_satisfied = final_latency <= self.latency_constraint
        
        logger.info(f"Optimization complete: lambda={best_lambda:.3f}, "
                   f"latency={final_latency:.1f}ms, quality={final_quality:.3f}")
        
        return OptimizationResult(
            optimal_lambda=best_lambda,
            achieved_latency=final_latency,
            achieved_quality=final_quality,
            constraint_satisfied=constraint_satisfied,
            iterations=iterations
        )
    
    def optimize_pareto_front(
        self,
        evaluate_function: Callable[[float], Tuple[float, float]],
        num_points: int = 20
    ) -> List[Tuple[float, float, float]]:
        """
        Compute Pareto front for quality-latency tradeoff
        
        Args:
            evaluate_function: Function that takes lambda and returns (latency, quality)
            num_points: Number of points to sample
            
        Returns:
            List of (lambda, latency, quality) tuples
        """
        lambda_values = np.logspace(
            np.log10(self.lambda_bounds[0]),
            np.log10(self.lambda_bounds[1]),
            num_points
        )
        
        pareto_points = []
        
        for lam in lambda_values:
            latency, quality = evaluate_function(lam)
            pareto_points.append((lam, latency, quality))
            logger.debug(f"Lambda={lam:.3f}: latency={latency:.1f}ms, quality={quality:.3f}")
        
        # Sort by latency
        pareto_points.sort(key=lambda x: x[1])
        
        return pareto_points
    
    def find_balanced_lambda(
        self,
        evaluate_function: Callable[[float], Tuple[float, float]],
        quality_weight: float = 0.5
    ) -> OptimizationResult:
        """
        Find lambda that balances quality and speed
        
        Args:
            evaluate_function: Function that takes lambda and returns (latency, quality)
            quality_weight: Weight for quality in [0,1], (1-weight) for speed
            
        Returns:
            OptimizationResult with balanced lambda
        """
        logger.info(f"Finding balanced lambda with quality_weight={quality_weight}")
        
        def objective(lam: float) -> float:
            """Weighted combination of normalized quality and speed"""
            latency, quality = evaluate_function(lam)
            
            # Normalize latency (lower is better)
            normalized_latency = latency / 1000.0  # Convert ms to seconds
            
            # Combined objective (maximize quality, minimize latency)
            score = quality_weight * quality - (1 - quality_weight) * normalized_latency
            
            return -score  # Minimize negative score
        
        # Optimize
        result = minimize_scalar(
            objective,
            bounds=self.lambda_bounds,
            method='bounded'
        )
        
        optimal_lambda = result.x
        final_latency, final_quality = evaluate_function(optimal_lambda)
        
        logger.info(f"Balanced optimization: lambda={optimal_lambda:.3f}, "
                   f"latency={final_latency:.1f}ms, quality={final_quality:.3f}")
        
        return OptimizationResult(
            optimal_lambda=optimal_lambda,
            achieved_latency=final_latency,
            achieved_quality=final_quality,
            constraint_satisfied=True,
            iterations=result.nit
        )


def find_optimal_lambda(
    pipeline,
    test_prompts: List[str],
    constraint_type: str = "latency",
    constraint_value: float = 1000.0,
    num_evaluations: int = 10
) -> float:
    """
    Convenience function to find optimal lambda for a pipeline
    
    Args:
        pipeline: Adaptive speculative decoding pipeline
        test_prompts: List of test prompts for evaluation
        constraint_type: "latency" or "quality"
        constraint_value: Constraint threshold
        num_evaluations: Number of evaluations per lambda
        
    Returns:
        optimal_lambda: Best lambda value
    """
    
    def evaluate_lambda(lam: float) -> Tuple[float, float]:
        """Evaluate pipeline performance for given lambda"""
        pipeline.lambda_value = lam
        
        latencies = []
        qualities = []
        
        for prompt in test_prompts[:num_evaluations]:
            result = pipeline.process_request(prompt)
            latencies.append(result["latency_ms"])
            
            # Simple quality estimate (can be improved with reference)
            output_length = len(result["output"].split())
            quality = min(1.0, output_length / 50.0)  # Rough heuristic
            qualities.append(quality)
        
        avg_latency = np.mean(latencies)
        avg_quality = np.mean(qualities)
        
        return avg_latency, avg_quality
    
    optimizer = LambdaOptimizer()
    
    if constraint_type == "latency":
        optimizer.latency_constraint = constraint_value
        result = optimizer.optimize_for_latency_constraint(evaluate_lambda)
    else:
        result = optimizer.find_balanced_lambda(evaluate_lambda)
    
    return result.optimal_lambda


class GridSearchOptimizer:
    """
    Grid search for lambda optimization with multiple metrics
    """
    
    def __init__(
        self,
        lambda_grid: Optional[List[float]] = None,
        metrics: List[str] = ["latency", "quality", "cost"]
    ):
        if lambda_grid is None:
            lambda_grid = np.logspace(-2, 2, 20).tolist()  # 0.01 to 100
        
        self.lambda_grid = lambda_grid
        self.metrics = metrics
    
    def search(
        self,
        pipeline,
        test_data: List[Dict],
        evaluation_func: Optional[Callable] = None
    ) -> Dict[str, List]:
        """
        Perform grid search over lambda values
        
        Args:
            pipeline: Pipeline to evaluate
            test_data: List of test samples
            evaluation_func: Custom evaluation function
            
        Returns:
            Dictionary with results for each lambda
        """
        results = {
            "lambda_values": [],
            "latencies": [],
            "qualities": [],
            "costs": [],
            "stage_distributions": []
        }
        
        for lam in self.lambda_grid:
            logger.info(f"Evaluating lambda = {lam:.3f}")
            
            pipeline.lambda_value = lam
            
            # Evaluate on test data
            latencies = []
            qualities = []
            costs = []
            stage_counts = [0] * 4
            
            for sample in test_data:
                if evaluation_func:
                    metrics = evaluation_func(pipeline, sample)
                else:
                    metrics = self._default_evaluation(pipeline, sample)
                
                latencies.append(metrics["latency"])
                qualities.append(metrics["quality"])
                costs.append(metrics["cost"])
                stage_counts[metrics["stopped_stage"]] += 1
            
            # Store results
            results["lambda_values"].append(lam)
            results["latencies"].append(np.mean(latencies))
            results["qualities"].append(np.mean(qualities))
            results["costs"].append(np.mean(costs))
            results["stage_distributions"].append(stage_counts)
            
            logger.info(f"  Avg latency: {np.mean(latencies):.1f}ms")
            logger.info(f"  Avg quality: {np.mean(qualities):.3f}")
            logger.info(f"  Stage dist: {stage_counts}")
        
        return results
    
    def _default_evaluation(self, pipeline, sample: Dict) -> Dict:
        """Default evaluation function"""
        prompt = sample["prompt"]
        result = pipeline.process_request(prompt)
        
        # Compute quality (placeholder - should use reference)
        quality = 1.0  # Default quality
        if "reference" in sample:
            # Could compute BLEU/ROUGE here
            pass
        
        return {
            "latency": result["latency_ms"],
            "quality": quality,
            "cost": sum(result["costs"]),
            "stopped_stage": result["stopped_at_stage"]
        }