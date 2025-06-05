"""
Dynamic Programming solver for optimal stopping rule
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def optimal_stopping_rule(
    p: List[float],
    C: List[float],
    lam: float,
    risk_adjustment: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[int, List[float]]:
    """
    Compute optimal stopping stage using dynamic programming
    
    Args:
        p: List of acceptance probabilities for each stage
        C: List of costs for each stage
        lam: Quality weight parameter (lambda)
        risk_adjustment: Whether to apply Bayesian risk adjustment
        alpha, beta: Beta distribution parameters for risk adjustment
        
    Returns:
        k_star: Optimal stopping stage (0-indexed)
        J: List of expected costs from each stage
    """
    if len(p) != len(C):
        raise ValueError("p and C must have the same length")
    
    L = len(C)
    
    # Apply risk adjustment if requested
    if risk_adjustment:
        p = [bayesian_adjustment(pi, n_obs=100, alpha=alpha, beta=beta) for pi in p]
    
    # Precompute cumulative probabilities
    p_bar = [1.0]
    for i in range(L):
        p_bar.append(p_bar[-1] * p[i])
    
    # Initialize DP table
    J = [0.0] * (L + 1)  # J[L] = 0 (boundary condition)
    stop_decision = [False] * L
    
    # Backward induction
    for i in reversed(range(L)):
        # Cost if we stop at stage i
        cost_if_stop = C[i] + lam * (1 - p_bar[i + 1])
        
        # Cost if we continue to next stage
        cost_if_continue = C[i] + J[i + 1]
        
        # Make optimal decision
        if cost_if_stop <= cost_if_continue:
            stop_decision[i] = True
            J[i] = cost_if_stop
        else:
            stop_decision[i] = False
            J[i] = cost_if_continue
    
    # Find first stopping stage
    k_star = next((i for i, should_stop in enumerate(stop_decision) if should_stop), L - 1)
    
    return k_star, J


def compute_expected_cost(
    p: List[float],
    C: List[float],
    lam: float,
    stopping_stage: int
) -> float:
    """
    Compute expected cost for a given stopping strategy
    
    Args:
        p: List of acceptance probabilities
        C: List of costs
        lam: Quality weight
        stopping_stage: Stage at which to stop (0-indexed)
        
    Returns:
        expected_cost: Expected cost of the strategy
    """
    # Cumulative probability up to stopping stage
    p_bar = 1.0
    for i in range(stopping_stage + 1):
        p_bar *= p[i]
    
    # Expected computation cost
    computation_cost = sum(C[:stopping_stage + 1])
    
    # Expected quality loss
    quality_loss = lam * (1 - p_bar)
    
    return computation_cost + quality_loss


def bayesian_adjustment(
    p_hat: float,
    n_obs: int,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Apply Bayesian risk adjustment using Beta prior
    
    Args:
        p_hat: Observed probability estimate
        n_obs: Number of observations
        alpha, beta: Beta prior parameters
        
    Returns:
        adjusted_p: Risk-adjusted probability
    """
    # Beta-Binomial conjugate update
    posterior_alpha = n_obs * p_hat + alpha
    posterior_beta = n_obs * (1 - p_hat) + beta
    
    # Posterior mean
    adjusted_p = posterior_alpha / (posterior_alpha + posterior_beta)
    
    return adjusted_p


class OptimalStoppingTable:
    """
    Precomputed table for optimal stopping decisions
    Speeds up online inference by avoiding repeated DP computations
    """
    
    def __init__(
        self,
        lambda_values: List[float],
        num_stages: int = 4
    ):
        self.lambda_values = lambda_values
        self.num_stages = num_stages
        self.table = {}
        
    def precompute(
        self,
        cost_ratios: List[float],
        prob_grid: List[List[float]]
    ):
        """
        Precompute stopping decisions for various probability scenarios
        
        Args:
            cost_ratios: Relative costs of each stage
            prob_grid: Grid of probability values to consider
        """
        logger.info("Precomputing optimal stopping table...")
        
        for lam in self.lambda_values:
            self.table[lam] = {}
            
            for prob_scenario in prob_grid:
                k_star, _ = optimal_stopping_rule(
                    prob_scenario, cost_ratios, lam
                )
                
                # Use rounded probabilities as key
                prob_key = tuple(round(p, 2) for p in prob_scenario)
                self.table[lam][prob_key] = k_star
        
        logger.info(f"Precomputed table for {len(self.lambda_values)} lambda values")
    
    def lookup(
        self,
        probabilities: List[float],
        lambda_value: float,
        fallback_to_dp: bool = True
    ) -> int:
        """
        Fast lookup of optimal stopping stage
        
        Args:
            probabilities: Current probability estimates
            lambda_value: Current lambda value
            fallback_to_dp: Whether to compute DP if not in table
            
        Returns:
            k_star: Optimal stopping stage
        """
        # Find closest lambda value
        closest_lam = min(self.lambda_values, key=lambda x: abs(x - lambda_value))
        
        # Round probabilities for lookup
        prob_key = tuple(round(p, 2) for p in probabilities)
        
        if closest_lam in self.table and prob_key in self.table[closest_lam]:
            return self.table[closest_lam][prob_key]
        
        # Fallback to DP computation
        if fallback_to_dp:
            logger.debug(f"Table miss for lambda={lambda_value}, computing DP")
            cost_ratios = [1.0, 1.6, 4.2, 8.8][:len(probabilities)]
            k_star, _ = optimal_stopping_rule(probabilities, cost_ratios, lambda_value)
            return k_star
        
        # Default to continuing
        return len(probabilities) - 1


class AdaptiveStopping:
    """
    Adaptive stopping with online learning and confidence bounds
    """
    
    def __init__(
        self,
        initial_lambda: float = 1.0,
        confidence_level: float = 0.1
    ):
        self.lambda_value = initial_lambda
        self.confidence_level = confidence_level
        
        # Online statistics
        self.stage_counts = np.zeros(4)
        self.stage_rewards = np.zeros(4)
        self.total_steps = 0
        
    def update_statistics(
        self,
        chosen_stage: int,
        observed_quality: float,
        observed_latency: float
    ):
        """
        Update online statistics based on observed outcomes
        
        Args:
            chosen_stage: Stage where we stopped
            observed_quality: Quality of the output (0-1)
            observed_latency: Latency in milliseconds
        """
        self.stage_counts[chosen_stage] += 1
        
        # Compute reward (quality - normalized latency cost)
        normalized_latency = observed_latency / 1000.0  # Convert to seconds
        reward = observed_quality - self.lambda_value * normalized_latency
        
        # Update running average
        n = self.stage_counts[chosen_stage]
        self.stage_rewards[chosen_stage] = (
            (n - 1) * self.stage_rewards[chosen_stage] + reward
        ) / n
        
        self.total_steps += 1
    
    def get_confidence_bounds(self, stage: int) -> Tuple[float, float]:
        """
        Compute confidence bounds for stage performance
        Using Hoeffding's inequality
        """
        n = self.stage_counts[stage]
        if n == 0:
            return -np.inf, np.inf
        
        # Confidence radius
        confidence_radius = np.sqrt(-np.log(self.confidence_level / 2) / (2 * n))
        
        mean_reward = self.stage_rewards[stage]
        lower_bound = mean_reward - confidence_radius
        upper_bound = mean_reward + confidence_radius
        
        return lower_bound, upper_bound
    
    def should_explore(self, stage: int) -> bool:
        """
        Determine if we should explore a particular stage
        Based on upper confidence bounds
        """
        if self.stage_counts[stage] < 10:  # Minimum exploration
            return True
        
        # Check if this stage has competitive upper bound
        upper_bounds = [self.get_confidence_bounds(i)[1] for i in range(4)]
        stage_upper = upper_bounds[stage]
        
        # Explore if this stage could be optimal
        return stage_upper >= max(upper_bounds) - 0.1