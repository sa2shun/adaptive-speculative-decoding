"""
Regret bounds and sample complexity analysis.

This module provides theoretical guarantees for our adaptive stopping algorithm.
"""

import numpy as np
from typing import Tuple, Dict
import scipy.special as special


def derive_regret_bound(n_stages: int, 
                       T: int,
                       quality_gaps: np.ndarray,
                       confidence: float = 0.95) -> Dict[str, float]:
    """
    Derive regret bound for hierarchical optimal stopping.
    
    Based on UCB-style analysis extended to sequential decisions.
    
    Args:
        n_stages: Number of model stages
        T: Time horizon (number of rounds)
        quality_gaps: Gaps between optimal and suboptimal qualities
        confidence: Confidence level for bound
        
    Returns:
        Dictionary with regret bounds and related quantities
    """
    # Minimum quality gap
    Delta_min = np.min(quality_gaps[quality_gaps > 0])
    
    # UCB exploration constant
    alpha = 2 * np.log(1 / (1 - confidence))
    
    # Problem-dependent bound (Theorem 3.1)
    C_problem = 8 * np.sum(1 / quality_gaps[quality_gaps > 0])
    regret_problem_dependent = C_problem * np.log(T) + alpha * n_stages
    
    # Problem-independent bound (Theorem 3.2)  
    C_independent = 2 * np.sqrt(2 * n_stages)
    regret_problem_independent = C_independent * np.sqrt(T * np.log(T))
    
    # High-probability bound (Theorem 3.3)
    delta = 1 - confidence
    regret_high_prob = (regret_problem_independent + 
                       np.sqrt(2 * np.log(1/delta) * T))
    
    # Minimax bound
    regret_minimax = np.sqrt(n_stages * T)
    
    return {
        "problem_dependent": regret_problem_dependent,
        "problem_independent": regret_problem_independent,
        "high_probability": regret_high_prob,
        "minimax": regret_minimax,
        "Delta_min": Delta_min,
        "exploration_constant": alpha
    }


def compute_sample_complexity(epsilon: float,
                            delta: float, 
                            n_stages: int,
                            quality_variance: float = 0.1) -> Dict[str, int]:
    """
    Compute sample complexity for ε-optimal policy.
    
    Theorem 4: Sample complexity for (ε,δ)-PAC learning of optimal thresholds.
    
    Args:
        epsilon: Accuracy parameter
        delta: Failure probability
        n_stages: Number of stages
        quality_variance: Variance in quality estimates
        
    Returns:
        Sample complexities under different assumptions
    """
    # Basic Hoeffding-based bound
    n_hoeffding = int(np.ceil(
        2 * quality_variance * np.log(2 * n_stages / delta) / (epsilon ** 2)
    ))
    
    # Bernstein-based bound (tighter for small variance)
    n_bernstein = int(np.ceil(
        2 * quality_variance * np.log(3 * n_stages / delta) / 
        (epsilon * (epsilon - quality_variance/3))
    ))
    
    # Median-of-means bound (robust to outliers)
    k_blocks = int(np.ceil(8 * np.log(2 / delta)))
    n_median = k_blocks * int(np.ceil(32 * quality_variance / (epsilon ** 2)))
    
    # Information-theoretic lower bound
    n_lower = int(np.ceil(
        quality_variance * np.log(1 / delta) / (2 * epsilon ** 2)
    ))
    
    return {
        "hoeffding": n_hoeffding,
        "bernstein": n_bernstein,
        "median_of_means": n_median,
        "lower_bound": n_lower,
        "recommended": max(n_hoeffding, n_lower)
    }


def concentration_inequality(n_samples: int,
                           confidence: float,
                           range_bound: float = 1.0) -> float:
    """
    Compute concentration bound for empirical quality estimates.
    
    Using McDiarmid's inequality for bounded differences.
    
    Args:
        n_samples: Number of samples
        confidence: Desired confidence level
        range_bound: Range of quality scores
        
    Returns:
        Width of confidence interval
    """
    delta = 1 - confidence
    width = range_bound * np.sqrt(np.log(2 / delta) / (2 * n_samples))
    return width


def martingale_concentration(T: int,
                           variance_bound: float,
                           confidence: float) -> float:
    """
    Azuma-Hoeffding bound for martingale sequences.
    
    Used for online regret analysis.
    
    Args:
        T: Time horizon
        variance_bound: Bound on conditional variance
        confidence: Confidence level
        
    Returns:
        Concentration bound
    """
    delta = 1 - confidence
    bound = np.sqrt(2 * variance_bound * T * np.log(1 / delta))
    return bound


def finite_sample_bound(n_stages: int,
                       n_samples: int,
                       delta: float) -> Tuple[float, float]:
    """
    Finite-sample generalization bound.
    
    Combines Rademacher complexity with union bound over stages.
    
    Args:
        n_stages: Number of model stages  
        n_samples: Number of training samples
        delta: Failure probability
        
    Returns:
        Tuple of (generalization_gap, confidence_radius)
    """
    # Rademacher complexity for linear threshold functions
    rademacher = np.sqrt(2 * np.log(2 * n_stages) / n_samples)
    
    # Union bound correction
    union_correction = np.sqrt(np.log(n_stages / delta) / (2 * n_samples))
    
    generalization_gap = 2 * rademacher + union_correction
    confidence_radius = 3 * union_correction
    
    return generalization_gap, confidence_radius


class TheoreticalAnalysis:
    """Complete theoretical analysis of the algorithm."""
    
    @staticmethod
    def main_theorem() -> str:
        """Main theoretical result."""
        return r"""
        \begin{theorem}[Main Result]
        For the hierarchical inference problem with $n$ stages, our adaptive
        stopping algorithm achieves:
        
        1. \textbf{Regret Bound}: $R(T) \leq O(\sqrt{nT\log T})$
        
        2. \textbf{Sample Complexity}: $O\left(\frac{n}{\epsilon^2}\log\frac{n}{\delta}\right)$
           samples suffice for $(\epsilon,\delta)$-optimal policy
           
        3. \textbf{Computational Complexity}: $O(n)$ per decision
        
        4. \textbf{Optimality Gap}: The policy is within factor $(1+\epsilon)$ of optimal
           with probability $1-\delta$
        \end{theorem}
        """
    
    @staticmethod  
    def lower_bound() -> str:
        """Information-theoretic lower bound."""
        return r"""
        \begin{theorem}[Lower Bound]
        Any algorithm for hierarchical inference must incur regret
        $R(T) \geq \Omega(\sqrt{nT})$ in the worst case.
        
        This matches our upper bound up to logarithmic factors.
        \end{theorem}
        """