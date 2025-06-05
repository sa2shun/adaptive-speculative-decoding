"""
Theoretical foundations for adaptive speculative decoding.

This module provides the mathematical framework and theoretical guarantees
for our hierarchical optimal stopping approach.
"""

from .optimal_stopping import OptimalStoppingTheory, RegretAnalyzer
from .regret_bounds import derive_regret_bound, compute_sample_complexity
from .convergence import ConvergenceAnalysis, TheoreticalGuarantees

__all__ = [
    "OptimalStoppingTheory",
    "RegretAnalyzer", 
    "derive_regret_bound",
    "compute_sample_complexity",
    "ConvergenceAnalysis",
    "TheoreticalGuarantees",
]